"""Define the creative writing agent graph structure.

Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-02-11 22:32:58
Current User's Login: fortunestoldco
"""

from typing import Dict, List, Optional, Callable, Any, Annotated, TypeVar, cast
import os
import json
from datetime import datetime
from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent
from langchain.agents.format_scratchpad import format_to_openai_functions
from langgraph.graph import StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from pydantic import BaseModel, Field

from src.team.configuration import (
    State, Configuration, StoryParameters, 
    MarketResearch, MessageWrapper
)
from src.team.prompts import (
    RESEARCH_SYSTEM_PROMPT,
    MARKET_ANALYST_PROMPT,
    REVIEW_ANALYST_PROMPT,
    WRITING_SYSTEM_PROMPT,
    DOC_WRITER_PROMPT,
    NOTE_TAKER_PROMPT,
)
from src.team.tools import RESEARCH_TOOLS, WRITING_TOOLS

# Define input types
class StoryInput(BaseModel):
    """Input parameters for the story creation process."""
    starting_point: str = Field(
        ...,
        description="The initial situation or scene that starts the story"
    )
    plot_points: List[str] = Field(
        ...,
        description="Key plot points that must be included in the story"
    )
    intended_ending: str = Field(
        ...,
        description="The desired conclusion of the story"
    )

def create_initial_state(story_input: Dict) -> State:
    """Create the initial state from the story input."""
    story_parameters = StoryParameters(
        starting_point=story_input["starting_point"],
        plot_points=story_input["plot_points"],
        intended_ending=story_input["intended_ending"]
    )
    
    initial_message = SystemMessage(
        content=f"Beginning story development with parameters:\n{story_parameters.to_prompt()}"
    )
    
    return State(
        messages=[MessageWrapper.from_message(initial_message)],
        story_parameters=story_parameters,
        market_research=MarketResearch(),
        next=""
    )

def create_agent_node(
    llm: ChatOpenAI,
    tools: List,
    system_prompt: str,
    name: str,
    config: Configuration
) -> Callable:
    """Create an agent node that can use tools."""
    tool_executor = ToolExecutor(tools=tools)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)

    def run_agent(state: State) -> Dict:
        """Execute agent with tools."""
        message_history = config.get_message_history(state.session_id)
        
        context = []
        if state.story_parameters:
            context.append(SystemMessage(content=state.story_parameters.to_prompt()))
        if state.market_research and state.market_research.similar_books:
            context.append(SystemMessage(content=
                f"Market Research:\n{json.dumps(state.market_research.dict(), indent=2)}"
            ))
        
        agent_input = {
            "history": message_history.messages + context,
            "messages": state.get_messages(),
            "agent_scratchpad": format_to_openai_functions([])
        }
        
        output = agent.invoke(agent_input)
        response = output.return_values["output"]
        
        message_history.add_ai_message(AIMessage(content=response))
        
        return {
            "messages": state.messages + [MessageWrapper.from_message(AIMessage(content=response))],
            "next": "supervisor"
        }
    
    return run_agent

def create_supervisor_node(
    llm: ChatOpenAI,
    team_members: List[str],
    config: Configuration,
    name: str = "supervisor"
) -> Callable:
    """Create a supervisor node that manages team members."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""You are the {name} coordinating the following team: {', '.join(team_members)}.
Your responsibilities:
1. Review current progress and team outputs
2. Determine next steps based on the story requirements
3. Ensure all plot points are being addressed
4. Maintain consistency with target demographic
5. Coordinate between research and writing teams

For research phase:
- Ensure market analysis is thorough
- Verify demographic targeting is clear
- Check that improvement opportunities are identified

For writing phase:
- Verify plot points are being incorporated
- Ensure tone matches target demographic
- Check that market research insights are being used

Respond with next team member to act or 'FINISH' if complete."""),
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="messages")
    ])

    def decide_next(state: State) -> Dict:
        """Route between team members."""
        message_history = config.get_message_history(state.session_id)
        
        context_messages = []
        if state.story_parameters:
            context_messages.append(SystemMessage(content=
                f"Story Parameters:\n{state.story_parameters.to_prompt()}"
            ))
        if state.market_research and state.market_research.similar_books:
            context_messages.append(SystemMessage(content=
                f"Market Research:\n{json.dumps(state.market_research.dict(), indent=2)}"
            ))
        
        formatted_prompt = prompt.format_messages(
            history=message_history.messages + context_messages,
            messages=state.get_messages()
        )
        
        response = llm.invoke(formatted_prompt)
        next_step = response.content.strip()
        
        message_history.add_ai_message(response)
        
        return {
            "messages": state.messages + [MessageWrapper.from_message(response)],
            "next": next_step
        }
    
    return decide_next

def create_graph(config: Optional[Configuration] = None) -> StateGraph:
    """Create the creative writing agent graph."""
    if config is None:
        config = Configuration()

    # Create the workflow
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("start", create_initial_state)
    
    # Create LLM
    llm = ChatOpenAI(
        model=config.model,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create research team nodes
    research_agents = {
        "market_researcher": create_agent_node(
            llm, RESEARCH_TOOLS, MARKET_ANALYST_PROMPT, "market_researcher", config
        ),
        "review_analyst": create_agent_node(
            llm, RESEARCH_TOOLS, REVIEW_ANALYST_PROMPT, "review_analyst", config
        )
    }
    
    # Create writing team nodes
    writing_agents = {
        "writer": create_agent_node(
            llm, WRITING_TOOLS, DOC_WRITER_PROMPT, "writer", config
        ),
        "editor": create_agent_node(
            llm, WRITING_TOOLS, NOTE_TAKER_PROMPT, "editor", config
        )
    }

    # Add all agent nodes
    for name, agent in {**research_agents, **writing_agents}.items():
        workflow.add_node(name, agent)

    # Create and add supervisors
    research_supervisor = create_supervisor_node(
        llm, list(research_agents.keys()), config, "research_supervisor"
    )
    writing_supervisor = create_supervisor_node(
        llm, list(writing_agents.keys()), config, "writing_supervisor"
    )
    top_supervisor = create_supervisor_node(
        llm, ["research_team", "writing_team"], config, "top_supervisor"
    )

    workflow.add_node("research_supervisor", research_supervisor)
    workflow.add_node("writing_supervisor", writing_supervisor)
    workflow.add_node("supervisor", top_supervisor)

    # Add conditional edges using state's next field
    workflow.add_edge("start", "supervisor")
    
    # Set up agent routing
    def route_agents(state: State) -> List[str]:
        next_step = state.next
        if next_step == "FINISH":
            return []
        return [next_step] if next_step in {**research_agents, **writing_agents} else []

    def route_supervisors(state: State) -> List[str]:
        next_step = state.next
        if next_step == "FINISH":
            return []
        return [next_step] if next_step in ["research_supervisor", "writing_supervisor", "supervisor"] else []

    # Add conditional edges
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisors,
        ["research_supervisor", "writing_supervisor"]
    )

    workflow.add_conditional_edges(
        "research_supervisor",
        route_agents,
        list(research_agents.keys())
    )

    workflow.add_conditional_edges(
        "writing_supervisor",
        route_agents,
        list(writing_agents.keys())
    )

    # Add return edges to supervisors
    for agent in research_agents:
        workflow.add_edge(agent, "research_supervisor")
    
    for agent in writing_agents:
        workflow.add_edge(agent, "writing_supervisor")

    workflow.add_edge("research_supervisor", "supervisor")
    workflow.add_edge("writing_supervisor", "supervisor")

    # Set entry point
    workflow.set_entry_point("start")

    return workflow.compile()

# Create the graph instance
graph = create_graph()
