"""Define the creative writing agent graph structure.

Current Date and Time (UTC): 2025-02-11 22:20:21
Current User's Login: fortunestoldco
"""

from typing import Dict, List, Optional, Callable, Any
import os
import json
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent
from langchain.agents.format_scratchpad import format_to_openai_functions
from langgraph.graph import StateGraph, Graph, START
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

def create_initial_state(story_input: StoryInput) -> State:
    """Create the initial state from the story input."""
    story_parameters = StoryParameters(
        starting_point=story_input.starting_point,
        plot_points=story_input.plot_points,
        intended_ending=story_input.intended_ending
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

def create_supervisor_node(
    llm: ChatOpenAI,
    team_members: List[str],
    config: Configuration,
    name: str = "supervisor"
) -> Callable:
    """Create a supervisor node that manages team members.
    
    Args:
        llm: Language model to use
        team_members: List of team member names to supervise
        config: Configuration instance
        name: Name of this supervisor
        
    Returns:
        Callable that processes state and returns next steps
    """
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
    
    def supervisor(state: State) -> Dict:
        """Route between team members."""
        message_history = config.get_message_history(state.session_id)
        
        # Include context about story parameters and research
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
    
    return supervisor

def create_team_graph(
    supervisor_node: Callable,
    team_nodes: Dict[str, Callable],
    config: Configuration,
    name: str
) -> Graph:
    """Create a team graph with supervisor and team members.
    
    Args:
        supervisor_node: The supervisor node function
        team_nodes: Dictionary of team member node functions
        config: Configuration instance
        name: Name of this team graph
        
    Returns:
        Compiled team graph
    """
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    for node_name, node_fn in team_nodes.items():
        workflow.add_node(node_name, node_fn)
    
    # Add end node
    def end_node(state: State) -> Dict:
        """End state node."""
        return {
            "messages": state.messages,
            "next": "END"
        }
    
    workflow.add_node("end", end_node)
    
    # Add edges
    for node_name in team_nodes:
        workflow.add_edge("supervisor", node_name)
        workflow.add_edge(node_name, "supervisor")
    
    # Add conditional edges from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: "end" if x["next"] == "FINISH" else x["next"],
        {**{name: name for name in team_nodes.keys()}, "end": "end"}
    )
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    return workflow.compile()

def create_agent_node(
    llm: ChatOpenAI,
    tools: List,
    system_prompt: str,
    name: str,
    config: Configuration
) -> Callable:
    """Create an agent node that can use tools.
    
    Args:
        llm: Language model to use
        tools: List of tools the agent can use
        system_prompt: System prompt for the agent
        name: Name of this agent
        config: Configuration instance
        
    Returns:
        Callable that processes state and returns next steps
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    def agent_node(state: State) -> Dict:
        """Execute agent with tools."""
        message_history = config.get_message_history(state.session_id)
        
        # Include story parameters and research in context
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
    
    return agent_node

def create_graph(config: Optional[Configuration] = None) -> StateGraph:
    """Create the creative writing agent graph.
    
    Args:
        config: Optional configuration instance
        
    Returns:
        Compiled state graph
    """
    if config is None:
        config = Configuration()

    workflow = StateGraph(State)
    
    # Add input validator
    def validate_and_initialize(story_input: StoryInput) -> State:
        """Validate input and create initial state."""
        return create_initial_state(story_input)
    
    workflow.add_node("start", validate_and_initialize)
    
    # Create teams and supervisors
    llm = ChatOpenAI(
        model=config.model,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create research team
    research_agents = {
        "market_researcher": create_agent_node(
            llm, RESEARCH_TOOLS, MARKET_ANALYST_PROMPT, "market_researcher", config
        ),
        "review_analyst": create_agent_node(
            llm, RESEARCH_TOOLS, REVIEW_ANALYST_PROMPT, "review_analyst", config
        )
    }
    
    research_supervisor = create_supervisor_node(
        llm, list(research_agents.keys()), config, "research_supervisor"
    )
    
    # Create writing team
    writing_agents = {
        "writer": create_agent_node(
            llm, WRITING_TOOLS, DOC_WRITER_PROMPT, "writer", config
        ),
        "editor": create_agent_node(
            llm, WRITING_TOOLS, NOTE_TAKER_PROMPT, "editor", config
        )
    }
    
    writing_supervisor = create_supervisor_node(
        llm, list(writing_agents.keys()), config, "writing_supervisor"
    )

    # Add team nodes
    workflow.add_node("research_team", create_team_graph(
        research_supervisor, research_agents, config, "research_team"
    ))
    workflow.add_node("writing_team", create_team_graph(
        writing_supervisor, writing_agents, config, "writing_team"
    ))
    
    # Create top-level supervisor
    top_supervisor = create_supervisor_node(
        llm, ["research_team", "writing_team"], config, "top_supervisor"
    )
    workflow.add_node("supervisor", top_supervisor)
    
    # Add end node
    def end_node(state: State) -> Dict:
        """End state node."""
        return {
            "messages": state.messages,
            "next": "END"
        }
    
    workflow.add_node("end", end_node)
    
    # Add edges
    workflow.add_edge("start", "supervisor")
    workflow.add_edge("supervisor", "research_team")
    workflow.add_edge("supervisor", "writing_team")
    workflow.add_edge("research_team", "supervisor")
    workflow.add_edge("writing_team", "supervisor")
    
    # Set conditional edges
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: "end" if x["next"] == "FINISH" else x["next"],
        {
            "research_team": "research_team", 
            "writing_team": "writing_team",
            "end": "end"
        }
    )
    
    # Set entry point
    workflow.set_entry_point("start")
    
    # Set input type
    workflow.set_input_type(StoryInput)
    
    # Compile the graph
    final_graph = workflow.compile()
    final_graph.name = "Creative Writing Agent"
    
    return final_graph

# Create the graph instance
graph = create_graph()
