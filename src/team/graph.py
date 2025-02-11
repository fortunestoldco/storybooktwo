"""Define the creative writing agent graph structure.

Current Date and Time (UTC): 2025-02-11 22:00:07
Current User's Login: fortunestoldco
"""

from typing import Dict, List, Optional
import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent
from langchain.agents.format_scratchpad import format_to_openai_functions
from langgraph.graph import StateGraph, Graph, START

from src.team.configuration import State, Configuration, StoryParameters, MarketResearch
from src.team.prompts import (
    RESEARCH_SYSTEM_PROMPT,
    MARKET_ANALYST_PROMPT,
    REVIEW_ANALYST_PROMPT,
    WRITING_SYSTEM_PROMPT,
    DOC_WRITER_PROMPT,
    NOTE_TAKER_PROMPT,
)
from src.team.tools import RESEARCH_TOOLS, WRITING_TOOLS

def create_agent_node(
    llm: ChatOpenAI,
    tools: List,
    system_prompt: str,
    name: str,
    config: Configuration
):
    """Create an agent node that can use tools."""
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
                "Market Research:\n" + json.dumps(state.market_research.__dict__, indent=2)
            ))
        
        agent_input = {
            "history": message_history.messages + context,
            "messages": state.get("messages", []),
            "agent_scratchpad": format_to_openai_functions([])
        }
        
        output = agent.invoke(agent_input)
        response = output.return_values["output"]
        
        message_history.add_ai_message(SystemMessage(content=response))
        
        return {
            "messages": state.get("messages", []) + [SystemMessage(content=response)],
            "next": "supervisor"
        }
    
    return agent_node

def create_supervisor_node(
    llm: ChatOpenAI,
    team_members: List[str],
    config: Configuration,
    name: str = "supervisor"
):
    """Create a supervisor node."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""You are the {name} managing: {', '.join(team_members)}.
Based on the current state of the story development:
1. Review the research team's findings
2. Guide the writing team in implementing improvements
3. Ensure all required plot points are being addressed
4. Maintain consistency with target demographic preferences
5. Decide which team member should act next

Respond with their name or 'FINISH' if the task is complete."""),
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    def supervisor(state: State) -> Dict:
        """Route between team members."""
        message_history = config.get_message_history(state.session_id)
        
        formatted_prompt = prompt.format_messages(
            history=message_history.messages,
            messages=state.get("messages", [])
        )
        
        response = llm.invoke(formatted_prompt)
        next_step = response.content.strip()
        
        message_history.add_ai_message(response)
        
        return {
            "messages": state.get("messages", []) + [response],
            "next": next_step
        }
    
    return supervisor

def create_graph(config: Optional[Configuration] = None) -> StateGraph:
    """Create the creative writing agent graph."""
    if config is None:
        config = Configuration()

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

    # Create main workflow
    workflow = StateGraph(State)
    
    # Add nodes
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
        return {"messages": state.get("messages", []), "next": "END"}
    
    workflow.add_node("end", end_node)
    
    # Add edges
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
    workflow.set_entry_point("supervisor")
    
    # Compile the graph
    final_graph = workflow.compile()
    final_graph.name = "Creative Writing Agent"
    
    return final_graph

# Create the graph instance
graph = create_graph()
