"""Graph definition for the creative writing agent system.

Current Date and Time (UTC): 2025-02-12 00:34:16
Current User's Login: fortunestoldco
"""

import os
import json
from typing import Dict, Optional, Any, Callable
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from team.configuration import Configuration, State, StoryState, StoryDict
from team.state import TeamState
from team.prompts import (
    MARKET_RESEARCH_PROMPT,
    AUDIENCE_RESEARCH_PROMPT,
    WRITER_PROMPT,
    NOTE_TAKER_PROMPT,
)
from team.tools import (
    tavily_tool,
    scrape_webpages,
    write_document,
    edit_document,
    read_document,
    create_outline
)

def create_agent_node(
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str,
    name: str,
    config: Configuration
) -> Callable[[State], StoryState]:
    """Create an agent node with proper LangSmith instrumentation."""
    def agent_node(state: State) -> StoryState:
        run_metadata = config.create_run_metadata(state)
        
        with config.langsmith_client.tracing(
            project_name=config.langsmith_project,
            metadata=run_metadata,
            tags=["agent_node", name]
        ) as run:
            messages = state.get("messages", [])
            story_params = state.get("story_parameters", {})
            
            # Add story parameters to system prompt
            enhanced_prompt = f"{system_prompt}\n\nStory Parameters:\n"
            enhanced_prompt += f"Starting Point: {story_params.get('start', '')}\n"
            enhanced_prompt += f"Key Plot Points: {', '.join(story_params.get('plot_points', []))}\n"
            enhanced_prompt += f"Intended Ending: {story_params.get('ending', '')}\n"
            
            if state.get("research_data"):
                enhanced_prompt += f"\nResearch Data:\n{json.dumps(state.research_data, indent=2)}"
            
            result = llm.invoke([
                {"role": "system", "content": enhanced_prompt},
                *messages
            ])
            
            return state.to_dict() | {
                "messages": [{"role": "assistant", "content": result.content}],
                "next": "supervisor"
            }
            
    return agent_node

def create_supervisor_node(
    llm: ChatOpenAI,
    team_members: list,
    config: Configuration,
    name: str
) -> Callable[[State], StoryState]:
    """Create a supervisor node that manages the story creation workflow."""
    def supervisor_node(state: State) -> StoryState:
        state.ensure_story_parameters()
        run_metadata = config.create_run_metadata(state)
        
        with config.langsmith_client.tracing(
            project_name=config.langsmith_project,
            metadata=run_metadata,
            tags=["supervisor_node", name]
        ) as run:
            messages = state.get("messages", [])
            current_phase = state.input_parameters.get("current_phase", "market_research")
            research_data = state.get("research_data", {})
            story_params = state.get("story_parameters", {})
            
            system_message = f"""You are supervising a creative writing team.
Current phase: {current_phase}
Available team members: {', '.join(team_members)}

Story Parameters:
Start: {story_params.get('start', '')}
Plot Points: {', '.join(story_params.get('plot_points', []))}
Ending: {story_params.get('ending', '')}

Workflow phases:
1. Market Research - Find similar novels and their reviews
2. Audience Research - Determine and analyze target demographic
3. Market Analysis - Analyze improvement opportunities
4. Writing - Create the story based on research

Current research data:
{json.dumps(research_data, indent=2)}

Decide next action or FINISH if the story is complete."""

            result = llm.invoke([
                {"role": "system", "content": system_message},
                *messages
            ])
            next_step = result.content.strip()
            
            # Update phase if moving to a new team
            if next_step == "writer" and current_phase != "writing":
                state.input_parameters["current_phase"] = "writing"
            
            return state.to_dict() | {
                "messages": [{"role": "assistant", "content": result.content}],
                "next": next_step
            }
            
    return supervisor_node

def create_team_graph(
    supervisor: Callable,
    agents: Dict[str, Callable],
    config: Configuration,
    name: str
) -> StateGraph:
    """Create a team graph with proper LangSmith instrumentation."""
    workflow = StateGraph(TeamState)
    
    # Add supervisor and agent nodes
    workflow.add_node("supervisor", supervisor)
    for agent_name, agent in agents.items():
        workflow.add_node(agent_name, agent)
    
    # Add end state node
    def end_node(state: TeamState) -> StoryState:
        """End state for team graph."""
        return state.to_dict() | {"next": "FINISH"}
    
    workflow.add_node("end", end_node)
    
    # Add edges between nodes
    for agent_name in agents:
        workflow.add_edge(agent_name, "supervisor")
    
    # Set conditional edges for supervisor
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            **{name: name for name in agents.keys()},
            "FINISH": "end"
        }
    )
    
    workflow.set_entry_point("supervisor")
    
    return workflow.compile()

def create_graph(config: Optional[Configuration] = None) -> StateGraph:
    """Create the complete creative writing graph with LangSmith instrumentation."""
    if config is None:
        config = Configuration()

    llm = ChatOpenAI(
        model=config.model,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create research team
    research_agents = {
        "market_researcher": create_agent_node(
            llm, 
            [tavily_tool, scrape_webpages], 
            MARKET_RESEARCH_PROMPT, 
            "market_researcher", 
            config
        ),
        "audience_researcher": create_agent_node(
            llm, 
            [tavily_tool, scrape_webpages], 
            AUDIENCE_RESEARCH_PROMPT, 
            "audience_researcher", 
            config
        )
    }
    research_supervisor = create_supervisor_node(
        llm, 
        list(research_agents.keys()), 
        config, 
        "research_supervisor"
    )
    research_team = create_team_graph(
        research_supervisor,
        research_agents,
        config,
        "research_team"
    )

    # Create writing team
    writing_agents = {
        "writer": create_agent_node(
            llm, 
            [write_document, edit_document, read_document], 
            WRITER_PROMPT, 
            "writer", 
            config
        ),
        "note_taker": create_agent_node(
            llm, 
            [create_outline, read_document], 
            NOTE_TAKER_PROMPT, 
            "note_taker", 
            config
        )
    }
    writing_supervisor = create_supervisor_node(
        llm, 
        list(writing_agents.keys()), 
        config, 
        "writing_supervisor"
    )
    writing_team = create_team_graph(
        writing_supervisor,
        writing_agents,
        config,
        "writing_team"
    )

    # Create main workflow graph
    workflow = StateGraph(State)
    
    # Add teams as nodes
    workflow.add_node("research_team", research_team)
    workflow.add_node("writing_team", writing_team)
    
    # Create top-level supervisor
    top_supervisor = create_supervisor_node(
        llm, 
        ["research_team", "writing_team"], 
        config, 
        "top_supervisor"
    )
    workflow.add_node("supervisor", top_supervisor)
    
    # Add end state node
    def end_node(state: State) -> StoryState:
        return state.to_dict() | {"next": "END"}
    
    workflow.add_node("end", end_node)
    
    # Add edges
    workflow.add_edge("supervisor", "research_team")
    workflow.add_edge("supervisor", "writing_team")
    workflow.add_edge("research_team", "supervisor")
    workflow.add_edge("writing_team", "supervisor")
    
    # Set conditional edges
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "research_team": "research_team", 
            "writing_team": "writing_team",
            "FINISH": "end"
        }
    )
    
    workflow.set_entry_point("supervisor")
    
    return workflow.compile()

# Create the graph instance for the API server
graph = create_graph()
