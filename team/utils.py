"""Utility functions for the hierarchical team agent.

Current Date and Time (UTC): 2025-02-12 00:02:08
Current User's Login: fortunestoldco
"""

from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

from team.configuration import State, Configuration
from team.state import TeamState

def make_supervisor_node(
    llm: ChatOpenAI,
    team_members: List[str],
    config: Configuration,
    name: str
):
    """Create a supervisor node with proper LangSmith instrumentation."""
    def supervisor_node(state: State) -> Dict:
        run_metadata = config.create_run_metadata(state)
        
        with config.langsmith_client.tracing(
            project_name=config.langsmith_project,
            metadata=run_metadata,
            tags=["supervisor_node", name]
        ) as run:
            messages = state.get("messages", [])
            result = llm.invoke([
                {"role": "system", "content": f"You are supervising: {', '.join(team_members)}. Decide who acts next or FINISH if done."},
                *messages
            ])
            next_step = result.content.strip()
            
            return {
                "messages": [{"role": "assistant", "content": result.content}],
                "next": next_step,
                "session_id": state.session_id,
                "input_parameters": state.input_parameters
            }
            
    return supervisor_node

def create_team_node(
    llm: ChatOpenAI,
    tools: List,
    system_prompt: str,
    name: str,
    config: Configuration
):
    """Create a team member node with proper LangSmith instrumentation."""
    def team_node(state: State) -> Dict:
        run_metadata = config.create_run_metadata(state)
        
        with config.langsmith_client.tracing(
            project_name=config.langsmith_project,
            metadata=run_metadata,
            tags=["team_node", name]
        ) as run:
            messages = state.get("messages", [])
            result = llm.invoke([
                {"role": "system", "content": system_prompt},
                *messages
            ])
            
            return {
                "messages": [{"role": "assistant", "content": result.content}],
                "next": "supervisor",
                "session_id": state.session_id,
                "input_parameters": state.input_parameters
            }
            
    return team_node

def build_team_graph(
    supervisor,
    agents: Dict,
    config: Configuration,
    name: str
) -> StateGraph:
    """Build a team graph with proper node structure."""
    workflow = StateGraph(TeamState)
    
    # Add supervisor and agent nodes
    workflow.add_node("supervisor", supervisor)
    for agent_name, agent in agents.items():
        workflow.add_node(agent_name, agent)
    
    # Add end state node
    def end_node(state: TeamState) -> Dict:
        """End state for team graph."""
        return {
            "messages": state.messages,
            "next": "FINISH",
            "input_parameters": state.input_parameters
        }
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
