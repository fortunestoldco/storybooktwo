"""Graph definition for the hierarchical team agent."""

from typing import Dict, Optional
from langchain_openai import ChatOpenAI
import os
from langgraph.graph import StateGraph
from .configuration import Configuration, State
from .state import TeamState
from .prompts import (
    RESEARCH_SYSTEM_PROMPT,
    DOC_WRITER_PROMPT,
    NOTE_TAKER_PROMPT,
    CHART_GENERATOR_PROMPT,
)
from .tools import (
    tavily_tool,
    scrape_webpages,
    write_document,
    edit_document,
    read_document,
    create_outline,
    python_repl_tool,
)


def create_agent_node(llm, tools, system_prompt, name, config: Configuration):
    """Create an agent node with proper LangSmith instrumentation."""
    def agent_node(state: State) -> Dict:
        # Create run metadata for LangSmith
        run_metadata = config.create_run_metadata(state)
        
        # Run agent with metadata
        with config.langsmith_client.tracing(
            project_name=config.langsmith_project,
            metadata=run_metadata,
            tags=["agent_node", name]
        ):
            # Execute agent logic here
            # ... (rest of the agent implementation)
            pass
            
    return agent_node

def create_supervisor_node(llm, team_members, config: Configuration, name: str):
    """Create a supervisor node with proper LangSmith instrumentation."""
    def supervisor_node(state: State) -> Dict:
        # Create run metadata for LangSmith
        run_metadata = config.create_run_metadata(state)
        
        # Run supervisor with metadata
        with config.langsmith_client.tracing(
            project_name=config.langsmith_project,
            metadata=run_metadata,
            tags=["supervisor_node", name]
        ):
            # Execute supervisor logic here
            # ... (rest of the supervisor implementation)
            pass
            
    return supervisor_node

def create_team_graph(supervisor, agents, config: Configuration, name: str) -> StateGraph:
    """Create a team graph with proper LangSmith instrumentation."""
    workflow = StateGraph(TeamState)
    
    # Add supervisor and agent nodes
    workflow.add_node("supervisor", supervisor)
    for agent_name, agent in agents.items():
        workflow.add_node(agent_name, agent)
    
    # Add edges between nodes
    for agent_name in agents:
        workflow.add_edge(agent_name, "supervisor")
    
    # Set conditional edges for supervisor
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: "end" if x["next"] == "FINISH" else x["next"],
        {**{name: name for name in agents.keys()}, "end": "end"}
    )
    
    workflow.set_entry_point("supervisor")
    
    return workflow.compile()

def create_graph(config: Optional[Configuration] = None) -> StateGraph:
    """Create the complete hierarchical team graph with LangSmith instrumentation."""
    if config is None:
        config = Configuration()

    llm = ChatOpenAI(
        model=config.model,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create the graph components with LangSmith instrumentation
    research_agents = {
        "search": create_agent_node(llm, [tavily_tool], RESEARCH_SYSTEM_PROMPT, "search", config),
        "web_scraper": create_agent_node(llm, [scrape_webpages], RESEARCH_SYSTEM_PROMPT, "web_scraper", config)
    }
    research_supervisor = create_supervisor_node(llm, list(research_agents.keys()), config, "research_supervisor")
    research_team = create_team_graph(research_supervisor, research_agents, config, "research_team")

    writing_agents = {
        "doc_writer": create_agent_node(llm, [write_document, edit_document, read_document], DOC_WRITER_PROMPT, "doc_writer", config),
        "note_taker": create_agent_node(llm, [create_outline, read_document], NOTE_TAKER_PROMPT, "note_taker", config),
        "chart_generator": create_agent_node(llm, [read_document, python_repl_tool], CHART_GENERATOR_PROMPT, "chart_generator", config)
    }
    writing_supervisor = create_supervisor_node(llm, list(writing_agents.keys()), config, "writing_supervisor")
    writing_team = create_team_graph(writing_supervisor, writing_agents, config, "writing_team")

    # Create main workflow graph
    workflow = StateGraph(State)
    
    # Add teams as nodes
    workflow.add_node("research_team", research_team)
    workflow.add_node("writing_team", writing_team)
    
    # Create top-level supervisor
    top_supervisor = create_supervisor_node(
        llm, ["research_team", "writing_team"], config, "top_supervisor"
    )
    workflow.add_node("supervisor", top_supervisor)
    
    # Add end state node
    def end_node(state: State) -> Dict:
        return {
            "messages": state.messages,
            "next": "END",
            "input_parameters": state.input_parameters
        }
    
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
    
    workflow.set_entry_point("supervisor")
    
    final_graph = workflow.compile()
    final_graph.name = "Hierarchical Team Agent"
    
    return final_graph

# Create the graph instance for the API server
graph = create_graph()
