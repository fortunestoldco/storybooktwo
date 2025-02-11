"""Define the hierarchical team agent graph."""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

# Configuration
@dataclass
class Configuration:
    """Configuration for the hierarchical team agent."""
    model: str = "gpt-4"
    max_search_results: int = 5
    working_directory: str = "workspace"

# State classes
@dataclass
class AgentState:
    """Base state for the agent."""
    messages: List[BaseMessage] = field(default_factory=list)
    next_step: str = field(default="research")
    is_done: bool = field(default=False)

def create_graph(config: Optional[Configuration] = None) -> StateGraph:
    """Create the agent graph."""
    if config is None:
        config = Configuration()

    # Initialize LLM
    llm = ChatOpenAI(model=config.model)

    # Create the agent node
    agent = create_react_agent(
        llm,
        [],  # No tools for this simplified version
        f"""You are a helpful assistant. Your task is to help users with their requests.
        Current time: {datetime.now(timezone.utc).isoformat()}"""
    )

    def agent_node(state: AgentState) -> Dict:
        """Execute the agent node."""
        result = agent.invoke({"messages": state.messages})
        return {
            "messages": state.messages + [result["messages"][-1]],
            "next_step": "FINISH" if state.is_done else "continue",
        }

    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add the agent node
    workflow.add_node("agent", agent_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    
    def router(state: AgentState) -> str:
        """Route between states."""
        return "__end__" if state.next_step == "FINISH" else "agent"
    
    workflow.add_conditional_edges(
        "agent",
        router,
    )
    
    # Compile the graph
    return workflow.compile()

# Create the graph instance for the API server
graph = create_graph()
