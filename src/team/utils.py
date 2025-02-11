"""Helper functions for the hierarchical team agent."""

from typing import Dict, List, Literal, TypedDict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from .configuration import State

def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    """Create a supervisor node for managing team members."""
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""
        next: Literal[*options]

    def supervisor_node(state: State) -> Dict:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = "__end__"
        return {"next": goto, "goto": goto}

    return supervisor_node

def build_team_graph(
    supervisor_node,
    team_nodes: Dict,
    state_class: type = State
) -> StateGraph:
    """Build a team graph with a supervisor and team members."""
    builder = StateGraph(state_class)
    
    # Add supervisor
    builder.add_node("supervisor", supervisor_node)
    
    # Add team members
    for name, node in team_nodes.items():
        builder.add_node(name, node)
    
    # Add edges
    builder.add_edge("START", "supervisor")
    for name in team_nodes:
        builder.add_edge(name, "supervisor")
    
    return builder.compile()

def create_team_node(agent, name: str):
    """Create a team node with standardized message handling."""
    def node(state: State) -> Dict:
        result = agent.invoke(state)
        return {
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name=name)
            ],
            "goto": "supervisor",
        }
    return node
