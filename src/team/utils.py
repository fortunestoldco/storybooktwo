"""Utility functions for the hierarchical team agent."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Type, TypeVar, Dict, Literal

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import create_react_agent

T = TypeVar("T")


def get_current_time() -> str:
    """Get the current time in ISO format."""
    return datetime.now(tz=timezone.utc).isoformat()


def load_chat_model(model_name: str) -> BaseChatModel:
    """Load a chat model based on the model name."""
    return ChatOpenAI(model=model_name)


def create_team_supervisor(
    llm: BaseChatModel,
    members: list[str],
    system_prompt: str,
    state_class: Optional[Type[T]] = None,
) -> tuple:
    """Create a supervisor node for a team."""
    def supervisor_node(state):
        messages = [
            {"role": "system", "content": system_prompt.format(system_time=get_current_time())},
        ] + state.messages
        
        response = llm.invoke(messages)
        
        if state.is_last_step:
            return {
                "messages": [
                    AIMessage(
                        content="Task completed in allocated steps.",
                    )
                ]
            }
        
        return {"messages": [response]}

    return supervisor_node


def create_worker_node(
    llm: BaseChatModel,
    tools: list,
    system_prompt: str,
    state_class: Optional[Type[T]] = None,
):
    """Create a worker node with specified tools."""
    agent = create_react_agent(
        llm,
        tools,
        system_prompt.format(system_time=get_current_time())
    )

    def worker_node(state):
        result = agent.invoke(state)
        
        if state.is_last_step:
            return {
                "messages": [
                    AIMessage(
                        content="Task completed in allocated steps.",
                    )
                ]
            }
        
        return {
            "messages": [
                AIMessage(content=result["messages"][-1].content)
            ]
        }

    return worker_node


def build_team_graph(
    supervisor_node,
    worker_nodes: dict,
    state_class: Optional[Type[T]] = None,
) -> StateGraph:
    """Build a team graph with a supervisor and workers."""
    builder = StateGraph(state_class)
    
    # Add supervisor node
    builder.add_node("supervisor", supervisor_node)
    
    # Add worker nodes
    for name, node in worker_nodes.items():
        builder.add_node(name, node)
    
    # Add edges
    builder.add_edge(START, "supervisor")
    
    # Add conditional edges
    def route_team(state) -> Literal[tuple(worker_nodes.keys()) + ("__end__",)]:
        if state.next == "FINISH":
            return "__end__"
        return state.next
    
    builder.add_conditional_edges(
        "supervisor",
        route_team,
    )
    
    # Add edges back to supervisor
    for name in worker_nodes:
        builder.add_edge(name, "supervisor")
    
    return builder.compile()
