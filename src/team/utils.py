"""Helper functions for the hierarchical team agent.

Current Date and Time (UTC): 2025-02-11 21:01:56
Current User's Login: fortunestoldco
"""

from typing import Dict, List, Literal, TypedDict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START
from src.team.configuration import State

def make_supervisor_node(llm: BaseChatModel, members: list[str]):
    """Create a supervisor node for managing team members."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""You are a supervisor managing: {', '.join(members)}.
Given the current context, decide which team member should act next.
Respond with their name or 'FINISH' if the task is complete."""),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    def supervisor_node(state: State) -> Dict:
        """Route between team members based on supervisor's decision."""
        result = llm.invoke(prompt.format_messages(messages=state.messages))
        next_step = result.content.strip()
        return {
            "goto": "__end__" if next_step == "FINISH" else next_step,
            "next": next_step
        }
    
    return supervisor_node

def build_team_graph(supervisor_node, team_nodes: Dict, state_class: type = State) -> StateGraph:
    """Build a team graph with supervisor and team members."""
    builder = StateGraph(state_class)
    builder.add_node("supervisor", supervisor_node)
    
    for name, node in team_nodes.items():
        builder.add_node(name, node)
    
    builder.add_edge(START, "supervisor")
    for name in team_nodes:
        builder.add_edge(name, "supervisor")
    
    return builder.compile()

def create_team_node(agent, name: str):
    """Create a team node that properly formats messages and handles state."""
    def node(state: State) -> Dict:
        result = agent.invoke(state.messages)
        return {
            "messages": [HumanMessage(content=result.content, name=name)],
            "goto": "supervisor",
        }
    return node
