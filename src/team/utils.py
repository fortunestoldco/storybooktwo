"""Helper functions for the hierarchical team agent.

Current Date and Time (UTC): 2025-02-11 21:08:35
Current User's Login: fortunestoldco
"""

from typing import Dict, List, Literal, TypedDict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START
from src.team.configuration import State, Configuration

def make_supervisor_node(llm: BaseChatModel, members: list[str], config: Configuration):
    """Create a supervisor node for managing team members."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""You are a supervisor managing: {', '.join(members)}.
Given the current context and history, decide which team member should act next.
Respond with their name or 'FINISH' if the task is complete."""),
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    def supervisor_node(state: State) -> Dict:
        """Route between team members based on supervisor's decision."""
        # Get message history
        message_history = config.get_message_history(state.session_id)
        
        # Format messages
        formatted_prompt = prompt.format_messages(
            history=message_history.messages,
            messages=state.get("messages", [])
        )
        
        result = llm.invoke(formatted_prompt)
        next_step = result.content.strip()
        
        # Save to history
        message_history.add_ai_message(result)
        
        return {
            "goto": "__end__" if next_step == "FINISH" else next_step,
            "next": next_step,
            "session_id": state.session_id
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

def create_team_node(agent, name: str, config: Configuration):
    """Create a team node that properly formats messages and handles state."""
    def node(state: State) -> Dict:
        # Get message history
        message_history = config.get_message_history(state.session_id)
        
        result = agent.invoke(state.messages)
        
        # Save to history
        message_history.add_ai_message(result)
        
        return {
            "messages": [HumanMessage(content=result.content, name=name)],
            "goto": "supervisor",
            "session_id": state.session_id
        }
    return node
