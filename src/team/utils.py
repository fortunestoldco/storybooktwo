"""Helper functions for the hierarchical team agent.

Current Date and Time (UTC): 2025-02-12 01:39:53
Current User's Login: fortunestoldco
"""

from typing import Dict, List, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.channels import Channel, EntityChannel
from src.team.configuration import State, Configuration

def make_supervisor_node(
    llm: BaseChatModel, 
    members: list[str], 
    config: Configuration,
    channel_id: str = "supervisor"
):
    """Create a supervisor node for managing team members."""
    channel = Channel(id=channel_id)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""You are a supervisor managing: {', '.join(members)}.
Given the current context and history, decide which team member should act next.
Respond with their name or 'FINISH' if the task is complete."""),
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    def supervisor_node(state: State) -> Dict[str, Any]:
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
        
        # Write to channel
        channel.write({
            "decision": next_step,
            "timestamp": state.get("timestamp")
        })
        
        return {
            "messages": [result],
            "next": next_step,
            "session_id": state.session_id
        }
    
    return supervisor_node

def build_team_graph(
    supervisor_node, 
    team_nodes: Dict[str, callable], 
    state_class: type = State
) -> StateGraph:
    """Build a team graph with supervisor and team members."""
    builder = StateGraph(state_class)
    
    # Add supervisor node
    builder.add_node("supervisor", supervisor_node)
    
    # Add team member nodes
    for name, node in team_nodes.items():
        builder.add_node(name, node)
    
    # Add edges
    builder.add_edge(START, "supervisor")
    
    # Add conditional edges from supervisor to team members
    for name in team_nodes:
        builder.add_conditional_edge(
            "supervisor",
            name,
            lambda x, name=name: x["next"] == name
        )
        builder.add_edge(name, "supervisor")
    
    # Add end condition
    builder.add_conditional_edge(
        "supervisor",
        END,
        lambda x: x["next"] == "FINISH"
    )
    
    return builder.compile()

def create_team_node(
    agent, 
    name: str, 
    config: Configuration,
    channel_id: Optional[str] = None
):
    """Create a team node that properly formats messages and handles state."""
    channel = Channel(id=channel_id or f"agent_{name}")
    
    def node(state: State) -> Dict[str, Any]:
        # Get message history
        message_history = config.get_message_history(state.session_id)
        
        result = agent.invoke(state.messages)
        
        # Save to history
        message_history.add_ai_message(result)
        
        # Write to channel
        channel.write({
            "action": result.content,
            "agent": name,
            "timestamp": state.get("timestamp")
        })
        
        return {
            "messages": [HumanMessage(content=result.content, name=name)],
            "next": "supervisor",
            "session_id": state.session_id
        }
    
    return node
