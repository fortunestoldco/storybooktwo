"""Define the hierarchical team agent graph structure.

Current Date and Time (UTC): 2025-02-12 01:41:17
Current User's Login: fortunestoldco
"""

import os
from typing import Dict, List, Optional, TypeVar, Any
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langgraph.graph import StateGraph, Graph
from langgraph.channels import Channel, EntityChannel

from src.team.configuration import State, Configuration
from src.team.prompts import (
    RESEARCH_SYSTEM_PROMPT,
    WRITING_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    DOC_WRITER_PROMPT,
    NOTE_TAKER_PROMPT,
    CHART_GENERATOR_PROMPT,
)
from src.team.tools import (
    create_tools,
)

# Type variable for state
S = TypeVar("S", bound=State)

def create_agent_node(
    llm: ChatOpenAI,
    tools: List,
    system_prompt: str,
    name: str,
    config: Configuration
):
    """Create an agent node that can use tools."""
    channel = Channel(id=f"agent_{name}")
    current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create the agent prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        SystemMessage(content=f"Current Date and Time (UTC): {current_time}\nAgent: {name}")
    ])
    
    # Create the agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    def agent_node(state: State) -> Dict[str, Any]:
        """Execute agent with tools."""
        # Get message history
        message_history = config.get_message_history(state.session_id)
        
        # Prepare the input
        agent_input = {
            "history": message_history.messages,
            "messages": state.get("messages", []),
            "agent_scratchpad": format_to_openai_functions([])
        }
        
        # Execute agent
        output = agent.invoke(agent_input)
        response = output.return_values["output"]
        
        # Save to history
        message_history.add_ai_message(SystemMessage(content=response))
        
        # Write to channel
        channel.write({
            "action": response,
            "agent": name,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Return state updates
        return {
            "messages": [SystemMessage(content=response)],
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
    channel = Channel(id=f"{name}_channel")
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"You are the supervisor managing: {', '.join(team_members)}.\n"
                    "Given the current context and history, decide which team member should act next.\n"
                    "Respond with their name or 'FINISH' if the task is complete."),
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    def supervisor(state: State) -> Dict[str, Any]:
        """Route between team members."""
        # Get message history
        message_history = config.get_message_history(state.session_id)
        
        # Format messages
        formatted_prompt = prompt.format_messages(
            history=message_history.messages,
            messages=state.get("messages", [])
        )
        
        # Get decision
        response = llm.invoke(formatted_prompt)
        next_step = response.content.strip()
        
        # Save to history
        message_history.add_ai_message(response)
        
        # Write to channel
        channel.write({
            "decision": next_step,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Return state updates
        return {
            "messages": [response],
            "next": next_step
        }
    
    return supervisor

def create_team_graph(config: Configuration) -> Graph:
    """Create the complete hierarchical team graph."""
    # Initialize LLM
    llm = ChatOpenAI(
        model=config.model,
        openai_api_key=config.openai_api_key
    )
    
    # Create tools
    tools = create_tools(config)
    
    # Create research team
    research_agents = {
        "search": create_agent_node(llm, [tools["tavily_tool"]], RESEARCH_SYSTEM_PROMPT, "search", config),
        "web_scraper": create_agent_node(llm, [tools["scrape_webpages"]], RESEARCH_SYSTEM_PROMPT, "web_scraper", config)
    }
    research_supervisor = create_supervisor_node(llm, list(research_agents.keys()), config, "research_supervisor")
    
    # Create writing team
    writing_agents = {
        "doc_writer": create_agent_node(llm, [tools["write_document"], tools["edit_document"], tools["read_document"]], 
                                      DOC_WRITER_PROMPT, "doc_writer", config),
        "note_taker": create_agent_node(llm, [tools["create_outline"], tools["read_document"]], 
                                      NOTE_TAKER_PROMPT, "note_taker", config),
        "chart_generator": create_agent_node(llm, [tools["read_document"], tools["python_repl_tool"]], 
                                           CHART_GENERATOR_PROMPT, "chart_generator", config)
    }
    writing_supervisor = create_supervisor_node(llm, list(writing_agents.keys()), config, "writing_supervisor")
    
    # Create main workflow graph
    workflow = StateGraph(State)
    
    # Add all nodes
    workflow.add_node("research_supervisor", research_supervisor)
    workflow.add_node("writing_supervisor", writing_supervisor)
    for name, node in research_agents.items():
        workflow.add_node(f"research_{name}", node)
    for name, node in writing_agents.items():
        workflow.add_node(f"writing_{name}", node)
    
    # Add top-level supervisor
    top_supervisor = create_supervisor_node(llm, ["research", "writing"], config)
    workflow.add_node("supervisor", top_supervisor)
    
    # Add edges
    workflow.add_edge("START", "supervisor")
    
    # Add conditional edges from supervisor to teams
    workflow.add_conditional_edge(
        "supervisor",
        "research_supervisor",
        lambda x: x["next"] == "research"
    )
    workflow.add_conditional_edge(
        "supervisor",
        "writing_supervisor",
        lambda x: x["next"] == "writing"
    )
    
    # Add research team edges
    for name in research_agents:
        workflow.add_conditional_edge(
            "research_supervisor",
            f"research_{name}",
            lambda x, name=name: x["next"] == name
        )
        workflow.add_edge(f"research_{name}", "research_supervisor")
    
    # Add writing team edges
    for name in writing_agents:
        workflow.add_conditional_edge(
            "writing_supervisor",
            f"writing_{name}",
            lambda x, name=name: x["next"] == name
        )
        workflow.add_edge(f"writing_{name}", "writing_supervisor")
    
    # Add edges back to main supervisor
    workflow.add_edge("research_supervisor", "supervisor")
    workflow.add_edge("writing_supervisor", "supervisor")
    
    # Add end condition
    workflow.add_conditional_edge(
        "supervisor",
        "END",
        lambda x: x["next"] == "FINISH"
    )
    
    # Compile the graph
    graph = workflow.compile()
    
    # Set graph metadata
    graph.metadata = {
        "name": "Hierarchical Team Agent",
        "description": "A multi-agent system for research and writing tasks",
        "version": "0.1.0",
        "channels": {
            "supervisor": {"type": "memory", "retention": "session"},
            "research": {"type": "memory", "retention": "session"},
            "writing": {"type": "memory", "retention": "session"}
        }
    }
    
    return graph

# Create the graph instance for the API server
graph = create_graph(Configuration())
