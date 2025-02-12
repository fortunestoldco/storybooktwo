"""Define the hierarchical team agent graph structure.

Current Date and Time (UTC): 2025-02-11 21:21:50
Current User's Login: fortunestoldco
"""

from typing import Dict, List, Optional
import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langgraph.graph import StateGraph, Graph, START

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
    tavily_tool,
    scrape_webpages,
    create_outline,
    read_document,
    write_document,
    edit_document,
    python_repl_tool,
)

def create_agent_node(
    llm: ChatOpenAI,
    tools: List,
    system_prompt: str,
    name: str,
    config: Configuration
):
    """Create an agent node that can use tools."""
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
    
    def agent_node(state: State) -> Dict:
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
        
        return {
            "messages": state.get("messages", []) + [SystemMessage(content=response)],
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
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"You are the {name} managing: {', '.join(team_members)}.\n"
                    "Given the current context and history, decide which team member should act next.\n"
                    "Respond with their name or 'FINISH' if the task is complete."),
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    def supervisor(state: State) -> Dict:
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
        
        # Return next step
        return {
            "messages": state.get("messages", []) + [response],
            "next": next_step
        }
    
    return supervisor

def create_team_graph(
    supervisor_node,
    team_nodes: Dict[str, callable],
    config: Configuration,
    name: str
) -> Graph:
    """Create a team graph."""
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    for node_name, node_fn in team_nodes.items():
        workflow.add_node(node_name, node_fn)
    
    # Add edges from supervisor to all team members
    for node_name in team_nodes:
        workflow.add_edge("supervisor", node_name)
        workflow.add_edge(node_name, "supervisor")
    
    # Add end state node
    def end_node(state: State) -> Dict:
        """End state node."""
        return {"messages": state.get("messages", []), "next": "END"}
    
    workflow.add_node("end", end_node)
    
    # Set conditional edges
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: "end" if x["next"] == "FINISH" else x["next"],
        {**{name: name for name in team_nodes.keys()}, "end": "end"}
    )
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    return workflow.compile()

def create_graph(config: Optional[Configuration] = None) -> StateGraph:
    """Create the complete hierarchical team graph."""
    if config is None:
        config = Configuration()

    llm = ChatOpenAI(
        model=config.model,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create research team
    research_agents = {
        "search": create_agent_node(llm, [tavily_tool], RESEARCH_SYSTEM_PROMPT, "search", config),
        "web_scraper": create_agent_node(llm, [scrape_webpages], RESEARCH_SYSTEM_PROMPT, "web_scraper", config)
    }
    research_supervisor = create_supervisor_node(llm, list(research_agents.keys()), config, "research_supervisor")
    research_team = create_team_graph(research_supervisor, research_agents, config, "research_team")

    # Create writing team
    writing_agents = {
        "doc_writer": create_agent_node(llm, [write_document, edit_document, read_document], DOC_WRITER_PROMPT, "doc_writer", config),
        "note_taker": create_agent_node(llm, [create_outline, read_document], NOTE_TAKER_PROMPT, "note_taker", config),
        "chart_generator": create_agent_node(llm, [read_document, python_repl_tool], CHART_GENERATOR_PROMPT, "chart_generator", config)
    }
    writing_supervisor = create_supervisor_node(llm, list(writing_agents.keys()), config, "writing_supervisor")
    writing_team = create_team_graph(writing_supervisor, writing_agents, config, "writing_team")

    # Create main workflow graph
    workflow = StateGraph(State)
    
    # Add the teams as nodes
    workflow.add_node("research_team", research_team)
    workflow.add_node("writing_team", writing_team)
    
    # Create top-level supervisor
    top_supervisor = create_supervisor_node(
        llm, ["research_team", "writing_team"], config, "top_supervisor"
    )
    workflow.add_node("supervisor", top_supervisor)
    
    # Add end state node
    def end_node(state: State) -> Dict:
        """End state node."""
        return {"messages": state.get("messages", []), "next": "END"}
    
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
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Compile the graph
    final_graph = workflow.compile()
    final_graph.name = "Hierarchical Team Agent"
    
    return final_graph

# Create the graph instance for the API server
graph = create_graph()
