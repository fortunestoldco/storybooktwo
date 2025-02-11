"""Define the hierarchical team agent graph structure."""

from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START

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
from src.team.utils import make_supervisor_node, build_team_graph, create_team_node

def create_agent(
    llm: ChatOpenAI,
    tools: List,
    system_prompt: str,
    name: str
):
    """Create an agent with tools and system prompt."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        SystemMessage(content=f"Current Date and Time (UTC): 2025-02-11 21:00:47\nAgent: {name}")
    ])
    
    def agent_node(state: State) -> Dict:
        """Execute agent with current state."""
        messages = state.get("messages", [])
        formatted_prompt = prompt.format_messages(messages=messages)
        
        # Include tools in a structured format
        tools_description = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
        formatted_prompt.insert(1, SystemMessage(content=f"Available tools:\n{tools_description}"))
        
        response = llm.invoke(formatted_prompt)
        return {
            "messages": messages + [response],
            "next": "supervisor"
        }
    
    return agent_node

def build_research_team(llm: ChatOpenAI) -> StateGraph:
    """Build the research team graph."""
    # Create research team members
    search_agent = create_agent(
        llm=llm,
        tools=[tavily_tool],
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        name="search"
    )
    
    web_scraper_agent = create_agent(
        llm=llm,
        tools=[scrape_webpages],
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        name="web_scraper"
    )

    research_supervisor = make_supervisor_node(
        llm, ["search", "web_scraper"]
    )

    return build_team_graph(
        research_supervisor,
        {
            "search": search_agent,
            "web_scraper": web_scraper_agent,
        }
    )

def build_writing_team(llm: ChatOpenAI) -> StateGraph:
    """Build the writing team graph."""
    doc_writer_agent = create_agent(
        llm=llm,
        tools=[write_document, edit_document, read_document],
        system_prompt=DOC_WRITER_PROMPT,
        name="doc_writer"
    )
    
    note_taker_agent = create_agent(
        llm=llm,
        tools=[create_outline, read_document],
        system_prompt=NOTE_TAKER_PROMPT,
        name="note_taker"
    )
    
    chart_generator_agent = create_agent(
        llm=llm,
        tools=[read_document, python_repl_tool],
        system_prompt=CHART_GENERATOR_PROMPT,
        name="chart_generator"
    )

    writing_supervisor = make_supervisor_node(
        llm, ["doc_writer", "note_taker", "chart_generator"]
    )

    return build_team_graph(
        writing_supervisor,
        {
            "doc_writer": doc_writer_agent,
            "note_taker": note_taker_agent,
            "chart_generator": chart_generator_agent,
        }
    )

def create_graph(config: Optional[Configuration] = None) -> StateGraph:
    """Create the complete hierarchical team graph."""
    if config is None:
        config = Configuration()

    llm = ChatOpenAI(model=config.model)

    # Build team graphs
    research_graph = build_research_team(llm)
    writing_graph = build_writing_team(llm)

    def call_research_team(state: State) -> Dict:
        """Call the research team graph."""
        response = research_graph.invoke({"messages": state["messages"][-1]})
        return {
            "messages": [
                HumanMessage(content=response["messages"][-1].content, name="research_team")
            ],
            "goto": "supervisor",
        }

    def call_writing_team(state: State) -> Dict:
        """Call the writing team graph."""
        response = writing_graph.invoke({"messages": state["messages"][-1]})
        return {
            "messages": [
                HumanMessage(content=response["messages"][-1].content, name="writing_team")
            ],
            "goto": "supervisor",
        }

    # Create supervisor
    teams_supervisor = make_supervisor_node(llm, ["research_team", "writing_team"])

    # Build super graph
    super_graph = build_team_graph(
        teams_supervisor,
        {
            "research_team": call_research_team,
            "writing_team": call_writing_team,
        }
    )

    super_graph.name = "Hierarchical Team Agent"
    return super_graph

# Create the graph instance for the API server
graph = create_graph()
