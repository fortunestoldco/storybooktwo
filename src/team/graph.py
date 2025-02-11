"""Define the hierarchical team agent graph structure."""

from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import create_react_agent

from .configuration import State, Configuration
from .prompts import (
    RESEARCH_SYSTEM_PROMPT,
    WRITING_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    DOC_WRITER_PROMPT,
    NOTE_TAKER_PROMPT,
    CHART_GENERATOR_PROMPT,
)
from .tools import (
    tavily_tool,
    scrape_webpages,
    create_outline,
    read_document,
    write_document,
    edit_document,
    python_repl_tool,
)
from .utils import make_supervisor_node, build_team_graph, create_team_node

def build_research_team(llm: ChatOpenAI) -> StateGraph:
    """Build the research team graph."""
    # Create research team members
    search_agent = create_react_agent(llm, [tavily_tool])
    search_node = create_team_node(search_agent, "search")

    web_scraper_agent = create_react_agent(llm, [scrape_webpages])
    web_scraper_node = create_team_node(web_scraper_agent, "web_scraper")

    # Create research supervisor
    research_supervisor = make_supervisor_node(
        llm, ["search", "web_scraper"]
    )

    return build_team_graph(
        research_supervisor,
        {
            "search": search_node,
            "web_scraper": web_scraper_node,
        }
    )

def build_writing_team(llm: ChatOpenAI) -> StateGraph:
    """Build the writing team graph."""
    # Create writing team members
    doc_writer_agent = create_react_agent(
        llm,
        [write_document, edit_document, read_document],
        DOC_WRITER_PROMPT,
    )
    doc_writer_node = create_team_node(doc_writer_agent, "doc_writer")

    note_taker_agent = create_react_agent(
        llm,
        [create_outline, read_document],
        NOTE_TAKER_PROMPT,
    )
    note_taker_node = create_team_node(note_taker_agent, "note_taker")

    chart_generator_agent = create_react_agent(
        llm,
        [read_document, python_repl_tool],
        CHART_GENERATOR_PROMPT,
    )
    chart_generator_node = create_team_node(chart_generator_agent, "chart_generator")

    # Create writing supervisor
    writing_supervisor = make_supervisor_node(
        llm, ["doc_writer", "note_taker", "chart_generator"]
    )

    return build_team_graph(
        writing_supervisor,
        {
            "doc_writer": doc_writer_node,
            "note_taker": note_taker_node,
            "chart_generator": chart_generator_node,
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
