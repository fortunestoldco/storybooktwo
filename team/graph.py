from typing import Dict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from team.configuration import Configuration, State
from team.prompts import (
    MARKET_RESEARCH_PROMPT,
    AUDIENCE_RESEARCH_PROMPT,
    WRITER_PROMPT,
    NOTE_TAKER_PROMPT,
)
from team.tools import (
    tavily_tool,
    scrape_webpages,
    write_document,
    edit_document,
    read_document,
    create_outline
)

def create_agent_node(llm: ChatOpenAI, tools: List, system_prompt: str, name: str, config: Configuration):
    """Create an agent node with proper LangSmith instrumentation."""
    def agent_node(state: State) -> Dict:
        run_metadata = config.create_run_metadata(state)
        with config.langsmith_client.tracing(
            project_name=config.langsmith_project,
            metadata=run_metadata,
            tags=["agent_node", name]
        ) as run:
            messages = state.get("messages", [])
            story_params = state.get("story_parameters", {})
            # Add story parameters to system prompt
            enhanced_prompt = f"{system_prompt}\n\nStory Parameters:\n"
            enhanced_prompt += f"Starting Point: {story_params.get('start', '')}\n"
            enhanced_prompt += f"Key Plot Points: {', '.join(story_params.get('plot_points', []))}\n"
            enhanced_prompt += f"Intended Ending: {story_params.get('ending', '')}\n"
            if state.get("research_data"):
                enhanced_prompt += f"\nResearch Data:\n{json.dumps(state.research_data, indent=2)}"
            result = llm.invoke([
                {"role": "system", "content": enhanced_prompt},
                *messages
            ])
            return {
                "messages": [{"role": "assistant", "content": result.content}],
                "next": "supervisor",
                "session_id": state.session_id,
                "input_parameters": state.input_parameters,
                "story_parameters": story_params,
                "research_data": state.get("research_data", {})
            }
    return agent_node

def create_supervisor_node(llm: ChatOpenAI, team_members: List[str], config: Configuration, name: str):
    """Create a supervisor node that manages the story creation workflow."""
    def supervisor_node(state: State) -> Dict:
        run_metadata = config.create_run_metadata(state)
        # Initialize story parameters if not present
        if not state.story_parameters and state.input_parameters.get("initial_request"):
            state.story_parameters = StoryParameters(
                start=state.input_parameters["initial_request"].get("start", ""),
                plot_points=state.input_parameters["initial_request"].get("plot_points", []),
                ending=state.input_parameters["initial_request"].get("ending", ""),
                genre=state.input_parameters["initial_request"].get("genre", None),
                target_length=state.input_parameters["initial_request"].get("target_length", None)
            )
        with config.langsmith_client.tracing(
            project_name=config.langsmith_project,
            metadata=run_metadata,
            tags=["supervisor_node", name]
        ) as run:
            messages = state.get("messages", [])
            current_phase = state.input_parameters.get("current_phase", "market_research")
            research_data = state.get("research_data", {})
            story_params = state.get("story_parameters", {})
            system_message = f"""You are supervising a creative writing team.
Current phase: {current_phase}
Available team members: {', '.join(team_members)}
Story Parameters:
Start: {story_params.get('start', '')}
Plot Points: {', '.join(story_params.get('plot_points', []))}
Ending: {story_params.get('ending', '')}
"""
            result = llm.invoke([
                {"role": "system", "content": system_message},
                *messages
            ])
            next_step = result.content.strip()
            return {
                "messages": [{"role": "assistant", "content": result.content}],
                "next": next_step,
                "session_id": state.session_id,
                "input_parameters": state.input_parameters,
                "story_parameters": story_params,
                "research_data": research_data
            }
    return supervisor_node