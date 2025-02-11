"""Define the creative writing agent graph structure.

Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-02-11 22:55:13
Current User's Login: fortunestoldco
"""

from typing import Dict, List, Optional, Callable, Any, Annotated, TypeVar, cast
import os
import json
from datetime import datetime
from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent
from langchain.agents.format_scratchpad import format_to_openai_functions
from langgraph.graph import Graph
from langgraph.prebuilt.tool_executor import ToolExecutor
from pydantic import BaseModel, Field, validator

from src.team.configuration import (
    State, Configuration, StoryParameters, 
    MarketResearch, MessageWrapper,
    ChapterStatus
)
from src.team.prompts import (
    RESEARCH_SYSTEM_PROMPT,
    MARKET_ANALYST_PROMPT,
    REVIEW_ANALYST_PROMPT,
    WRITING_SYSTEM_PROMPT,
    DOC_WRITER_PROMPT,
    NOTE_TAKER_PROMPT,
)
from src.team.tools import RESEARCH_TOOLS, WRITING_TOOLS

class StoryInput(BaseModel):
    """Input parameters for the story creation process."""
    starting_point: str = Field(
        ...,
        description="The initial situation or scene that starts the story"
    )
    plot_points: List[str] = Field(
        ...,
        description="Key plot points that must be included in the story"
    )
    intended_ending: str = Field(
        ...,
        description="The desired conclusion of the story"
    )
    num_chapters: int = Field(
        default=10,
        description="Number of chapters in the story"
    )
    words_per_chapter: int = Field(
        default=2000,
        description="Target words per chapter"
    )

    @validator('num_chapters')
    def validate_num_chapters(cls, v):
        """Validate number of chapters."""
        if v < 1:
            raise ValueError("Number of chapters must be at least 1")
        if v > 50:
            raise ValueError("Number of chapters cannot exceed 50")
        return v

    @validator('words_per_chapter')
    def validate_words_per_chapter(cls, v):
        """Validate words per chapter."""
        if v < 500:
            raise ValueError("Words per chapter must be at least 500")
        if v > 10000:
            raise ValueError("Words per chapter cannot exceed 10,000")
        return v

    def to_parameters(self) -> StoryParameters:
        """Convert input to StoryParameters."""
        return StoryParameters(
            starting_point=self.starting_point,
            plot_points=self.plot_points,
            intended_ending=self.intended_ending,
            num_chapters=self.num_chapters,
            words_per_chapter=self.words_per_chapter
        )

def process_input(inputs: Dict) -> Dict:
    """Process and validate the input dictionary."""
    try:
        # Convert dict to StoryInput for validation
        story_input = StoryInput(**inputs)
        return {
            "starting_point": story_input.starting_point,
            "plot_points": story_input.plot_points,
            "intended_ending": story_input.intended_ending,
            "num_chapters": story_input.num_chapters,
            "words_per_chapter": story_input.words_per_chapter
        }
    except Exception as e:
        raise ValueError(f"Invalid input parameters: {str(e)}") from e

def validate_and_initialize(inputs: Dict) -> State:
    """Validate input and create initial state."""
    # Process and validate inputs
    processed_inputs = process_input(inputs)
    
    # Create StoryInput instance
    story_input = StoryInput(**processed_inputs)
    
    # Convert to StoryParameters
    story_parameters = story_input.to_parameters()
    
    initial_message = SystemMessage(
        content=f"""Beginning story development with parameters:
{story_parameters.to_prompt()}"""
    )
    
    state = State(
        messages=[MessageWrapper.from_message(initial_message)],
        story_parameters=story_parameters,
        market_research=MarketResearch(),
        next=""
    )
    
    state.initialize_chapters()
    state.market_research.chapter_distribution = story_parameters.get_chapter_distribution()
    
    return state

def create_agent_node(
    llm: ChatOpenAI,
    tools: List,
    system_prompt: str,
    name: str,
    config: Configuration
) -> Callable:
    """Create an agent node that can use tools."""
    tool_executor = ToolExecutor(tools=tools)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)

    def run_agent(state: Dict) -> Dict:
        """Execute agent with tools."""
        current_state = State(**state)
        message_history = config.get_message_history(current_state.session_id)
        
        context = []
        if current_state.story_parameters:
            context.append(SystemMessage(content=current_state.story_parameters.to_prompt()))
        if current_state.market_research and current_state.market_research.similar_books:
            context.append(SystemMessage(content=
                f"Market Research:\n{json.dumps(current_state.market_research.dict(), indent=2)}"
            ))
        
        # Add chapter status
        chapter_status = current_state.get_chapter_status()
        context.append(SystemMessage(content=
            f"Chapter Status:\n{json.dumps(chapter_status, indent=2)}"
        ))
        
        agent_input = {
            "history": message_history.messages + context,
            "messages": current_state.get_messages(),
            "agent_scratchpad": format_to_openai_functions([])
        }
        
        output = agent.invoke(agent_input)
        response = output.return_values["output"]
        
        message_history.add_ai_message(AIMessage(content=response))
        
        current_state.add_message(AIMessage(content=response))
        return current_state.dict()
    
    return run_agent

def create_supervisor_node(
    llm: ChatOpenAI,
    team_members: List[str],
    config: Configuration,
    name: str = "supervisor"
) -> Callable:
    """Create a supervisor node that manages team members."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""You are the {name} coordinating the following team: {', '.join(team_members)}.
Your responsibilities:
1. Review current progress and team outputs
2. Determine next steps based on the story requirements
3. Ensure all plot points are being addressed
4. Maintain consistency with target demographic
5. Track chapter progress and word count goals

For research phase:
- Ensure market analysis is thorough
- Verify demographic targeting is clear
- Check that improvement opportunities are identified

For writing phase:
- Track chapter completion status
- Ensure word count targets are being met
- Verify plot points are being incorporated
- Ensure tone matches target demographic
- Check that market research insights are being used

Respond with next team member to act or 'FINISH' if complete."""),
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="messages")
    ])

    def decide_next(state: Dict) -> Dict:
        """Route between team members."""
        current_state = State(**state)
        message_history = config.get_message_history(current_state.session_id)
        
        context_messages = []
        if current_state.story_parameters:
            context_messages.append(SystemMessage(content=
                f"Story Parameters:\n{current_state.story_parameters.to_prompt()}"
            ))
        if current_state.market_research and current_state.market_research.similar_books:
            context_messages.append(SystemMessage(content=
                f"Market Research:\n{json.dumps(current_state.market_research.dict(), indent=2)}"
            ))
            
        # Add chapter status
        chapter_status = current_state.get_chapter_status()
        context_messages.append(SystemMessage(content=
            f"Chapter Status:\n{json.dumps(chapter_status, indent=2)}"
        ))
        
        formatted_prompt = prompt.format_messages(
            history=message_history.messages + context_messages,
            messages=current_state.get_messages()
        )
        
        response = llm.invoke(formatted_prompt)
        next_step = response.content.strip()
        
        message_history.add_ai_message(response)
        
        current_state.add_message(response)
        current_state.next = next_step
        return current_state.dict()
    
    return decide_next

def create_graph(config: Optional[Configuration] = None) -> Graph:
    """Create the creative writing agent graph."""
    if config is None:
        config = Configuration()

    workflow = Graph()
    
    # Add input validator
    workflow.add_node("start", validate_and_initialize)
    
    # Create LLM
    llm = ChatOpenAI(
        model=config.model,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create research team nodes
    research_agents = {
        "market_researcher": create_agent_node(
            llm, RESEARCH_TOOLS, MARKET_ANALYST_PROMPT, "market_researcher", config
        ),
        "review_analyst": create_agent_node(
            llm, RESEARCH_TOOLS, REVIEW_ANALYST_PROMPT, "review_analyst", config
        )
    }
    
    # Create writing team nodes
    writing_agents = {
        "writer": create_agent_node(
            llm, WRITING_TOOLS, DOC_WRITER_PROMPT, "writer", config
        ),
        "editor": create_agent_node(
            llm, WRITING_TOOLS, NOTE_TAKER_PROMPT, "editor", config
        )
    }

    # Add all agent nodes
    for name, agent in {**research_agents, **writing_agents}.items():
        workflow.add_node(name, agent)

    # Create and add supervisors
    research_supervisor = create_supervisor_node(
        llm, list(research_agents.keys()), config, "research_supervisor"
    )
    writing_supervisor = create_supervisor_node(
        llm, list(writing_agents.keys()), config, "writing_supervisor"
    )
    top_supervisor = create_supervisor_node(
        llm, ["research_team", "writing_team"], config, "top_supervisor"
    )

    workflow.add_node("research_supervisor", research_supervisor)
    workflow.add_node("writing_supervisor", writing_supervisor)
    workflow.add_node("supervisor", top_supervisor)

    # Add edges
    workflow.add_edge("start", "supervisor")
    
    # Add conditional edges
    def route_to_next(state: Dict) -> List[str]:
        next_step = state.get("next", "")
        if next_step == "FINISH":
            return []
        
        valid_nodes = {
            **research_agents,
            **writing_agents,
            "research_supervisor": research_supervisor,
            "writing_supervisor": writing_supervisor,
            "supervisor": top_supervisor
        }
        
        return [next_step] if next_step in valid_nodes else []

    # Connect nodes with conditional routing
    for node in [
        "supervisor",
        "research_supervisor",
        "writing_supervisor",
        *research_agents.keys(),
        *writing_agents.keys()
    ]:
        workflow.add_conditional_edges(
            node,
            route_to_next,
            [
                "supervisor",
                "research_supervisor",
                "writing_supervisor",
                *research_agents.keys(),
                *writing_agents.keys()
            ]
        )

    # Set entry point
    workflow.set_entry_point("start")

    return workflow.compile()

# Create the graph instance
graph = create_graph()
