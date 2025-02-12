"""Configuration for the creative writing system.

Current Date and Time (UTC): 2025-02-12 00:43:52
Current User's Login: fortunestoldco
"""

import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, TypedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from langchain_core.messages import BaseMessage
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langsmith import Client
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv(Path(__file__).parent.parent / '.env')

# Set default user agent
os.environ["USER_AGENT"] = "storybooktwo/0.1.0"

required_env_vars = [
    "MONGODB_CONNECTION_STRING",
    "MONGODB_DATABASE_NAME",
    "OPENAI_API_KEY",
    "TAVILY_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
]

# Check for required environment variables
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize working directory
_TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)

class StoryDict(TypedDict):
    """TypedDict for story parameters."""
    start: str
    plot_points: List[str]
    ending: str
    genre: Optional[str]
    target_length: Optional[int]

@dataclass
class StoryParameters:
    """Parameters for story generation."""
    start: str
    plot_points: List[str]
    ending: str
    genre: Optional[str] = None
    target_length: Optional[int] = None

    def to_dict(self) -> StoryDict:
        """Convert to dictionary."""
        return StoryDict(
            start=self.start,
            plot_points=self.plot_points,
            ending=self.ending,
            genre=self.genre,
            target_length=self.target_length
        )

class StoryState(TypedDict):
    """Type definition for story state."""
    messages: List[BaseMessage]
    next: str
    session_id: str
    input_parameters: Dict[str, Any]
    story_parameters: StoryDict
    research_data: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class State:
    """Base state for story creation operations."""
    messages: List[BaseMessage] = field(default_factory=list)
    next: str = field(default="")
    session_id: str = field(default="default")
    input_parameters: Dict[str, Any] = field(default_factory=dict)
    story_parameters: StoryDict = field(default_factory=lambda: StoryDict(
        start="",
        plot_points=[],
        ending="",
        genre=None,
        target_length=None
    ))
    research_data: Dict[str, Any] = field(default_factory=lambda: {
        "market_analysis": {},
        "audience_data": {},
        "improvement_opportunities": []
    })
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "app_name": "storybooktwo",
        "version": "0.1.0",
        "api_base_url": "http://127.0.0.1:2024"
    })

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a value from the state dict."""
        if hasattr(self, key):
            return getattr(self, key)
        return default

    def to_dict(self) -> StoryState:
        """Convert state to dictionary."""
        return StoryState(
            messages=self.messages,
            next=self.next,
            session_id=self.session_id,
            input_parameters=self.input_parameters,
            story_parameters=self.story_parameters,
            research_data=self.research_data,
            metadata=self.metadata
        )

    def ensure_story_parameters(self) -> None:
        """Ensure story parameters are properly initialized."""
        if "initial_request" in self.input_parameters:
            req = self.input_parameters["initial_request"]
            self.story_parameters = StoryDict(
                start=req.get("start", ""),
                plot_points=req.get("plot_points", []),
                ending=req.get("ending", ""),
                genre=req.get("genre"),
                target_length=req.get("target_length")
            )

@dataclass
class Configuration:
    """Configuration for the creative writing system."""
    model: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4")
    max_search_results: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    working_directory: Path = WORKING_DIRECTORY
    mongodb_connection: str = os.getenv("MONGODB_CONNECTION_STRING")
    mongodb_db_name: str = os.getenv("MONGODB_DATABASE_NAME")
    mongodb_collection: str = os.getenv("MONGODB_COLLECTION_NAME", "story_histories")
    api_base_url: str = os.getenv("API_BASE_URL", "http://127.0.0.1:2024")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "storybooktwo")
    
    def __post_init__(self):
        """Initialize LangSmith client."""
        self.langsmith_client = Client()

    def get_message_history(self, session_id: str) -> MongoDBChatMessageHistory:
        """Get MongoDB message history for a session."""
        return MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=self.mongodb_connection,
            database_name=self.mongodb_db_name,
            collection_name=self.mongodb_collection
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for langgraph."""
        return {
            "app_name": "storybooktwo",
            "version": "0.1.0",
            "api_base_url": self.api_base_url,
            "model": self.model,
            "project": self.langsmith_project
        }

    def create_run_metadata(self, state: State) -> Dict[str, Any]:
        """Create run metadata for LangSmith tracking."""
        state.ensure_story_parameters()
        return {
            **self.get_metadata(),
            **state.input_parameters,
            **state.story_parameters,  # Spread story parameters directly at root level
            "session_id": state.session_id,
            "research_data": state.research_data,
            "current_phase": state.input_parameters.get("current_phase", "market_research")
        }