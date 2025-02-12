"""Configuration and state classes for the hierarchical team agent.

Current Date and Time (UTC): 2025-02-11 23:55:35
Current User's Login: fortunestoldco
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
from tempfile import TemporaryDirectory
from langchain_core.messages import BaseMessage
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langsmith import Client
from dotenv import load_dotenv

# Load environment variables from project root
load_dotenv(Path(__file__).parent.parent / '.env')

# Set default user agent for langchain
os.environ["USER_AGENT"] = "storybooktwo/0.1.0"

# Required environment variables
required_env_vars = [
    "MONGODB_CONNECTION_STRING",
    "MONGODB_DATABASE_NAME",
    "OPENAI_API_KEY",
    "TAVILY_API_KEY",
    "LANGSMITH_API_KEY",  # Updated from LANGCHAIN_API_KEY
    "LANGSMITH_PROJECT",  # Updated from LANGCHAIN_PROJECT
]

# Check for required environment variables
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize working directory
_TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)

@dataclass
class State:
    """Base state for team operations."""
    messages: List[BaseMessage] = field(default_factory=list)
    next: str = field(default="")
    session_id: str = field(default="default")
    input_parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict = field(default_factory=lambda: {
        "app_name": "storybooktwo",
        "version": "0.1.0",
        "api_base_url": "http://127.0.0.1:2024"
    })
    
    def get(self, key: str, default: Optional[any] = None) -> any:
        """Get a value from the state dict."""
        if key == "messages":
            return self.messages
        if key == "session_id":
            return self.session_id
        if key == "metadata":
            return self.metadata
        if key == "input_parameters":
            return self.input_parameters
        return default

@dataclass
class Configuration:
    """Configuration for the agent system."""
    model: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4")
    max_search_results: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    working_directory: Path = WORKING_DIRECTORY
    mongodb_connection: str = os.getenv("MONGODB_CONNECTION_STRING")
    mongodb_db_name: str = os.getenv("MONGODB_DATABASE_NAME")
    mongodb_collection: str = os.getenv("MONGODB_COLLECTION_NAME", "chat_histories")
    api_base_url: str = os.getenv("API_BASE_URL", "http://127.0.0.1:2024")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "storybooktwo")  # Updated from LANGCHAIN_PROJECT
    
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

    def get_metadata(self) -> Dict:
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
        return {
            "session_id": state.session_id,
            "input_parameters": state.input_parameters,
            **self.get_metadata()
        }
