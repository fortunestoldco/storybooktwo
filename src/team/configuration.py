"""Configuration and state classes for the hierarchical team agent.

Current Date and Time (UTC): 2025-02-12 01:37:44
Current User's Login: fortunestoldco
"""

import os
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from urllib.parse import urljoin, urlparse
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langgraph.channels import ChannelMetadata
from .env_config import EnvironmentConfig

def ensure_url_format(url: str) -> str:
    """Ensure URL is properly formatted with scheme."""
    if not url.startswith(('http://', 'https://')):
        return f"http://{url}"
    return url

def ensure_path_format(path: str) -> str:
    """Ensure path starts with /."""
    return f"/{path.lstrip('/')}"

# Define API and service configuration
API_BASE_URL = ensure_url_format(os.getenv("API_BASE_URL", "127.0.0.1:2024"))
SERVICE_PATH = ensure_path_format(os.getenv("SERVICE_PATH", "api/v1"))

# Define working directory - use temp directory as default
DEFAULT_WORKING_DIR = Path(tempfile.gettempdir()) / "storybooktwo_workspace"
WORKING_DIRECTORY = Path(os.getenv("WORKING_DIRECTORY", str(DEFAULT_WORKING_DIR)))

# Ensure the directory exists
try:
    WORKING_DIRECTORY.mkdir(parents=True, exist_ok=True)
except (PermissionError, OSError) as e:
    # Fallback to temp directory if the specified directory is not writable
    WORKING_DIRECTORY = Path(tempfile.mkdtemp(prefix="storybooktwo_"))
    print(f"Warning: Could not create specified working directory. Using temporary directory: {WORKING_DIRECTORY}")

@dataclass
class State:
    """Base state for team operations."""
    messages: List[BaseMessage] = field(default_factory=list)
    next: str = field(default="")
    session_id: str = field(default="default")
    channels: Dict[str, ChannelMetadata] = field(default_factory=dict)
    
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a value from the state dict."""
        if hasattr(self, key):
            return getattr(self, key)
        return default

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the state's message history."""
        self.messages.append(message)

@dataclass
class Configuration:
    """Configuration for the agent system."""
    model: str = "gpt-4"
    env_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    working_directory: Path = field(default=WORKING_DIRECTORY)
    api_base_url: str = field(default=API_BASE_URL)
    service_path: str = field(default=SERVICE_PATH)
    
    def __post_init__(self):
        """Ensure URLs and paths are properly formatted after initialization."""
        self.api_base_url = ensure_url_format(self.api_base_url)
        self.service_path = ensure_path_format(self.service_path)
    
    @property
    def mongodb_connection(self) -> str:
        return self.env_config.mongodb_connection
    
    @property
    def mongodb_db_name(self) -> str:
        return self.env_config.mongodb_db_name
    
    @property
    def mongodb_collection(self) -> str:
        return self.env_config.mongodb_collection
    
    @property
    def anthropic_api_key(self) -> str:
        return self.env_config.anthropic_api_key
    
    @property
    def openai_api_key(self) -> str:
        return self.env_config.openai_api_key
    
    @property
    def tavily_api_key(self) -> str:
        return self.env_config.tavily_api_key

    @property
    def api_endpoint(self) -> str:
        """Get the full API endpoint with proper path formatting."""
        return urljoin(self.api_base_url, self.service_path)

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
            "name": "storybooktwo",
            "version": "0.1.0",
            "description": "A hierarchical team agent for creative writing",
            "model": self.model,
            "base_url": self.api_base_url,
            "api_path": self.service_path,
            "working_directory": str(self.working_directory),
            "channels": {
                "supervisor": {
                    "type": "memory",
                    "retention": "session"
                },
                "research_team": {
                    "type": "memory",
                    "retention": "session"
                },
                "writing_team": {
                    "type": "memory",
                    "retention": "session"
                }
            }
        }
