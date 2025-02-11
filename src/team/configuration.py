"""Configuration and state classes for the hierarchical team agent.

Current Date and Time (UTC): 2025-02-11 21:01:56
Current User's Login: fortunestoldco
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
from tempfile import TemporaryDirectory
from langchain_core.messages import BaseMessage

# Initialize working directory
_TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)

@dataclass
class State:
    """Base state for team operations."""
    messages: List[BaseMessage] = field(default_factory=list)
    next: str = field(default="")
    
    def get(self, key: str, default: Optional[any] = None) -> any:
        """Get a value from the state dict."""
        if key == "messages":
            return self.messages
        return default

@dataclass
class Configuration:
    """Configuration for the agent system."""
    model: str = "gpt-4"
    max_search_results: int = 5
    working_directory: Path = WORKING_DIRECTORY
