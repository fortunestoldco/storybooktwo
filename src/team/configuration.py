"""Configuration and state classes for the hierarchical team agent."""

from dataclasses import dataclass
from typing import List
from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from pathlib import Path
from tempfile import TemporaryDirectory

# Initialize working directory
_TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)

# Define state class
class State(MessagesState):
    """State class for the agent system."""
    next: str

@dataclass
class Configuration:
    """Configuration for the agent system."""
    model: str = "gpt-4"
    max_search_results: int = 5
    working_directory: Path = WORKING_DIRECTORY
