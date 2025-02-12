"""Configuration and state classes for the creative writing agent.

Current Date and Time (UTC): 2025-02-11 21:58:52
Current User's Login: fortunestoldco
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from langchain_core.messages import BaseMessage
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

@dataclass
class StoryParameters:
    """Parameters for story generation."""
    starting_point: str
    plot_points: List[str]
    intended_ending: str
    
    def to_prompt(self) -> str:
        """Convert parameters to a prompt string."""
        plot_points_str = "\n".join(f"- {point}" for point in self.plot_points)
        return f"""Story Parameters:
Starting Point: {self.starting_point}

Required Plot Points:
{plot_points_str}

Intended Ending: {self.intended_ending}"""

@dataclass
class MarketResearch:
    """Container for market research data."""
    similar_books: List[Dict[str, str]] = field(default_factory=list)
    target_demographic: Dict[str, any] = field(default_factory=dict)
    market_analysis: Dict[str, any] = field(default_factory=dict)
    improvement_opportunities: List[str] = field(default_factory=list)

@dataclass
class State:
    """Base state for team operations."""
    messages: List[BaseMessage] = field(default_factory=list)
    next: str = field(default="")
    session_id: str = field(default="default")
    story_parameters: Optional[StoryParameters] = None
    market_research: MarketResearch = field(default_factory=MarketResearch)
    
    def get(self, key: str, default: Optional[any] = None) -> any:
        """Get a value from the state dict."""
        if key == "messages":
            return self.messages
        if key == "session_id":
            return self.session_id
        if key == "story_parameters":
            return self.story_parameters
        if key == "market_research":
            return self.market_research
        return default

@dataclass
class Configuration:
    """Configuration for the agent system."""
    model: str = "gpt-4"
    mongodb_connection: str = ""
    mongodb_db_name: str = "creative_writing"
    mongodb_collection: str = "story_development"
    
    def get_message_history(self, session_id: str) -> MongoDBChatMessageHistory:
        """Get MongoDB message history for a session."""
        return MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=self.mongodb_connection,
            database_name=self.mongodb_db_name,
            collection_name=self.mongodb_collection
        )
