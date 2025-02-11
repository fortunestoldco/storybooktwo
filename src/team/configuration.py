"""Configuration and state classes for the creative writing agent.

Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-02-11 22:31:12
Current User's Login: fortunestoldco
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

class MessageWrapper(BaseModel):
    """Wrapper for BaseMessage to make it JSON serializable."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    type: str = Field(..., description="Message type (human/ai/system)")
    content: str = Field(..., description="Message content")
    
    @classmethod
    def from_message(cls, message: BaseMessage) -> "MessageWrapper":
        """Create a wrapper from a BaseMessage."""
        return cls(
            type=message.__class__.__name__.lower().replace('message', ''),
            content=message.content
        )
    
    def to_message(self) -> BaseMessage:
        """Convert wrapper back to BaseMessage."""
        message_types = {
            'human': HumanMessage,
            'ai': AIMessage,
            'system': SystemMessage
        }
        message_class = message_types.get(self.type, SystemMessage)
        return message_class(content=self.content)

class StoryParameters(BaseModel):
    """Parameters for story generation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    starting_point: str = Field(..., description="The starting point of the story")
    plot_points: List[str] = Field(default_factory=list, description="Required plot points")
    intended_ending: str = Field(..., description="The intended ending of the story")

    def to_prompt(self) -> str:
        """Convert parameters to a prompt string."""
        plot_points_str = "\n".join(f"- {point}" for point in self.plot_points)
        return f"""Story Parameters:
Starting Point: {self.starting_point}

Required Plot Points:
{plot_points_str}

Intended Ending: {self.intended_ending}"""

class SimilarBook(BaseModel):
    """Information about a similar book."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    title: str = Field(..., description="Book title")
    author: str = Field(..., description="Book author")
    rating: float = Field(..., description="Average rating")
    popularity: float = Field(..., description="Popularity score")
    reviews: List[str] = Field(default_factory=list, description="Notable reviews")

class Demographics(BaseModel):
    """Target demographic information."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    age_range: str = Field(..., description="Target age range")
    interests: List[str] = Field(default_factory=list, description="Key interests")
    reading_level: str = Field(..., description="Reading level")
    market_segment: str = Field(..., description="Market segment")

class MarketAnalysis(BaseModel):
    """Market analysis data."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    trends: List[str] = Field(default_factory=list, description="Current market trends")
    opportunities: List[str] = Field(default_factory=list, description="Market opportunities")
    competition: List[str] = Field(default_factory=list, description="Competing works")

class MarketResearch(BaseModel):
    """Container for market research data."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    similar_books: List[SimilarBook] = Field(default_factory=list)
    target_demographic: Optional[Demographics] = None
    market_analysis: MarketAnalysis = Field(default_factory=MarketAnalysis)
    improvement_opportunities: List[str] = Field(default_factory=list)

class State(BaseModel):
    """Base state for team operations."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    messages: List[MessageWrapper] = Field(default_factory=list)
    next: str = Field(default="")
    session_id: str = Field(default="default")
    story_parameters: Optional[StoryParameters] = None
    market_research: MarketResearch = Field(default_factory=MarketResearch)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the state dict."""
        if hasattr(self, key):
            return getattr(self, key)
        return default
    
    def get_messages(self) -> List[BaseMessage]:
        """Get the actual messages from wrappers."""
        return [msg.to_message() for msg in self.messages]

class Configuration:
    """Configuration for the agent system."""
    def __init__(
        self,
        model: str = "gpt-4",
        mongodb_connection: str = "",
        mongodb_db_name: str = "creative_writing",
        mongodb_collection: str = "story_development"
    ):
        self.model = model
        self.mongodb_connection = mongodb_connection
        self.mongodb_db_name = mongodb_db_name
        self.mongodb_collection = mongodb_collection
    
    def get_message_history(self, session_id: str) -> MongoDBChatMessageHistory:
        """Get MongoDB message history for a session."""
        return MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=self.mongodb_connection,
            database_name=self.mongodb_db_name,
            collection_name=self.mongodb_collection
        )
