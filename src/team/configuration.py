"""Configuration and state classes for the creative writing agent.

Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-02-11 22:45:16
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
    additional_kwargs: Dict = Field(default_factory=dict, description="Additional message kwargs")
    
    @classmethod
    def from_message(cls, message: BaseMessage) -> "MessageWrapper":
        """Create a wrapper from a BaseMessage."""
        return cls(
            type=message.__class__.__name__.lower().replace('message', ''),
            content=message.content,
            additional_kwargs=message.additional_kwargs
        )
    
    def to_message(self) -> BaseMessage:
        """Convert wrapper back to BaseMessage."""
        message_types = {
            'human': HumanMessage,
            'ai': AIMessage,
            'system': SystemMessage
        }
        message_class = message_types.get(self.type, SystemMessage)
        return message_class(
            content=self.content,
            additional_kwargs=self.additional_kwargs
        )

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
    success_factors: List[str] = Field(default_factory=list, description="Key success factors")

class MarketResearch(BaseModel):
    """Container for market research data."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    similar_books: List[SimilarBook] = Field(default_factory=list)
    target_demographic: Optional[Demographics] = None
    market_analysis: MarketAnalysis = Field(default_factory=MarketAnalysis)
    improvement_opportunities: List[str] = Field(default_factory=list)
    
    def to_prompt(self) -> str:
        """Convert market research to a prompt string."""
        sections = ["Market Research Summary:"]
        
        if self.similar_books:
            sections.append("\nSimilar Books:")
            for book in self.similar_books:
                sections.append(f"- {book.title} by {book.author} (Rating: {book.rating})")
        
        if self.target_demographic:
            sections.append(f"\nTarget Demographic:")
            sections.append(f"- Age Range: {self.target_demographic.age_range}")
            sections.append(f"- Reading Level: {self.target_demographic.reading_level}")
            sections.append(f"- Market Segment: {self.target_demographic.market_segment}")
            if self.target_demographic.interests:
                sections.append("- Interests: " + ", ".join(self.target_demographic.interests))
        
        if self.market_analysis.trends:
            sections.append("\nMarket Trends:")
            sections.extend(f"- {trend}" for trend in self.market_analysis.trends)
        
        if self.improvement_opportunities:
            sections.append("\nImprovement Opportunities:")
            sections.extend(f"- {opp}" for opp in self.improvement_opportunities)
        
        return "\n".join(sections)

class State(BaseModel):
    """Base state for team operations."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    messages: List[MessageWrapper] = Field(default_factory=list)
    story_parameters: Optional[StoryParameters] = None
    market_research: MarketResearch = Field(default_factory=MarketResearch)
    next: str = Field(default="")
    session_id: str = Field(default="default")
    
    def get_messages(self) -> List[BaseMessage]:
        """Get the actual messages from wrappers."""
        return [msg.to_message() for msg in self.messages]
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a new message to the state."""
        self.messages.append(MessageWrapper.from_message(message))
    
    def get_last_message(self) -> Optional[BaseMessage]:
        """Get the last message if any exist."""
        if not self.messages:
            return None
        return self.messages[-1].to_message()

class Configuration:
    """Configuration for the agent system."""
    def __init__(
        self,
        model: str = "gpt-4",
        mongodb_connection: Optional[str] = None,
        mongodb_db_name: str = "creative_writing",
        mongodb_collection: str = "story_development"
    ):
        """Initialize configuration.
        
        Args:
            model: The model to use for chat completions
            mongodb_connection: MongoDB connection string (optional)
            mongodb_db_name: MongoDB database name
            mongodb_collection: MongoDB collection name
        """
        self.model = model
        self.mongodb_connection = mongodb_connection
        self.mongodb_db_name = mongodb_db_name
        self.mongodb_collection = mongodb_collection
    
    def get_message_history(self, session_id: str) -> MongoDBChatMessageHistory:
        """Get MongoDB message history for a session."""
        if not self.mongodb_connection:
            return MongoDBChatMessageHistory(
                session_id=session_id,
                connection_string="mongodb://localhost:27017/",
                database_name=self.mongodb_db_name,
                collection_name=self.mongodb_collection
            )
        
        return MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=self.mongodb_connection,
            database_name=self.mongodb_db_name,
            collection_name=self.mongodb_collection
        )
