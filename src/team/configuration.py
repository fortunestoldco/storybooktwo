"""Configuration and state classes for the creative writing agent.

Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-02-11 22:49:55
Current User's Login: fortunestoldco
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, validator
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
    num_chapters: int = Field(default=10, description="Number of chapters in the story", ge=1, le=50)
    words_per_chapter: int = Field(default=2000, description="Target words per chapter", ge=500, le=10000)

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

    def to_prompt(self) -> str:
        """Convert parameters to a prompt string."""
        plot_points_str = "\n".join(f"- {point}" for point in self.plot_points)
        total_words = self.num_chapters * self.words_per_chapter
        
        return f"""Story Parameters:
Starting Point: {self.starting_point}

Required Plot Points:
{plot_points_str}

Story Structure:
- Number of Chapters: {self.num_chapters}
- Words per Chapter: {self.words_per_chapter:,}
- Total Word Count Target: {total_words:,}

Intended Ending: {self.intended_ending}"""

    def get_chapter_distribution(self) -> Dict[str, int]:
        """Get the distribution of words across chapters."""
        return {
            "total_words": self.num_chapters * self.words_per_chapter,
            "num_chapters": self.num_chapters,
            "words_per_chapter": self.words_per_chapter,
            "average_scene_length": self.words_per_chapter // 3  # Assuming ~3 scenes per chapter
        }

class SimilarBook(BaseModel):
    """Information about a similar book."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    title: str = Field(..., description="Book title")
    author: str = Field(..., description="Book author")
    rating: float = Field(..., description="Average rating", ge=0, le=5)
    popularity: float = Field(..., description="Popularity score", ge=0, le=100)
    reviews: List[str] = Field(default_factory=list, description="Notable reviews")
    word_count: Optional[int] = Field(None, description="Total word count if available")
    chapter_count: Optional[int] = Field(None, description="Number of chapters if available")

class Scene(BaseModel):
    """Scene structure within a chapter."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    scene_number: int = Field(..., description="Scene number within the chapter")
    title: Optional[str] = Field(None, description="Scene title")
    summary: Optional[str] = Field(None, description="Brief summary of the scene")
    word_count: int = Field(default=0, description="Current word count")
    target_word_count: int = Field(..., description="Target word count for this scene")
    status: str = Field(default="not_started", description="Scene status")
    draft: Optional[str] = Field(None, description="Current draft of the scene")
    
    @validator('status')
    def validate_status(cls, v):
        """Validate scene status."""
        valid_statuses = {"not_started", "in_progress", "draft_complete", "in_review", "complete"}
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {', '.join(valid_statuses)}")
        return v

class Demographics(BaseModel):
    """Target demographic information."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    age_range: str = Field(..., description="Target age range")
    interests: List[str] = Field(default_factory=list, description="Key interests")
    reading_level: str = Field(..., description="Reading level")
    market_segment: str = Field(..., description="Market segment")
    genre_preferences: List[str] = Field(default_factory=list, description="Preferred genres")
    content_preferences: Dict[str, Any] = Field(default_factory=dict, description="Content preferences")

class MarketAnalysis(BaseModel):
    """Market analysis data."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    trends: List[str] = Field(default_factory=list, description="Current market trends")
    opportunities: List[str] = Field(default_factory=list, description="Market opportunities")
    competition: List[str] = Field(default_factory=list, description="Competing works")
    success_factors: List[str] = Field(default_factory=list, description="Key success factors")
    genre_analysis: Dict[str, Any] = Field(default_factory=dict, description="Genre-specific analysis")
    word_count_analysis: Dict[str, Any] = Field(default_factory=dict, description="Word count trends")

class ChapterStatus(BaseModel):
    """Status tracking for individual chapters."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    chapter_number: int = Field(..., description="Chapter number")
    title: Optional[str] = Field(None, description="Chapter title")
    word_count: int = Field(default=0, description="Current word count")
    target_word_count: int = Field(..., description="Target word count")
    status: str = Field(default="not_started", description="Chapter status")
    scenes: List[Scene] = Field(default_factory=list, description="Scene breakdowns")
    draft_complete: bool = Field(default=False, description="Whether the chapter draft is complete")
    review_comments: List[str] = Field(default_factory=list, description="Review feedback")
    
    @validator('status')
    def validate_status(cls, v):
        """Validate chapter status."""
        valid_statuses = {"not_started", "in_progress", "draft_complete", "in_review", "complete"}
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {', '.join(valid_statuses)}")
        return v

    def initialize_scenes(self, average_scene_length: int) -> None:
        """Initialize scenes for the chapter."""
        num_scenes = 3  # Default to 3 scenes per chapter
        scene_word_count = self.target_word_count // num_scenes
        
        for i in range(num_scenes):
            self.scenes.append(Scene(
                scene_number=i+1,
                target_word_count=scene_word_count
            ))

class MarketResearch(BaseModel):
    """Container for market research data."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    similar_books: List[SimilarBook] = Field(default_factory=list)
    target_demographic: Optional[Demographics] = None
    market_analysis: MarketAnalysis = Field(default_factory=MarketAnalysis)
    improvement_opportunities: List[str] = Field(default_factory=list)
    chapter_distribution: Dict[str, Any] = Field(default_factory=dict)
    
    def to_prompt(self) -> str:
        """Convert market research to a prompt string."""
        sections = ["Market Research Summary:"]
        
        if self.similar_books:
            sections.append("\nSimilar Books:")
            for book in self.similar_books:
                book_info = [f"- {book.title} by {book.author} (Rating: {book.rating})"]
                if book.word_count:
                    book_info.append(f"  Word Count: {book.word_count:,}")
                if book.chapter_count:
                    book_info.append(f"  Chapters: {book.chapter_count}")
                sections.append("\n".join(book_info))
        
        if self.target_demographic:
            sections.append(f"\nTarget Demographic:")
            sections.append(f"- Age Range: {self.target_demographic.age_range}")
            sections.append(f"- Reading Level: {self.target_demographic.reading_level}")
            sections.append(f"- Market Segment: {self.target_demographic.market_segment}")
            if self.target_demographic.interests:
                sections.append("- Interests: " + ", ".join(self.target_demographic.interests))
            if self.target_demographic.genre_preferences:
                sections.append("- Genre Preferences: " + ", ".join(self.target_demographic.genre_preferences))
        
        if self.chapter_distribution:
            sections.append("\nChapter Distribution:")
            sections.append(f"- Total Words: {self.chapter_distribution.get('total_words', 0):,}")
            sections.append(f"- Chapters: {self.chapter_distribution.get('num_chapters', 0)}")
            sections.append(f"- Words per Chapter: {self.chapter_distribution.get('words_per_chapter', 0):,}")
            sections.append(f"- Average Scene Length: {self.chapter_distribution.get('average_scene_length', 0):,}")
        
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
    chapters: Dict[int, ChapterStatus] = Field(default_factory=dict)
    
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
    
    def initialize_chapters(self) -> None:
        """Initialize chapter tracking."""
        if not self.story_parameters:
            return
            
        for chapter_num in range(1, self.story_parameters.num_chapters + 1):
            if chapter_num not in self.chapters:
                chapter = ChapterStatus(
                    chapter_number=chapter_num,
                    target_word_count=self.story_parameters.words_per_chapter
                )
                chapter.initialize_scenes(self.story_parameters.words_per_chapter // 3)
                self.chapters[chapter_num] = chapter
    
    def get_chapter_status(self) -> Dict[str, Any]:
        """Get overall chapter status."""
        total_words = sum(chapter.word_count for chapter in self.chapters.values())
        target_words = self.story_parameters.num_chapters * self.story_parameters.words_per_chapter if self.story_parameters else 0
        
        return {
            "total_words": total_words,
            "target_words": target_words,
            "completion_percentage": (total_words / target_words * 100) if target_words else 0,
            "chapters_complete": sum(1 for chapter in self.chapters.values() if chapter.status == "complete"),
            "chapters_in_progress": sum(1 for chapter in self.chapters.values() if chapter.status == "in_progress"),
            "chapters_not_started": sum(1 for chapter in self.chapters.values() if chapter.status == "not_started"),
            "scenes_complete": sum(
                sum(1 for scene in chapter.scenes if scene.status == "complete")
                for chapter in self.chapters.values()
            )
        }

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
