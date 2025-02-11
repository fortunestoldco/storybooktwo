"""Configuration and state classes for the creative writing agent.

Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-02-11 23:12:22
Current User's Login: fortunestoldco
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, validator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

class InputConfig(BaseModel):
    """Input configuration for story generation."""
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
        description="Number of chapters in the story",
        ge=1,
        le=50
    )
    words_per_chapter: int = Field(
        default=2000,
        description="Target words per chapter",
        ge=500,
        le=10000
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

[Previous MessageWrapper, StoryParameters, Scene, ChapterStatus, SimilarBook, 
Demographics, MarketAnalysis, MarketResearch classes remain exactly the same]

class State(BaseModel):
    """Base state for team operations."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    messages: List[MessageWrapper] = Field(default_factory=list)
    story_parameters: Optional[StoryParameters] = None
    market_research: MarketResearch = Field(default_factory=MarketResearch)
    next: str = Field(default="")
    current_supervisor: str = Field(default="supervisor")
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
        if not self.story_parameters:
            return {}
            
        total_words = sum(chapter.word_count for chapter in self.chapters.values())
        target_words = self.story_parameters.num_chapters * self.story_parameters.words_per_chapter
        
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

def process_input(inputs: Dict) -> InputConfig:
    """Process and validate the input dictionary."""
    try:
        return InputConfig(**inputs)
    except Exception as e:
        raise ValueError(f"Invalid input parameters: {str(e)}") from e

def create_initial_state(validated_input: InputConfig) -> State:
    """Create initial state from validated input."""
    story_parameters = StoryParameters(
        starting_point=validated_input.starting_point,
        plot_points=validated_input.plot_points,
        intended_ending=validated_input.intended_ending,
        num_chapters=validated_input.num_chapters,
        words_per_chapter=validated_input.words_per_chapter
    )
    
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
