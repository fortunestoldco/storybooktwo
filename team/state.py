"""State definitions for the creative writing system.

Current Date and Time (UTC): 2025-02-12 00:43:52
Current User's Login: fortunestoldco
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from team.configuration import StoryDict, StoryState

@dataclass
class TeamState:
    """State for team operations."""
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

    def to_dict(self) -> StoryState:
        """Convert to dictionary representation."""
        return StoryState(
            messages=self.messages,
            next=self.next,
            session_id=self.session_id,
            input_parameters=self.input_parameters,
            story_parameters=self.story_parameters,
            research_data=self.research_data,
            metadata={
                "app_name": "storybooktwo",
                "version": "0.1.0"
            }
        )