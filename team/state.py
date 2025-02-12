from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage

@dataclass
class TeamState:
    """State for team operations."""
    messages: List[BaseMessage] = field(default_factory=list)
    next: str = field(default="")
    session_id: str = field(default="default")
    input_parameters: Dict[str, Any] = field(default_factory=dict)
    story_parameters: Optional[Dict[str, Any]] = None
    research_data: Dict[str, Any] = field(default_factory=lambda: {
        "market_analysis": {},
        "audience_data": {},
        "improvement_opportunities": []
    })
