"""Define state structures for the hierarchical team agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Optional, List, Dict
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


@dataclass
class InputState:
    """Input state for the agent."""

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )


@dataclass
class TeamState(InputState):
    """Base state for team-specific states."""

    team_name: str = field(default="")
    is_last_step: IsLastStep = field(default=False)
    next: str = field(default="")


@dataclass
class ResearchTeamState(TeamState):
    """State specific to the research team."""

    search_results: List[Dict] = field(default_factory=list)
    scraped_content: Dict[str, str] = field(default_factory=dict)


@dataclass
class WritingTeamState(TeamState):
    """State specific to the writing team."""

    current_document: Optional[str] = field(default=None)
    outline: List[str] = field(default_factory=list)


@dataclass
class SupervisorState(InputState):
    """State for the top-level supervisor."""

    is_last_step: IsLastStep = field(default=False)
    next: str = field(default="")
    research_complete: bool = field(default=False)
    writing_complete: bool = field(default=False)
