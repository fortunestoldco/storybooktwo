"""Define the configurable parameters for the hierarchical team agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig, ensure_config
from src.team import prompts


@dataclass(kw_only=True)
class Configuration:
    """Configuration for the hierarchical team agent."""

    research_system_prompt: str = field(
        default=prompts.RESEARCH_SYSTEM_PROMPT,
        metadata={
            "description": "System prompt for research team interactions."
        },
    )

    writing_system_prompt: str = field(
        default=prompts.WRITING_SYSTEM_PROMPT,
        metadata={
            "description": "System prompt for writing team interactions."
        },
    )

    supervisor_system_prompt: str = field(
        default=prompts.SUPERVISOR_SYSTEM_PROMPT,
        metadata={
            "description": "System prompt for top-level supervisor interactions."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="gpt-4",
        metadata={
            "description": "Language model name for agent interactions."
        },
    )

    max_search_results: int = field(
        default=5,
        metadata={
            "description": "Maximum number of search results per query."
        },
    )

    working_directory: str = field(
        default="workspace",
        metadata={
            "description": "Directory for document storage and access."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create Configuration from RunnableConfig."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
