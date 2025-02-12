"""Configuration and state classes for the hierarchical team agent.

Current Date and Time (UTC): 2025-02-11 21:13:25
Current User's Login: fortunestoldco
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from langchain_core.messages import BaseMessage
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from .env_config import EnvironmentConfig

@dataclass
class Configuration:
    """Configuration for the agent system."""
    model: str = "gpt-4"
    env_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    
    @property
    def mongodb_connection(self) -> str:
        return self.env_config.mongodb_connection
    
    @property
    def mongodb_db_name(self) -> str:
        return self.env_config.mongodb_db_name
    
    @property
    def mongodb_collection(self) -> str:
        return self.env_config.mongodb_collection
    
    def get_message_history(self, session_id: str) -> MongoDBChatMessageHistory:
        """Get MongoDB message history for a session."""
        return MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=self.mongodb_connection,
            database_name=self.mongodb_db_name,
            collection_name=self.mongodb_collection
        )
    
    def get_message_history(self, session_id: str) -> MongoDBChatMessageHistory:
        """Get MongoDB message history for a session."""
        return MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=self.mongodb_connection,
            database_name=self.mongodb_db_name,
            collection_name=self.mongodb_collection
        )

    def get_metadata(self) -> Dict:
        """Get metadata for langgraph."""
        return {
            "app_name": "storybooktwo",
            "version": "0.1.0",
            "api_base_url": self.api_base_url,
            "model": self.model,
        }
