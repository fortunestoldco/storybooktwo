"""Environment configuration module for Cloud Run compatibility."""

import os
from typing import Optional

class EnvironmentConfig:
    """Environment configuration handler."""
    
    @staticmethod
    def get_required_env(key: str) -> str:
        """Get a required environment variable or raise an error."""
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    @staticmethod
    def get_optional_env(key: str, default: str) -> str:
        """Get an optional environment variable with a default value."""
        return os.getenv(key, default)
    
    # API Keys
    @property
    def anthropic_api_key(self) -> str:
        return self.get_required_env("ANTHROPIC_API_KEY")
    
    @property
    def openai_api_key(self) -> str:
        return self.get_required_env("OPENAI_API_KEY")
    
    @property
    def tavily_api_key(self) -> str:
        return self.get_required_env("TAVILY_API_KEY")
    
    # MongoDB Configuration
    @property
    def mongodb_connection(self) -> str:
        return self.get_required_env("MONGODB_CONNECTION")
    
    @property
    def mongodb_db_name(self) -> str:
        return self.get_optional_env("MONGODB_DB_NAME", "creative_writing")
    
    @property
    def mongodb_collection(self) -> str:
        return self.get_optional_env("MONGODB_COLLECTION", "story_development")
