"""Application configuration."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    app_name: str = "Semantic Detector"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # Ollama settings
    ollama_endpoint: str = "http://localhost:11434"
    default_embedding_model: str = "nomic-embed-text"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

