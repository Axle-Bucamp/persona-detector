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
    
    # File upload settings
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: list[str] = [".txt", ".md", ".csv", ".json", ".pdf"]
    
    # Processing settings
    default_clustering_method: str = "kmeans"
    default_n_clusters: Optional[int] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

