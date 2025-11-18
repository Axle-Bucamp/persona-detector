"""Pydantic models for request/response validation."""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class ClusteringMethod(str, Enum):
    """Clustering method options."""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"


class AnalysisRequest(BaseModel):
    """Request model for text analysis."""
    
    text: Optional[str] = Field(None, description="Text content to analyze")
    file_path: Optional[str] = Field(None, description="Path to text file")
    ollama_endpoint: Optional[str] = Field(None, description="Ollama API endpoint")
    embedding_model: Optional[str] = Field(None, description="Embedding model name")
    n_clusters: Optional[int] = Field(None, ge=1, description="Number of clusters")
    clustering_method: ClusteringMethod = Field(
        ClusteringMethod.KMEANS, 
        description="Clustering algorithm"
    )
    
    @validator('text', 'file_path')
    def validate_input(cls, v, values):
        """Ensure either text or file_path is provided."""
        if not v and not values.get('file_path') and not values.get('text'):
            raise ValueError("Either 'text' or 'file_path' must be provided")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "The quick brown fox jumps. This is a test sentence.",
                "ollama_endpoint": "http://localhost:11434",
                "embedding_model": "nomic-embed-text",
                "n_clusters": 3,
                "clustering_method": "kmeans"
            }
        }


class StyleSearchRequest(BaseModel):
    """Request model for style search."""
    
    sentence_index: int = Field(ge=0, description="Index of sentence to find similar styles for")
    k: int = Field(5, ge=1, le=50, description="Number of similar styles to return")
    in_cluster_only: bool = Field(False, description="Search only within same cluster")
    out_cluster_only: bool = Field(False, description="Search only outside cluster")


class ClusterStats(BaseModel):
    """Cluster statistics model."""
    
    cluster_id: int
    count: int
    avg_length: float
    sample_texts: List[str]


class FingerprintSummary(BaseModel):
    """Fingerprint summary model."""
    
    top_words: List[str]
    top_bigrams: List[str]
    top_trigrams: List[str]
    avg_sentence_length: float
    vocab_richness: float
    punctuation_style: Dict[str, float]


class StyleMatch(BaseModel):
    """Style match result model."""
    
    index: int
    similarity: float
    cluster_id: int
    text: str


class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    
    success: bool
    message: str
    num_sentences: int
    num_clusters: int
    cluster_stats: Dict[str, ClusterStats]
    fingerprint_summaries: Dict[str, FingerprintSummary]
    output_files: Dict[str, str]
    metadata: Dict[str, Any]
    tfidf_features: Optional[Dict[str, List[Union[List, Dict]]]] = None
    sentiment_analysis: Optional[List[Dict[str, Any]]] = None
    cluster_sentiments: Optional[Dict[str, Dict[str, Any]]] = None
    overall_sentiment: Optional[Dict[str, Any]] = None
    overall_person_score: Optional[float] = None
    coords_3d: Optional[List[List[float]]] = None
    labels: Optional[List[int]] = None
    sentences: Optional[List[str]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Analysis completed successfully",
                "num_sentences": 10,
                "num_clusters": 3,
                "cluster_stats": {},
                "fingerprint_summaries": {},
                "output_files": {},
                "metadata": {}
            }
        }


class StyleSearchResponse(BaseModel):
    """Response model for style search."""
    
    query_index: int
    query_text: str
    query_cluster: int
    in_cluster_matches: List[StyleMatch]
    out_cluster_matches: List[StyleMatch]
    fingerprint_summary: FingerprintSummary

