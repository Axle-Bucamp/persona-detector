"""
Embedding generator using Ollama API.
Supports multiple embedding models with configurable endpoints.
"""

import requests
import json
from typing import List, Optional, Dict
import numpy as np
import time


class EmbeddingGenerator:
    """Generate embeddings using Ollama API."""
    
    def __init__(self, endpoint: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        """
        Initialize embedding generator.
        
        Args:
            endpoint: Ollama API endpoint URL
            model: Model name to use for embeddings
        """
        self.endpoint = endpoint.rstrip('/')
        self.model = model
        # Try both endpoint formats (older: /api/embeddings, newer: /api/embed)
        self.api_urls = [
            f"{self.endpoint}/api/embeddings",
            f"{self.endpoint}/api/embed"
        ]
        self.api_url = self.api_urls[0]  # Default to first
        self.embedding_dim = None  # Will be detected from first embedding
    
    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        # Try both endpoint formats
        for api_url in self.api_urls:
            try:
                # Try with 'prompt' (older format)
                payload = {"model": self.model, "prompt": text}
                response = requests.post(api_url, json=payload, timeout=30)
                
                # If 404, try with 'input' (newer format)
                if response.status_code == 404 and api_url.endswith('/embed'):
                    payload = {"model": self.model, "input": text}
                    response = requests.post(api_url, json=payload, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    embedding = np.array(data.get('embedding', []))
                    
                    # Update working endpoint
                    self.api_url = api_url
                    
                    # Store dimension for fallback
                    if self.embedding_dim is None and len(embedding) > 0:
                        self.embedding_dim = len(embedding)
                    
                    return embedding
                elif response.status_code == 404:
                    # Model not found - continue to next endpoint or return None
                    continue
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.ConnectionError:
                # Connection error - skip this endpoint
                continue
            except requests.exceptions.HTTPError:
                # HTTP error - try next endpoint
                continue
            except Exception:
                # Other errors - try next endpoint
                continue
        
        # All endpoints failed - likely model not installed
        return None
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 10, delay: float = 0.1) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts with batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in parallel
            delay: Delay between batches in seconds
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        failed_count = 0
        first_error_shown = False
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch:
                embedding = self.generate_embedding(text)
                if embedding is not None:
                    batch_embeddings.append(embedding)
                else:
                    failed_count += 1
                    # Use zero vector as fallback
                    dim = self.embedding_dim if self.embedding_dim else 768
                    batch_embeddings.append(np.zeros(dim))
                    
                    # Show error message only once
                    if not first_error_shown:
                        print(f"[WARNING] Model '{self.model}' not found or embeddings unavailable.")
                        print(f"[INFO] Install model: ollama pull {self.model}")
                        print(f"[INFO] Using fallback zero vectors for embeddings.")
                        first_error_shown = True
                
                time.sleep(delay)
            
            embeddings.extend(batch_embeddings)
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
        
        if failed_count > 0:
            print(f"[INFO] {failed_count}/{len(texts)} embeddings failed, using fallback vectors")
        
        return embeddings
    
    def test_connection(self) -> bool:
        """Test if Ollama endpoint is accessible."""
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available embedding models."""
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return models
        except:
            return []

