"""
Clustering module for identifying speakers and contexts.
Uses multiple clustering algorithms and feature combinations.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class SpeakerClusterer:
    """Cluster texts to identify speakers and contexts."""
    
    def __init__(self, n_clusters: Optional[int] = None, method: str = 'kmeans'):
        """
        Initialize clusterer.
        
        Args:
            n_clusters: Number of clusters (None for auto-detection)
            method: Clustering method ('kmeans', 'dbscan', 'hierarchical')
        """
        self.n_clusters = n_clusters
        self.method = method
        self.scaler = StandardScaler()
        self.clusterer = None
        self.labels_ = None
    
    def fit(self, features: np.ndarray, embeddings: Optional[np.ndarray] = None):
        """
        Fit clustering model.
        
        Args:
            features: Language fingerprint features (deprecated - kept for backward compatibility)
            embeddings: Embedding vectors (primary clustering input)
        
        Note: Clustering now uses embeddings only. Features parameter is kept for backward compatibility
        but is ignored when embeddings are provided.
        """
        # Use embeddings only if provided (new approach)
        if embeddings is not None:
            # Normalize embeddings (L2 normalization)
            embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            combined = embeddings_norm
        elif features is not None:
            # Fallback to features if no embeddings (backward compatibility)
            combined = self.scaler.fit_transform(features)
        else:
            raise ValueError("Either embeddings or features must be provided")
        
        # Determine number of clusters if not specified
        n_clusters = self.n_clusters
        if n_clusters is None:
            n_clusters = self._estimate_clusters(combined)
        
        # Initialize clusterer
        if self.method == 'kmeans':
            self.clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif self.method == 'dbscan':
            self.clusterer = DBSCAN(eps=0.5, min_samples=3)
        elif self.method == 'hierarchical':
            self.clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.labels_ = self.clusterer.fit_predict(combined)
        
        return self.labels_
    
    def _estimate_clusters(self, data: np.ndarray) -> int:
        """Estimate optimal number of clusters."""
        n_samples = len(data)
        if n_samples < 2:
            return 1
        
        # Use elbow method or simple heuristic
        max_clusters = min(10, n_samples // 2)
        if max_clusters < 2:
            return 2
        
        # Simple heuristic: use sqrt of sample size, capped
        estimated = max(2, int(np.sqrt(n_samples)))
        return min(estimated, max_clusters)
    
    def predict(self, features: np.ndarray, embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            features: Language fingerprint features (deprecated - kept for backward compatibility)
            embeddings: Embedding vectors (primary input)
            
        Returns:
            Cluster labels
        """
        if self.clusterer is None:
            raise ValueError("Model must be fitted first")
        
        # Use embeddings only if provided (matching fit() behavior)
        if embeddings is not None:
            embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            combined = embeddings_norm
        elif features is not None:
            # Fallback to features if no embeddings (backward compatibility)
            combined = self.scaler.transform(features)
        else:
            raise ValueError("Either embeddings or features must be provided")
        
        if hasattr(self.clusterer, 'predict'):
            return self.clusterer.predict(combined)
        else:
            # DBSCAN doesn't have predict, need to refit
            return self.clusterer.fit_predict(combined)
    
    def get_cluster_stats(self, texts: List[str]) -> Dict[int, Dict]:
        """
        Get statistics for each cluster.
        
        Args:
            texts: List of texts corresponding to labels
            
        Returns:
            Dictionary mapping cluster ID to statistics
        """
        if self.labels_ is None:
            return {}
        
        stats = {}
        for cluster_id in set(self.labels_):
            if cluster_id == -1:  # Noise in DBSCAN
                continue
            
            cluster_texts = [texts[i] for i, label in enumerate(self.labels_) if label == cluster_id]
            
            stats[cluster_id] = {
                'count': len(cluster_texts),
                'avg_length': np.mean([len(t) for t in cluster_texts]),
                'sample_texts': cluster_texts[:3],  # Sample texts
            }
        
        return stats

