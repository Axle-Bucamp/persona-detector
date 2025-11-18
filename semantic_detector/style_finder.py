"""
Style finder module for identifying closest styles using embeddings and clusters.
Implements in-cluster and out-of-cluster fingerprint matching.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from collections import defaultdict


class StyleFinder:
    """Find closest styles using embeddings and cluster information."""
    
    def __init__(self, embeddings: np.ndarray, fingerprints: np.ndarray, labels: np.ndarray):
        """
        Initialize style finder.
        
        Args:
            embeddings: Embedding vectors (n_samples, embedding_dim)
            fingerprints: Fingerprint vectors (n_samples, fingerprint_dim)
            labels: Cluster labels (n_samples,)
        """
        self.embeddings = embeddings
        self.fingerprints = fingerprints
        self.labels = labels
        self.n_samples = len(embeddings)
        
        # Normalize embeddings for cosine similarity
        self.embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Build cluster indices
        self.cluster_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.cluster_indices[label].append(idx)
    
    def find_closest_styles(
        self,
        query_idx: int,
        k: int = 5,
        use_embeddings: bool = True,
        use_fingerprints: bool = True,
        exclude_self: bool = True,
        in_cluster_only: bool = False,
        out_cluster_only: bool = False
    ) -> List[Tuple[int, float, str]]:
        """
        Find closest styles to a given text.
        
        Args:
            query_idx: Index of query text
            k: Number of closest styles to return
            use_embeddings: Whether to use embedding similarity
            use_fingerprints: Whether to use fingerprint similarity
            exclude_self: Whether to exclude the query text itself
            in_cluster_only: Only search within the same cluster
            out_cluster_only: Only search outside the query's cluster
            
        Returns:
            List of tuples (index, similarity_score, cluster_info)
        """
        if query_idx < 0 or query_idx >= self.n_samples:
            raise ValueError(f"Invalid query index: {query_idx}")
        
        query_cluster = self.labels[query_idx]
        
        # Determine search space
        if in_cluster_only:
            search_indices = [i for i in self.cluster_indices[query_cluster] if i != query_idx or not exclude_self]
        elif out_cluster_only:
            search_indices = [i for i in range(self.n_samples) 
                            if self.labels[i] != query_cluster and (i != query_idx or not exclude_self)]
        else:
            search_indices = list(range(self.n_samples))
            if exclude_self:
                search_indices.remove(query_idx)
        
        if not search_indices:
            return []
        
        # Calculate similarities
        similarities = np.zeros(len(search_indices))
        
        if use_embeddings:
            query_emb = self.embeddings_norm[query_idx:query_idx+1]
            search_embs = self.embeddings_norm[search_indices]
            emb_sim = cosine_similarity(query_emb, search_embs)[0]
            similarities += emb_sim * 0.6  # Weight embeddings
        
        if use_fingerprints:
            query_fp = self.fingerprints[query_idx:query_idx+1]
            search_fps = self.fingerprints[search_indices]
            fp_sim = cosine_similarity(query_fp, search_fps)[0]
            similarities += fp_sim * 0.4  # Weight fingerprints
        
        # Get top k
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            original_idx = search_indices[idx]
            similarity = similarities[idx]
            cluster_label = self.labels[original_idx]
            cluster_info = f"Cluster {cluster_label}" + (
                " (same)" if cluster_label == query_cluster else " (different)"
            )
            results.append((original_idx, float(similarity), cluster_info))
        
        return results
    
    def find_closest_in_cluster(
        self,
        query_idx: int,
        k: int = 5,
        exclude_self: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Find closest styles within the same cluster (excluding self).
        
        Args:
            query_idx: Index of query text
            k: Number of closest styles to return
            exclude_self: Whether to exclude the query text itself
            
        Returns:
            List of tuples (index, similarity_score)
        """
        results = self.find_closest_styles(
            query_idx,
            k=k,
            exclude_self=exclude_self,
            in_cluster_only=True
        )
        return [(idx, sim) for idx, sim, _ in results]
    
    def find_closest_out_cluster(
        self,
        query_idx: int,
        k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find closest styles outside the query's cluster.
        
        Args:
            query_idx: Index of query text
            k: Number of closest styles to return
            
        Returns:
            List of tuples (index, similarity_score)
        """
        results = self.find_closest_styles(
            query_idx,
            k=k,
            exclude_self=False,  # Can include self if needed
            out_cluster_only=True
        )
        return [(idx, sim) for idx, sim, _ in results]
    
    def get_cluster_style_profile(self, cluster_id: int) -> Dict[str, any]:
        """
        Get style profile for a cluster (average fingerprint).
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            Dictionary with cluster style profile
        """
        if cluster_id not in self.cluster_indices:
            return {}
        
        indices = self.cluster_indices[cluster_id]
        
        # Average embeddings
        avg_embedding = np.mean(self.embeddings[indices], axis=0)
        
        # Average fingerprints
        avg_fingerprint = np.mean(self.fingerprints[indices], axis=0)
        
        return {
            'cluster_id': cluster_id,
            'size': len(indices),
            'avg_embedding': avg_embedding,
            'avg_fingerprint': avg_fingerprint,
            'indices': indices,
        }
    
    def compare_clusters(self, cluster_id1: int, cluster_id2: int) -> Dict[str, float]:
        """
        Compare two clusters' style profiles.
        
        Args:
            cluster_id1: First cluster ID
            cluster_id2: Second cluster ID
            
        Returns:
            Dictionary with comparison metrics
        """
        profile1 = self.get_cluster_style_profile(cluster_id1)
        profile2 = self.get_cluster_style_profile(cluster_id2)
        
        if not profile1 or not profile2:
            return {}
        
        # Compare embeddings
        emb_sim = cosine_similarity(
            profile1['avg_embedding'].reshape(1, -1),
            profile2['avg_embedding'].reshape(1, -1)
        )[0][0]
        
        # Compare fingerprints
        fp_sim = cosine_similarity(
            profile1['avg_fingerprint'].reshape(1, -1),
            profile2['avg_fingerprint'].reshape(1, -1)
        )[0][0]
        
        return {
            'embedding_similarity': float(emb_sim),
            'fingerprint_similarity': float(fp_sim),
            'combined_similarity': float(emb_sim * 0.6 + fp_sim * 0.4),
        }

