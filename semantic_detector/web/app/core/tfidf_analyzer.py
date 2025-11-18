"""TF-IDF vectorization for style identification."""

from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple
import numpy as np
from collections import Counter


class TFIDFAnalyzer:
    """TF-IDF analysis for word vectorization and style identification."""
    
    def __init__(self, max_features: int = 1000, ngram_range: Tuple[int, int] = (1, 3)):
        """
        Initialize TF-IDF analyzer.
        
        Args:
            max_features: Maximum number of features
            ngram_range: Range of n-grams to extract
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=1,
            max_df=0.95
        )
        self.feature_names = None
        self.tfidf_matrix = None
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit TF-IDF vectorizer and transform texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            TF-IDF matrix
        """
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self.tfidf_matrix.toarray()
    
    def get_top_features(self, text_index: int, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top TF-IDF features for a specific text.
        
        Args:
            text_index: Index of the text
            top_n: Number of top features to return
            
        Returns:
            List of (feature, score) tuples
        """
        if self.tfidf_matrix is None:
            raise ValueError("Must call fit_transform first")
        
        scores = self.tfidf_matrix[text_index].toarray()[0]
        top_indices = np.argsort(scores)[::-1][:top_n]
        
        return [
            (self.feature_names[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]
    
    def get_cluster_features(self, texts: List[str], cluster_labels: np.ndarray, cluster_id: int, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top TF-IDF features for a cluster.
        
        Args:
            texts: List of all texts
            cluster_labels: Cluster labels for each text
            cluster_id: ID of the cluster
            top_n: Number of top features to return
            
        Returns:
            List of (feature, score) tuples
        """
        # Get texts in cluster
        cluster_texts = [texts[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
        
        if not cluster_texts:
            return []
        
        # Combine cluster texts
        combined_text = ' '.join(cluster_texts)
        
        # Transform combined text
        combined_vector = self.vectorizer.transform([combined_text]).toarray()[0]
        
        # Get top features
        top_indices = np.argsort(combined_vector)[::-1][:top_n]
        
        return [
            (self.feature_names[idx], float(combined_vector[idx]))
            for idx in top_indices
            if combined_vector[idx] > 0
        ]
    
    def get_style_vector(self, text: str) -> Dict[str, float]:
        """
        Get TF-IDF vector for a text as a dictionary.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping features to TF-IDF scores
        """
        vector = self.vectorizer.transform([text]).toarray()[0]
        return {
            feature: float(score)
            for feature, score in zip(self.feature_names, vector)
            if score > 0
        }

