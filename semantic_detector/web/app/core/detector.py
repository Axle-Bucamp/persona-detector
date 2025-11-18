"""Core detector service wrapper."""

import os
import numpy as np
from typing import Optional, Dict, Any
import tempfile

from semantic_detector.core.detector import SemanticDetector
from semantic_detector.web.app.core.config import settings
from semantic_detector.web.app.core.tfidf_analyzer import TFIDFAnalyzer
from semantic_detector.web.app.core.sentiment_analyzer import SentimentAnalyzer


class DetectorService:
    """Service wrapper for SemanticDetector."""
    
    def __init__(self):
        """Initialize detector service."""
        self.detector = None
        self.current_result: Optional[Dict[str, Any]] = None
        self.tfidf_analyzer = TFIDFAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def initialize(
        self,
        ollama_endpoint: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """Initialize detector with configuration."""
        endpoint = ollama_endpoint or settings.ollama_endpoint
        model = embedding_model or settings.default_embedding_model
        self.detector = SemanticDetector(endpoint, model)
        return self.detector
    
    def process_text(
        self,
        text: str,
        n_clusters: Optional[int] = None,
        clustering_method: str = "kmeans"
    ) -> Dict[str, Any]:
        """Process text and return results."""
        if self.detector is None:
            self.initialize()
        
        # Save text to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(text)
            temp_path = f.name
        
        try:
            result = self.detector.process_text_file(
                temp_path,
                n_clusters=n_clusters,
                clustering_method=clustering_method
            )
            
            # Add TF-IDF and sentiment analysis
            result = self._enhance_result(result)
            
            self.current_result = result
            return result
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def process_file(
        self,
        file_path: str,
        n_clusters: Optional[int] = None,
        clustering_method: str = "kmeans"
    ) -> Dict[str, Any]:
        """Process file and return results."""
        if self.detector is None:
            self.initialize()
        
        result = self.detector.process_text_file(
            file_path,
            n_clusters=n_clusters,
            clustering_method=clustering_method
        )
        
        # Add TF-IDF and sentiment analysis
        result = self._enhance_result(result)
        
        self.current_result = result
        return result
    
    def get_style_matches(
        self,
        sentence_index: int,
        k: int = 5,
        in_cluster_only: bool = False,
        out_cluster_only: bool = False
    ) -> Dict[str, Any]:
        """Get style matches for a sentence."""
        if self.current_result is None:
            raise ValueError("No analysis result available. Process text first.")
        
        style_finder = self.current_result['style_finder']
        sentences = self.current_result['sentences']
        labels = self.current_result['labels']
        
        if sentence_index >= len(sentences):
            raise ValueError(f"Sentence index {sentence_index} out of range")
        
        results = {}
        
        if in_cluster_only:
            matches = style_finder.find_closest_in_cluster(sentence_index, k=k)
            results['in_cluster'] = [
                {
                    'index': idx,
                    'similarity': float(sim),
                    'cluster_id': int(labels[idx]),
                    'text': sentences[idx]
                }
                for idx, sim in matches
            ]
        elif out_cluster_only:
            matches = style_finder.find_closest_out_cluster(sentence_index, k=k)
            results['out_cluster'] = [
                {
                    'index': idx,
                    'similarity': float(sim),
                    'cluster_id': int(labels[idx]),
                    'text': sentences[idx]
                }
                for idx, sim in matches
            ]
        else:
            # Both
            in_matches = style_finder.find_closest_in_cluster(sentence_index, k=k)
            out_matches = style_finder.find_closest_out_cluster(sentence_index, k=k)
            
            results['in_cluster'] = [
                {
                    'index': idx,
                    'similarity': float(sim),
                    'cluster_id': int(labels[idx]),
                    'text': sentences[idx]
                }
                for idx, sim in in_matches
            ]
            results['out_cluster'] = [
                {
                    'index': idx,
                    'similarity': float(sim),
                    'cluster_id': int(labels[idx]),
                    'text': sentences[idx]
                }
                for idx, sim in out_matches
            ]
        
        # Get fingerprint summary
        fp_summary = self.detector.fingerprint_extractor.get_fingerprint_summary(
            sentences[sentence_index]
        )
        
        results['query_index'] = sentence_index
        results['query_text'] = sentences[sentence_index]
        results['query_cluster'] = int(labels[sentence_index])
        results['fingerprint_summary'] = fp_summary
        
        return results
    
    def _enhance_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance result with TF-IDF and sentiment analysis."""
        sentences = result.get('sentences', [])
        labels = result.get('labels', np.array([]))
        
        if not sentences:
            return result
        
        # TF-IDF analysis
        try:
            tfidf_matrix = self.tfidf_analyzer.fit_transform(sentences)
            result['tfidf_matrix'] = tfidf_matrix.tolist()
            result['tfidf_features'] = self.tfidf_analyzer.feature_names.tolist()
            
            # Get cluster TF-IDF features
            cluster_tfidf = {}
            unique_clusters = set(labels) if len(labels) > 0 else set()
            for cluster_id in unique_clusters:
                cluster_tfidf[str(cluster_id)] = self.tfidf_analyzer.get_cluster_features(
                    sentences, labels, cluster_id, top_n=20
                )
            result['cluster_tfidf_features'] = cluster_tfidf
        except Exception as e:
            print(f"[WARNING] TF-IDF analysis failed: {e}")
            result['tfidf_error'] = str(e)
        
        # Sentiment analysis
        try:
            sentiment_results = self.sentiment_analyzer.analyze_batch(sentences)
            result['sentiment_analysis'] = sentiment_results
            
            # Average sentiment per cluster
            cluster_sentiments = {}
            unique_clusters = set(labels) if len(labels) > 0 else set()
            for cluster_id in unique_clusters:
                cluster_texts = [sentences[i] for i, label in enumerate(labels) if label == cluster_id]
                cluster_sentiments[str(cluster_id)] = {
                    'average': self.sentiment_analyzer.get_average_sentiment(cluster_texts),
                    'person_score': self.sentiment_analyzer.get_person_score(cluster_texts)
                }
            result['cluster_sentiments'] = cluster_sentiments
            
            # Overall average
            result['overall_sentiment'] = self.sentiment_analyzer.get_average_sentiment(sentences)
            result['overall_person_score'] = self.sentiment_analyzer.get_person_score(sentences)
        except Exception as e:
            print(f"[WARNING] Sentiment analysis failed: {e}")
            result['sentiment_error'] = str(e)
        
        return result


# Global detector service instance
detector_service = DetectorService()

