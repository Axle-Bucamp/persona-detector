"""Core detector service wrapper."""

import os
import numpy as np
from typing import Optional, Dict, Any, List
import tempfile
import re
from collections import Counter

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
        
        # Save text to temporary file with safe encoding
        from semantic_detector.web.app.core.text_utils import parse_raw_text
        # Sanitize text before writing
        safe_text = parse_raw_text(text)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', errors='replace', suffix='.txt', delete=False) as f:
            f.write(safe_text)
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
    
    def process_structured_data(
        self,
        texts: List[str],
        n_clusters: Optional[int] = None,
        clustering_method: str = "kmeans",
        ollama_endpoint: Optional[str] = None,
        embedding_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process pre-parsed structured data (from CSV/JSON/regex).
        
        Args:
            texts: List of text strings to analyze
            n_clusters: Number of clusters
            clustering_method: Clustering algorithm
            ollama_endpoint: Optional Ollama endpoint
            embedding_model: Optional embedding model
            
        Returns:
            Dictionary with analysis results including unified format data
        """
        if self.detector is None:
            self.initialize(ollama_endpoint=ollama_endpoint, embedding_model=embedding_model)
        elif ollama_endpoint or embedding_model:
            self.initialize(ollama_endpoint=ollama_endpoint, embedding_model=embedding_model)
        
        if not texts:
            raise ValueError("No texts provided")
        
        # Generate embeddings
        embeddings = self.detector.embedding_generator.generate_embeddings_batch(texts)
        embeddings_array = np.array(embeddings)
        
        # Extract language fingerprints (for analysis, not clustering)
        fingerprints = self.detector.fingerprint_extractor.extract_fingerprints_batch(texts)
        
        # Perform clustering using embeddings only (fingerprints kept for analysis)
        from semantic_detector.clustering import SpeakerClusterer
        clusterer = SpeakerClusterer(n_clusters=n_clusters, method=clustering_method)
        # Pass None for features to use embeddings only
        labels = clusterer.fit(np.zeros((len(texts), 1)), embeddings_array)  # Dummy features, embeddings used
        
        # Get cluster statistics
        cluster_stats = clusterer.get_cluster_stats(texts)
        
        # Generate 3D UMAP coordinates
        coords_3d = self.detector.visualizer.fit_transform(embeddings_array)
        
        # Initialize style finder
        from semantic_detector.style_finder import StyleFinder
        style_finder = StyleFinder(embeddings_array, fingerprints, labels)
        
        # Build result structure similar to process_text_file
        result = {
            'sentences': texts,
            'labels': labels,
            'cluster_stats': cluster_stats,
            'fingerprint_summaries': {},
            'style_finder': style_finder,
            'coords_3d': coords_3d
        }
        
        # Get fingerprint summaries per cluster
        unique_clusters = set(labels)
        for cluster_id in unique_clusters:
            cluster_texts = [texts[i] for i, label in enumerate(labels) if label == cluster_id]
            if cluster_texts:
                # Get summary for first text in cluster as representative
                summary = self.detector.fingerprint_extractor.get_fingerprint_summary(cluster_texts[0])
                result['fingerprint_summaries'][str(cluster_id)] = summary
        
        # Enhance with TF-IDF and sentiment
        result = self._enhance_result(result)
        
        # Calculate word frequencies per text
        word_frequencies = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            word_counts = Counter(words)
            total_words = len(words)
            if total_words > 0:
                word_freq = {word: count / total_words for word, count in word_counts.items()}
            else:
                word_freq = {}
            word_frequencies.append(word_freq)
        
        # Get TF-IDF features per text
        tfidf_features_per_text = []
        if 'tfidf_matrix' in result:
            for idx in range(len(texts)):
                top_features = self.tfidf_analyzer.get_top_features(idx, top_n=10)
                tfidf_features_per_text.append(top_features)
        else:
            tfidf_features_per_text = [[] for _ in texts]
        
        # Store for unified formatting
        result['word_frequencies'] = word_frequencies
        result['tfidf_features_per_text'] = tfidf_features_per_text
        
        self.current_result = result
        return result
    
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

