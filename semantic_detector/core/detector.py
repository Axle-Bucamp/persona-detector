"""
Main SemanticDetector class combining all components.
"""

import os
import numpy as np
from typing import List, Optional, Dict, Any
import json

from semantic_detector.sentence_splitter import SentenceSplitter
from semantic_detector.embedding_generator import EmbeddingGenerator
from semantic_detector.language_fingerprint import LanguageFingerprint
from semantic_detector.clustering import SpeakerClusterer
from semantic_detector.visualization import EmbeddingVisualizer
from semantic_detector.style_finder import StyleFinder


class SemanticDetector:
    """Main class combining all components."""
    
    def __init__(
        self,
        ollama_endpoint: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text"
    ):
        """
        Initialize semantic detector.
        
        Args:
            ollama_endpoint: Ollama API endpoint
            embedding_model: Model name for embeddings
        """
        self.sentence_splitter = SentenceSplitter()
        self.embedding_generator = EmbeddingGenerator(ollama_endpoint, embedding_model)
        self.fingerprint_extractor = LanguageFingerprint()
        self.visualizer = EmbeddingVisualizer()
        self.clusterer = None
        
        # Check connection
        if not self.embedding_generator.test_connection():
            print(f"[WARNING] Cannot connect to Ollama at {ollama_endpoint}")
            print("[INFO] Analysis will continue with fallback embeddings if needed.")
    
    def process_text_file(
        self,
        file_path: str,
        n_clusters: Optional[int] = None,
        clustering_method: str = 'kmeans'
    ) -> dict:
        """
        Process a text file and generate analysis.
        
        Args:
            file_path: Path to text file
            n_clusters: Number of clusters (None for auto)
            clustering_method: Clustering algorithm
            
        Returns:
            Dictionary with results
        """
        print(f"[INFO] Processing file: {file_path}")
        
        # Step 1: Split into sentences
        print("[INFO] Step 1: Splitting text into sentences...")
        sentences = self.sentence_splitter.split_file(file_path)
        print(f"[INFO] Found {len(sentences)} sentences")
        
        if len(sentences) == 0:
            return {"error": "No sentences found in file"}
        
        # Step 2: Generate embeddings
        print("[INFO] Step 2: Generating embeddings...")
        embeddings = self.embedding_generator.generate_embeddings_batch(sentences)
        embeddings_array = np.array(embeddings)
        print(f"[INFO] Generated {len(embeddings)} embeddings of dimension {embeddings_array.shape[1]}")
        
        # Step 3: Extract language fingerprints
        print("[INFO] Step 3: Extracting language fingerprints...")
        fingerprints = self.fingerprint_extractor.extract_fingerprints_batch(sentences)
        print(f"[INFO] Extracted {len(fingerprints)} fingerprints with {fingerprints.shape[1]} features")
        
        # Step 4: Perform clustering
        print("[INFO] Step 4: Clustering texts...")
        self.clusterer = SpeakerClusterer(n_clusters=n_clusters, method=clustering_method)
        labels = self.clusterer.fit(fingerprints, embeddings_array)
        print(f"[INFO] Identified {len(set(labels))} clusters")
        
        # Get cluster statistics
        cluster_stats = self.clusterer.get_cluster_stats(sentences)
        print("\n[INFO] Cluster Statistics:")
        for cluster_id, stats in cluster_stats.items():
            print(f"  Cluster {cluster_id}: {stats['count']} sentences")
            print(f"    Avg length: {stats['avg_length']:.1f} chars")
        
        # Step 5: Create 3D UMAP projection
        print("[INFO] Step 5: Creating 3D UMAP projection...")
        coords_3d = self.visualizer.fit_transform(embeddings_array)
        print("[INFO] UMAP projection complete")
        
        # Step 5.5: Initialize style finder
        print("[INFO] Step 5.5: Initializing style finder...")
        style_finder = StyleFinder(embeddings_array, fingerprints, labels)
        
        # Step 6: Create visualizations
        print("[INFO] Step 6: Generating visualizations...")
        figures = self.visualizer.create_combined_plot(
            coords_3d,
            labels,
            sentences,
            include_ngram_view=True,
            include_sentence_analytics=True
        )
        
        # Save all visualizations
        base_name = file_path.replace('.txt', '').replace('.md', '')
        output_files = {}
        
        # Main 3D projection
        output_files['3d'] = f"{base_name}_3d_projection.html"
        figures['3d_projection'].write_html(output_files['3d'])
        print(f"[INFO] 3D projection saved to: {output_files['3d']}")
        
        # N-gram views
        if 'bigrams' in figures:
            output_files['bigrams'] = f"{base_name}_bigrams.html"
            figures['bigrams'].write_html(output_files['bigrams'])
            print(f"[INFO] Bigrams view saved to: {output_files['bigrams']}")
        
        if 'trigrams' in figures:
            output_files['trigrams'] = f"{base_name}_trigrams.html"
            figures['trigrams'].write_html(output_files['trigrams'])
            print(f"[INFO] Trigrams view saved to: {output_files['trigrams']}")
        
        # Sentence analytics
        if 'sentence_analytics' in figures:
            output_files['sentence_analytics'] = f"{base_name}_sentence_analytics.html"
            figures['sentence_analytics'].write_html(output_files['sentence_analytics'])
            print(f"[INFO] Sentence analytics saved to: {output_files['sentence_analytics']}")
        
        # Generate fingerprint summaries for each cluster
        print("\n[INFO] Step 7: Generating fingerprint summaries...")
        fingerprint_summaries = {}
        for cluster_id in set(labels):
            cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
            cluster_texts = [sentences[i] for i in cluster_indices]
            
            # Combine texts in cluster for summary
            combined_text = ' '.join(cluster_texts)
            fingerprint_summaries[cluster_id] = self.fingerprint_extractor.get_fingerprint_summary(combined_text)
        
        # Generate style comparison data
        print("[INFO] Step 8: Generating style comparisons...")
        style_comparisons = {}
        unique_clusters = sorted(set(labels))
        for i, c1 in enumerate(unique_clusters):
            for c2 in unique_clusters[i+1:]:
                comparison = style_finder.compare_clusters(c1, c2)
                style_comparisons[f"{c1}_vs_{c2}"] = comparison
        
        # Save metadata with fingerprint summaries
        metadata = {
            'file': file_path,
            'num_sentences': len(sentences),
            'num_clusters': len(set(labels)),
            'cluster_stats': {
                str(k): {
                    'count': v['count'],
                    'avg_length': float(v['avg_length'])
                }
                for k, v in cluster_stats.items()
            },
            'embedding_dim': int(embeddings_array.shape[1]),
            'fingerprint_dim': int(fingerprints.shape[1]),
            'fingerprint_summaries': {
                str(k): {
                    'top_words': v['top_words'],
                    'top_bigrams': v['top_bigrams'],
                    'top_trigrams': v['top_trigrams'],
                    'avg_sentence_length': float(v['avg_sentence_length']),
                    'vocab_richness': float(v['vocab_richness']),
                }
                for k, v in fingerprint_summaries.items()
            },
            'style_comparisons': {
                k: {kk: float(vv) for kk, vv in v.items()}
                for k, v in style_comparisons.items()
            },
            'output_files': output_files,
        }
        
        metadata_file = base_name + '_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\n[INFO] Metadata saved to: {metadata_file}")
        
        return {
            'sentences': sentences,
            'embeddings': embeddings_array,
            'fingerprints': fingerprints,
            'labels': labels,
            'coords_3d': coords_3d,
            'cluster_stats': cluster_stats,
            'figures': figures,
            'output_files': output_files,
            'style_finder': style_finder,
            'fingerprint_summaries': fingerprint_summaries,
            'style_comparisons': style_comparisons,
            'metadata': metadata
        }

