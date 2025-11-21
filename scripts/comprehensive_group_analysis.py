#!/usr/bin/env python3
"""
Comprehensive Group Analysis: Embedding clustering, similarity scores, and n-gram comparison
with statistical significance tests across all groups.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
import sys
import warnings
warnings.filterwarnings('ignore')

# Statistical tests
try:
    from scipy.stats import ks_2samp, chi2_contingency, mannwhitneyu
    from scipy.spatial.distance import cosine
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARNING] scipy not installed. Statistical tests will be limited.")

# ML libraries
try:
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Import project modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from semantic_detector.embedding_generator import EmbeddingGenerator
    from semantic_detector.language_fingerprint import LanguageFingerprint
    from semantic_detector.web.app.core.text_utils import parse_raw_text
    HAS_SEMANTIC = True
except ImportError as e:
    HAS_SEMANTIC = False
    print(f"[WARNING] Semantic modules not available: {e}")


class ComprehensiveGroupAnalyzer:
    """Comprehensive analysis of all groups with embeddings, similarity, and n-grams."""
    
    def __init__(self, csv_path: str, ollama_endpoint: str = "http://localhost:11434",
                 embedding_model: str = "nomic-embed-text"):
        """Initialize analyzer."""
        self.csv_path = csv_path
        self.ollama_endpoint = ollama_endpoint
        self.embedding_model = embedding_model
        self.df = None
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        if HAS_SEMANTIC:
            self.embedding_generator = EmbeddingGenerator(endpoint=ollama_endpoint, model=embedding_model)
            self.fingerprint_extractor = LanguageFingerprint()
        
        # Data storage
        self.group_data = {}  # {group_name: {messages: [], embeddings: [], ...}}
        self.group_embeddings = {}  # {group_name: np.array}
        self.group_ngrams = {}  # {group_name: {bigrams: [], trigrams: [], words: []}}
        self.similarity_matrix = {}  # {group1: {group2: similarity_score}}
        
    def load_data(self):
        """Load and prepare CSV data."""
        print("[INFO] Loading CSV data...")
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            from io import StringIO
            self.df = pd.read_csv(StringIO(content))
        except Exception as e:
            print(f"[ERROR] Failed to load CSV: {e}")
            return False
        
        # Sanitize text columns
        text_cols = ['content', 'chat_id', 'party', 'group_name']
        for col in text_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(
                    lambda x: str(x).strip() if pd.notna(x) else ''
                )
        
        print(f"[INFO] Loaded {len(self.df)} rows")
        return True
    
    def extract_group_messages(self):
        """Extract messages for each group (attacker only)."""
        print("[INFO] Extracting group messages...")
        
        # Get unique groups
        groups = self.df['group_name'].dropna().unique()
        print(f"[INFO] Found {len(groups)} groups")
        
        for group in groups:
            if not group or str(group).strip() == '':
                continue
            
            group_str = str(group).strip()
            
            # Filter: group_name matches AND party is NOT "Victim"
            group_df = self.df[
                (self.df['group_name'] == group) &
                (~self.df['party'].str.contains('Victim', case=False, na=False))
            ]
            
            messages = group_df['content'].dropna().tolist()
            messages = [str(m).strip() for m in messages if str(m).strip()]
            
            if len(messages) >= 5:  # Minimum threshold
                self.group_data[group_str] = {
                    'messages': messages,
                    'count': len(messages),
                    'chat_ids': group_df['chat_id'].unique().tolist() if 'chat_id' in group_df.columns else []
                }
                print(f"  - {group_str}: {len(messages)} messages")
        
        print(f"[INFO] Extracted messages for {len(self.group_data)} groups")
        return len(self.group_data) > 0
    
    def generate_embeddings_per_group(self):
        """Generate embeddings for each group using Ollama, with caching."""
        if not HAS_SEMANTIC:
            print("[WARNING] Embedding generator not available")
            return {}
        
        print("[INFO] Generating embeddings per group using Ollama (nomic-embed-text)...")
        
        # Load cached embeddings
        cache_file = self.output_dir / 'group_embeddings_cache.json'
        cached_embeddings = {}
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    for group, data in cached_data.items():
                        if 'embeddings' in data:
                            cached_embeddings[group] = np.array(data['embeddings'])
                    print(f"[INFO] Loaded {len(cached_embeddings)} cached group embeddings")
            except Exception as e:
                print(f"[WARNING] Failed to load embedding cache: {e}")
        
        results = {}
        new_embeddings_data = {}
        
        for group_name, data in self.group_data.items():
            messages = data['messages']
            
            # Check cache first
            if group_name in cached_embeddings:
                embeddings_normalized = cached_embeddings[group_name]
                print(f"[INFO] Using cached embeddings for {group_name}: {len(embeddings_normalized)} embeddings")
            else:
                print(f"[INFO] Processing {group_name}: {len(messages)} messages...")
                
                try:
                    # Generate embeddings batch
                    embeddings = self.embedding_generator.generate_embeddings_batch(messages)
                    embeddings_array = np.array(embeddings)
                    
                    # L2 normalize for cosine similarity
                    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
                    embeddings_normalized = embeddings_array / norms
                    
                    # Store for saving
                    new_embeddings_data[group_name] = {
                        'embeddings': embeddings_normalized.tolist(),
                        'count': len(messages),
                        'dimension': embeddings_array.shape[1] if len(embeddings_array) > 0 else 0
                    }
                    
                    print(f"  - {group_name}: {len(embeddings)} embeddings, dim={embeddings_array.shape[1]}")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to generate embeddings for {group_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            self.group_embeddings[group_name] = embeddings_normalized
            results[group_name] = {
                'embeddings': embeddings_normalized.tolist(),
                'count': len(messages),
                'dimension': embeddings_normalized.shape[1] if len(embeddings_normalized) > 0 else 0
            }
        
        # Save new embeddings to cache
        if new_embeddings_data:
            try:
                # Load existing cache and merge
                if cache_file.exists():
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        existing_cache = json.load(f)
                else:
                    existing_cache = {}
                
                # Update with new embeddings
                existing_cache.update(new_embeddings_data)
                
                # Save
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_cache, f, indent=2)
                print(f"[INFO] Saved {len(new_embeddings_data)} new group embeddings to cache")
            except Exception as e:
                print(f"[WARNING] Failed to save embedding cache: {e}")
        
        return results
    
    def calculate_group_similarities(self):
        """Calculate cosine similarity between group embeddings."""
        print("[INFO] Calculating group similarity scores...")
        
        if not self.group_embeddings:
            print("[WARNING] No embeddings available for similarity calculation")
            return {}
        
        similarities = {}
        groups = list(self.group_embeddings.keys())
        
        for i, group1 in enumerate(groups):
            similarities[group1] = {}
            emb1 = self.group_embeddings[group1]
            
            # Calculate mean embedding for group1
            if len(emb1) > 0:
                mean_emb1 = np.mean(emb1, axis=0)
                mean_emb1 = mean_emb1 / (np.linalg.norm(mean_emb1) + 1e-8)  # Normalize
            else:
                continue
            
            for j, group2 in enumerate(groups):
                if i == j:
                    similarities[group1][group2] = 1.0
                    continue
                
                emb2 = self.group_embeddings[group2]
                
                if len(emb2) > 0:
                    # Calculate mean embedding for group2
                    mean_emb2 = np.mean(emb2, axis=0)
                    mean_emb2 = mean_emb2 / (np.linalg.norm(mean_emb2) + 1e-8)  # Normalize
                    
                    # Cosine similarity
                    similarity = np.dot(mean_emb1, mean_emb2)
                    similarities[group1][group2] = float(similarity)
                else:
                    similarities[group1][group2] = 0.0
        
        self.similarity_matrix = similarities
        return similarities
    
    def extract_ngrams_per_group(self):
        """Extract n-grams for each group."""
        print("[INFO] Extracting n-grams per group...")
        
        results = {}
        
        for group_name, data in self.group_data.items():
            messages = data['messages']
            
            # Combine all messages for the group
            combined_text = ' '.join([str(m).strip() for m in messages if str(m).strip()])
            
            if not combined_text:
                continue
            
            if HAS_SEMANTIC:
                try:
                    # Use LanguageFingerprint
                    bigrams = self.fingerprint_extractor.extract_ngram_distribution(combined_text, n=2, top_n=50)
                    trigrams = self.fingerprint_extractor.extract_ngram_distribution(combined_text, n=3, top_n=50)
                    word_dist = self.fingerprint_extractor.extract_word_distribution(combined_text, top_n=50)
                    
                    results[group_name] = {
                        'bigrams': list(bigrams.items()),
                        'trigrams': list(trigrams.items()),
                        'words': list(word_dist.items()),
                        'message_count': len(messages)
                    }
                except Exception as e:
                    print(f"[WARNING] Failed to extract n-grams for {group_name}: {e}")
                    results[group_name] = self._extract_basic_ngrams(combined_text, len(messages))
            else:
                results[group_name] = self._extract_basic_ngrams(combined_text, len(messages))
        
        self.group_ngrams = results
        return results
    
    def _extract_basic_ngrams(self, text: str, msg_count: int) -> dict:
        """Basic n-gram extraction without LanguageFingerprint."""
        import re
        from collections import Counter
        
        def extract_ngrams(text, n):
            words = re.findall(r'\b\w+\b', text.lower())
            if len(words) < n:
                return []
            ngrams = []
            for i in range(len(words) - n + 1):
                ngrams.append(tuple(words[i:i+n]))
            return ngrams
        
        def get_top_words(text, top_n=50):
            words = re.findall(r'\b\w+\b', text.lower())
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = [w for w in words if w not in stopwords and len(w) > 2]
            word_counts = Counter(words)
            return dict(word_counts.most_common(top_n))
        
        bigrams = Counter(extract_ngrams(text, 2))
        trigrams = Counter(extract_ngrams(text, 3))
        words = get_top_words(text, 50)
        
        # Normalize frequencies
        total_bigrams = sum(bigrams.values()) if bigrams else 1
        total_trigrams = sum(trigrams.values()) if trigrams else 1
        total_words = sum(words.values()) if words else 1
        
        return {
            'bigrams': [(bg, count/total_bigrams) for bg, count in bigrams.most_common(50)],
            'trigrams': [(tg, count/total_trigrams) for tg, count in trigrams.most_common(50)],
            'words': [(w, count/total_words) for w, count in words.items()],
            'message_count': msg_count
        }
    
    def compare_ngram_distributions(self, reference_group: str = "BlackBasta") -> dict:
        """Compare n-gram distributions between groups with statistical tests."""
        print(f"[INFO] Comparing n-gram distributions (reference: {reference_group})...")
        
        if reference_group not in self.group_ngrams:
            print(f"[WARNING] Reference group {reference_group} not found")
            return {}
        
        ref_ngrams = self.group_ngrams[reference_group]
        comparisons = {}
        
        for group_name, group_ngrams in self.group_ngrams.items():
            if group_name == reference_group:
                continue
            
            comparison = {
                'group': group_name,
                'reference': reference_group,
            }
            
            # Compare bigrams
            if 'bigrams' in ref_ngrams and 'bigrams' in group_ngrams:
                bigram_sim = self._compare_ngram_lists(
                    ref_ngrams['bigrams'],
                    group_ngrams['bigrams']
                )
                comparison['bigram_similarity'] = bigram_sim
            
            # Compare trigrams
            if 'trigrams' in ref_ngrams and 'trigrams' in group_ngrams:
                trigram_sim = self._compare_ngram_lists(
                    ref_ngrams['trigrams'],
                    group_ngrams['trigrams']
                )
                comparison['trigram_similarity'] = trigram_sim
            
            # Compare words
            if 'words' in ref_ngrams and 'words' in group_ngrams:
                word_sim = self._compare_ngram_lists(
                    ref_ngrams['words'],
                    group_ngrams['words']
                )
                comparison['word_similarity'] = word_sim
            
            # Statistical significance tests
            if HAS_SCIPY:
                comparison['statistical_tests'] = self._statistical_ngram_test(
                    ref_ngrams, group_ngrams
                )
            
            comparisons[group_name] = comparison
        
        # Sort by similarity
        sorted_comparisons = sorted(
            comparisons.items(),
            key=lambda x: (
                x[1].get('bigram_similarity', 0) +
                x[1].get('trigram_similarity', 0) +
                x[1].get('word_similarity', 0)
            ) / 3,
            reverse=True
        )
        
        return {
            'reference_group': reference_group,
            'comparisons': dict(sorted_comparisons),
            'sorted_by_similarity': [g for g, _ in sorted_comparisons]
        }
    
    def _compare_ngram_lists(self, list1: List[Tuple], list2: List[Tuple]) -> float:
        """Compare two n-gram lists and return similarity score."""
        # Convert to dictionaries for easier lookup
        dict1 = {str(k): v for k, v in list1}
        dict2 = {str(k): v for k, v in list2}
        
        # Get common n-grams
        common = set(dict1.keys()) & set(dict2.keys())
        
        if not common:
            return 0.0
        
        # Calculate weighted similarity (Jaccard-like with frequency weighting)
        total_weight = 0.0
        common_weight = 0.0
        
        for ngram in common:
            weight1 = dict1.get(ngram, 0)
            weight2 = dict2.get(ngram, 0)
            common_weight += min(weight1, weight2)
        
        for ngram in set(dict1.keys()) | set(dict2.keys()):
            weight1 = dict1.get(ngram, 0)
            weight2 = dict2.get(ngram, 0)
            total_weight += max(weight1, weight2)
        
        if total_weight == 0:
            return 0.0
        
        return common_weight / total_weight
    
    def _statistical_ngram_test(self, ref_ngrams: dict, group_ngrams: dict) -> dict:
        """Perform statistical tests on n-gram distributions."""
        if not HAS_SCIPY:
            return {}
        
        tests = {}
        
        # Convert to frequency vectors for statistical testing
        def ngram_to_vector(ngram_list, all_ngrams_set):
            vector = []
            ngram_dict = {str(k): v for k, v in ngram_list}
            for ngram in all_ngrams_set:
                vector.append(ngram_dict.get(str(ngram), 0.0))
            return np.array(vector)
        
        # Test bigrams
        if 'bigrams' in ref_ngrams and 'bigrams' in group_ngrams:
            ref_bigrams = ref_ngrams['bigrams']
            group_bigrams = group_ngrams['bigrams']
            
            # Get union of all bigrams
            all_bigrams = set([str(k) for k, _ in ref_bigrams]) | set([str(k) for k, _ in group_bigrams])
            
            if len(all_bigrams) > 0:
                ref_vec = ngram_to_vector(ref_bigrams, all_bigrams)
                group_vec = ngram_to_vector(group_bigrams, all_bigrams)
                
                # Kolmogorov-Smirnov test
                try:
                    ks_stat, ks_p = ks_2samp(ref_vec, group_vec)
                    tests['bigram_ks'] = {
                        'statistic': float(ks_stat),
                        'pvalue': float(ks_p),
                        'significant': ks_p < 0.05
                    }
                except Exception:
                    pass
        
        # Test trigrams
        if 'trigrams' in ref_ngrams and 'trigrams' in group_ngrams:
            ref_trigrams = ref_ngrams['trigrams']
            group_trigrams = group_ngrams['trigrams']
            
            all_trigrams = set([str(k) for k, _ in ref_trigrams]) | set([str(k) for k, _ in group_trigrams])
            
            if len(all_trigrams) > 0:
                ref_vec = ngram_to_vector(ref_trigrams, all_trigrams)
                group_vec = ngram_to_vector(group_trigrams, all_trigrams)
                
                try:
                    ks_stat, ks_p = ks_2samp(ref_vec, group_vec)
                    tests['trigram_ks'] = {
                        'statistic': float(ks_stat),
                        'pvalue': float(ks_p),
                        'significant': ks_p < 0.05
                    }
                except Exception:
                    pass
        
        return tests
    
    def cluster_groups_by_embeddings(self, n_clusters: Optional[int] = None):
        """Cluster groups based on their embedding centroids."""
        print("[INFO] Clustering groups by embedding similarity...")
        
        if not self.group_embeddings:
            print("[WARNING] No embeddings available for clustering")
            return {}
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            # Calculate centroids for each group
            centroids = []
            group_names = []
            
            for group_name, embeddings in self.group_embeddings.items():
                if len(embeddings) > 0:
                    centroid = np.mean(embeddings, axis=0)
                    centroids.append(centroid)
                    group_names.append(group_name)
            
            if len(centroids) < 2:
                print("[WARNING] Not enough groups for clustering")
                return {}
            
            centroids_array = np.array(centroids)
            
            # Auto-determine clusters if not specified
            if n_clusters is None:
                n_clusters = max(2, min(10, len(centroids) // 2))
            
            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(centroids_array)
            
            # Calculate silhouette score
            silhouette = silhouette_score(centroids_array, cluster_labels) if len(set(cluster_labels)) > 1 else 0.0
            
            # Organize results
            cluster_groups = defaultdict(list)
            for group_name, label in zip(group_names, cluster_labels):
                cluster_groups[int(label)].append(group_name)
            
            results = {
                'n_clusters': n_clusters,
                'silhouette_score': float(silhouette),
                'cluster_assignments': dict(cluster_groups),
                'group_to_cluster': {group: int(label) for group, label in zip(group_names, cluster_labels)}
            }
            
            print(f"[INFO] Clustered {len(group_names)} groups into {n_clusters} clusters (silhouette: {silhouette:.3f})")
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Clustering failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def generate_comprehensive_report(self) -> dict:
        """Generate comprehensive analysis report."""
        print("[INFO] Generating comprehensive analysis report...")
        
        report = {
            'groups_analyzed': list(self.group_data.keys()),
            'group_counts': {g: d['count'] for g, d in self.group_data.items()},
            'embedding_similarities': self.similarity_matrix,
            'ngram_comparisons': {},
            'clustering_results': {},
        }
        
        # N-gram comparisons (BlackBasta as reference)
        if 'BlackBasta' in self.group_ngrams:
            ngram_comparison = self.compare_ngram_distributions('BlackBasta')
            report['ngram_comparisons']['blackbasta_reference'] = ngram_comparison
        
        # Clustering results
        clustering = self.cluster_groups_by_embeddings()
        if clustering:
            report['clustering_results'] = clustering
        
        # Find most similar groups to BlackBasta
        if 'BlackBasta' in self.similarity_matrix:
            bb_similarities = self.similarity_matrix['BlackBasta']
            sorted_similar = sorted(
                [(g, s) for g, s in bb_similarities.items() if g != 'BlackBasta'],
                key=lambda x: x[1],
                reverse=True
            )
            report['most_similar_to_blackbasta'] = [
                {'group': g, 'similarity': s} for g, s in sorted_similar[:10]
            ]
        
        return report
    
    def save_results(self, report: dict):
        """Save analysis results to JSON."""
        output_file = self.output_dir / 'comprehensive_group_analysis.json'
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_for_json(item) for item in obj)
            # Handle numpy scalar types generically
            elif hasattr(obj, 'item'):
                return obj.item()
            return obj
        
        report_serializable = convert_for_json(report)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Results saved to {output_file}")
        return output_file


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive group analysis with embeddings and n-grams')
    parser.add_argument('--csv', type=str, default='output/ransom_chats.csv',
                       help='Path to CSV file')
    parser.add_argument('--ollama-endpoint', type=str, default='http://localhost:11434',
                       help='Ollama endpoint URL')
    parser.add_argument('--embedding-model', type=str, default='nomic-embed-text',
                       help='Embedding model name')
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV file not found: {csv_path}")
        return
    
    analyzer = ComprehensiveGroupAnalyzer(
        csv_path=str(csv_path),
        ollama_endpoint=args.ollama_endpoint,
        embedding_model=args.embedding_model
    )
    
    # Load data
    if not analyzer.load_data():
        return
    
    # Extract group messages
    if not analyzer.extract_group_messages():
        print("[ERROR] No group messages extracted")
        return
    
    # Generate embeddings
    analyzer.generate_embeddings_per_group()
    
    # Calculate similarities
    analyzer.calculate_group_similarities()
    
    # Extract n-grams
    analyzer.extract_ngrams_per_group()
    
    # Generate report
    report = analyzer.generate_comprehensive_report()
    
    # Save results
    analyzer.save_results(report)
    
    print("\n[SUCCESS] Comprehensive analysis complete!")
    print(f"[INFO] Analyzed {len(analyzer.group_data)} groups")
    print(f"[INFO] Generated embeddings for {len(analyzer.group_embeddings)} groups")
    print(f"[INFO] Calculated similarities for {len(analyzer.similarity_matrix)} groups")


if __name__ == '__main__':
    main()

