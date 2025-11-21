"""
Sentence-Level Embedding Clustering Analysis
Analyzes ransom chat messages using embedding-based clustering per party.
Focuses on isolating victims from attackers and matching attacker sentence styles.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML libraries
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Import project modules
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from semantic_detector.embedding_generator import EmbeddingGenerator
from semantic_detector.sentence_splitter import SentenceSplitter
from semantic_detector.language_fingerprint import LanguageFingerprint
from semantic_detector.web.app.core.text_utils import parse_raw_text, parse_identifier

# Set style
if HAS_SEABORN and hasattr(sns, 'set_style'):
    sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class SentenceClusteringAnalyzer:
    """Analyze sentence-level clustering using embeddings."""
    
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
        self.embedding_generator = EmbeddingGenerator(endpoint=ollama_endpoint, model=embedding_model)
        self.sentence_splitter = SentenceSplitter()
        self.fingerprint_extractor = LanguageFingerprint()
        
        # Data storage
        self.attacker_messages = []
        self.victim_messages = []
        self.attacker_embeddings = None
        self.victim_embeddings = None
        self.attacker_clusters = None
        self.victim_clusters = None
        
    def load_and_prepare_messages(self, max_message_length: int = 500):
        """Load CSV and prepare messages with embeddings."""
        print("[INFO] Loading and preparing messages...")
        
        # Read CSV with safe encoding
        try:
            with open(self.csv_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            from io import StringIO
            self.df = pd.read_csv(StringIO(content))
        except Exception:
            with open(self.csv_path, 'r', encoding='latin-1', errors='replace') as f:
                content = f.read()
            from io import StringIO
            self.df = pd.read_csv(StringIO(content))
        
        # Sanitize text fields
        text_cols = ['content', 'chat_id', 'party', 'group_name']
        for col in text_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(
                    lambda x: parse_raw_text(str(x)) if pd.notna(x) else ''
                )
        
        # Filter and prepare messages
        attacker_data = []
        victim_data = []
        
        for idx, row in self.df.iterrows():
            content = str(row.get('content', '')).strip()
            if not content:
                continue
            
            party = str(row.get('party', '')).lower()
            group_name = str(row.get('group_name', ''))
            chat_id = str(row.get('chat_id', ''))
            
            # Determine if attacker or victim
            is_victim = party in ['victim', 'client', 'target']
            is_attacker = not is_victim and party
            
            # Split long messages
            if len(content) > max_message_length:
                sentences = self.sentence_splitter.split_by_period(content)
                for sent_idx, sentence in enumerate(sentences):
                    if len(sentence.strip()) > 10:  # Minimum length
                        msg_data = {
                            'content': sentence.strip(),
                            'group_name': group_name,
                            'chat_id': chat_id,
                            'party': party,
                            'original_index': idx,
                            'sentence_index': sent_idx
                        }
                        if is_attacker:
                            attacker_data.append(msg_data)
                        elif is_victim:
                            victim_data.append(msg_data)
            else:
                msg_data = {
                    'content': content,
                    'group_name': group_name,
                    'chat_id': chat_id,
                    'party': party,
                    'original_index': idx,
                    'sentence_index': 0
                }
                if is_attacker:
                    attacker_data.append(msg_data)
                elif is_victim:
                    victim_data.append(msg_data)
        
        self.attacker_messages = attacker_data
        self.victim_messages = victim_data
        
        print(f"[INFO] Prepared {len(self.attacker_messages)} attacker messages")
        print(f"[INFO] Prepared {len(self.victim_messages)} victim messages")
        
        return len(self.attacker_messages), len(self.victim_messages)
    
    def generate_embeddings(self):
        """Generate embeddings for all messages."""
        print("[INFO] Generating embeddings for attacker messages...")
        attacker_texts = [msg['content'] for msg in self.attacker_messages]
        self.attacker_embeddings = np.array(
            self.embedding_generator.generate_embeddings_batch(attacker_texts)
        )
        
        print("[INFO] Generating embeddings for victim messages...")
        victim_texts = [msg['content'] for msg in self.victim_messages]
        self.victim_embeddings = np.array(
            self.embedding_generator.generate_embeddings_batch(victim_texts)
        )
        
        print(f"[INFO] Attacker embeddings shape: {self.attacker_embeddings.shape}")
        print(f"[INFO] Victim embeddings shape: {self.victim_embeddings.shape}")
    
    def cluster_by_embeddings(self, n_clusters: Optional[int] = None, method: str = 'kmeans'):
        """Cluster messages using embeddings only."""
        print(f"[INFO] Clustering attacker messages (method: {method})...")
        
        # Normalize embeddings (L2 normalization)
        attacker_emb_norm = self.attacker_embeddings / (
            np.linalg.norm(self.attacker_embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        # Cluster attackers
        if method == 'kmeans':
            if n_clusters is None:
                n_clusters = min(10, max(2, int(np.sqrt(len(attacker_emb_norm)))))
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.attacker_clusters = clusterer.fit_predict(attacker_emb_norm)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.3, min_samples=5)
            self.attacker_clusters = clusterer.fit_predict(attacker_emb_norm)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"[INFO] Attacker clusters: {len(set(self.attacker_clusters))} unique clusters")
        
        # Cluster victims separately (for comparison)
        if len(self.victim_embeddings) > 0:
            print("[INFO] Clustering victim messages...")
            victim_emb_norm = self.victim_embeddings / (
                np.linalg.norm(self.victim_embeddings, axis=1, keepdims=True) + 1e-8
            )
            
            if method == 'kmeans':
                if n_clusters is None:
                    n_clusters = min(5, max(2, int(np.sqrt(len(victim_emb_norm)))))
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                self.victim_clusters = clusterer.fit_predict(victim_emb_norm)
            elif method == 'dbscan':
                clusterer = DBSCAN(eps=0.3, min_samples=5)
                self.victim_clusters = clusterer.fit_predict(victim_emb_norm)
            
            print(f"[INFO] Victim clusters: {len(set(self.victim_clusters))} unique clusters")
        else:
            self.victim_clusters = np.array([])
        
        return self.attacker_clusters, self.victim_clusters
    
    def calculate_separability_metrics(self) -> Dict[str, float]:
        """Calculate metrics showing attacker vs victim separability."""
        print("[INFO] Calculating separability metrics...")
        
        if len(self.victim_embeddings) == 0:
            return {'error': 'No victim messages'}
        
        # Normalize embeddings
        attacker_emb_norm = self.attacker_embeddings / (
            np.linalg.norm(self.attacker_embeddings, axis=1, keepdims=True) + 1e-8
        )
        victim_emb_norm = self.victim_embeddings / (
            np.linalg.norm(self.victim_embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        # Calculate inter-cluster distance (attacker centroid vs victim centroid)
        attacker_centroid = np.mean(attacker_emb_norm, axis=0)
        victim_centroid = np.mean(victim_emb_norm, axis=0)
        inter_cluster_dist = cosine(attacker_centroid, victim_centroid)
        
        # Calculate intra-cluster distances
        attacker_intra = np.mean([
            cosine(attacker_emb_norm[i], attacker_centroid)
            for i in range(len(attacker_emb_norm))
        ])
        victim_intra = np.mean([
            cosine(victim_emb_norm[i], victim_centroid)
            for i in range(len(victim_emb_norm))
        ])
        
        # Separability ratio (higher = better separation)
        separability_ratio = inter_cluster_dist / (attacker_intra + victim_intra + 1e-8)
        
        # Combined clustering for silhouette score
        combined_embeddings = np.vstack([attacker_emb_norm, victim_emb_norm])
        combined_labels = np.concatenate([
            np.ones(len(attacker_emb_norm)),  # Label 1 for attackers
            np.zeros(len(victim_emb_norm))   # Label 0 for victims
        ])
        
        silhouette = silhouette_score(combined_embeddings, combined_labels)
        
        return {
            'inter_cluster_distance': float(inter_cluster_dist),
            'attacker_intra_cluster_distance': float(attacker_intra),
            'victim_intra_cluster_distance': float(victim_intra),
            'separability_ratio': float(separability_ratio),
            'silhouette_score': float(silhouette),
            'attacker_count': len(attacker_emb_norm),
            'victim_count': len(victim_emb_norm)
        }
    
    def extract_ngrams(self, texts: List[str], n: int = 2, top_n: int = 20) -> List[Tuple[Tuple[str, ...], int]]:
        """Extract n-grams from texts."""
        from nltk.tokenize import word_tokenize
        
        all_ngrams = []
        for text in texts:
            words = word_tokenize(text.lower())
            words = [w for w in words if w.isalpha()]
            
            if len(words) >= n:
                for i in range(len(words) - n + 1):
                    ngram = tuple(words[i:i+n])
                    all_ngrams.append(ngram)
        
        ngram_counts = Counter(all_ngrams)
        return ngram_counts.most_common(top_n)
    
    def analyze_blackbasta(self) -> Dict[str, Any]:
        """Focused analysis on BlackBasta group."""
        print("[INFO] Analyzing BlackBasta group...")
        
        # Find BlackBasta messages
        bb_attacker_msgs = [
            msg for msg in self.attacker_messages
            if 'blackbasta' in msg['group_name'].lower() or 'black basta' in msg['group_name'].lower()
        ]
        bb_victim_msgs = [
            msg for msg in self.victim_messages
            if 'blackbasta' in msg.get('group_name', '').lower() or 'black basta' in msg.get('group_name', '').lower()
        ]
        
        # Get corresponding embeddings
        bb_attacker_indices = [
            i for i, msg in enumerate(self.attacker_messages)
            if 'blackbasta' in msg['group_name'].lower() or 'black basta' in msg['group_name'].lower()
        ]
        bb_attacker_emb = self.attacker_embeddings[bb_attacker_indices] if bb_attacker_indices else None
        
        # Extract n-grams
        bb_attacker_texts = [msg['content'] for msg in bb_attacker_msgs]
        bb_victim_texts = [msg['content'] for msg in bb_victim_msgs]
        
        bb_attacker_unigrams = self.extract_ngrams(bb_attacker_texts, n=1, top_n=20)
        bb_attacker_bigrams = self.extract_ngrams(bb_attacker_texts, n=2, top_n=20)
        bb_attacker_trigrams = self.extract_ngrams(bb_attacker_texts, n=3, top_n=20)
        
        bb_victim_unigrams = self.extract_ngrams(bb_victim_texts, n=1, top_n=20) if bb_victim_texts else []
        bb_victim_bigrams = self.extract_ngrams(bb_victim_texts, n=2, top_n=20) if bb_victim_texts else []
        bb_victim_trigrams = self.extract_ngrams(bb_victim_texts, n=3, top_n=20) if bb_victim_texts else []
        
        # Compare with other attacker groups
        other_attacker_groups = defaultdict(list)
        for msg in self.attacker_messages:
            group = msg['group_name']
            if group and 'blackbasta' not in group.lower() and 'black basta' not in group.lower():
                other_attacker_groups[group].append(msg['content'])
        
        # Calculate similarity to other groups
        group_similarities = {}
        if bb_attacker_emb is not None and len(bb_attacker_emb) > 0:
            bb_centroid = np.mean(bb_attacker_emb / (np.linalg.norm(bb_attacker_emb, axis=1, keepdims=True) + 1e-8), axis=0)
            
            for group_name, group_texts in other_attacker_groups.items():
                if len(group_texts) < 5:
                    continue
                
                # Get embeddings for this group
                group_indices = [
                    i for i, msg in enumerate(self.attacker_messages)
                    if msg['group_name'] == group_name
                ]
                if len(group_indices) > 0:
                    group_emb = self.attacker_embeddings[group_indices]
                    group_emb_norm = group_emb / (np.linalg.norm(group_emb, axis=1, keepdims=True) + 1e-8)
                    group_centroid = np.mean(group_emb_norm, axis=0)
                    
                    similarity = 1 - cosine(bb_centroid, group_centroid)
                    group_similarities[group_name] = float(similarity)
        
        return {
            'attacker_messages': len(bb_attacker_msgs),
            'victim_messages': len(bb_victim_msgs),
            'attacker_unigrams': [(tuple([ng]) if isinstance(ng, str) else ng, count) for ng, count in bb_attacker_unigrams],
            'attacker_bigrams': bb_attacker_bigrams,
            'attacker_trigrams': bb_attacker_trigrams,
            'victim_unigrams': [(tuple([ng]) if isinstance(ng, str) else ng, count) for ng, count in bb_victim_unigrams],
            'victim_bigrams': bb_victim_bigrams,
            'victim_trigrams': bb_victim_trigrams,
            'group_similarities': group_similarities,
            'similar_groups': sorted(group_similarities.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def analyze_by_original_groups(self) -> Dict[str, Dict[str, Any]]:
        """Analyze n-grams for each original attacker group."""
        print("[INFO] Analyzing n-grams by original attacker groups...")
        
        group_analysis = {}
        
        # Group messages by group_name
        groups = defaultdict(list)
        group_indices = defaultdict(list)
        
        for i, msg in enumerate(self.attacker_messages):
            group_name = msg['group_name']
            if group_name:
                groups[group_name].append(msg['content'])
                group_indices[group_name].append(i)
        
        for group_name, texts in groups.items():
            if len(texts) < 3:  # Skip groups with too few messages
                continue
            
            unigrams = self.extract_ngrams(texts, n=1, top_n=20)
            bigrams = self.extract_ngrams(texts, n=2, top_n=20)
            trigrams = self.extract_ngrams(texts, n=3, top_n=20)
            
            # Get embeddings for this group
            indices = group_indices[group_name]
            group_emb = self.attacker_embeddings[indices]
            group_emb_norm = group_emb / (np.linalg.norm(group_emb, axis=1, keepdims=True) + 1e-8)
            group_centroid = np.mean(group_emb_norm, axis=0)
            
            group_analysis[group_name] = {
                'message_count': len(texts),
                'unigrams': [(tuple([ng]) if isinstance(ng, str) else ng, count) for ng, count in unigrams],
                'bigrams': bigrams,
                'trigrams': trigrams,
                'centroid': group_centroid.tolist()
            }
        
        return group_analysis
    
    def analyze_by_clusters(self) -> Dict[int, Dict[str, Any]]:
        """Analyze n-grams for each embedding-based cluster."""
        print("[INFO] Analyzing n-grams by embedding clusters...")
        
        cluster_analysis = {}
        
        for cluster_id in set(self.attacker_clusters):
            if cluster_id == -1:  # Skip noise in DBSCAN
                continue
            
            cluster_indices = [i for i, cid in enumerate(self.attacker_clusters) if cid == cluster_id]
            cluster_texts = [self.attacker_messages[i]['content'] for i in cluster_indices]
            
            if len(cluster_texts) < 3:
                continue
            
            unigrams = self.extract_ngrams(cluster_texts, n=1, top_n=20)
            bigrams = self.extract_ngrams(cluster_texts, n=2, top_n=20)
            trigrams = self.extract_ngrams(cluster_texts, n=3, top_n=20)
            
            # Get group distribution in this cluster
            cluster_groups = Counter([self.attacker_messages[i]['group_name'] for i in cluster_indices])
            
            cluster_analysis[int(cluster_id)] = {
                'message_count': len(cluster_texts),
                'group_distribution': dict(cluster_groups),
                'unigrams': [(tuple([ng]) if isinstance(ng, str) else ng, count) for ng, count in unigrams],
                'bigrams': bigrams,
                'trigrams': trigrams
            }
        
        return cluster_analysis
    
    def generate_visualizations(self, plots_dir: Path):
        """Generate all visualizations."""
        print("[INFO] Generating visualizations...")
        
        # 1. Separability visualization (attacker vs victim)
        self._plot_separability(plots_dir)
        
        # 2. Cluster visualization
        self._plot_clusters(plots_dir)
        
        # 3. BlackBasta n-grams
        self._plot_blackbasta_ngrams(plots_dir)
        
        # 4. Group comparison
        self._plot_group_comparison(plots_dir)
        
        return plots_dir
    
    def _plot_separability(self, plots_dir: Path):
        """Plot attacker vs victim separability."""
        if len(self.victim_embeddings) == 0:
            return
        
        # Use PCA for 2D visualization
        combined_embeddings = np.vstack([
            self.attacker_embeddings / (np.linalg.norm(self.attacker_embeddings, axis=1, keepdims=True) + 1e-8),
            self.victim_embeddings / (np.linalg.norm(self.victim_embeddings, axis=1, keepdims=True) + 1e-8)
        ])
        
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(combined_embeddings)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot attackers
        attacker_2d = emb_2d[:len(self.attacker_embeddings)]
        ax.scatter(attacker_2d[:, 0], attacker_2d[:, 1], 
                  c='red', alpha=0.6, label='Attacker', s=50)
        
        # Plot victims
        victim_2d = emb_2d[len(self.attacker_embeddings):]
        ax.scatter(victim_2d[:, 0], victim_2d[:, 1], 
                  c='blue', alpha=0.6, label='Victim', s=50)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title('Attacker vs Victim Separability\n(Embedding-Based Clustering)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'attacker_victim_separability.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_clusters(self, plots_dir: Path):
        """Plot attacker clusters."""
        # Use PCA for 2D visualization
        attacker_emb_norm = self.attacker_embeddings / (
            np.linalg.norm(self.attacker_embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(attacker_emb_norm)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        unique_clusters = sorted(set(self.attacker_clusters))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
        
        for cluster_id, color in zip(unique_clusters, colors):
            if cluster_id == -1:
                continue
            cluster_mask = self.attacker_clusters == cluster_id
            ax.scatter(emb_2d[cluster_mask, 0], emb_2d[cluster_mask, 1],
                      c=[color], alpha=0.6, label=f'Cluster {cluster_id}', s=50)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title('Attacker Message Clusters\n(Embedding-Based)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'attacker_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_blackbasta_ngrams(self, plots_dir: Path):
        """Plot BlackBasta n-gram analysis."""
        bb_analysis = self.analyze_blackbasta()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Unigrams
        if bb_analysis['attacker_unigrams']:
            unigrams, counts = zip(*bb_analysis['attacker_unigrams'][:15])
            unigram_strs = [' '.join(ng) for ng in unigrams]
            axes[0].barh(range(len(unigram_strs)), counts)
            axes[0].set_yticks(range(len(unigram_strs)))
            axes[0].set_yticklabels(unigram_strs)
            axes[0].set_xlabel('Frequency')
            axes[0].set_title('BlackBasta Attacker - Top Unigrams')
            axes[0].invert_yaxis()
        
        # Bigrams
        if bb_analysis['attacker_bigrams']:
            bigrams, counts = zip(*bb_analysis['attacker_bigrams'][:15])
            bigram_strs = [' '.join(bg) for bg in bigrams]
            axes[1].barh(range(len(bigram_strs)), counts)
            axes[1].set_yticks(range(len(bigram_strs)))
            axes[1].set_yticklabels(bigram_strs)
            axes[1].set_xlabel('Frequency')
            axes[1].set_title('BlackBasta Attacker - Top Bigrams')
            axes[1].invert_yaxis()
        
        # Trigrams
        if bb_analysis['attacker_trigrams']:
            trigrams, counts = zip(*bb_analysis['attacker_trigrams'][:15])
            trigram_strs = [' '.join(tg) for tg in trigrams]
            axes[2].barh(range(len(trigram_strs)), counts)
            axes[2].set_yticks(range(len(trigram_strs)))
            axes[2].set_yticklabels(trigram_strs)
            axes[2].set_xlabel('Frequency')
            axes[2].set_title('BlackBasta Attacker - Top Trigrams')
            axes[2].invert_yaxis()
        
        # Similar groups
        if bb_analysis['similar_groups']:
            groups, similarities = zip(*bb_analysis['similar_groups'])
            axes[3].barh(range(len(groups)), similarities)
            axes[3].set_yticks(range(len(groups)))
            axes[3].set_yticklabels(groups)
            axes[3].set_xlabel('Cosine Similarity')
            axes[3].set_title('BlackBasta - Similar Attacker Groups')
            axes[3].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'blackbasta_ngrams.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_group_comparison(self, plots_dir: Path):
        """Plot n-gram comparison across groups."""
        group_analysis = self.analyze_by_original_groups()
        
        # Focus on top groups
        top_groups = sorted(group_analysis.items(), key=lambda x: x[1]['message_count'], reverse=True)[:5]
        
        if not top_groups:
            return
        
        fig, axes = plt.subplots(len(top_groups), 1, figsize=(16, 4*len(top_groups)))
        if len(top_groups) == 1:
            axes = [axes]
        
        for idx, (group_name, analysis) in enumerate(top_groups):
            if analysis['bigrams']:
                bigrams, counts = zip(*analysis['bigrams'][:10])
                bigram_strs = [' '.join(bg) for bg in bigrams]
                axes[idx].barh(range(len(bigram_strs)), counts)
                axes[idx].set_yticks(range(len(bigram_strs)))
                axes[idx].set_yticklabels(bigram_strs)
                axes[idx].set_xlabel('Frequency')
                axes[idx].set_title(f'{group_name} - Top Bigrams (n={analysis["message_count"]})')
                axes[idx].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'group_ngram_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        print("[INFO] Generating analysis report...")
        
        separability = self.calculate_separability_metrics()
        bb_analysis = self.analyze_blackbasta()
        group_analysis = self.analyze_by_original_groups()
        cluster_analysis = self.analyze_by_clusters()
        
        return {
            'separability_metrics': separability,
            'blackbasta_analysis': bb_analysis,
            'group_analysis': group_analysis,
            'cluster_analysis': cluster_analysis,
            'attacker_cluster_count': len(set(self.attacker_clusters)),
            'victim_cluster_count': len(set(self.victim_clusters)) if len(self.victim_clusters) > 0 else 0
        }


def main():
    """Main execution function."""
    csv_path = "output/ransom_chats.csv"
    
    if not Path(csv_path).exists():
        print(f"[ERROR] CSV file not found: {csv_path}")
        return
    
    analyzer = SentenceClusteringAnalyzer(csv_path)
    
    # Load and prepare
    analyzer.load_and_prepare_messages()
    
    # Generate embeddings
    analyzer.generate_embeddings()
    
    # Cluster
    analyzer.cluster_by_embeddings(method='kmeans', n_clusters=None)
    
    # Generate visualizations
    plots_dir = analyzer.output_dir / "sentence_clustering_plots"
    plots_dir.mkdir(exist_ok=True)
    analyzer.generate_visualizations(plots_dir)
    
    # Generate analysis
    analysis_report = analyzer.generate_analysis_report()
    
    # Save report
    report_path = analyzer.output_dir / "sentence_clustering_analysis.json"
    with open(report_path, 'w', encoding='utf-8', errors='replace') as f:
        json.dump(analysis_report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n[SUCCESS] Analysis complete!")
    print(f"[INFO] Report saved to: {report_path}")
    print(f"[INFO] Plots saved to: {plots_dir}")
    
    return analyzer, analysis_report


if __name__ == "__main__":
    main()

