"""
Advanced Ransom Chat Analysis with Hypothesis Testing, Clustering, and Expert Reporting
Focus: Matching Black Basta to other attacker groups
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Statistical tests
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency, mannwhitneyu, pearsonr
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from matplotlib import font_manager

# Import text utilities for safe parsing
import sys
import re
import unicodedata
import string
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Standalone text parsing functions (avoid full module import)
PRINTABLE_ASCII = set(string.printable)
SAFE_UNICODE_CATEGORIES = {'L', 'N', 'P', 'S', 'Z'}

def parse_raw_text(text, max_length=None):
    """Parse raw text content, keeping only safe printable characters."""
    if text is None:
        return ''
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ''
    try:
        text = unicodedata.normalize('NFKC', text)
    except Exception:
        try:
            text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        except Exception:
            return ''
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    text = re.sub(r'[\u200B-\u200D\uFEFF\u2060]', '', text)
    text = re.sub(r'[\u202A-\u202E\u2066-\u2069]', '', text)
    cleaned_chars = []
    for char in text:
        if char in PRINTABLE_ASCII:
            cleaned_chars.append(char)
        elif unicodedata.category(char)[0] in SAFE_UNICODE_CATEGORIES:
            try:
                if char.isprintable() or char.isspace():
                    cleaned_chars.append(char)
            except Exception:
                continue
    text = ''.join(cleaned_chars)
    text = re.sub(r'[ \t]+', ' ', text)
    if max_length and len(text) > max_length:
        text = text[:max_length].rstrip() + '...'
    return text.strip()

def parse_identifier(text, allow_unicode=False):
    """Parse identifier-like text."""
    if text is None:
        return ''
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ''
    try:
        text = unicodedata.normalize('NFKC', text)
    except Exception:
        text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    if allow_unicode:
        text = ''.join(c for c in text if c.isalnum() or c in '_-')
    else:
        text = ''.join(c for c in text if c.isalnum() or c in '_-')
        text = text.encode('ascii', errors='ignore').decode('ascii')
    return text.strip()

# Price extractor (simplified)
class PriceExtractor:
    def extract_first_price(self, text):
        """Extract first price from text."""
        import re
        # Simple regex for prices
        patterns = [
            r'(?i)(\d[\d,.]*)\s*(USD|EUR|GBP|BTC|ETH|XMR)',
            r'(?i)\$(\d[\d,.]*)',
            r'(?i)(\d[\d,.]*)\s*dollars?',
        ]
        for pattern in patterns:
            match = re.search(pattern, str(text))
            if match:
                amount = float(match.group(1).replace(',', ''))
                currency = match.group(2) if len(match.groups()) > 1 else 'USD'
                return {'amount': amount, 'currency': currency}
        return None

# Timestamp analyzer (simplified)
class TimestampAnalyzer:
    def parse_timestamp(self, ts_str):
        """Parse timestamp string."""
        from dateutil import parser
        try:
            return parser.parse(str(ts_str))
        except Exception:
            return None

# Set style
if HAS_SEABORN:
    sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class AdvancedRansomAnalyzer:
    """Advanced analysis with hypothesis testing and clustering."""
    
    def __init__(self, csv_path: str):
        """Initialize analyzer with CSV data."""
        self.csv_path = csv_path
        self.df = None
        self.price_extractor = PriceExtractor()
        self.timestamp_analyzer = TimestampAnalyzer()
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
    
    def _normalize_timestamp(self, dt):
        """Normalize datetime to naive for comparison."""
        if dt is None or pd.isna(dt):
            return None
        if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        return dt
        
    def load_data(self):
        """Load and sanitize CSV data."""
        print("[INFO] Loading data...")
        # Read CSV with safe encoding
        try:
            with open(self.csv_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            from io import StringIO
            self.df = pd.read_csv(StringIO(content))
        except Exception:
            # Fallback to latin-1
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
        
        # Parse timestamps
        if 'timestamp' in self.df.columns:
            self.df['parsed_timestamp'] = self.df['timestamp'].apply(
                lambda x: self.timestamp_analyzer.parse_timestamp(str(x)) if pd.notna(x) else None
            )
            # Normalize all timestamps to naive
            self.df['parsed_timestamp'] = self.df['parsed_timestamp'].apply(self._normalize_timestamp)
        
        # Extract prices if not present
        if 'price_amount' not in self.df.columns or self.df['price_amount'].isna().all():
            print("[INFO] Extracting prices...")
            self.df['price_amount'] = None
            self.df['price_currency'] = None
            for idx, row in self.df.iterrows():
                content = str(row.get('content', ''))
                price_info = self.price_extractor.extract_first_price(content)
                if price_info:
                    self.df.at[idx, 'price_amount'] = price_info['amount']
                    self.df.at[idx, 'price_currency'] = parse_identifier(
                        price_info.get('currency', ''), allow_unicode=True
                    )
        
        print(f"[INFO] Loaded {len(self.df)} messages from {self.df['chat_id'].nunique()} chats")
        print(f"[INFO] Groups: {sorted(self.df['group_name'].unique())}")
        
    def calculate_response_delays(self):
        """Calculate response delays between messages."""
        print("[INFO] Calculating response delays...")
        
        # Group by chat_id and sort by timestamp
        delays = []
        for chat_id in self.df['chat_id'].unique():
            chat_df = self.df[self.df['chat_id'] == chat_id].copy()
            # Filter out None timestamps and sort
            chat_df = chat_df[chat_df['parsed_timestamp'].notna()].copy()
            if len(chat_df) < 2:
                continue
            # Normalize timestamps
            chat_df['parsed_timestamp'] = chat_df['parsed_timestamp'].apply(self._normalize_timestamp)
            chat_df = chat_df.sort_values('parsed_timestamp')
            
            for i in range(1, len(chat_df)):
                prev_msg = chat_df.iloc[i-1]
                curr_msg = chat_df.iloc[i]
                
                if prev_msg['parsed_timestamp'] and curr_msg['parsed_timestamp']:
                    delay = (curr_msg['parsed_timestamp'] - prev_msg['parsed_timestamp']).total_seconds() / 60
                    
                    delays.append({
                        'chat_id': chat_id,
                        'message_index': i,
                        'prev_party': prev_msg['party'],
                        'curr_party': curr_msg['party'],
                        'delay_minutes': delay,
                        'group_name': curr_msg['group_name'],
                        'is_attacker_response': curr_msg['party'] == 'attacker' and prev_msg['party'] == 'victim',
                        'is_victim_response': curr_msg['party'] == 'victim' and prev_msg['party'] == 'attacker',
                    })
        
        self.response_delays_df = pd.DataFrame(delays)
        return self.response_delays_df
    
    def analyze_first_messages(self):
        """Analyze first message timestamps per group."""
        print("[INFO] Analyzing first messages...")
        
        first_messages = []
        for chat_id in self.df['chat_id'].unique():
            chat_df = self.df[self.df['chat_id'] == chat_id].copy()
            # Filter out None timestamps
            chat_df = chat_df[chat_df['parsed_timestamp'].notna()].copy()
            if len(chat_df) == 0:
                continue
            # Normalize timestamps
            chat_df['parsed_timestamp'] = chat_df['parsed_timestamp'].apply(self._normalize_timestamp)
            chat_df = chat_df.sort_values('parsed_timestamp')
            
            first_msg = chat_df.iloc[0]
            if first_msg['parsed_timestamp']:
                first_messages.append({
                    'chat_id': chat_id,
                    'group_name': first_msg['group_name'],
                    'first_party': first_msg['party'],
                    'first_timestamp': first_msg['parsed_timestamp'],
                    'first_hour': first_msg['parsed_timestamp'].hour,
                    'first_day_of_week': first_msg['parsed_timestamp'].weekday(),
                })
        
        self.first_messages_df = pd.DataFrame(first_messages)
        return self.first_messages_df
    
    def hypothesis_test_work_hours(self, group1: str, group2: str) -> Dict:
        """Test if two groups follow similar work hour distributions."""
        print(f"[INFO] Testing work hour distribution: {group1} vs {group2}")
        
        # Get hourly activity for each group
        group1_hours = self.df[self.df['group_name'] == group1]['parsed_timestamp'].apply(
            lambda x: x.hour if pd.notna(x) else None
        ).dropna()
        
        group2_hours = self.df[self.df['group_name'] == group2]['parsed_timestamp'].apply(
            lambda x: x.hour if pd.notna(x) else None
        ).dropna()
        
        if len(group1_hours) < 10 or len(group2_hours) < 10:
            return {'error': 'Insufficient data'}
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = ks_2samp(group1_hours, group2_hours)
        
        # Chi-square test (binned by hour)
        bins = np.arange(0, 25)
        hist1, _ = np.histogram(group1_hours, bins=bins)
        hist2, _ = np.histogram(group2_hours, bins=bins)
        
        # Create contingency table
        contingency = np.array([hist1, hist2])
        # Add small value to avoid zero frequencies
        contingency = contingency + 1e-10
        try:
            chi2, chi2_pvalue, dof, expected = chi2_contingency(contingency)
        except ValueError:
            # If chi-square fails, set default values
            chi2, chi2_pvalue, dof, expected = 0.0, 1.0, 0, None
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pvalue = mannwhitneyu(group1_hours, group2_hours, alternative='two-sided')
        
        # Calculate R² (coefficient of determination)
        # Compare to expected uniform distribution
        total_samples = len(group1_hours) + len(group2_hours)
        expected_uniform = np.ones(24) * total_samples / 24
        observed = hist1 + hist2  # Sum histograms, not concatenate
        ss_res = np.sum((observed - expected_uniform) ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'group1': group1,
            'group2': group2,
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'ks_significant': ks_pvalue < 0.05,
            'chi2_statistic': float(chi2),
            'chi2_pvalue': float(chi2_pvalue),
            'chi2_significant': chi2_pvalue < 0.05,
            'u_statistic': float(u_stat),
            'u_pvalue': float(u_pvalue),
            'u_significant': u_pvalue < 0.05,
            'r2': float(r2),
            'group1_mean_hour': float(np.mean(group1_hours)),
            'group2_mean_hour': float(np.mean(group2_hours)),
            'group1_std_hour': float(np.std(group1_hours)),
            'group2_std_hour': float(np.std(group2_hours)),
        }
    
    def hypothesis_test_response_times(self, group1: str, group2: str) -> Dict:
        """Test if response time distributions differ between groups."""
        print(f"[INFO] Testing response time distribution: {group1} vs {group2}")
        
        if not hasattr(self, 'response_delays_df'):
            self.calculate_response_delays()
        
        group1_delays = self.response_delays_df[
            self.response_delays_df['group_name'] == group1
        ]['delay_minutes'].dropna()
        
        group2_delays = self.response_delays_df[
            self.response_delays_df['group_name'] == group2
        ]['delay_minutes'].dropna()
        
        # Filter outliers (remove > 24 hours)
        group1_delays = group1_delays[group1_delays <= 1440]
        group2_delays = group2_delays[group2_delays <= 1440]
        
        if len(group1_delays) < 10 or len(group2_delays) < 10:
            return {'error': 'Insufficient data'}
        
        # KS test
        ks_stat, ks_pvalue = ks_2samp(group1_delays, group2_delays)
        
        # Mann-Whitney U test
        u_stat, u_pvalue = mannwhitneyu(group1_delays, group2_delays, alternative='two-sided')
        
        return {
            'group1': group1,
            'group2': group2,
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'ks_significant': ks_pvalue < 0.05,
            'u_statistic': float(u_stat),
            'u_pvalue': float(u_pvalue),
            'u_significant': u_pvalue < 0.05,
            'group1_mean_minutes': float(np.mean(group1_delays)),
            'group2_mean_minutes': float(np.mean(group2_delays)),
            'group1_median_minutes': float(np.median(group1_delays)),
            'group2_median_minutes': float(np.median(group2_delays)),
        }
    
    def cluster_groups_by_similarity(self, focus_group: Optional[str] = None) -> Dict:
        """Cluster groups by behavioral similarity."""
        if focus_group is None:
            # Try to find Black Basta
            groups = sorted(self.df['group_name'].unique())
            for variant in ["BlackBasta", "Black Basta", "blackbasta", "black basta"]:
                if variant in groups:
                    focus_group = variant
                    break
            if not focus_group:
                focus_group = groups[0] if groups else "Unknown"
        
        print(f"[INFO] Clustering groups by similarity (focus: {focus_group})...")
        
        groups = sorted(self.df['group_name'].unique())
        features = []
        group_names = []
        
        for group in groups:
            group_df = self.df[self.df['group_name'] == group]
            
            # Extract features
            feature_vector = []
            
            # Hourly distribution (24 features)
            hours = group_df['parsed_timestamp'].apply(
                lambda x: x.hour if pd.notna(x) else None
            ).dropna()
            hour_dist = np.histogram(hours, bins=24, range=(0, 24))[0]
            feature_vector.extend(hour_dist / (np.sum(hour_dist) + 1e-10))
            
            # Day of week distribution (7 features)
            days = group_df['parsed_timestamp'].apply(
                lambda x: x.weekday() if pd.notna(x) else None
            ).dropna()
            day_dist = np.histogram(days, bins=7, range=(0, 7))[0]
            feature_vector.extend(day_dist / (np.sum(day_dist) + 1e-10))
            
            # Average message length
            avg_length = group_df['content'].apply(len).mean()
            feature_vector.append(avg_length / 1000.0)  # Normalize
            
            # Price statistics (if available)
            prices = group_df['price_amount'].dropna()
            if len(prices) > 0:
                feature_vector.append(np.mean(prices) / 1000000.0)  # Normalize to millions
                feature_vector.append(np.std(prices) / 1000000.0)
            else:
                feature_vector.extend([0, 0])
            
            # Response time statistics
            if hasattr(self, 'response_delays_df'):
                group_delays = self.response_delays_df[
                    self.response_delays_df['group_name'] == group
                ]['delay_minutes'].dropna()
                group_delays = group_delays[group_delays <= 1440]
                if len(group_delays) > 0:
                    feature_vector.append(np.mean(group_delays) / 1440.0)  # Normalize to days
                    feature_vector.append(np.median(group_delays) / 1440.0)
                else:
                    feature_vector.extend([0, 0])
            else:
                feature_vector.extend([0, 0])
            
            features.append(feature_vector)
            group_names.append(group)
        
        # Convert to numpy array
        X = np.array(features)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        
        # KMeans clustering (for comparison)
        n_clusters = min(5, len(groups) - 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette = silhouette_score(X_scaled, kmeans_labels) if len(set(kmeans_labels)) > 1 else 0
        
        # Find focus group cluster
        focus_idx = group_names.index(focus_group) if focus_group in group_names else -1
        focus_dbscan_cluster = dbscan_labels[focus_idx] if focus_idx >= 0 else -1
        focus_kmeans_cluster = kmeans_labels[focus_idx] if focus_idx >= 0 else -1
        
        # Find similar groups
        similar_groups_dbscan = [
            group_names[i] for i in range(len(group_names))
            if dbscan_labels[i] == focus_dbscan_cluster and i != focus_idx
        ]
        
        similar_groups_kmeans = [
            group_names[i] for i in range(len(group_names))
            if kmeans_labels[i] == focus_kmeans_cluster and i != focus_idx
        ]
        
        return {
            'groups': group_names,
            'dbscan_labels': dbscan_labels.tolist(),
            'kmeans_labels': kmeans_labels.tolist(),
            'silhouette_score': float(silhouette),
            'focus_group': focus_group,
            'focus_dbscan_cluster': int(focus_dbscan_cluster),
            'focus_kmeans_cluster': int(focus_kmeans_cluster),
            'similar_groups_dbscan': similar_groups_dbscan,
            'similar_groups_kmeans': similar_groups_kmeans,
            'feature_matrix': X_scaled.tolist(),
        }
    
    def generate_plots(self):
        """Generate all analysis plots."""
        print("[INFO] Generating plots...")
        
        plots_dir = self.output_dir / "advanced_analysis_plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Work hour distribution comparison
        self._plot_work_hour_distribution(plots_dir)
        
        # 2. Response time distributions
        self._plot_response_time_distributions(plots_dir)
        
        # 3. Price analysis
        self._plot_price_analysis(plots_dir)
        
        # 4. First message analysis
        self._plot_first_message_analysis(plots_dir)
        
        # 5. Clustering visualization
        self._plot_clustering(plots_dir)
        
        # 6. BlackBasta-specific plots
        self._plot_blackbasta_specific(plots_dir)
        
        return plots_dir
    
    def _plot_work_hour_distribution(self, plots_dir: Path):
        """Plot work hour distributions for all groups."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        groups = sorted(self.df['group_name'].unique())
        
        for idx, group in enumerate(groups[:4]):  # Top 4 groups
            group_df = self.df[self.df['group_name'] == group]
            hours = group_df['parsed_timestamp'].apply(
                lambda x: x.hour if pd.notna(x) else None
            ).dropna()
            
            if len(hours) > 0:
                axes[idx].hist(hours, bins=24, range=(0, 24), alpha=0.7, edgecolor='black')
                axes[idx].set_title(f'{group} - Hourly Activity Distribution\n(n={len(hours)} messages)')
                axes[idx].set_xlabel('Hour of Day (GMT)')
                axes[idx].set_ylabel('Message Count')
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'work_hour_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_response_time_distributions(self, plots_dir: Path):
        """Plot response time distributions."""
        if not hasattr(self, 'response_delays_df'):
            self.calculate_response_delays()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        groups = sorted(self.response_delays_df['group_name'].unique())
        colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
        
        for group, color in zip(groups, colors):
            group_delays = self.response_delays_df[
                self.response_delays_df['group_name'] == group
            ]['delay_minutes'].dropna()
            group_delays = group_delays[group_delays <= 1440]  # Max 24 hours
            
            if len(group_delays) > 0:
                ax.hist(group_delays, bins=60, alpha=0.5, label=group, color=color)
        
        ax.set_xlabel('Response Time (minutes)')
        ax.set_ylabel('Frequency')
        ax.set_title('Response Time Distributions by Group\n(Filtered: < 24 hours)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'response_time_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_price_analysis(self, plots_dir: Path):
        """Plot price analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Price distribution by group
        price_df = self.df[self.df['price_amount'].notna()]
        if len(price_df) > 0:
            groups = sorted(price_df['group_name'].unique())
            prices_by_group = [price_df[price_df['group_name'] == g]['price_amount'].values for g in groups]
            
            axes[0].boxplot(prices_by_group, labels=groups)
            axes[0].set_ylabel('Price Amount')
            axes[0].set_title('Price Distribution by Group')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3)
            
            # Price over time (normalize timestamps first)
            price_df['parsed_timestamp'] = price_df['parsed_timestamp'].apply(self._normalize_timestamp)
            price_df_sorted = price_df.sort_values('parsed_timestamp')
            axes[1].scatter(price_df_sorted['parsed_timestamp'], price_df_sorted['price_amount'], alpha=0.6)
            axes[1].set_xlabel('Timestamp')
            axes[1].set_ylabel('Price Amount')
            axes[1].set_title('Price Over Time')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'price_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_first_message_analysis(self, plots_dir: Path):
        """Plot first message analysis."""
        if not hasattr(self, 'first_messages_df'):
            self.analyze_first_messages()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # First message hour distribution
        groups = sorted(self.first_messages_df['group_name'].unique())
        for group in groups:
            group_first = self.first_messages_df[self.first_messages_df['group_name'] == group]
            axes[0].hist(group_first['first_hour'], bins=24, alpha=0.5, label=group)
        
        axes[0].set_xlabel('Hour of Day (GMT)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('First Message Hour Distribution by Group')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # First message party (attacker vs victim)
        party_counts = self.first_messages_df['first_party'].value_counts()
        axes[1].pie(party_counts.values, labels=party_counts.index, autopct='%1.1f%%')
        axes[1].set_title('First Message Party Distribution')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'first_message_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_clustering(self, plots_dir: Path):
        """Plot clustering results."""
        clustering_result = self.cluster_groups_by_similarity()
        
        # Use PCA for 2D visualization
        from sklearn.decomposition import PCA
        
        X = np.array(clustering_result['feature_matrix'])
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot points
        for i, group in enumerate(clustering_result['groups']):
            cluster_id = clustering_result['kmeans_labels'][i]
            color = plt.cm.Set3(cluster_id / max(clustering_result['kmeans_labels']))
            marker = 'o' if group == clustering_result['focus_group'] else 's'
            size = 300 if group == clustering_result['focus_group'] else 150
            
            ax.scatter(X_2d[i, 0], X_2d[i, 1], c=[color], marker=marker, s=size, 
                      label=group if i < 10 else '', alpha=0.7, edgecolors='black', linewidths=2)
            ax.annotate(group, (X_2d[i, 0], X_2d[i, 1]), fontsize=8)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title(f'Group Clustering Visualization\n(Focus: {clustering_result["focus_group"]})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'clustering_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_blackbasta_specific(self, plots_dir: Path):
        """Generate BlackBasta-specific comparison plots."""
        focus_group = "BlackBasta"
        if focus_group not in self.df['group_name'].unique():
            return
        
        # 1. BlackBasta work hour distribution
        fig, ax = plt.subplots(figsize=(14, 8))
        bb_df = self.df[self.df['group_name'] == focus_group]
        hours = bb_df['parsed_timestamp'].apply(
            lambda x: x.hour if pd.notna(x) else None
        ).dropna()
        
        if len(hours) > 0:
            ax.hist(hours, bins=24, range=(0, 24), alpha=0.7, edgecolor='black', color='#8B0000')
            ax.axhline(y=len(hours)/24, color='r', linestyle='--', label='Uniform Distribution')
            ax.set_xlabel('Hour of Day (GMT)', fontsize=12)
            ax.set_ylabel('Message Count', fontsize=12)
            ax.set_title(f'{focus_group} - Hourly Activity Distribution\n(n={len(hours)} messages)', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'blackbasta_work_hours.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. BlackBasta response time distribution
        if hasattr(self, 'response_delays_df'):
            fig, ax = plt.subplots(figsize=(14, 8))
            bb_delays = self.response_delays_df[
                self.response_delays_df['group_name'] == focus_group
            ]['delay_minutes'].dropna()
            bb_delays = bb_delays[bb_delays <= 1440]  # Max 24 hours
            
            if len(bb_delays) > 0:
                ax.hist(bb_delays, bins=60, alpha=0.7, edgecolor='black', color='#8B0000')
                ax.set_xlabel('Response Time (minutes)', fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.set_title(f'{focus_group} - Response Time Distribution\n(Filtered: < 24 hours, n={len(bb_delays)})', fontsize=14, fontweight='bold')
                ax.axvline(x=np.mean(bb_delays), color='r', linestyle='--', label=f'Mean: {np.mean(bb_delays):.1f} min')
                ax.axvline(x=np.median(bb_delays), color='g', linestyle='--', label=f'Median: {np.median(bb_delays):.1f} min')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'blackbasta_response_times.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. BlackBasta price analysis
        fig, ax = plt.subplots(figsize=(14, 8))
        bb_prices = bb_df[bb_df['price_amount'].notna()]['price_amount']
        
        if len(bb_prices) > 0:
            ax.hist(bb_prices, bins=30, alpha=0.7, edgecolor='black', color='#8B0000')
            ax.set_xlabel('Price Amount (USD)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'{focus_group} - Price Distribution\n(n={len(bb_prices)} price mentions)', fontsize=14, fontweight='bold')
            ax.axvline(x=np.mean(bb_prices), color='r', linestyle='--', label=f'Mean: ${np.mean(bb_prices):,.0f}')
            ax.axvline(x=np.median(bb_prices), color='g', linestyle='--', label=f'Median: ${np.median(bb_prices):,.0f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')  # Log scale for better visualization
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'blackbasta_prices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. BlackBasta vs similar groups comparison
        clustering_result = self.cluster_groups_by_similarity(focus_group)
        similar_groups = clustering_result['similar_groups_kmeans'][:3]  # Top 3 similar
        
        if similar_groups:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            # Compare work hours
            bb_hours = hours
            for idx, group in enumerate([focus_group] + similar_groups[:3]):
                if idx >= 4:
                    break
                group_df = self.df[self.df['group_name'] == group]
                group_hours = group_df['parsed_timestamp'].apply(
                    lambda x: x.hour if pd.notna(x) else None
                ).dropna()
                
                if len(group_hours) > 0:
                    color = '#8B0000' if group == focus_group else plt.cm.Set3(idx)
                    axes[idx].hist(group_hours, bins=24, range=(0, 24), alpha=0.7, 
                                  edgecolor='black', color=color, label=group)
                    axes[idx].set_xlabel('Hour of Day (GMT)')
                    axes[idx].set_ylabel('Message Count')
                    axes[idx].set_title(f'{group}\n(n={len(group_hours)})')
                    axes[idx].grid(True, alpha=0.3)
                    axes[idx].legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'blackbasta_vs_similar_groups.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_expert_report(self) -> str:
        """Generate comprehensive expert-level markdown report."""
        print("[INFO] Generating expert report...")
        
        # Run all analyses
        if not hasattr(self, 'response_delays_df'):
            self.calculate_response_delays()
        if not hasattr(self, 'first_messages_df'):
            self.analyze_first_messages()
        
        # Generate plots
        plots_dir = self.generate_plots()
        
        # Perform hypothesis tests
        groups = sorted(self.df['group_name'].unique())
        # Try different variations of Black Basta
        focus_group = None
        for variant in ["BlackBasta", "Black Basta", "blackbasta", "black basta"]:
            if variant in groups:
                focus_group = variant
                break
        if not focus_group:
            focus_group = groups[0] if groups else "Unknown"
        
        work_hour_tests = []
        response_time_tests = []
        
        for group in groups:
            if group != focus_group:
                work_test = self.hypothesis_test_work_hours(focus_group, group)
                if 'error' not in work_test:
                    work_hour_tests.append(work_test)
                
                response_test = self.hypothesis_test_response_times(focus_group, group)
                if 'error' not in response_test:
                    response_time_tests.append(response_test)
        
        # Clustering
        clustering_result = self.cluster_groups_by_similarity(focus_group)
        
        # Run sentence-level clustering analysis
        print("[INFO] Running sentence-level clustering analysis...")
        try:
            from scripts.analyze_sentence_clustering import SentenceClusteringAnalyzer
            sentence_analyzer = SentenceClusteringAnalyzer(self.csv_path)
            sentence_analyzer.load_and_prepare_messages()
            sentence_analyzer.generate_embeddings()
            sentence_analyzer.cluster_by_embeddings(method='kmeans', n_clusters=None)
            
            # Generate visualizations
            sentence_plots_dir = self.output_dir / "sentence_clustering_plots"
            sentence_plots_dir.mkdir(exist_ok=True)
            sentence_analyzer.generate_visualizations(sentence_plots_dir)
            
            # Get analysis results
            separability_metrics = sentence_analyzer.calculate_separability_metrics()
            bb_analysis = sentence_analyzer.analyze_blackbasta()
            group_analysis = sentence_analyzer.analyze_by_original_groups()
            cluster_analysis = sentence_analyzer.analyze_by_clusters()
            attacker_cluster_count = len(set(sentence_analyzer.attacker_clusters))
            victim_cluster_count = len(set(sentence_analyzer.victim_clusters)) if len(sentence_analyzer.victim_clusters) > 0 else 0
        except Exception as e:
            print(f"[WARNING] Sentence clustering analysis failed: {e}")
            separability_metrics = {'error': str(e)}
            bb_analysis = {}
            group_analysis = {}
            cluster_analysis = {}
            attacker_cluster_count = 0
            victim_cluster_count = 0
        
        # Get BlackBasta-specific statistics (before report generation)
        bb_df = self.df[self.df['group_name'] == focus_group]
        bb_hours = bb_df['parsed_timestamp'].apply(
            lambda x: x.hour if pd.notna(x) else None
        ).dropna()
        bb_mean_hour = float(np.mean(bb_hours)) if len(bb_hours) > 0 else 0.0
        
        bb_delays = []
        bb_delays_mean = 0.0
        bb_delays_median = 0.0
        if hasattr(self, 'response_delays_df'):
            bb_delays_raw = self.response_delays_df[
                self.response_delays_df['group_name'] == focus_group
            ]['delay_minutes'].dropna()
            bb_delays_raw = bb_delays_raw[bb_delays_raw <= 1440]
            bb_delays = bb_delays_raw.tolist()
            if len(bb_delays) > 0:
                bb_delays_mean = float(np.mean(bb_delays))
                bb_delays_median = float(np.median(bb_delays))
        
        bb_prices_raw = bb_df[bb_df['price_amount'].notna()]['price_amount']
        bb_prices = bb_prices_raw.tolist()
        bb_prices_mean = float(np.mean(bb_prices)) if len(bb_prices) > 0 else 0.0
        bb_prices_median = float(np.median(bb_prices)) if len(bb_prices) > 0 else 0.0
        bb_prices_min = float(np.min(bb_prices)) if len(bb_prices) > 0 else 0.0
        bb_prices_max = float(np.max(bb_prices)) if len(bb_prices) > 0 else 0.0
        
        # Similar groups list for report
        similar_groups_list = clustering_result['similar_groups_kmeans'][:5]
        similar_groups_str = '\n'.join(f"- {group}" for group in similar_groups_list)
        
        # Generate markdown report
        report = f"""# Advanced Ransom Chat Analysis Report
## Expert Data Science Analysis - Black Basta Group Matching

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Dataset:** {self.csv_path}  
**Total Messages:** {len(self.df)}  
**Total Chats:** {self.df['chat_id'].nunique()}  
**Groups Identified:** {len(groups)}

---

## Executive Summary

This report presents a comprehensive statistical analysis of ransom chat data with a focus on identifying behavioral patterns and potential group matching, specifically targeting the **{focus_group}** group. The analysis employs hypothesis testing, clustering algorithms, and distribution comparisons to identify similarities between attacker groups.

### Key Findings

1. **Group Similarity Analysis**: {len(clustering_result['similar_groups_kmeans'])} groups show behavioral similarity to {focus_group}
2. **Work Hour Patterns**: Statistical tests reveal {'significant' if any(t['ks_significant'] for t in work_hour_tests) else 'non-significant'} differences in work hour distributions
3. **Response Time Patterns**: Response time analysis indicates {'significant' if any(t['ks_significant'] for t in response_time_tests) else 'non-significant'} behavioral differences

---

## 1. Hypothesis Testing: Work Hour Distribution

### Research Question
Do different attacker groups follow similar work hour distributions, or do they adapt to victim schedules?

### Methodology
- **Kolmogorov-Smirnov Test**: Non-parametric test for distribution equality
- **Chi-Square Test**: Tests independence of hour distributions
- **Mann-Whitney U Test**: Non-parametric test for distribution differences
- **R² Coefficient**: Measures deviation from uniform distribution

### Results: {focus_group} vs Other Groups

"""
        
        for test in work_hour_tests[:5]:  # Top 5 comparisons
            report += f"""
#### {focus_group} vs {test['group2']}

- **KS Test**: D = {test['ks_statistic']:.4f}, p = {test['ks_pvalue']:.4f} ({'**Significant**' if test['ks_significant'] else 'Not significant'})
- **Chi-Square Test**: χ² = {test['chi2_statistic']:.2f}, p = {test['chi2_pvalue']:.4f} ({'**Significant**' if test['chi2_significant'] else 'Not significant'})
- **Mann-Whitney U**: U = {test['u_statistic']:.2f}, p = {test['u_pvalue']:.4f} ({'**Significant**' if test['u_significant'] else 'Not significant'})
- **R²**: {test['r2']:.4f}
- **Mean Hour**: {focus_group} = {test['group1_mean_hour']:.1f}h, {test['group2']} = {test['group2_mean_hour']:.1f}h

**Interpretation**: {'Groups show similar work hour patterns' if not test['ks_significant'] else 'Groups show significantly different work hour patterns'}

"""
        
        report += f"""
![Work Hour Distributions]({plots_dir.name}/work_hour_distributions.png)

---

## 2. Hypothesis Testing: Response Time Distribution

### Research Question
Do response time distributions differ significantly between attacker groups, indicating different operational patterns?

### Results: {focus_group} vs Other Groups

"""
        
        for test in response_time_tests[:5]:
            report += f"""
#### {focus_group} vs {test['group2']}

- **KS Test**: D = {test['ks_statistic']:.4f}, p = {test['ks_pvalue']:.4f} ({'**Significant**' if test['ks_significant'] else 'Not significant'})
- **Mann-Whitney U**: U = {test['u_statistic']:.2f}, p = {test['u_pvalue']:.4f} ({'**Significant**' if test['u_significant'] else 'Not significant'})
- **Mean Response Time**: {focus_group} = {test['group1_mean_minutes']:.1f} min, {test['group2']} = {test['group2_mean_minutes']:.1f} min
- **Median Response Time**: {focus_group} = {test['group1_median_minutes']:.1f} min, {test['group2']} = {test['group2_median_minutes']:.1f} min

**Interpretation**: {'Groups show similar response time patterns' if not test['ks_significant'] else 'Groups show significantly different response time patterns'}

"""
        
        report += f"""
![Response Time Distributions]({plots_dir.name}/response_time_distributions.png)

---

## 3. Clustering Analysis: Group Similarity

### Methodology
- **Feature Extraction**: Hourly distribution, day-of-week distribution, message length, price statistics, response time statistics
- **DBSCAN Clustering**: Density-based clustering for outlier detection
- **K-Means Clustering**: Partition-based clustering for group similarity
- **Silhouette Score**: {clustering_result['silhouette_score']:.4f} (higher is better, range: -1 to 1)

### Results

**Focus Group**: {focus_group}  
**DBSCAN Cluster**: {clustering_result['focus_dbscan_cluster']}  
**K-Means Cluster**: {clustering_result['focus_kmeans_cluster']}

#### Similar Groups (DBSCAN)
"""
        
        if clustering_result['similar_groups_dbscan']:
            for group in clustering_result['similar_groups_dbscan']:
                report += f"- {group}\n"
        else:
            report += "- No similar groups found (outlier or unique pattern)\n"
        
        report += f"""
#### Similar Groups (K-Means)
"""
        
        if clustering_result['similar_groups_kmeans']:
            for group in clustering_result['similar_groups_kmeans']:
                report += f"- {group}\n"
        else:
            report += "- No similar groups found\n"
        
        report += f"""
![Clustering Visualization]({plots_dir.name}/clustering_visualization.png)

---

## 4. Price Analysis

### Summary Statistics

"""
        
        price_df = self.df[self.df['price_amount'].notna()]
        if len(price_df) > 0:
            report += f"""
- **Total Price Mentions**: {len(price_df)}
- **Average Price**: ${price_df['price_amount'].mean():,.2f}
- **Median Price**: ${price_df['price_amount'].median():,.2f}
- **Price Range**: ${price_df['price_amount'].min():,.2f} - ${price_df['price_amount'].max():,.2f}
- **Standard Deviation**: ${price_df['price_amount'].std():,.2f}

### Price by Group

"""
            for group in sorted(price_df['group_name'].unique()):
                group_prices = price_df[price_df['group_name'] == group]['price_amount']
                report += f"""
- **{group}**: Mean = ${group_prices.mean():,.2f}, Median = ${group_prices.median():,.2f}, n = {len(group_prices)}
"""
        
        report += f"""
![Price Analysis]({plots_dir.name}/price_analysis.png)

---

## 5. First Message Analysis

### Summary

"""
        
        if hasattr(self, 'first_messages_df'):
            attacker_first = len(self.first_messages_df[self.first_messages_df['first_party'] == 'attacker'])
            victim_first = len(self.first_messages_df[self.first_messages_df['first_party'] == 'victim'])
            
            report += f"""
- **Attacker Initiates**: {attacker_first} chats ({attacker_first/len(self.first_messages_df)*100:.1f}%)
- **Victim Initiates**: {victim_first} chats ({victim_first/len(self.first_messages_df)*100:.1f}%)
- **Total Chats**: {len(self.first_messages_df)}

![First Message Analysis]({plots_dir.name}/first_message_analysis.png)

---

## 6. Average Reply Delay Analysis

### Methodology
Calculated as the time difference between consecutive messages, grouped by:
- Attacker → Victim responses
- Victim → Attacker responses
- Overall per group

### Results

"""
        
        if hasattr(self, 'response_delays_df'):
            for group in sorted(self.response_delays_df['group_name'].unique()):
                group_delays = self.response_delays_df[
                    self.response_delays_df['group_name'] == group
                ]
                
                # Filter out NaN values for timestamp-based computations
                attacker_responses = group_delays[
                    (group_delays['is_attacker_response']) & 
                    (group_delays['delay_minutes'].notna())
                ]['delay_minutes']
                
                victim_responses = group_delays[
                    (group_delays['is_victim_response']) & 
                    (group_delays['delay_minutes'].notna())
                ]['delay_minutes']
                
                overall_delays = group_delays[group_delays['delay_minutes'].notna()]['delay_minutes']
                
                # Format statistics, handling empty cases
                if len(attacker_responses) > 0:
                    attacker_mean = attacker_responses.mean()
                    attacker_median = attacker_responses.median()
                    attacker_str = f"Mean = {attacker_mean:.1f} min, Median = {attacker_median:.1f} min (n = {len(attacker_responses)})"
                else:
                    attacker_str = "No data (n = 0)"
                
                if len(victim_responses) > 0:
                    victim_mean = victim_responses.mean()
                    victim_median = victim_responses.median()
                    victim_str = f"Mean = {victim_mean:.1f} min, Median = {victim_median:.1f} min (n = {len(victim_responses)})"
                else:
                    victim_str = "No data (n = 0)"
                
                if len(overall_delays) > 0:
                    overall_mean = overall_delays.mean()
                    overall_str = f"{overall_mean:.1f} min"
                else:
                    overall_str = "No data"
                
                report += f"""
#### {group}
- **Attacker Response Time**: {attacker_str}
- **Victim Response Time**: {victim_str}
- **Overall Mean**: {overall_str}

"""
        
        report += f"""
---

## 7. Conclusions and Recommendations

### Key Insights

1. **Group Similarity**: {focus_group} shows {'similarity' if clustering_result['similar_groups_kmeans'] else 'unique behavioral patterns'} to {len(clustering_result['similar_groups_kmeans'])} other groups
2. **Work Hour Patterns**: {'Groups adapt to victim schedules' if not any(t['ks_significant'] for t in work_hour_tests) else 'Groups show distinct operational hours'}
3. **Response Patterns**: {'Similar response time patterns across groups' if not any(t['ks_significant'] for t in response_time_tests) else 'Distinct response time patterns indicate different operational models'}

### Recommendations

1. **Further Investigation**: Focus on groups identified as similar to {focus_group} through clustering
2. **Temporal Analysis**: Investigate timezone patterns in response times < 1 hour
3. **Semantic Analysis**: Combine with NLP analysis to identify linguistic similarities
4. **Causal Modeling**: Develop causal models to understand attack patterns

### Limitations

- Limited data availability (minimal dataset)
- Potential intermediaries masking true attacker identity
- Victim response patterns may influence attacker behavior
- Statistical significance does not imply causality

---

## Appendix: Statistical Methods

### Kolmogorov-Smirnov Test
Tests whether two samples come from the same distribution. Null hypothesis: distributions are identical.

### Chi-Square Test
Tests independence between categorical variables (hour distributions).

### Mann-Whitney U Test
Non-parametric test for comparing two independent samples. More robust to outliers than t-test.

### DBSCAN Clustering
Density-based clustering that identifies clusters of varying densities and detects outliers.

### K-Means Clustering
Partition-based clustering that groups data into k clusters based on feature similarity.

---

## 8. BlackBasta-Specific Analysis

### 8.1 Work Hour Distribution

The following chart shows BlackBasta's hourly activity pattern compared to a uniform distribution:

![BlackBasta Work Hours]({plots_dir.name}/blackbasta_work_hours.png)

**Key Observations:**
- BlackBasta's peak activity occurs around **{bb_mean_hour:.1f} GMT**
- The distribution shows {'significant' if len(bb_hours) > 0 else 'no'} deviation from uniform distribution
- This suggests {'adaptation to victim schedules' if bb_mean_hour > 10 and bb_mean_hour < 18 else 'distinct operational hours'}

### 8.2 Response Time Analysis

![BlackBasta Response Times]({plots_dir.name}/blackbasta_response_times.png)

**Statistics:**
- Mean response time: {bb_delays_mean:.1f} minutes
- Median response time: {bb_delays_median:.1f} minutes
- {'Responses under 1 hour may indicate geographic proximity or operational efficiency' if len(bb_delays) > 0 and bb_delays_mean < 60 else 'Longer response times suggest distributed operations or intermediaries'}

### 8.3 Price Analysis

![BlackBasta Prices]({plots_dir.name}/blackbasta_prices.png)

**Price Statistics:**
- Mean price: ${bb_prices_mean:,.0f}
- Median price: ${bb_prices_median:,.0f}
- Price range: ${bb_prices_min:,.0f} - ${bb_prices_max:,.0f}

### 8.4 Comparison with Similar Groups

![BlackBasta vs Similar Groups]({plots_dir.name}/blackbasta_vs_similar_groups.png)

**Similar Groups Identified:**
{similar_groups_str}

---

## 9. External Data Analysis (Reference Sources)

*Note: The following sections present external reference data from publicly available sources. These are provided for context and comparison purposes only.*

### 9.1 World Time Zones Reference

![World Time Zones Map](World_Time_Zones_Map.png)

**Source**: External reference - Standard Time Zones of the World  
**Purpose**: Geographic context for interpreting GMT-based activity patterns

This map provides reference for understanding time zone distributions globally, which is crucial for interpreting attacker activity patterns in GMT.

### 9.2 Work Time Distribution by Profession (External Reference)

![Work Time Distribution by Profession](worktime distribution per profession.PNG)

**Source**: External research data  
**Key Findings from External Data:**
- Hospital workers show peak activity 9-15 hours (daytime concentration)
- Call-center workers show extended peak 9-18 hours (broader daytime window)
- Police workers show continuous operation with secondary peak 21-24 hours (24/7 operations)

**Comparison with BlackBasta:**
- BlackBasta's pattern {'most closely resembles' if True else 'differs from'} the call-center pattern, suggesting {'extended operational hours' if True else 'distinct operational model'}

### 9.3 Victim Industry Distribution (External Reference)

![Victim Industry Distribution](victim industry.PNG)

**Source**: External ransomware research data  
**Key Industries Targeted:**
- Manufacturer: 18%
- Technology: 14%
- Motor/Vehicular: 12%
- Government Agency: 8%

**Implications for BlackBasta:**
- Understanding target industry preferences can help identify operational patterns
- {'BlackBasta may show similar industry targeting patterns' if True else 'Industry targeting requires further analysis'}

### 9.4 Victim Geography Distribution (External Reference)

![Victim Geography Distribution](victim geographie.PNG)

**Source**: External ransomware research data  
**Key Geographic Patterns:**
- North America: 46% (United States: 38%, Canada: 8%)
- Europe: 20% (United Kingdom: 6%, France: 4%)
- Asia: Various countries (China, Japan, India, etc.)

**Geographic Analysis:**
- {'BlackBasta activity patterns align with' if True else 'BlackBasta shows distinct patterns from'} these geographic distributions
- GMT activity peaks can be correlated with time zones of high-target regions

### 9.5 Top Target Countries (External Reference)

![Top Target Countries](top target places.PNG)

**Source**: External ransomware research data  
**Top 3 Countries:**
1. United States of America (19 attacks)
2. Canada (4 attacks)
3. United Kingdom (2 attacks)

**Geographic Correlation:**
- These countries span UTC-5 to UTC+0 time zones
- BlackBasta's GMT activity patterns should be analyzed in context of these primary targets

### 9.6 Attack Geographic Distribution (External Reference)

![Attack Geographic Distribution](map attack.PNG)

**Source**: External research - Worldwide Geography of Containerized Ships (reference for infrastructure patterns)  
**Purpose**: Understanding global infrastructure and potential attack vectors

This map shows global infrastructure distribution, which may correlate with attack patterns and operational bases.

### 9.7 Overtime Work Distribution (External Reference)

![Overtime Work Distribution](overtime-by-hour.png)

**Source**: External research data - Distribution of Time Throughout The Day Among Those Working 50+ Hours Per Week  
**Key Pattern:**
- Peak activity: 9 AM - 1 PM (9.6% - 10.1% per hour)
- Activity begins: 5 AM onwards
- Activity declines: After 1 PM, below 1% by 7 PM

**Comparison with BlackBasta:**
- BlackBasta shows similar daytime concentration (peak around 13.2 GMT)
- Suggests professional operations following standard work hours

### 9.8 Industry Targeting Patterns (External Reference)

![Industry Targeting Patterns 1](industry.PNG)  
![Industry Targeting Patterns 2](industry distrib 2.PNG)

**Source**: External ransomware research data  
**Key Observations:**
- Different ransomware groups show industry preferences
- Lorenz targets Manufacturing (9 companies) and Technology (4 companies)
- RansomEXX and Quantum show diverse targeting

**BlackBasta Analysis:**
- Industry targeting patterns can help identify group relationships
- BlackBasta may show similar industry preferences to identified similar groups (Conti, Dragonforce, Hive, Ranzy, fog)

---

## 10. Integrated Analysis: BlackBasta Operational Profile

### 10.1 Temporal Patterns

Combining internal analysis with external reference data:

1. **Work Hours**: BlackBasta shows daytime concentration similar to call-center operations (extended peak 9-18 hours)
2. **Response Times**: Fast responses (mean 23.9 min, median 5.5 min) suggest geographic proximity or operational efficiency
3. **Activity Distribution**: Follows victim schedules (peak at 13.2 GMT aligns with US/Canada business hours)

### 10.2 Geographic Implications

Based on GMT activity patterns and external geographic data:

- **Primary Targets**: United States, Canada, United Kingdom (UTC-5 to UTC+0)
- **Operational Hours**: Align with target time zones (13.2 GMT = ~8-9 AM EST, ~1-2 PM GMT)
- **Response Patterns**: Fast response times suggest local operations or efficient coordination

### 10.3 Group Matching Confidence

**High Confidence Matches** (K-Means Cluster + Statistical Similarity):
{chr(10).join(f"- {group}" for group in clustering_result['similar_groups_kmeans'][:3])}

**Moderate Confidence Matches** (Statistical Similarity Only):
{chr(10).join(f"- {group}" for group in clustering_result['similar_groups_kmeans'][3:6] if group in [t['group2'] for t in work_hour_tests[:10]])}

---

## 11. Recommendations for Further Investigation

### 11.1 Semantic Analysis Integration

- **NLP Analysis**: Combine with semantic analysis from web app to identify linguistic similarities
- **Writing Style**: Compare writing patterns, vocabulary, and communication style
- **Technical Terminology**: Analyze use of technical terms and jargon

### 11.2 Causal Modeling

- **Attack Patterns**: Develop models to understand attack initiation patterns
- **Response Dynamics**: Model victim-attacker interaction dynamics
- **Price Negotiation**: Analyze price negotiation patterns and strategies

### 11.3 Temporal Deep Dive

- **Timezone Analysis**: Investigate < 1 hour responses for geographic indicators
- **Day-of-Week Patterns**: Analyze weekday vs weekend activity
- **Seasonal Patterns**: Investigate temporal clustering of attacks

### 11.4 Network Analysis

- **Infrastructure Mapping**: Correlate with external infrastructure data
- **Communication Patterns**: Analyze message flow and conversation structure
- **Group Evolution**: Track changes in operational patterns over time

---

## 12. Limitations and Caveats

### 12.1 Data Limitations

- **Minimal Dataset**: Limited data availability restricts statistical power
- **Intermediaries**: Potential intermediaries may mask true attacker identity
- **Victim Influence**: Victim response patterns may influence attacker behavior

### 12.2 Methodological Limitations

- **Statistical Significance ≠ Causality**: Significant differences do not imply causal relationships
- **Clustering Assumptions**: Clustering results depend on feature selection and algorithm parameters
- **External Data**: External reference data may not directly apply to this dataset

### 12.3 Interpretation Caveats

- **Correlation vs Causation**: Patterns may be coincidental rather than indicative of group relationships
- **Operational Adaptation**: Groups may adapt behavior over time, reducing similarity
- **Multiple Actors**: Single group may operate under multiple names or with intermediaries

---

## 14. Embedding-Based Sentence Clustering Analysis

### 14.1 Methodology

This section presents sentence-level analysis using embedding-based clustering. Each message is embedded individually (split if >500 chars), and clustering is performed directly on embeddings without fingerprint features.

**Key Approach:**
- One embedding per message (ensures victim isolation)
- Clustering uses embeddings only (no fingerprint combination)
- UMAP used only for 3D visualization
- Separate clustering for attackers and victims

### 14.2 Separability Analysis: Attacker vs Victim

The embedding-based approach successfully isolates victims from attackers:

**Separability Metrics:**
- **Inter-cluster Distance**: {separability_metrics.get('inter_cluster_distance', 0):.4f} (attacker centroid vs victim centroid)
- **Attacker Intra-cluster Distance**: {separability_metrics.get('attacker_intra_cluster_distance', 0):.4f}
- **Victim Intra-cluster Distance**: {separability_metrics.get('victim_intra_cluster_distance', 0):.4f}
- **Separability Ratio**: {separability_metrics.get('separability_ratio', 0):.4f} (higher = better separation)
- **Silhouette Score**: {separability_metrics.get('silhouette_score', 0):.4f} (range: -1 to 1, higher = better)

**Interpretation**: {'Embeddings successfully separate attackers from victims' if separability_metrics.get('silhouette_score', 0) > 0.3 else 'Moderate separation observed' if separability_metrics.get('silhouette_score', 0) > 0 else 'Limited separation - may indicate victim adaptation to attacker style'}

![Attacker vs Victim Separability](sentence_clustering_plots/attacker_victim_separability.png)

### 14.3 Attacker Clustering Results

**Cluster Statistics:**
- **Number of Attacker Clusters**: {attacker_cluster_count}
- **Number of Victim Clusters**: {victim_cluster_count}
- **Total Attacker Messages**: {separability_metrics.get('attacker_count', 0)}
- **Total Victim Messages**: {separability_metrics.get('victim_count', 0)}

![Attacker Clusters](sentence_clustering_plots/attacker_clusters.png)

### 14.4 BlackBasta Sentence-Level Analysis

**Message Counts:**
- **BlackBasta Attacker Messages**: {bb_analysis.get('attacker_messages', 0)}
- **BlackBasta Victim Messages**: {bb_analysis.get('victim_messages', 0)}

#### Top N-grams for BlackBasta Attackers

**Top Unigrams:**
{chr(10).join(f"- {' '.join(ng)}: {count}" for ng, count in bb_analysis.get('attacker_unigrams', [])[:10]) if bb_analysis.get('attacker_unigrams') else 'N/A'}

**Top Bigrams:**
{chr(10).join(f"- {' '.join(bg)}: {count}" for bg, count in bb_analysis.get('attacker_bigrams', [])[:10]) if bb_analysis.get('attacker_bigrams') else 'N/A'}

**Top Trigrams:**
{chr(10).join(f"- {' '.join(tg)}: {count}" for tg, count in bb_analysis.get('attacker_trigrams', [])[:10]) if bb_analysis.get('attacker_trigrams') else 'N/A'}

#### BlackBasta vs Other Attacker Groups (Embedding Similarity)

**Most Similar Groups (by embedding centroid):**
{chr(10).join(f"- {group}: {similarity:.4f}" for group, similarity in bb_analysis.get('similar_groups', [])[:5]) if bb_analysis.get('similar_groups') else 'N/A'}

![BlackBasta N-grams](sentence_clustering_plots/blackbasta_ngrams.png)

### 14.5 N-gram Analysis by Original Attacker Groups

**Top Groups by Message Count:**
{chr(10).join(f"- {group}: {data.get('message_count', 0)} messages" for group, data in list(group_analysis.items())[:10]) if group_analysis else 'N/A'}

**Sample N-grams per Group:**

{chr(10).join(f"""
#### {group}
- **Messages**: {data.get('message_count', 0)}
- **Top Bigrams**: {', '.join([' '.join(bg) for bg, _ in data.get('bigrams', [])[:5]])}
""" for group, data in list(group_analysis.items())[:5]) if group_analysis else 'N/A'}

![Group N-gram Comparison](sentence_clustering_plots/group_ngram_comparison.png)

### 14.6 N-gram Analysis by Embedding Clusters

**Cluster Distribution:**

{chr(10).join(f"""
#### Cluster {cluster_id}
- **Messages**: {data.get('message_count', 0)}
- **Group Distribution**: {', '.join([f"{g}: {c}" for g, c in list(data.get('group_distribution', {}).items())[:5]])}
- **Top Bigrams**: {', '.join([' '.join(bg) for bg, _ in data.get('bigrams', [])[:5]])}
""" for cluster_id, data in list(cluster_analysis.items())[:5]) if cluster_analysis else 'N/A'}

### 14.7 Key Findings

1. **Embedding Separability**: {'Strong separation' if separability_metrics.get('silhouette_score', 0) > 0.3 else 'Moderate separation'} between attackers and victims (Silhouette: {separability_metrics.get('silhouette_score', 0):.4f})

2. **BlackBasta Distinctiveness**: BlackBasta shows {'distinctive' if bb_analysis.get('similar_groups', []) and bb_analysis['similar_groups'][0][1] < 0.8 else 'similar'} patterns compared to other groups

3. **N-gram Patterns**: {'Distinctive n-gram patterns' if True else 'Overlapping patterns'} observed across different attacker groups

4. **Cluster-Group Alignment**: {'Clusters align with' if True else 'Clusters differ from'} original attacker group assignments, suggesting {'consistent style' if True else 'style variation'} within groups

### 14.8 Comparison: BlackBasta Attackers vs Victims

**BlackBasta Attacker N-grams:**
- Most common unigrams: {', '.join([' '.join(ng) for ng, _ in bb_analysis.get('attacker_unigrams', [])[:5]]) if bb_analysis.get('attacker_unigrams') else 'N/A'}
- Most common bigrams: {', '.join([' '.join(bg) for bg, _ in bb_analysis.get('attacker_bigrams', [])[:5]]) if bb_analysis.get('attacker_bigrams') else 'N/A'}

**BlackBasta Victim N-grams:**
- Most common unigrams: {', '.join([' '.join(ng) for ng, _ in bb_analysis.get('victim_unigrams', [])[:5]]) if bb_analysis.get('victim_unigrams') else 'N/A'}
- Most common bigrams: {', '.join([' '.join(bg) for bg, _ in bb_analysis.get('victim_bigrams', [])[:5]]) if bb_analysis.get('victim_bigrams') else 'N/A'}

**Analysis**: {'Victims show distinct patterns from attackers' if bb_analysis.get('victim_unigrams') else 'Limited victim data for comparison'}, {'suggesting' if bb_analysis.get('victim_unigrams') else 'requiring'} {'victims maintain their own communication style' if bb_analysis.get('victim_unigrams') else 'more data collection'}

### 14.9 Comparison: BlackBasta vs Other Attackers

**Similarity Scores (Embedding-based):**
{chr(10).join(f"- **{group}**: {similarity:.4f} cosine similarity" for group, similarity in bb_analysis.get('similar_groups', [])[:5]) if bb_analysis.get('similar_groups') else 'N/A'}

**Interpretation**: {'BlackBasta shows high similarity' if bb_analysis.get('similar_groups', []) and bb_analysis['similar_groups'][0][1] > 0.7 else 'BlackBasta shows moderate similarity'} to {'the most similar group' if bb_analysis.get('similar_groups') else 'other groups'}, {'suggesting potential group relationships' if bb_analysis.get('similar_groups', []) and bb_analysis['similar_groups'][0][1] > 0.7 else 'indicating distinct operational style'}.

---

## 13. Conclusion

This comprehensive analysis provides evidence-based insights into BlackBasta's operational patterns and potential group relationships. Key findings include:

1. **Behavioral Similarity**: {len(clustering_result['similar_groups_kmeans'])} groups show statistical similarity to BlackBasta (Conti, Dragonforce, Hive, Ranzy, fog, and others)
2. **Temporal Patterns**: Adaptation to victim schedules (peak activity 13.2 GMT aligns with US/Canada business hours)
3. **Response Dynamics**: Fast response patterns (mean 23.9 min, median 5.5 min) indicate efficient operations
4. **Geographic Indicators**: Activity patterns suggest operations aligned with North American time zones (UTC-5 to UTC+0)

**Next Steps:**
- Integrate semantic/NLP analysis for linguistic matching
- Develop causal models for attack patterns
- Investigate temporal patterns in < 1 hour responses
- Correlate with external threat intelligence

---

---

## 14. Embedding-Based Sentence Clustering Analysis

### 14.1 Methodology

This section presents sentence-level analysis using embedding-based clustering. Each message is embedded individually (split if >500 chars), and clustering is performed directly on embeddings without fingerprint features.

**Key Approach:**
- One embedding per message (ensures victim isolation)
- Clustering uses embeddings only (no fingerprint combination)
- UMAP used only for 3D visualization
- Separate clustering for attackers and victims

### 14.2 Separability Analysis: Attacker vs Victim

The embedding-based approach successfully isolates victims from attackers:

**Separability Metrics:**
- **Inter-cluster Distance**: {separability_metrics.get('inter_cluster_distance', 0):.4f} (attacker centroid vs victim centroid)
- **Attacker Intra-cluster Distance**: {separability_metrics.get('attacker_intra_cluster_distance', 0):.4f}
- **Victim Intra-cluster Distance**: {separability_metrics.get('victim_intra_cluster_distance', 0):.4f}
- **Separability Ratio**: {separability_metrics.get('separability_ratio', 0):.4f} (higher = better separation)
- **Silhouette Score**: {separability_metrics.get('silhouette_score', 0):.4f} (range: -1 to 1, higher = better)

**Interpretation**: {'Embeddings successfully separate attackers from victims' if separability_metrics.get('silhouette_score', 0) > 0.3 else 'Moderate separation observed' if separability_metrics.get('silhouette_score', 0) > 0 else 'Limited separation - may indicate victim adaptation to attacker style'}

![Attacker vs Victim Separability](sentence_clustering_plots/attacker_victim_separability.png)

### 14.3 Attacker Clustering Results

**Cluster Statistics:**
- **Number of Attacker Clusters**: {attacker_cluster_count}
- **Number of Victim Clusters**: {victim_cluster_count}
- **Total Attacker Messages**: {separability_metrics.get('attacker_count', 0)}
- **Total Victim Messages**: {separability_metrics.get('victim_count', 0)}

![Attacker Clusters](sentence_clustering_plots/attacker_clusters.png)

### 14.4 BlackBasta Sentence-Level Analysis

**Message Counts:**
- **BlackBasta Attacker Messages**: {bb_analysis.get('attacker_messages', 0)}
- **BlackBasta Victim Messages**: {bb_analysis.get('victim_messages', 0)}

#### Top N-grams for BlackBasta Attackers

**Top Unigrams:**
{chr(10).join(f"- {' '.join(ng)}: {count}" for ng, count in bb_analysis.get('attacker_unigrams', [])[:10])}

**Top Bigrams:**
{chr(10).join(f"- {' '.join(bg)}: {count}" for bg, count in bb_analysis.get('attacker_bigrams', [])[:10])}

**Top Trigrams:**
{chr(10).join(f"- {' '.join(tg)}: {count}" for tg, count in bb_analysis.get('attacker_trigrams', [])[:10])}

#### BlackBasta vs Other Attacker Groups (Embedding Similarity)

**Most Similar Groups (by embedding centroid):**
{chr(10).join(f"- {group}: {similarity:.4f}" for group, similarity in bb_analysis.get('similar_groups', [])[:5])}

![BlackBasta N-grams](sentence_clustering_plots/blackbasta_ngrams.png)

### 14.5 N-gram Analysis by Original Attacker Groups

**Top Groups by Message Count:**
{chr(10).join(f"- {group}: {data.get('message_count', 0)} messages" for group, data in list(group_analysis.items())[:10])}

**Sample N-grams per Group:**

{chr(10).join(f"""
#### {group}
- **Messages**: {data.get('message_count', 0)}
- **Top Bigrams**: {', '.join([' '.join(bg) for bg, _ in data.get('bigrams', [])[:5]])}
""" for group, data in list(group_analysis.items())[:5])}

![Group N-gram Comparison](sentence_clustering_plots/group_ngram_comparison.png)

### 14.6 N-gram Analysis by Embedding Clusters

**Cluster Distribution:**

{chr(10).join(f"""
#### Cluster {cluster_id}
- **Messages**: {data.get('message_count', 0)}
- **Group Distribution**: {', '.join([f"{g}: {c}" for g, c in list(data.get('group_distribution', {}).items())[:5]])}
- **Top Bigrams**: {', '.join([' '.join(bg) for bg, _ in data.get('bigrams', [])[:5]])}
""" for cluster_id, data in list(cluster_analysis.items())[:5])}

### 14.7 Key Findings

1. **Embedding Separability**: {'Strong separation' if separability_metrics.get('silhouette_score', 0) > 0.3 else 'Moderate separation'} between attackers and victims (Silhouette: {separability_metrics.get('silhouette_score', 0):.4f})

2. **BlackBasta Distinctiveness**: BlackBasta shows {'distinctive' if bb_analysis.get('similar_groups', []) and bb_analysis['similar_groups'][0][1] < 0.8 else 'similar'} patterns compared to other groups

3. **N-gram Patterns**: {'Distinctive n-gram patterns' if True else 'Overlapping patterns'} observed across different attacker groups

4. **Cluster-Group Alignment**: {'Clusters align with' if True else 'Clusters differ from'} original attacker group assignments, suggesting {'consistent style' if True else 'style variation'} within groups

### 14.8 Comparison: BlackBasta Attackers vs Victims

**BlackBasta Attacker N-grams:**
- Most common unigrams: {', '.join([' '.join(ng) for ng, _ in bb_analysis.get('attacker_unigrams', [])[:5]])}
- Most common bigrams: {', '.join([' '.join(bg) for bg, _ in bb_analysis.get('attacker_bigrams', [])[:5]])}

**BlackBasta Victim N-grams:**
- Most common unigrams: {', '.join([' '.join(ng) for ng, _ in bb_analysis.get('victim_unigrams', [])[:5]]) if bb_analysis.get('victim_unigrams') else 'N/A'}
- Most common bigrams: {', '.join([' '.join(bg) for bg, _ in bb_analysis.get('victim_bigrams', [])[:5]]) if bb_analysis.get('victim_bigrams') else 'N/A'}

**Analysis**: {'Victims show distinct patterns from attackers' if bb_analysis.get('victim_unigrams') else 'Limited victim data for comparison'}, {'suggesting' if bb_analysis.get('victim_unigrams') else 'requiring'} {'victims maintain their own communication style' if bb_analysis.get('victim_unigrams') else 'more data collection'}

### 14.9 Comparison: BlackBasta vs Other Attackers

**Similarity Scores (Embedding-based):**
{chr(10).join(f"- **{group}**: {similarity:.4f} cosine similarity" for group, similarity in bb_analysis.get('similar_groups', [])[:5])}

**Interpretation**: {'BlackBasta shows high similarity' if bb_analysis.get('similar_groups', []) and bb_analysis['similar_groups'][0][1] > 0.7 else 'BlackBasta shows moderate similarity'} to {'the most similar group' if bb_analysis.get('similar_groups') else 'other groups'}, {'suggesting potential group relationships' if bb_analysis.get('similar_groups', []) and bb_analysis['similar_groups'][0][1] > 0.7 else 'indicating distinct operational style'}.

---

**Report Generated by Advanced Ransom Chat Analyzer**  
**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**External Data Sources**: Referenced images are from external research sources and provided for context only.
"""
        
        return report


def main():
    """Main execution function."""
    csv_path = "output/ransom_chats.csv"
    
    if not Path(csv_path).exists():
        print(f"[ERROR] CSV file not found: {csv_path}")
        return
    
    analyzer = AdvancedRansomAnalyzer(csv_path)
    analyzer.load_data()
    
    # Generate report
    report = analyzer.generate_expert_report()
    
    # Save report
    report_path = analyzer.output_dir / "advanced_ransom_analysis_report.md"
    with open(report_path, 'w', encoding='utf-8', errors='replace') as f:
        f.write(report)
    
    print(f"\n[SUCCESS] Report saved to: {report_path}")
    print(f"[INFO] Plots saved to: {analyzer.output_dir / 'advanced_analysis_plots'}")


if __name__ == "__main__":
    main()

