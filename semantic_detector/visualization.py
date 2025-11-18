"""
3D visualization using UMAP and Plotly.
Creates interactive 3D projections of embeddings.
"""

import numpy as np
import umap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Optional, Dict, Tuple
import pandas as pd
from collections import Counter, defaultdict
from semantic_detector.language_fingerprint import LanguageFingerprint


class EmbeddingVisualizer:
    """Create 3D visualizations of embeddings."""
    
    def __init__(self, n_components: int = 3, n_neighbors: int = 15, min_dist: float = 0.1):
        """
        Initialize visualizer.
        
        Args:
            n_components: Number of dimensions for UMAP (3 for 3D)
            n_neighbors: UMAP parameter
            min_dist: UMAP parameter
        """
        self.n_components = n_components
        self.umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit UMAP and transform embeddings to 3D.
        
        Args:
            embeddings: High-dimensional embeddings
            
        Returns:
            3D coordinates
        """
        return self.umap_model.fit_transform(embeddings)
    
    def create_3d_plot(
        self,
        coords_3d: np.ndarray,
        labels: Optional[np.ndarray] = None,
        texts: Optional[List[str]] = None,
        colors: Optional[np.ndarray] = None,
        title: str = "3D UMAP Projection"
    ) -> go.Figure:
        """
        Create interactive 3D plot.
        
        Args:
            coords_3d: 3D coordinates (n_samples, 3)
            labels: Cluster labels
            texts: Text snippets for hover
            colors: Color values for each point
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Prepare data
        df = pd.DataFrame({
            'x': coords_3d[:, 0],
            'y': coords_3d[:, 1],
            'z': coords_3d[:, 2],
        })
        
        if labels is not None:
            df['cluster'] = labels
            df['cluster'] = df['cluster'].astype(str)
        
        if texts is not None:
            # Truncate long texts for hover
            df['text'] = [t[:100] + '...' if len(t) > 100 else t for t in texts]
        else:
            df['text'] = [f'Sentence {i}' for i in range(len(coords_3d))]
        
        if colors is not None:
            df['color'] = colors
        
        # Create figure
        fig = go.Figure()
        
        if labels is not None:
            # Plot by cluster
            unique_labels = sorted(df['cluster'].unique())
            colors_palette = px.colors.qualitative.Set3
            
            for i, label in enumerate(unique_labels):
                cluster_data = df[df['cluster'] == label]
                color = colors_palette[i % len(colors_palette)]
                
                fig.add_trace(go.Scatter3d(
                    x=cluster_data['x'],
                    y=cluster_data['y'],
                    z=cluster_data['z'],
                    mode='markers',
                    name=f'Cluster {label}',
                    marker=dict(
                        size=5,
                        color=color,
                        opacity=0.7,
                        line=dict(width=0.5, color='black')
                    ),
                    text=cluster_data['text'],
                    hovertemplate='<b>%{text}</b><br>' +
                                  'X: %{x:.2f}<br>' +
                                  'Y: %{y:.2f}<br>' +
                                  'Z: %{z:.2f}<extra></extra>',
                ))
        else:
            # Single color plot
            color_scale = colors if colors is not None else 'Viridis'
            fig.add_trace(go.Scatter3d(
                x=df['x'],
                y=df['y'],
                z=df['z'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=df['color'] if colors is not None else None,
                    colorscale=color_scale if isinstance(color_scale, str) else None,
                    opacity=0.7,
                    line=dict(width=0.5, color='black')
                ),
                text=df['text'],
                hovertemplate='<b>%{text}</b><br>' +
                              'X: %{x:.2f}<br>' +
                              'Y: %{y:.2f}<br>' +
                              'Z: %{z:.2f}<extra></extra>',
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                zaxis_title='UMAP 3',
                bgcolor='white',
            ),
            width=1200,
            height=800,
            hovermode='closest',
        )
        
        return fig
    
    def create_ngram_view(
        self,
        texts: List[str],
        labels: Optional[np.ndarray] = None,
        n: int = 2,
        top_n: int = 20
    ) -> go.Figure:
        """
        Create n-gram frequency visualization.
        
        Args:
            texts: List of texts
            labels: Cluster labels (optional)
            n: N-gram size (2 for bigrams, 3 for trigrams)
            top_n: Number of top n-grams to show
            
        Returns:
            Plotly figure
        """
        fingerprint_extractor = LanguageFingerprint()
        
        # Collect all n-grams
        all_ngrams = Counter()
        cluster_ngrams = defaultdict(Counter)
        
        for idx, text in enumerate(texts):
            ngram_dist = fingerprint_extractor.extract_ngram_distribution(text, n=n, top_n=100)
            all_ngrams.update(ngram_dist)
            
            if labels is not None:
                cluster_id = labels[idx]
                cluster_ngrams[cluster_id].update(ngram_dist)
        
        # Get top n-grams overall
        top_ngrams = all_ngrams.most_common(top_n)
        ngram_names = [' '.join(ngram) if isinstance(ngram, tuple) else str(ngram) 
                      for ngram, _ in top_ngrams]
        
        # Create figure
        fig = go.Figure()
        
        if labels is not None and cluster_ngrams:
            # Show by cluster
            unique_labels = sorted(set(labels))
            colors_palette = px.colors.qualitative.Set3
            
            for i, cluster_id in enumerate(unique_labels):
                cluster_freqs = []
                for ngram, _ in top_ngrams:
                    freq = cluster_ngrams[cluster_id].get(ngram, 0)
                    cluster_freqs.append(freq)
                
                fig.add_trace(go.Bar(
                    x=ngram_names,
                    y=cluster_freqs,
                    name=f'Cluster {cluster_id}',
                    marker_color=colors_palette[i % len(colors_palette)],
                    opacity=0.7
                ))
        else:
            # Show overall frequencies
            freqs = [count for _, count in top_ngrams]
            fig.add_trace(go.Bar(
                x=ngram_names,
                y=freqs,
                marker_color='steelblue',
                opacity=0.7
            ))
        
        fig.update_layout(
            title=f'Top {top_n} {n}-gram Frequencies',
            xaxis_title=f'{n}-grams',
            yaxis_title='Frequency',
            barmode='group' if labels is not None else 'stack',
            height=600,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_sentence_analytics_view(
        self,
        texts: List[str],
        labels: Optional[np.ndarray] = None
    ) -> go.Figure:
        """
        Create sentence analytics visualization.
        
        Args:
            texts: List of texts
            labels: Cluster labels (optional)
            
        Returns:
            Plotly figure with subplots
        """
        fingerprint_extractor = LanguageFingerprint()
        
        # Extract sentence-level features
        sentence_lengths = []
        word_counts = []
        vocab_richness = []
        cluster_data = defaultdict(lambda: {'lengths': [], 'words': [], 'richness': []})
        
        for idx, text in enumerate(texts):
            features = fingerprint_extractor.extract_features(text)
            length = features.get('char_count', 0)
            words = features.get('word_count', 0)
            richness = features.get('type_token_ratio', 0)
            
            sentence_lengths.append(length)
            word_counts.append(words)
            vocab_richness.append(richness)
            
            if labels is not None:
                cluster_id = labels[idx]
                cluster_data[cluster_id]['lengths'].append(length)
                cluster_data[cluster_id]['words'].append(words)
                cluster_data[cluster_id]['richness'].append(richness)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentence Length Distribution', 'Word Count Distribution',
                           'Vocabulary Richness', 'Length vs Word Count'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        if labels is not None and cluster_data:
            # Plot by cluster
            unique_labels = sorted(set(labels))
            colors_palette = px.colors.qualitative.Set3
            
            for i, cluster_id in enumerate(unique_labels):
                color = colors_palette[i % len(colors_palette)]
                data = cluster_data[cluster_id]
                
                # Length distribution
                fig.add_trace(go.Histogram(
                    x=data['lengths'],
                    name=f'Cluster {cluster_id}',
                    marker_color=color,
                    opacity=0.7,
                    legendgroup=f'cluster_{cluster_id}',
                    showlegend=True
                ), row=1, col=1)
                
                # Word count distribution
                fig.add_trace(go.Histogram(
                    x=data['words'],
                    name=f'Cluster {cluster_id}',
                    marker_color=color,
                    opacity=0.7,
                    legendgroup=f'cluster_{cluster_id}',
                    showlegend=False
                ), row=1, col=2)
                
                # Vocabulary richness
                fig.add_trace(go.Histogram(
                    x=data['richness'],
                    name=f'Cluster {cluster_id}',
                    marker_color=color,
                    opacity=0.7,
                    legendgroup=f'cluster_{cluster_id}',
                    showlegend=False
                ), row=2, col=1)
                
                # Scatter: Length vs Word Count
                fig.add_trace(go.Scatter(
                    x=data['lengths'],
                    y=data['words'],
                    mode='markers',
                    name=f'Cluster {cluster_id}',
                    marker_color=color,
                    opacity=0.6,
                    legendgroup=f'cluster_{cluster_id}',
                    showlegend=False
                ), row=2, col=2)
        else:
            # Plot overall
            fig.add_trace(go.Histogram(x=sentence_lengths, name='All', marker_color='steelblue'), row=1, col=1)
            fig.add_trace(go.Histogram(x=word_counts, name='All', marker_color='steelblue'), row=1, col=2)
            fig.add_trace(go.Histogram(x=vocab_richness, name='All', marker_color='steelblue'), row=2, col=1)
            fig.add_trace(go.Scatter(x=sentence_lengths, y=word_counts, mode='markers',
                                   marker_color='steelblue', opacity=0.6), row=2, col=2)
        
        fig.update_xaxes(title_text="Characters", row=1, col=1)
        fig.update_xaxes(title_text="Words", row=1, col=2)
        fig.update_xaxes(title_text="Type-Token Ratio", row=2, col=1)
        fig.update_xaxes(title_text="Characters", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Words", row=2, col=2)
        
        fig.update_layout(
            title_text="Sentence Analytics",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_combined_plot(
        self,
        coords_3d: np.ndarray,
        labels: np.ndarray,
        texts: List[str],
        fingerprint_features: Optional[np.ndarray] = None,
        include_ngram_view: bool = True,
        include_sentence_analytics: bool = True
    ) -> Dict[str, go.Figure]:
        """
        Create combined visualization with multiple views.
        
        Args:
            coords_3d: 3D coordinates
            labels: Cluster labels
            texts: Text snippets
            fingerprint_features: Optional fingerprint features for additional views
            include_ngram_view: Whether to include n-gram view
            include_sentence_analytics: Whether to include sentence analytics
            
        Returns:
            Dictionary of Plotly figures
        """
        figures = {
            '3d_projection': self.create_3d_plot(coords_3d, labels, texts, title="3D UMAP Projection")
        }
        
        if include_ngram_view:
            figures['bigrams'] = self.create_ngram_view(texts, labels, n=2)
            figures['trigrams'] = self.create_ngram_view(texts, labels, n=3)
        
        if include_sentence_analytics:
            figures['sentence_analytics'] = self.create_sentence_analytics_view(texts, labels)
        
        return figures

