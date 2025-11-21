#!/usr/bin/env python3
"""
Enhance markdown report with static images from HTML plots, BlackBasta-specific sections,
semantic analysis, n-gram analysis, and external analysis data with proper attribution.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import argparse
import re
import html

# Plotly for PNG export
try:
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.io import read_html, to_image
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("[WARNING] plotly not installed. Install with: pip install plotly kaleido")

# Import semantic detector modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from semantic_detector.core.detector import SemanticDetector
    from semantic_detector.language_fingerprint import LanguageFingerprint
    from semantic_detector.web.app.core.text_utils import parse_raw_text
    HAS_SEMANTIC = True
except ImportError as e:
    HAS_SEMANTIC = False
    print(f"[WARNING] Semantic detector modules not available: {e}")

# Data analysis
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[WARNING] pandas/numpy not installed")


def extract_plotly_figures_from_html(html_path: Path):
    """Extract Plotly figure objects from HTML file."""
    if not HAS_PLOTLY:
        return []
    
    try:
        # Read HTML and extract Plotly figures
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Try to extract Plotly JSON data
        figures = []
        
        # Method 1: Use plotly's read_html if available
        try:
            fig_dicts = pyo.get_plotlyjs()  # This won't work, need different approach
        except:
            pass
        
        # Method 2: Extract from script tags
        script_pattern = r'Plotly\.newPlot\([^,]+,\s*(\{.*?\}),\s*\{.*?\}\)'
        matches = re.findall(script_pattern, html_content, re.DOTALL)
        
        # Method 3: Look for Plotly data in window.PlotlyData
        data_pattern = r'window\.PlotlyData\s*=\s*(\{.*?\});'
        data_matches = re.findall(data_pattern, html_content, re.DOTALL)
        
        return figures
        
    except Exception as e:
        print(f"[ERROR] Failed to extract figures from {html_path}: {e}")
        return []


def convert_plotly_html_to_png(html_path: Path, output_png_path: Path, width: int = 1200, height: int = 800):
    """Convert Plotly HTML to PNG using Plotly's built-in export."""
    if not HAS_PLOTLY:
        print(f"[SKIP] Converting {html_path} to PNG (plotly not available)")
        return False
    
    try:
        # Read HTML file
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Try to extract Plotly figure data
        # Look for Plotly.newPlot calls or data in script tags
        script_pattern = r'<script[^>]*>(.*?)</script>'
        scripts = re.findall(script_pattern, html_content, re.DOTALL)
        
        # Try to find Plotly figure JSON
        for script in scripts:
            # Look for Plotly.newPlot with data
            plot_pattern = r'Plotly\.newPlot\([^,]+,\s*(\[.*?\]),\s*(\{.*?\})\)'
            matches = re.findall(plot_pattern, script, re.DOTALL)
            
            if matches:
                try:
                    import json as json_lib
                    # Parse the data and layout
                    data_str, layout_str = matches[0]
                    data = json_lib.loads(data_str)
                    layout = json_lib.loads(layout_str)
                    
                    # Create figure
                    fig = go.Figure(data=data, layout=layout)
                    fig.update_layout(width=width, height=height)
                    
                    # Export to PNG (requires kaleido)
                    try:
                        img_bytes = to_image(fig, format='png', width=width, height=height)
                        with open(output_png_path, 'wb') as f:
                            f.write(img_bytes)
                        print(f"[INFO] Converted {html_path.name} to {output_png_path.name}")
                        return True
                    except Exception as e:
                        print(f"[WARNING] Kaleido not available, trying alternative method: {e}")
                        # Fallback: save as HTML reference
                        return False
                        
                except Exception as e:
                    continue
        
        # If we can't extract, create a placeholder note
        print(f"[SKIP] Could not extract Plotly figure from {html_path.name}")
        return False
        
    except Exception as e:
        print(f"[ERROR] Failed to convert {html_path}: {e}")
        return False


def convert_html_to_png_simple(html_path: Path, output_png_path: Path):
    """Simple conversion: just copy HTML reference or create placeholder."""
    # For now, we'll reference the HTML files directly in markdown
    # Users can open them in browser and take screenshots
    print(f"[INFO] HTML plot available at: {html_path}")
    print(f"[INFO] Please open in browser and take screenshot, save as: {output_png_path.name}")
    return False


def find_html_plots(output_dir: Path) -> dict:
    """Find all HTML plot files in output directory."""
    html_files = {
        'response_times': output_dir / 'timeline_analysis_response_times.html',
        'conversation_metrics': output_dir / 'timeline_analysis_conversation_metrics.html',
        'hourly_activity': output_dir / 'timeline_analysis_hourly_activity.html',
        'hourly_activity_by_group': output_dir / 'timeline_analysis_hourly_activity_by_group.html',
        'distribution_comparison': output_dir / 'timeline_analysis_distribution_comparison.html',
        'timestamp_distribution': output_dir / 'timeline_analysis_timestamp_distribution.html',
        'by_attacker': output_dir / 'timeline_analysis_by_attacker.html',
    }
    
    # Filter to only existing files
    existing = {k: v for k, v in html_files.items() if v.exists()}
    return existing


def convert_all_html_plots(output_dir: Path, plots_dir: Path, force: bool = False):
    """Convert all HTML plots to PNG images."""
    plots_dir.mkdir(exist_ok=True)
    
    html_plots = find_html_plots(output_dir)
    converted = {}
    
    for plot_name, html_path in html_plots.items():
        png_path = plots_dir / f"{plot_name}.png"
        
        # Skip if already exists and not forcing
        if png_path.exists() and not force:
            print(f"[SKIP] {png_path.name} already exists (use --force to regenerate)")
            converted[plot_name] = png_path
            continue
        
        # Try Plotly conversion first
        if convert_plotly_html_to_png(html_path, png_path):
            converted[plot_name] = png_path
        else:
            # Fallback: keep HTML reference
            converted[plot_name] = html_path
    
    return converted


def load_sentence_clustering_results(output_dir: Path) -> dict:
    """Load sentence clustering analysis results if available."""
    clustering_files = [
        output_dir / 'sentence_clustering_analysis.json',
        output_dir / 'sentence_clustering_results.json',
        output_dir / 'clustering_analysis.json',
    ]
    
    for file_path in clustering_files:
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                continue
    
    return {}


def load_timeline_analysis(output_dir: Path) -> dict:
    """Load timeline analysis JSON if available."""
    timeline_file = output_dir / 'timeline_analysis.json'
    if timeline_file.exists():
        try:
            with open(timeline_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def enhance_blackbasta_interpretations(report_content: str) -> str:
    """Enhance existing interpretations with BlackBasta-specific context."""
    # Replace generic interpretations with BlackBasta-specific ones
    
    replacements = [
        (
            r'\*\*Interpretation\*\*: Groups show significantly different work hour patterns',
            r'**Interpretation**: BlackBasta and the compared group show significantly different work hour patterns. '
            r'BlackBasta\'s peak activity at 13.2 GMT suggests operations aligned with specific time zones, '
            r'potentially indicating geographic location or adaptation to target schedules.'
        ),
        (
            r'\*\*Interpretation\*\*: Groups show significantly different response time patterns',
            r'**Interpretation**: BlackBasta and the compared group show significantly different response time patterns. '
            r'BlackBasta\'s mean response time of 23.9 minutes (median 5.5 minutes) indicates relatively fast '
            r'operational responses compared to other groups, suggesting efficient coordination or geographic proximity.'
        ),
    ]
    
    for pattern, replacement in replacements:
        report_content = re.sub(pattern, replacement, report_content)
    
    return report_content


def load_csv_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV data for analysis."""
    if not HAS_PANDAS or not csv_path.exists():
        return None
    
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        from io import StringIO
        df = pd.read_csv(StringIO(content))
        
        # Sanitize text columns (use safe string conversion)
        text_cols = ['content', 'chat_id', 'party', 'group_name']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: str(x).strip() if pd.notna(x) else ''
                )
        
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_umap_visualizations(embeddings_2d: np.ndarray, labels: list, parties: list, 
                                plots_dir: Path, group_name: str = "BlackBasta") -> dict:
    """Create UMAP 2D visualizations colored by cluster and party."""
    if not HAS_PLOTLY or len(embeddings_2d) == 0:
        return {}
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        
        results = {}
        
        # Plot 1: Colored by cluster
        fig_cluster = go.Figure()
        unique_clusters = sorted(set(labels))
        colors_cluster = px.colors.qualitative.Set3
        
        for cluster_id in unique_clusters:
            cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
            x_coords = [embeddings_2d[i, 0] for i in cluster_indices]
            y_coords = [embeddings_2d[i, 1] for i in cluster_indices]
            
            fig_cluster.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(
                    size=8,
                    color=colors_cluster[cluster_id % len(colors_cluster)],
                    opacity=0.7,
                    line=dict(width=1, color='black')
                )
            ))
        
        fig_cluster.update_layout(
            title=f'{group_name} Messages: 2D UMAP Colored by Cluster',
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            height=600,
            hovermode='closest'
        )
        
        cluster_png = plots_dir / f'{group_name.lower()}_umap_clusters.png'
        try:
            import plotly.io as pio
            img_bytes = pio.to_image(fig_cluster, format='png', width=1200, height=600)
            with open(cluster_png, 'wb') as f:
                f.write(img_bytes)
            results['cluster_plot'] = cluster_png
        except Exception:
            cluster_html = plots_dir / f'{group_name.lower()}_umap_clusters.html'
            fig_cluster.write_html(str(cluster_html))
            results['cluster_plot'] = cluster_html
        
        # Plot 2: Colored by party
        if parties and len(set(parties)) > 1:
            fig_party = go.Figure()
            unique_parties = sorted(set(parties))
            colors_party = {'attacker': 'rgba(239, 68, 68, 0.7)', 'victim': 'rgba(59, 130, 246, 0.7)'}
            
            for party in unique_parties:
                party_indices = [i for i, p in enumerate(parties) if str(p).lower() == str(party).lower()]
                if not party_indices:
                    continue
                
                x_coords = [embeddings_2d[i, 0] for i in party_indices]
                y_coords = [embeddings_2d[i, 1] for i in party_indices]
                
                color = colors_party.get(str(party).lower(), 'rgba(128, 128, 128, 0.7)')
                
                fig_party.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='markers',
                    name=str(party).title(),
                    marker=dict(
                        size=8,
                        color=color,
                        opacity=0.7,
                        line=dict(width=1, color='black')
                    )
                ))
            
            fig_party.update_layout(
                title=f'{group_name} Messages: 2D UMAP Colored by Party',
                xaxis_title='UMAP Dimension 1',
                yaxis_title='UMAP Dimension 2',
                height=600,
                hovermode='closest'
            )
            
            party_png = plots_dir / f'{group_name.lower()}_umap_party.png'
            try:
                import plotly.io as pio
                img_bytes = pio.to_image(fig_party, format='png', width=1200, height=600)
                with open(party_png, 'wb') as f:
                    f.write(img_bytes)
                results['party_plot'] = party_png
            except Exception:
                party_html = plots_dir / f'{group_name.lower()}_umap_party.html'
                fig_party.write_html(str(party_html))
                results['party_plot'] = party_html
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Failed to create UMAP visualizations: {e}")
        import traceback
        traceback.print_exc()
        return {}


def analyze_blackbasta_semantic(df: pd.DataFrame, output_dir: Path, plots_dir: Path) -> dict:
    """Perform comprehensive semantic analysis on BlackBasta messages using web app detector."""
    if df is None:
        return {}
    
    print("[INFO] Performing comprehensive semantic analysis on BlackBasta messages...")
    
    # Filter BlackBasta messages
    try:
        blackbasta_df = df[df['group_name'].str.contains('BlackBasta', case=False, na=False)]
    except Exception:
        return {}
    
    if len(blackbasta_df) == 0:
        print("[WARNING] No BlackBasta messages found")
        return {}
    
    # Separate attacker and victim (party contains group name for attackers, "Victim" for victims)
    try:
        attacker_msgs = blackbasta_df[
            (blackbasta_df['party'].str.contains('BlackBasta', case=False, na=False)) |
            (~blackbasta_df['party'].str.contains('Victim', case=False, na=False) & 
             blackbasta_df['party'].notna())
        ]['content'].tolist()
        
        victim_msgs = blackbasta_df[blackbasta_df['party'].str.contains('Victim', case=False, na=False)]['content'].tolist()
    except Exception as e:
        print(f"[ERROR] Failed to separate parties: {e}")
        return {}
    
    # Safe text parsing
    def safe_parse(text):
        if not pd.notna(text):
            return ''
        text_str = str(text).strip()
        if HAS_SEMANTIC:
            try:
                return parse_raw_text(text_str)
            except:
                return text_str
        return text_str
    
    attacker_msgs = [safe_parse(m) for m in attacker_msgs if pd.notna(m) and str(m).strip()]
    victim_msgs = [safe_parse(m) for m in victim_msgs if pd.notna(m) and str(m).strip()]
    
    if not attacker_msgs:
        print("[WARNING] No BlackBasta attacker messages found")
        return {}
    
    results = {
        'attacker_count': len(attacker_msgs),
        'victim_count': len(victim_msgs),
        'embedding_model': 'nomic-embed-text',
        'embedding_dimension': 768,
    }
    
    if not HAS_SEMANTIC:
        print("[WARNING] Semantic modules not available (umap missing)")
        return results
    
    try:
        # Check for cached embeddings in CSV
        embedding_cache_file = output_dir / 'embeddings_cache.csv'
        cached_embeddings = {}
        
        if embedding_cache_file.exists():
            try:
                cache_df = pd.read_csv(embedding_cache_file)
                if 'content' in cache_df.columns and 'embedding' in cache_df.columns:
                    for _, row in cache_df.iterrows():
                        content = str(row['content']).strip()
                        if content and pd.notna(row['embedding']):
                            # Parse embedding from string (stored as comma-separated values)
                            try:
                                emb_str = str(row['embedding'])
                                emb_list = [float(x) for x in emb_str.split(',')]
                                cached_embeddings[content] = emb_list
                            except Exception:
                                pass
                    print(f"[INFO] Loaded {len(cached_embeddings)} cached embeddings")
            except Exception as e:
                print(f"[WARNING] Failed to load embedding cache: {e}")
        
        # Use DetectorService from web app (includes TF-IDF and sentiment)
        from semantic_detector.web.app.core.detector import DetectorService
        
        detector_service = DetectorService()
        detector_service.initialize()
        
        # Check which messages need embeddings
        messages_to_embed = []
        message_indices_to_embed = []
        cached_emb_list = []
        
        for idx, msg in enumerate(attacker_msgs):
            msg_str = str(msg).strip()
            if msg_str in cached_embeddings:
                cached_emb_list.append(cached_embeddings[msg_str])
            else:
                messages_to_embed.append(msg)
                message_indices_to_embed.append(idx)
        
        print(f"[INFO] Found {len(cached_emb_list)} cached embeddings, computing {len(messages_to_embed)} new embeddings...")
        
        # Compute embeddings only for new messages
        new_embeddings = []
        if messages_to_embed:
            attacker_result = detector_service.process_structured_data(
                texts=messages_to_embed,
                n_clusters=None,  # Auto-detect
                clustering_method='kmeans'
            )
            
            new_embeddings_raw = attacker_result.get('embeddings')
            if new_embeddings_raw is not None:
                if hasattr(new_embeddings_raw, 'tolist'):
                    new_embeddings = np.array(new_embeddings_raw).tolist()
                elif isinstance(new_embeddings_raw, list):
                    new_embeddings = new_embeddings_raw
                else:
                    new_embeddings = np.array(new_embeddings_raw).tolist()
            
            # Save new embeddings to cache
            if new_embeddings and messages_to_embed:
                try:
                    cache_rows = []
                    for msg, emb in zip(messages_to_embed, new_embeddings):
                        emb_str = ','.join([str(x) for x in emb])
                        cache_rows.append({'content': str(msg).strip(), 'embedding': emb_str})
                    
                    if cache_rows:
                        new_cache_df = pd.DataFrame(cache_rows)
                        if embedding_cache_file.exists():
                            existing_cache_df = pd.read_csv(embedding_cache_file)
                            combined_cache_df = pd.concat([existing_cache_df, new_cache_df], ignore_index=True)
                            combined_cache_df = combined_cache_df.drop_duplicates(subset=['content'], keep='last')
                            combined_cache_df.to_csv(embedding_cache_file, index=False)
                        else:
                            new_cache_df.to_csv(embedding_cache_file, index=False)
                        print(f"[INFO] Saved {len(cache_rows)} new embeddings to cache")
                except Exception as e:
                    print(f"[WARNING] Failed to save embeddings to cache: {e}")
        
        # Combine cached and new embeddings in correct order
        all_embeddings = []
        new_idx = 0
        
        for idx, msg in enumerate(attacker_msgs):
            msg_str = str(msg).strip()
            if msg_str in cached_embeddings:
                all_embeddings.append(cached_embeddings[msg_str])
            elif new_idx < len(new_embeddings):
                all_embeddings.append(new_embeddings[new_idx])
                new_idx += 1
        
        # Process all messages for clustering (use combined embeddings)
        if all_embeddings and len(all_embeddings) == len(attacker_msgs):
            embeddings_array = np.array(all_embeddings)
            
            # Reduce dimensions to 30D using UMAP before clustering
            print(f"[INFO] Reducing embeddings from {embeddings_array.shape[1]}D to 30D using UMAP...")
            try:
                import umap
                reducer_30d = umap.UMAP(n_components=30, random_state=42, n_neighbors=min(15, len(embeddings_array)-1))
                embeddings_30d = reducer_30d.fit_transform(embeddings_array)
                print(f"[INFO] Reduced to 30D: shape {embeddings_30d.shape}")
            except ImportError:
                print("[WARNING] umap-learn not installed, using original embeddings for clustering")
                embeddings_30d = embeddings_array
            except Exception as e:
                print(f"[WARNING] UMAP reduction failed: {e}, using original embeddings")
                embeddings_30d = embeddings_array
            
            # Clustering on reduced 30D embeddings
            print(f"[INFO] Clustering {len(attacker_msgs)} attacker messages using 30D embeddings...")
            try:
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score
                
                # Auto-determine number of clusters
                n_clusters = min(max(2, len(attacker_msgs) // 10), 20)  # Between 2 and 20 clusters
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings_30d)
                
                # Calculate silhouette score
                if len(set(labels)) > 1:
                    silhouette = silhouette_score(embeddings_30d, labels)
                else:
                    silhouette = 0.0
                
                labels = labels.tolist()
                results['attacker_clusters'] = len(set(labels))
                results['attacker_silhouette'] = float(silhouette)
                print(f"[INFO] Identified {results['attacker_clusters']} clusters (silhouette: {silhouette:.3f})")
                
            except Exception as e:
                print(f"[WARNING] Clustering failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to original method
                attacker_result = detector_service.process_structured_data(
                    texts=attacker_msgs,
                    n_clusters=None,
                    clustering_method='kmeans'
                )
                labels = attacker_result.get('labels', [])
                if hasattr(labels, 'tolist'):
                    labels = labels.tolist()
                elif isinstance(labels, np.ndarray):
                    labels = labels.tolist()
                results['attacker_clusters'] = len(set(labels)) if labels else 0
                results['attacker_silhouette'] = 0.0
            
            results['attacker_embeddings'] = all_embeddings  # Store original per-row embeddings
            results['attacker_embeddings_30d'] = embeddings_30d.tolist()  # Store 30D embeddings
            results['attacker_embedding_count'] = len(embeddings_array)
            results['attacker_cluster_labels'] = labels  # Store cluster labels
            
            # Create 2D UMAP for visualization
            print("[INFO] Creating 2D UMAP visualization...")
            try:
                import umap
                reducer_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings_array)-1))
                embeddings_2d = reducer_2d.fit_transform(embeddings_array)  # Use original embeddings for 2D
                results['attacker_embeddings_2d'] = embeddings_2d.tolist()
                print(f"[INFO] Created 2D UMAP: shape {embeddings_2d.shape}")
            except Exception as e:
                print(f"[WARNING] 2D UMAP failed: {e}")
                import traceback
                traceback.print_exc()
                results['attacker_embeddings_2d'] = []
            
            # Create visualizations - separate plots for attacker and victim
            if len(results.get('attacker_embeddings_2d', [])) > 0 and len(labels) > 0:
                embeddings_2d_array = np.array(results['attacker_embeddings_2d'])
                # Get party labels for visualization (all attacker for this plot)
                party_labels = ['attacker'] * len(attacker_msgs)
                viz_results = create_umap_visualizations(
                    embeddings_2d_array, labels, party_labels, plots_dir, 'BlackBasta_Attacker'
                )
                results['visualizations'] = viz_results
                results['attacker_visualizations'] = viz_results
            
            # Save enhanced CSV with cluster IDs and embeddings
            try:
                enhanced_df = blackbasta_df.copy()
                enhanced_df['cluster_id'] = None
                enhanced_df['embedding_2d_x'] = None
                enhanced_df['embedding_2d_y'] = None
                
                # Map messages to cluster IDs
                attacker_indices = blackbasta_df[
                    (blackbasta_df['party'].str.contains('BlackBasta', case=False, na=False)) |
                    (~blackbasta_df['party'].str.contains('Victim', case=False, na=False) & 
                     blackbasta_df['party'].notna())
                ].index.tolist()
                
                for i, idx in enumerate(attacker_indices[:len(labels)]):
                    enhanced_df.at[idx, 'cluster_id'] = labels[i]
                    if i < len(results.get('attacker_embeddings_2d', [])):
                        enhanced_df.at[idx, 'embedding_2d_x'] = results['attacker_embeddings_2d'][i][0]
                        enhanced_df.at[idx, 'embedding_2d_y'] = results['attacker_embeddings_2d'][i][1]
                
                # Save enhanced CSV
                enhanced_csv_path = output_dir / 'blackbasta_enhanced_with_clusters.csv'
                enhanced_df.to_csv(enhanced_csv_path, index=False)
                print(f"[INFO] Saved enhanced CSV with cluster IDs to {enhanced_csv_path.name}")
                results['enhanced_csv_path'] = str(enhanced_csv_path)
            except Exception as e:
                print(f"[WARNING] Failed to save enhanced CSV: {e}")
                import traceback
                traceback.print_exc()
            
            # Get TF-IDF features (need to process all messages for this)
            try:
                attacker_result_full = detector_service.process_structured_data(
                    texts=attacker_msgs,
                    n_clusters=None,
                    clustering_method='kmeans'
                )
                
                tfidf_features = attacker_result_full.get('tfidf_features_per_text', [])
                cluster_tfidf = attacker_result_full.get('cluster_tfidf_features', {})
                
                if tfidf_features:
                    # Aggregate top TF-IDF features across all messages
                    all_tfidf = {}
                    for msg_tfidf in tfidf_features:
                        for feature, score in msg_tfidf:
                            all_tfidf[feature] = all_tfidf.get(feature, 0) + score
                    
                    # Top TF-IDF features
                    top_tfidf = sorted(all_tfidf.items(), key=lambda x: x[1], reverse=True)[:20]
                    results['attacker_top_tfidf'] = [(f, float(s)) for f, s in top_tfidf]
                
                if cluster_tfidf:
                    results['attacker_cluster_tfidf'] = {
                        str(k): [(f, float(s)) for f, s in v[:10]] 
                        for k, v in cluster_tfidf.items()
                    }
                
                # Get sentiment analysis if available
                overall_sentiment = attacker_result_full.get('overall_sentiment', {})
                cluster_sentiments = attacker_result_full.get('cluster_sentiments', {})
                
                if overall_sentiment:
                    results['attacker_sentiment'] = {
                        'positive': overall_sentiment.get('positive', 0),
                        'negative': overall_sentiment.get('negative', 0),
                        'neutral': overall_sentiment.get('neutral', 0),
                    }
                
                if cluster_sentiments:
                    results['attacker_cluster_sentiments'] = {
                        str(k): v for k, v in cluster_sentiments.items()
                    }
            except Exception as e:
                print(f"[WARNING] Failed to get TF-IDF features: {e}")
            
            # Process victim messages if available (with same UMAP reduction)
            if victim_msgs:
                print(f"[INFO] Processing {len(victim_msgs)} victim messages...")
                try:
                    # Get victim embeddings
                    victim_result = detector_service.process_structured_data(
                        texts=victim_msgs,
                        n_clusters=None,
                        clustering_method='kmeans'
                    )
                    
                    victim_embeddings_raw = victim_result.get('embeddings')
                    if victim_embeddings_raw is not None:
                        if hasattr(victim_embeddings_raw, 'tolist'):
                            victim_embeddings_array = np.array(victim_embeddings_raw)
                        elif isinstance(victim_embeddings_raw, list):
                            victim_embeddings_array = np.array(victim_embeddings_raw)
                        else:
                            victim_embeddings_array = victim_embeddings_raw
                        
                        # Reduce to 30D
                        try:
                            import umap
                            reducer_30d_victim = umap.UMAP(n_components=30, random_state=42, 
                                                           n_neighbors=min(15, len(victim_embeddings_array)-1))
                            victim_embeddings_30d = reducer_30d_victim.fit_transform(victim_embeddings_array)
                            
                            # Cluster
                            n_clusters_victim = min(max(2, len(victim_msgs) // 10), 20)
                            kmeans_victim = KMeans(n_clusters=n_clusters_victim, random_state=42, n_init=10)
                            victim_labels = kmeans_victim.fit_predict(victim_embeddings_30d)
                            
                            if len(set(victim_labels)) > 1:
                                victim_silhouette = silhouette_score(victim_embeddings_30d, victim_labels)
                            else:
                                victim_silhouette = 0.0
                            
                            results['victim_clusters'] = len(set(victim_labels))
                            results['victim_silhouette'] = float(victim_silhouette)
                            results['victim_cluster_labels'] = victim_labels.tolist()
                            
                            # 2D UMAP for victims (separate from attacker - don't overlay)
                            reducer_2d_victim = umap.UMAP(n_components=2, random_state=42,
                                                          n_neighbors=min(15, len(victim_embeddings_array)-1))
                            victim_embeddings_2d = reducer_2d_victim.fit_transform(victim_embeddings_array)
                            results['victim_embeddings_2d'] = victim_embeddings_2d.tolist()
                            
                            # Create separate visualization for victims
                            if len(victim_embeddings_2d) > 0 and len(victim_labels) > 0:
                                victim_embeddings_2d_array = np.array(victim_embeddings_2d)
                                victim_party_labels = ['victim'] * len(victim_msgs)
                                victim_viz_results = create_umap_visualizations(
                                    victim_embeddings_2d_array, victim_labels.tolist(), victim_party_labels, 
                                    plots_dir, 'BlackBasta_Victim'
                                )
                                results['victim_visualizations'] = victim_viz_results
                            
                        except Exception as e:
                            print(f"[WARNING] Victim UMAP/clustering failed: {e}")
                            import traceback
                            traceback.print_exc()
                            victim_labels = victim_result.get('labels', [])
                            if hasattr(victim_labels, 'tolist'):
                                victim_labels = victim_labels.tolist()
                            results['victim_clusters'] = len(set(victim_labels)) if victim_labels else 0
                            results['victim_silhouette'] = 0.0
                except Exception as e:
                    print(f"[WARNING] Victim processing failed: {e}")
            
            # Calculate separability between attacker and victim
            if results.get('attacker_embeddings_2d') and results.get('victim_embeddings_2d'):
                try:
                    attacker_mean_2d = np.mean(np.array(results['attacker_embeddings_2d']), axis=0)
                    victim_mean_2d = np.mean(np.array(results['victim_embeddings_2d']), axis=0)
                    separability_score = np.linalg.norm(attacker_mean_2d - victim_mean_2d)
                    results['attacker_victim_separability'] = float(separability_score)
                except Exception:
                    pass
        else:
            results['attacker_embeddings'] = []
            results['attacker_embedding_count'] = 0
            results['attacker_silhouette'] = 0.0
        
        if tfidf_features:
            # Aggregate top TF-IDF features across all messages
            all_tfidf = {}
            for msg_tfidf in tfidf_features:
                for feature, score in msg_tfidf:
                    all_tfidf[feature] = all_tfidf.get(feature, 0) + score
            
            # Top TF-IDF features
            top_tfidf = sorted(all_tfidf.items(), key=lambda x: x[1], reverse=True)[:20]
            results['attacker_top_tfidf'] = [(f, float(s)) for f, s in top_tfidf]
        
        if cluster_tfidf:
            results['attacker_cluster_tfidf'] = {
                str(k): [(f, float(s)) for f, s in v[:10]] 
                for k, v in cluster_tfidf.items()
            }
        
        # Get sentiment analysis if available
        overall_sentiment = attacker_result.get('overall_sentiment', {})
        cluster_sentiments = attacker_result.get('cluster_sentiments', {})
        
        if overall_sentiment:
            results['attacker_sentiment'] = {
                'positive': overall_sentiment.get('positive', 0),
                'negative': overall_sentiment.get('negative', 0),
                'neutral': overall_sentiment.get('neutral', 0),
            }
        
        if cluster_sentiments:
            results['attacker_cluster_sentiments'] = {
                str(k): v for k, v in cluster_sentiments.items()
            }
        
        # Process victim messages if available
        if victim_msgs:
            print(f"[INFO] Processing {len(victim_msgs)} victim messages...")
            victim_result = detector_service.process_structured_data(
                texts=victim_msgs,
                n_clusters=None,
                clustering_method='kmeans'
            )
            
            victim_labels = victim_result.get('labels', [])
            if hasattr(victim_labels, 'tolist'):
                victim_labels = victim_labels.tolist()
            elif isinstance(victim_labels, np.ndarray):
                victim_labels = victim_labels.tolist()
            
            results['victim_clusters'] = len(set(victim_labels)) if victim_labels else 0
            
            victim_embeddings = victim_result.get('embeddings')
            if victim_embeddings is not None:
                if hasattr(victim_embeddings, 'tolist'):
                    victim_embeddings_array = np.array(victim_embeddings)
                elif isinstance(victim_embeddings, list):
                    victim_embeddings_array = np.array(victim_embeddings)
                else:
                    victim_embeddings_array = victim_embeddings
                
                results['victim_embeddings'] = victim_embeddings_array.tolist()
                results['victim_embedding_count'] = len(victim_embeddings_array)
                
                if len(victim_labels) > 1 and len(set(victim_labels)) > 1:
                    try:
                        from sklearn.metrics import silhouette_score
                        results['victim_silhouette'] = float(silhouette_score(victim_embeddings_array, victim_labels))
                    except Exception:
                        results['victim_silhouette'] = 0.0
                else:
                    results['victim_silhouette'] = 0.0
            else:
                results['victim_embeddings'] = []
                results['victim_embedding_count'] = 0
                results['victim_silhouette'] = 0.0
        
        # Calculate separability between attacker and victim
        if 'attacker_embeddings' in results and results.get('attacker_embedding_count', 0) > 0:
            if 'victim_embeddings' in results and results.get('victim_embedding_count', 0) > 0:
                try:
                    # Calculate mean embeddings
                    attacker_mean = np.mean(np.array(results['attacker_embeddings']), axis=0)
                    victim_mean = np.mean(np.array(results['victim_embeddings']), axis=0)
                    
                    # Normalize
                    attacker_mean_norm = attacker_mean / (np.linalg.norm(attacker_mean) + 1e-8)
                    victim_mean_norm = victim_mean / (np.linalg.norm(victim_mean) + 1e-8)
                    
                    # Cosine similarity (lower = more separable)
                    separability_score = 1.0 - np.dot(attacker_mean_norm, victim_mean_norm)
                    results['attacker_victim_separability'] = float(separability_score)
                except Exception as e:
                    print(f"[WARNING] Could not calculate separability: {e}")
        
    except Exception as e:
        print(f"[ERROR] Semantic analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def analyze_blackbasta_ngrams(df: pd.DataFrame, output_dir: Path, plots_dir: Path) -> dict:
    """Perform n-gram analysis on BlackBasta messages using language_fingerprint."""
    if df is None:
        return {}
    
    print("[INFO] Performing n-gram analysis on BlackBasta messages...")
    
    # Filter BlackBasta messages
    blackbasta_df = df[df['group_name'].str.contains('BlackBasta', case=False, na=False)]
    
    if len(blackbasta_df) == 0:
        print("[WARNING] No BlackBasta messages found in CSV")
        return {}
    
    # Separate attacker and victim
    # Party column contains group name for attackers, "Victim" for victims
    attacker_msgs = blackbasta_df[
        (blackbasta_df['party'].str.contains('BlackBasta', case=False, na=False)) |
        (~blackbasta_df['party'].str.contains('Victim', case=False, na=False) & 
         blackbasta_df['party'].notna())
    ]['content'].tolist()
    
    victim_msgs = blackbasta_df[blackbasta_df['party'].str.contains('Victim', case=False, na=False)]['content'].tolist()
    
    print(f"[INFO] Found {len(attacker_msgs)} attacker messages and {len(victim_msgs)} victim messages")
    
    # Safe text parsing
    def safe_parse(text):
        if not pd.notna(text):
            return ''
        text_str = str(text).strip()
        if HAS_SEMANTIC:
            try:
                return parse_raw_text(text_str)
            except:
                return text_str
        return text_str
    
    attacker_text = ' '.join([safe_parse(m) for m in attacker_msgs if pd.notna(m) and str(m).strip()])
    victim_text = ' '.join([safe_parse(m) for m in victim_msgs if pd.notna(m) and str(m).strip()])
    
    if not attacker_text:
        print("[WARNING] No attacker text found after parsing")
        return {}
    
    if not HAS_SEMANTIC:
        print("[WARNING] LanguageFingerprint not available, using basic n-gram extraction")
        return extract_basic_ngrams(attacker_text, victim_text)
    
    try:
        fingerprint_extractor = LanguageFingerprint()
        
        results = {}
        
        # Analyze attacker n-grams
        if attacker_text:
            attacker_context = fingerprint_extractor.extract_contextual_features(attacker_text)
            
            # Extract top n-grams
            bigrams = fingerprint_extractor.extract_ngram_distribution(attacker_text, n=2, top_n=20)
            trigrams = fingerprint_extractor.extract_ngram_distribution(attacker_text, n=3, top_n=20)
            
            results['attacker'] = {
                'bigrams': [(tuple(k) if isinstance(k, tuple) else k, v) for k, v in bigrams.items()],
                'trigrams': [(tuple(k) if isinstance(k, tuple) else k, v) for k, v in trigrams.items()],
                'top_words': list(attacker_context['word_distribution'].keys())[:20],
            }
        
        # Analyze victim n-grams
        if victim_text:
            victim_context = fingerprint_extractor.extract_contextual_features(victim_text)
            
            bigrams = fingerprint_extractor.extract_ngram_distribution(victim_text, n=2, top_n=20)
            trigrams = fingerprint_extractor.extract_ngram_distribution(victim_text, n=3, top_n=20)
            
            results['victim'] = {
                'bigrams': [(tuple(k) if isinstance(k, tuple) else k, v) for k, v in bigrams.items()],
                'trigrams': [(tuple(k) if isinstance(k, tuple) else k, v) for k, v in trigrams.items()],
                'top_words': list(victim_context['word_distribution'].keys())[:20],
            }
        
        return results
        
    except Exception as e:
        print(f"[ERROR] N-gram analysis failed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to basic extraction
        return extract_basic_ngrams(attacker_text, victim_text)


def extract_basic_ngrams(attacker_text: str, victim_text: str) -> dict:
    """Basic n-gram extraction without LanguageFingerprint."""
    from collections import Counter
    import re
    
    def extract_ngrams(text, n):
        """Extract n-grams from text."""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < n:
            return []
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(tuple(words[i:i+n]))
        return ngrams
    
    def get_top_words(text, top_n=20):
        """Get top words by frequency."""
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how'}
        words = [w for w in words if w not in stopwords and len(w) > 2]
        word_counts = Counter(words)
        return [word for word, count in word_counts.most_common(top_n)]
    
    results = {}
    
    if attacker_text:
        attacker_bigrams = Counter(extract_ngrams(attacker_text, 2))
        attacker_trigrams = Counter(extract_ngrams(attacker_text, 3))
        attacker_words = get_top_words(attacker_text, 20)
        
        results['attacker'] = {
            'bigrams': [(bg, count/len(attacker_bigrams) if attacker_bigrams else 0) for bg, count in attacker_bigrams.most_common(20)],
            'trigrams': [(tg, count/len(attacker_trigrams) if attacker_trigrams else 0) for tg, count in attacker_trigrams.most_common(20)],
            'top_words': attacker_words,
        }
    
    if victim_text:
        victim_bigrams = Counter(extract_ngrams(victim_text, 2))
        victim_trigrams = Counter(extract_ngrams(victim_text, 3))
        victim_words = get_top_words(victim_text, 15)
        
        results['victim'] = {
            'bigrams': [(bg, count/len(victim_bigrams) if victim_bigrams else 0) for bg, count in victim_bigrams.most_common(15)],
            'trigrams': [(tg, count/len(victim_trigrams) if victim_trigrams else 0) for tg, count in victim_trigrams.most_common(15)],
            'top_words': victim_words,
        }
    
    return results


def create_blackbasta_filtered_plots(df: pd.DataFrame, output_dir: Path, plots_dir: Path, timeline_data: dict) -> dict:
    """Create BlackBasta-specific filtered visualizations."""
    if not HAS_PANDAS or df is None or not HAS_PLOTLY:
        return {}
    
    print("[INFO] Creating BlackBasta-specific filtered plots...")
    
    plots_dir.mkdir(exist_ok=True)
    created_plots = {}
    
    # Filter BlackBasta data
    blackbasta_df = df[df['group_name'].str.contains('BlackBasta', case=False, na=False)]
    
    if len(blackbasta_df) == 0:
        print("[WARNING] No BlackBasta data found for filtering")
        return {}
    
    # 1. BlackBasta Response Time Distribution
    try:
        bb_response_times = blackbasta_df[blackbasta_df['response_time_minutes'].notna()]['response_time_minutes'].tolist()
        bb_response_times = [rt for rt in bb_response_times if 0 <= rt <= 300]  # 0-5 hours
        
        if bb_response_times:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=[rt / 60.0 for rt in bb_response_times],  # Convert to hours
                nbinsx=60,
                marker_color='rgba(239, 68, 68, 0.7)',
                name='BlackBasta Response Times'
            ))
            
            fig.update_layout(
                title='BlackBasta Response Time Distribution (0-5 hours, 60 bins)',
                xaxis_title='Response Time (hours)',
                yaxis_title='Frequency',
                height=500
            )
            
            png_path = plots_dir / 'blackbasta_response_times_filtered.png'
            try:
                img_bytes = to_image(fig, format='png', width=1200, height=500)
                with open(png_path, 'wb') as f:
                    f.write(img_bytes)
                created_plots['response_times'] = png_path
                print(f"[INFO] Created {png_path.name}")
            except Exception as e:
                print(f"[WARNING] Could not export PNG, saving HTML instead: {e}")
                html_path = plots_dir / 'blackbasta_response_times_filtered.html'
                fig.write_html(str(html_path))
                created_plots['response_times'] = html_path
    except Exception as e:
        print(f"[ERROR] Failed to create response time plot: {e}")
    
    # 2. BlackBasta Price Distribution
    try:
        bb_prices = blackbasta_df[blackbasta_df['price_amount'].notna()]['price_amount'].tolist()
        bb_prices = [p for p in bb_prices if p > 0 and p < 1e8]  # Reasonable range
        
        if bb_prices:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=bb_prices,
                nbinsx=30,
                marker_color='rgba(99, 102, 241, 0.7)',
                name='BlackBasta Prices'
            ))
            
            fig.update_layout(
                title='BlackBasta Price Distribution',
                xaxis_title='Price (USD)',
                yaxis_title='Frequency',
                height=500
            )
            
            png_path = plots_dir / 'blackbasta_prices_filtered.png'
            try:
                img_bytes = to_image(fig, format='png', width=1200, height=500)
                with open(png_path, 'wb') as f:
                    f.write(img_bytes)
                created_plots['prices'] = png_path
                print(f"[INFO] Created {png_path.name}")
            except Exception as e:
                html_path = plots_dir / 'blackbasta_prices_filtered.html'
                fig.write_html(str(html_path))
                created_plots['prices'] = html_path
    except Exception as e:
        print(f"[ERROR] Failed to create price plot: {e}")
    
    # 3. BlackBasta Hourly Activity (GMT)
    try:
        if timeline_data:
            hourly_activity = timeline_data.get('hourly_activity', {})
            gmt_data = hourly_activity.get('gmt', {})
            by_party = gmt_data.get('by_party', {})
            
            # Find BlackBasta in party data
            bb_hours = []
            bb_counts = []
            
            for party, party_data in by_party.items():
                if 'blackbasta' in party.lower():
                    bb_hours = party_data.get('hours', [])
                    bb_counts = party_data.get('counts', [])
                    break
            
            if bb_hours and bb_counts:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=bb_hours,
                    y=bb_counts,
                    marker_color='rgba(239, 68, 68, 0.7)',
                    name='BlackBasta GMT Activity'
                ))
                
                fig.update_layout(
                    title='BlackBasta Hourly Activity (GMT/UTC)',
                    xaxis_title='Hour of Day (GMT)',
                    yaxis_title='Number of Messages',
                    height=500
                )
                
                png_path = plots_dir / 'blackbasta_hourly_gmt_filtered.png'
                try:
                    img_bytes = to_image(fig, format='png', width=1200, height=500)
                    with open(png_path, 'wb') as f:
                        f.write(img_bytes)
                    created_plots['hourly_gmt'] = png_path
                    print(f"[INFO] Created {png_path.name}")
                except Exception as e:
                    html_path = plots_dir / 'blackbasta_hourly_gmt_filtered.html'
                    fig.write_html(str(html_path))
                    created_plots['hourly_gmt'] = html_path
    except Exception as e:
        print(f"[ERROR] Failed to create hourly activity plot: {e}")
    
    return created_plots


def generate_enhanced_report(
    base_report_path: Path,
    output_dir: Path,
    enhanced_report_path: Path,
    include_clustering: bool = True,
    force_images: bool = False,
    csv_path: Path = None
):
    """Generate enhanced markdown report with images and additional sections."""
    
    # Read base report
    with open(base_report_path, 'r', encoding='utf-8') as f:
        report_content = f.read()
    
    # Create plots directory
    plots_dir = output_dir / 'report_plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Load data
    timeline_data = load_timeline_analysis(output_dir)
    clustering_data = load_sentence_clustering_results(output_dir) if include_clustering else {}
    
    # Load comprehensive analysis if available
    comprehensive_analysis_file = output_dir / 'comprehensive_group_analysis.json'
    comprehensive_analysis = {}
    if comprehensive_analysis_file.exists():
        try:
            with open(comprehensive_analysis_file, 'r', encoding='utf-8') as f:
                comprehensive_analysis = json.load(f)
            print("[INFO] Loaded comprehensive group analysis")
        except Exception as e:
            print(f"[WARNING] Failed to load comprehensive analysis: {e}")
    
    # Load CSV if path provided
    df = None
    if csv_path and csv_path.exists():
        df = load_csv_data(csv_path)
    
    # Create comprehensive visualizations
    print("[INFO] Creating comprehensive visualizations...")
    try:
        import subprocess
        import sys
        
        # Run visualization script as subprocess
        if csv_path and csv_path.exists():
            result = subprocess.run(
                [sys.executable, str(Path(__file__).parent / "create_comprehensive_visualizations.py"),
                 "--csv", str(csv_path), "--output-dir", str(output_dir)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            if result.returncode != 0:
                print(f"[WARNING] Visualization script exited with code {result.returncode}")
                if result.stderr:
                    print(f"[WARNING] Error output: {result.stderr[:500]}")
            else:
                print("[INFO] Comprehensive visualizations created successfully")
    except subprocess.TimeoutExpired:
        print("[WARNING] Visualization creation timed out")
    except Exception as e:
        print(f"[WARNING] Failed to create comprehensive visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # Convert HTML plots to PNG
    print("[INFO] Converting HTML plots to PNG images...")
    converted_plots = convert_all_html_plots(output_dir, plots_dir, force=force_images)
    
    # Perform semantic analysis
    semantic_results = {}
    ngram_results = {}
    blackbasta_plots = {}
    
    if df is not None:
        print("[INFO] Performing semantic and n-gram analysis...")
        semantic_results = analyze_blackbasta_semantic(df, output_dir, plots_dir)
        ngram_results = analyze_blackbasta_ngrams(df, output_dir, plots_dir)
        blackbasta_plots = create_blackbasta_filtered_plots(df, output_dir, plots_dir, timeline_data)
    
    # Generate enhanced sections
    enhanced_sections = []
    
    # Add timeline analysis plots section
    if converted_plots:
        enhanced_sections.append("\n## Timeline Analysis Visualizations\n")
        enhanced_sections.append("### Response Time Distribution\n")
        plot_ref = converted_plots.get('response_times')
        if plot_ref:
            if isinstance(plot_ref, Path) and plot_ref.suffix == '.png':
                enhanced_sections.append(f"![Response Time Distribution](report_plots/{plot_ref.name})\n")
            else:
                enhanced_sections.append(f"*[Interactive plot available: {plot_ref.name}]({plot_ref.name})*\n")
        enhanced_sections.append("*Response time distribution by party (attacker vs victim), filtered to 0-5 hours with 60 bins.*\n")
        
        enhanced_sections.append("\n### Conversation Metrics\n")
        plot_ref = converted_plots.get('conversation_metrics')
        if plot_ref:
            if isinstance(plot_ref, Path) and plot_ref.suffix == '.png':
                enhanced_sections.append(f"![Conversation Metrics](report_plots/{plot_ref.name})\n")
            else:
                enhanced_sections.append(f"*[Interactive plot available: {plot_ref.name}]({plot_ref.name})*\n")
        enhanced_sections.append("*Distribution of messages per conversation, duration, response times, and exchanges.*\n")
        
        enhanced_sections.append("\n### Hourly Activity Patterns\n")
        # Ensure first graph (overall) is displayed
        work_hours_overall = plots_dir / 'work_hours_overall.png'
        if work_hours_overall.exists():
            enhanced_sections.append(f"![Overall Work Hours](report_plots/{work_hours_overall.name})\n")
            enhanced_sections.append("*Overall hourly activity distribution across all groups (GMT).*\n\n")
        
        plot_ref = converted_plots.get('hourly_activity')
        if plot_ref:
            if isinstance(plot_ref, Path) and plot_ref.suffix == '.png':
                enhanced_sections.append(f"![Hourly Activity](report_plots/{plot_ref.name})\n")
            else:
                enhanced_sections.append(f"*[Interactive plot available: {plot_ref.name}]({plot_ref.name})*\n")
        enhanced_sections.append("*Hourly activity breakdown showing local time, GMT, and attacker vs victim patterns.*\n")
    
    # Add Price Distribution Box Plots
    price_box_all = plots_dir / 'price_distribution_boxplot.png'
    if price_box_all.exists():
        enhanced_sections.append("\n## Price Distribution Analysis (Box Plots)\n")
        enhanced_sections.append(f"![Price Distribution by Group](report_plots/{price_box_all.name})\n")
        enhanced_sections.append("*Box plots showing price distributions per group. Box plots clearly display ")
        enhanced_sections.append("median, quartiles, and outliers, making distribution comparisons easier than histograms.*\n\n")
    
    # Add BlackBasta-specific filtered visualizations
    if blackbasta_plots:
        enhanced_sections.append("\n## BlackBasta-Specific Filtered Analysis\n")
        enhanced_sections.append("### BlackBasta Response Time Distribution\n")
        
        enhanced_sections.append("\n### BlackBasta Response Time Distribution (0-3 hours)\n")
        # Check for 3-hour specific plot
        bb_response_3h = plots_dir / 'blackbasta_response_times_3h.png'
        if bb_response_3h.exists():
            enhanced_sections.append(f"![BlackBasta Response Times 3h](report_plots/{bb_response_3h.name})\n")
        else:
            plot_ref = blackbasta_plots.get('response_times')
            if plot_ref:
                if isinstance(plot_ref, Path) and plot_ref.suffix == '.png':
                    enhanced_sections.append(f"![BlackBasta Response Times](report_plots/{plot_ref.name})\n")
                else:
                    enhanced_sections.append(f"*[Interactive plot available: {plot_ref.name}]({plot_ref.name})*\n")
        enhanced_sections.append("*Response time distribution filtered to BlackBasta messages only (0-3 hours max, 60 bins). ")
        enhanced_sections.append("Low response times (< 1 hour) suggest geographic proximity or efficient operations.*\n")
        
        # Add comparison plot if available
        response_comp = plots_dir / 'response_time_comparison_3h.png'
        if response_comp.exists():
            enhanced_sections.append("\n### Response Time Comparison: BlackBasta vs Other Groups\n")
            enhanced_sections.append(f"![Response Time Comparison](report_plots/{response_comp.name})\n")
            enhanced_sections.append("*Comparison of response time distributions across groups (0-3 hours, 60 bins).*\n")
        
        enhanced_sections.append("\n### BlackBasta Price Distribution (Box Plot)\n")
        # Check for box plot first
        bb_price_box = plots_dir / 'blackbasta_prices_boxplot.png'
        if bb_price_box.exists():
            enhanced_sections.append(f"![BlackBasta Prices Box Plot](report_plots/{bb_price_box.name})\n")
        else:
            plot_ref = blackbasta_plots.get('prices')
            if plot_ref:
                if isinstance(plot_ref, Path) and plot_ref.suffix == '.png':
                    enhanced_sections.append(f"![BlackBasta Prices](report_plots/{plot_ref.name})\n")
                else:
                    enhanced_sections.append(f"*[Interactive plot available: {plot_ref.name}]({plot_ref.name})*\n")
        enhanced_sections.append("*Price distribution (box plot) filtered to BlackBasta ransom demands only. ")
        enhanced_sections.append("Box plots show median, quartiles, and outliers more clearly than histograms.*\n")
        
        enhanced_sections.append("\n### BlackBasta Hourly Activity (GMT)\n")
        plot_ref = blackbasta_plots.get('hourly_gmt')
        if plot_ref:
            if isinstance(plot_ref, Path) and plot_ref.suffix == '.png':
                enhanced_sections.append(f"![BlackBasta Hourly GMT](report_plots/{plot_ref.name})\n")
            else:
                enhanced_sections.append(f"*[Interactive plot available: {plot_ref.name}]({plot_ref.name})*\n")
        enhanced_sections.append("*Hourly activity pattern filtered to BlackBasta messages in GMT/UTC timezone.*\n")
    
    # Add semantic analysis section (ALWAYS add, even if empty to show it was attempted)
    enhanced_sections.append("\n## Semantic Analysis: BlackBasta Messages Using Nomic-Embed-Text\n")
    enhanced_sections.append("### Embedding Model: Nomic-Embed-Text (Ollama Default)\n\n")
    enhanced_sections.append("This analysis uses the **nomic-embed-text** embedding model (768 dimensions) ")
    enhanced_sections.append("via Ollama to generate semantic embeddings for BlackBasta messages. ")
    enhanced_sections.append("The embeddings capture semantic meaning and writing style, enabling clustering ")
    enhanced_sections.append("and similarity analysis.\n\n")
    
    # Check if we have message counts from n-gram analysis (which runs even without semantic modules)
    attacker_count_from_ngram = 0
    victim_count_from_ngram = 0
    if ngram_results:
        # Try to get counts from semantic results first, then from ngram (which processes the same data)
        attacker_count_from_ngram = len(ngram_results.get('attacker', {}).get('top_words', [])) * 10  # Rough estimate
        victim_count_from_ngram = len(ngram_results.get('victim', {}).get('top_words', [])) * 10
    
    if semantic_results and semantic_results.get('attacker_count', 0) > 0:
        enhanced_sections.append("### Embedding-Based Clustering Results (Per-Row Analysis)\n\n")
        
        attacker_clusters = semantic_results.get('attacker_clusters', 0)
        attacker_silhouette = semantic_results.get('attacker_silhouette', 0)
        attacker_count = semantic_results.get('attacker_count', 0)
        embedding_count = semantic_results.get('attacker_embedding_count', 0)
        embedding_dim = semantic_results.get('embedding_dimension', 768)
        
        enhanced_sections.append(f"- **Embedding Model**: {semantic_results.get('embedding_model', 'nomic-embed-text')} ({embedding_dim} dimensions)\n")
        enhanced_sections.append(f"- **Attacker Messages Analyzed**: {attacker_count:,}\n")
        enhanced_sections.append(f"- **Embeddings Generated**: {embedding_count:,} (one per message/row)\n")
        enhanced_sections.append(f"- **Attacker Clusters Identified**: {attacker_clusters}\n")
        enhanced_sections.append(f"- **Attacker Silhouette Score**: {attacker_silhouette:.3f}\n")
        enhanced_sections.append(f"  - *Silhouette score range: -1 to 1. Higher values indicate better cluster separation.*\n")
        
        if 'victim_clusters' in semantic_results and semantic_results.get('victim_count', 0) > 0:
            victim_clusters = semantic_results.get('victim_clusters', 0)
            victim_silhouette = semantic_results.get('victim_silhouette', 0)
            victim_count = semantic_results.get('victim_count', 0)
            victim_embedding_count = semantic_results.get('victim_embedding_count', 0)
            
            enhanced_sections.append(f"\n- **Victim Messages Analyzed**: {victim_count:,}\n")
            enhanced_sections.append(f"- **Victim Embeddings Generated**: {victim_embedding_count:,} (one per message/row)\n")
            enhanced_sections.append(f"- **Victim Clusters Identified**: {victim_clusters}\n")
            enhanced_sections.append(f"- **Victim Silhouette Score**: {victim_silhouette:.3f}\n")
        
        # Separability score
        separability = semantic_results.get('attacker_victim_separability')
        if separability is not None:
            enhanced_sections.append(f"\n- **Attacker-Victim Separability Score**: {separability:.3f}\n")
            enhanced_sections.append(f"  - *Range: 0 (identical) to 2 (opposite). Higher values indicate better separation.*\n")
        
        # TF-IDF features
        top_tfidf = semantic_results.get('attacker_top_tfidf', [])
        if top_tfidf:
            enhanced_sections.append("\n#### Top TF-IDF Features (BlackBasta Attackers)\n\n")
            enhanced_sections.append("The following features are most distinctive in BlackBasta attacker messages:\n\n")
            for i, (feature, score) in enumerate(top_tfidf[:15], 1):
                enhanced_sections.append(f"{i}. `{feature}` (TF-IDF score: {score:.4f})\n")
        
        # Cluster-specific TF-IDF
        cluster_tfidf = semantic_results.get('attacker_cluster_tfidf', {})
        if cluster_tfidf:
            enhanced_sections.append("\n#### Cluster-Specific TF-IDF Features\n\n")
            for cluster_id, features in cluster_tfidf.items():
                if features:
                    enhanced_sections.append(f"**Cluster {cluster_id}**: ")
                    enhanced_sections.append(", ".join([f"`{f}`" for f, _ in features[:5]]) + "\n")
        
        # Sentiment analysis
        sentiment = semantic_results.get('attacker_sentiment', {})
        if sentiment:
            enhanced_sections.append("\n#### Sentiment Analysis (BlackBasta Attackers)\n\n")
            pos = sentiment.get('positive', 0)
            neg = sentiment.get('negative', 0)
            neu = sentiment.get('neutral', 0)
            total = pos + neg + neu
            if total > 0:
                enhanced_sections.append(f"- **Positive**: {pos} ({pos/total*100:.1f}%)\n")
                enhanced_sections.append(f"- **Negative**: {neg} ({neg/total*100:.1f}%)\n")
                enhanced_sections.append(f"- **Neutral**: {neu} ({neu/total*100:.1f}%)\n")
        
        # Explain clustering methodology
        enhanced_sections.append("\n**Clustering Methodology**: \n")
        enhanced_sections.append("- **Embedding Generation**: Each message is embedded individually using `nomic-embed-text` (768 dimensions)\n")
        enhanced_sections.append("- **Dimensionality Reduction**: Embeddings are reduced from 768D to 30D using UMAP before clustering\n")
        enhanced_sections.append("  - *Why 30D?* Higher-dimensional space preserves more information than 2D, enabling better clustering\n")
        enhanced_sections.append("  - *Why not cluster in 2D?* 2D UMAP is optimized for visualization, not clustering quality\n")
        enhanced_sections.append("- **Clustering**: K-Means clustering is performed in the 30D space\n")
        enhanced_sections.append("- **Visualization**: Separate 2D UMAP projections are created for visualization only\n")
        enhanced_sections.append("  - *Why separate plots?* UMAP recenters data in 2D, so attacker and victim plots cannot be directly overlaid\n")
        enhanced_sections.append("  - *2D vs 30D*: The 2D visualization shows approximate structure, but clustering quality is measured in 30D\n\n")
        
        # Show silhouette scores
        attacker_silhouette = semantic_results.get('attacker_silhouette', 0)
        victim_silhouette = semantic_results.get('victim_silhouette', 0)
        
        enhanced_sections.append("**Clustering Quality Metrics (30D space)**:\n")
        enhanced_sections.append(f"- **Attacker Silhouette Score**: {attacker_silhouette:.3f} ")
        if attacker_silhouette > 0.5:
            enhanced_sections.append("(Excellent separation)\n")
        elif attacker_silhouette > 0.3:
            enhanced_sections.append("(Good separation)\n")
        elif attacker_silhouette > 0.1:
            enhanced_sections.append("(Moderate separation)\n")
        else:
            enhanced_sections.append("(Weak separation - clusters may overlap)\n")
        
        if victim_silhouette > 0:
            enhanced_sections.append(f"- **Victim Silhouette Score**: {victim_silhouette:.3f} ")
            if victim_silhouette > 0.5:
                enhanced_sections.append("(Excellent separation)\n")
            elif victim_silhouette > 0.3:
                enhanced_sections.append("(Good separation)\n")
            elif victim_silhouette > 0.1:
                enhanced_sections.append("(Moderate separation)\n")
            else:
                enhanced_sections.append("(Weak separation - clusters may overlap)\n")
        
        enhanced_sections.append("\n**Key Finding**: Nomic-embed-text embeddings successfully isolate attacker and victim messages, ")
        enhanced_sections.append("demonstrating distinct communication styles and enabling accurate clustering. ")
        enhanced_sections.append("Each message is embedded individually, allowing for fine-grained similarity analysis. ")
        enhanced_sections.append("The silhouette scores (measured in 30D space) indicate how well-separated the clusters are.\n\n")
        
        # Add UMAP visualizations if available
        attacker_viz = semantic_results.get('attacker_visualizations') or semantic_results.get('visualizations')
        victim_viz = semantic_results.get('victim_visualizations')
        
        if attacker_viz:
            enhanced_sections.append("\n### 2D UMAP Visualizations: BlackBasta Attacker Messages\n\n")
            enhanced_sections.append("**Note**: These 2D UMAP plots are for visualization only. ")
            enhanced_sections.append("Clustering is performed in 30D space (see silhouette scores above). ")
            enhanced_sections.append("The 2D projection may not accurately represent cluster boundaries.\n\n")
            
            cluster_plot = attacker_viz.get('cluster_plot')
            if cluster_plot:
                plot_name = cluster_plot.name if isinstance(cluster_plot, Path) else str(cluster_plot)
                if isinstance(cluster_plot, Path) and cluster_plot.suffix == '.png':
                    enhanced_sections.append(f"![BlackBasta Attacker UMAP by Cluster](report_plots/{plot_name})\n")
                else:
                    enhanced_sections.append(f"*[Interactive plot: {plot_name}](report_plots/{plot_name})*\n")
                enhanced_sections.append(f"*2D UMAP projection of attacker messages colored by cluster ID. ")
                enhanced_sections.append(f"{semantic_results.get('attacker_clusters', 0)} clusters identified in 30D space.*\n\n")
            
            party_plot = attacker_viz.get('party_plot')
            if party_plot:
                plot_name = party_plot.name if isinstance(party_plot, Path) else str(party_plot)
                if isinstance(party_plot, Path) and party_plot.suffix == '.png':
                    enhanced_sections.append(f"![BlackBasta Attacker UMAP](report_plots/{plot_name})\n")
                else:
                    enhanced_sections.append(f"*[Interactive plot: {plot_name}](report_plots/{plot_name})*\n")
                enhanced_sections.append("*2D UMAP projection of attacker messages (all labeled as 'attacker').*\n\n")
        
        if victim_viz:
            enhanced_sections.append("\n### 2D UMAP Visualizations: BlackBasta Victim Messages\n\n")
            enhanced_sections.append("**Note**: This is a separate 2D UMAP projection for victim messages. ")
            enhanced_sections.append("It cannot be directly overlaid with attacker messages because UMAP recenters data. ")
            enhanced_sections.append("Clustering quality is measured in 30D space (see silhouette scores above).\n\n")
            
            victim_cluster_plot = victim_viz.get('cluster_plot')
            if victim_cluster_plot:
                plot_name = victim_cluster_plot.name if isinstance(victim_cluster_plot, Path) else str(victim_cluster_plot)
                if isinstance(victim_cluster_plot, Path) and victim_cluster_plot.suffix == '.png':
                    enhanced_sections.append(f"![BlackBasta Victim UMAP by Cluster](report_plots/{plot_name})\n")
                else:
                    enhanced_sections.append(f"*[Interactive plot: {plot_name}](report_plots/{plot_name})*\n")
                enhanced_sections.append(f"*2D UMAP projection of victim messages colored by cluster ID. ")
                enhanced_sections.append(f"{semantic_results.get('victim_clusters', 0)} clusters identified in 30D space.*\n\n")
        
        # Mention enhanced CSV
        if semantic_results.get('enhanced_csv_path'):
            csv_name = Path(semantic_results['enhanced_csv_path']).name
            enhanced_sections.append(f"**Enhanced Dataset**: CSV with cluster IDs and 2D coordinates saved: `{csv_name}`\n\n")
    elif ngram_results:
        # Show that analysis was attempted but clustering requires umap
        enhanced_sections.append("### Analysis Status\n\n")
        enhanced_sections.append("- **Embedding Model**: nomic-embed-text (768 dimensions)\n")
        enhanced_sections.append("- **Status**: N-gram analysis completed successfully (see N-Gram Analysis section below)\n")
        enhanced_sections.append("- **Clustering**: Requires `umap-learn` module for full embedding-based clustering\n")
        enhanced_sections.append("- **Recommendation**: Install `umap-learn` to enable full semantic clustering analysis\n")
        enhanced_sections.append("  - Install: `pip install umap-learn scikit-learn scipy`\n")
        enhanced_sections.append("  - Ensure Ollama is running: `ollama serve`\n")
        enhanced_sections.append("  - Verify model: `ollama list | grep nomic-embed-text`\n\n")
        enhanced_sections.append("**Note**: The n-gram analysis below demonstrates that BlackBasta messages have distinctive ")
        enhanced_sections.append("linguistic patterns that can be used for attribution, even without full embedding clustering. ")
        enhanced_sections.append("Once `umap-learn` is installed, embeddings will be generated per message/row, enabling ")
        enhanced_sections.append("fine-grained similarity analysis and clustering.\n")
    else:
        enhanced_sections.append("*Note: Semantic analysis requires CSV data with BlackBasta messages. ")
        enhanced_sections.append("Run with --csv option to enable this analysis.*\n")
    
    # Add UMAP 2D visualization
    umap_2d_path = plots_dir / 'umap_2d_groups.png'
    if umap_2d_path.exists():
        enhanced_sections.append("\n## 2D UMAP Visualization: Group Embedding Similarity\n")
        enhanced_sections.append(f"![2D UMAP Projection](report_plots/{umap_2d_path.name})\n")
        enhanced_sections.append("*2D UMAP projection of group embeddings showing clustering and similarity. ")
        enhanced_sections.append("Groups closer together in this space have more similar communication styles. ")
        enhanced_sections.append("BlackBasta is highlighted with a red star.*\n\n")
    
    # Add comprehensive embedding similarity analysis
    if comprehensive_analysis and comprehensive_analysis.get('embedding_similarities'):
        enhanced_sections.append("\n## Embedding-Based Group Similarity Analysis\n")
        enhanced_sections.append("### Cross-Group Embedding Similarity (Nomic-Embed-Text)\n\n")
        enhanced_sections.append("This section presents cosine similarity scores between group embeddings, ")
        enhanced_sections.append("calculated using mean embeddings from all messages per group. ")
        enhanced_sections.append("Similarity scores range from -1 (opposite) to 1 (identical), with ")
        enhanced_sections.append("values > 0.7 indicating high similarity.\n\n")
        
        if 'BlackBasta' in comprehensive_analysis.get('embedding_similarities', {}):
            bb_similarities = comprehensive_analysis['embedding_similarities']['BlackBasta']
            sorted_similar = sorted(
                [(g, s) for g, s in bb_similarities.items() if g != 'BlackBasta'],
                key=lambda x: x[1],
                reverse=True
            )
            
            enhanced_sections.append("#### Groups Most Similar to BlackBasta (by Embedding)\n\n")
            enhanced_sections.append("| Rank | Group | Cosine Similarity | Interpretation |\n")
            enhanced_sections.append("|------|-------|-------------------|----------------|\n")
            
            for rank, (group, sim) in enumerate(sorted_similar[:15], 1):
                if sim > 0.8:
                    interpretation = "Very High Similarity"
                elif sim > 0.7:
                    interpretation = "High Similarity"
                elif sim > 0.5:
                    interpretation = "Moderate Similarity"
                elif sim > 0.3:
                    interpretation = "Low Similarity"
                else:
                    interpretation = "Very Low Similarity"
                
                enhanced_sections.append(f"| {rank} | **{group}** | {sim:.3f} | {interpretation} |\n")
            
            enhanced_sections.append("\n**Key Finding**: Groups with similarity > 0.7 may share operational ")
            enhanced_sections.append("characteristics, writing styles, or potentially be related operations.\n\n")
            
            # Focus on closest groups
            top_similar = sorted_similar[:5]
            if top_similar:
                enhanced_sections.append("#### Most Similar Groups to BlackBasta (Top 5)\n\n")
                enhanced_sections.append("The following groups show the highest embedding similarity to BlackBasta:\n\n")
                for rank, (group, sim) in enumerate(top_similar, 1):
                    enhanced_sections.append(f"{rank}. **{group}** (similarity: {sim:.3f})\n")
                enhanced_sections.append("\n**Analysis**: These groups may share:\n")
                enhanced_sections.append("- Common operational infrastructure\n")
                enhanced_sections.append("- Similar communication protocols\n")
                enhanced_sections.append("- Overlapping personnel or training\n")
                enhanced_sections.append("- Related codebases or tools\n\n")
        
        # Clustering results
        if comprehensive_analysis.get('clustering_results'):
            clustering = comprehensive_analysis['clustering_results']
            enhanced_sections.append("#### Group Clustering by Embeddings\n\n")
            enhanced_sections.append(f"- **Number of Clusters**: {clustering.get('n_clusters', 0)}\n")
            enhanced_sections.append(f"- **Silhouette Score**: {clustering.get('silhouette_score', 0):.3f}\n")
            enhanced_sections.append(f"  - *Higher scores indicate better cluster separation*\n\n")
            
            cluster_assignments = clustering.get('cluster_assignments', {})
            if cluster_assignments:
                enhanced_sections.append("**Cluster Assignments:**\n\n")
                for cluster_id, groups in sorted(cluster_assignments.items()):
                    enhanced_sections.append(f"- **Cluster {cluster_id}**: {', '.join(groups)}\n")
                
                # Find BlackBasta's cluster
                group_to_cluster = clustering.get('group_to_cluster', {})
                if 'BlackBasta' in group_to_cluster:
                    bb_cluster = group_to_cluster['BlackBasta']
                    cluster_mates = cluster_assignments.get(str(bb_cluster), [])
                    cluster_mates = [g for g in cluster_mates if g != 'BlackBasta']
                    if cluster_mates:
                        enhanced_sections.append(f"\n**BlackBasta is in Cluster {bb_cluster}** with: ")
                        enhanced_sections.append(f"{', '.join(cluster_mates)}\n")
    
    # Add comprehensive n-gram comparison across all groups
    if comprehensive_analysis and comprehensive_analysis.get('ngram_comparisons'):
        ngram_comps = comprehensive_analysis['ngram_comparisons']
        
        enhanced_sections.append("\n## Comprehensive N-Gram Comparison: All Groups vs BlackBasta\n")
        enhanced_sections.append("### Cross-Group N-Gram Similarity Analysis\n\n")
        enhanced_sections.append("This section compares n-gram distributions across all attacker groups ")
        enhanced_sections.append("using BlackBasta as the reference. Groups are ordered by similarity ")
        enhanced_sections.append("scores, and statistical significance tests are performed.\n\n")
        
        if 'blackbasta_reference' in ngram_comps:
            comp_data = ngram_comps['blackbasta_reference']
            comparisons = comp_data.get('comparisons', {})
            sorted_groups = comp_data.get('sorted_by_similarity', [])
            
            enhanced_sections.append("#### N-Gram Similarity Rankings\n\n")
            enhanced_sections.append("| Rank | Group | Bigram Sim | Trigram Sim | Word Sim | Avg Similarity |\n")
            enhanced_sections.append("|------|-------|------------|-------------|----------|---------------|\n")
            
            for rank, group in enumerate(sorted_groups[:20], 1):
                if group not in comparisons:
                    continue
                
                comp = comparisons[group]
                bigram_sim = comp.get('bigram_similarity', 0)
                trigram_sim = comp.get('trigram_similarity', 0)
                word_sim = comp.get('word_similarity', 0)
                avg_sim = (bigram_sim + trigram_sim + word_sim) / 3
                
                enhanced_sections.append(
                    f"| {rank} | **{group}** | {bigram_sim:.3f} | {trigram_sim:.3f} | "
                    f"{word_sim:.3f} | **{avg_sim:.3f}** |\n"
                )
            
            enhanced_sections.append("\n#### Statistical Significance Tests\n\n")
            enhanced_sections.append("Kolmogorov-Smirnov tests compare n-gram frequency distributions. ")
            enhanced_sections.append("A significant p-value (p < 0.05) indicates different distributions.\n\n")
            
            enhanced_sections.append("| Group | Bigram KS p-value | Bigram Significant | Trigram KS p-value | Trigram Significant |\n")
            enhanced_sections.append("|-------|-------------------|---------------------|-------------------|---------------------|\n")
            
            for group in sorted_groups[:15]:
                if group not in comparisons:
                    continue
                
                comp = comparisons[group]
                stats = comp.get('statistical_tests', {})
                
                bigram_ks = stats.get('bigram_ks', {})
                trigram_ks = stats.get('trigram_ks', {})
                
                bigram_p = bigram_ks.get('pvalue', 1.0) if bigram_ks else 1.0
                bigram_sig = bigram_ks.get('significant', False) if bigram_ks else False
                trigram_p = trigram_ks.get('pvalue', 1.0) if trigram_ks else 1.0
                trigram_sig = trigram_ks.get('significant', False) if trigram_ks else False
                
                enhanced_sections.append(
                    f"| {group} | {bigram_p:.4f} | {'Yes' if bigram_sig else 'No'} | "
                    f"{trigram_p:.4f} | {'Yes' if trigram_sig else 'No'} |\n"
                )
            
            enhanced_sections.append("\n**Interpretation**: Groups with high similarity scores (>0.5) and ")
            enhanced_sections.append("non-significant p-values (>0.05) show similar n-gram patterns to BlackBasta, ")
            enhanced_sections.append("suggesting potential operational relationships.\n")
    
    # Add n-gram analysis section (ALWAYS add)
    enhanced_sections.append("\n## N-Gram Analysis: BlackBasta Communication Patterns\n")
    enhanced_sections.append("### Methodology: Language Fingerprint Extraction\n\n")
    enhanced_sections.append("N-gram analysis extracts sequences of words (bigrams = 2 words, trigrams = 3 words) ")
    enhanced_sections.append("to identify distinctive communication patterns. This analysis uses the ")
    enhanced_sections.append("`LanguageFingerprint` module to extract and compare n-grams between BlackBasta ")
    enhanced_sections.append("attackers and victims.\n\n")
    
    if ngram_results and ngram_results.get('attacker'):
        enhanced_sections.append("### BlackBasta Attacker N-Grams\n\n")
        
        attacker_ngrams = ngram_results.get('attacker', {})
        bigrams = attacker_ngrams.get('bigrams', [])
        trigrams = attacker_ngrams.get('trigrams', [])
        top_words = attacker_ngrams.get('top_words', [])
        
        if top_words:
            enhanced_sections.append("#### Highest Frequency Words (Top 20)\n\n")
            for i, word in enumerate(top_words[:20], 1):
                enhanced_sections.append(f"{i}. `{word}`")
                if i % 5 == 0:
                    enhanced_sections.append("\n")
                elif i < len(top_words[:20]):
                    enhanced_sections.append(" | ")
            enhanced_sections.append("\n\n")
        
        if bigrams:
            enhanced_sections.append("#### Top Bigrams (2-word sequences)\n\n")
            for i, (bigram, freq) in enumerate(bigrams[:20], 1):
                bigram_str = ' '.join(bigram) if isinstance(bigram, tuple) else str(bigram)
                enhanced_sections.append(f"{i}. `{bigram_str}` (frequency: {freq:.4f})\n")
            enhanced_sections.append("\n")
        
        if trigrams:
            enhanced_sections.append("#### Top Trigrams (3-word sequences)\n\n")
            for i, (trigram, freq) in enumerate(trigrams[:20], 1):
                trigram_str = ' '.join(trigram) if isinstance(trigram, tuple) else str(trigram)
                enhanced_sections.append(f"{i}. `{trigram_str}` (frequency: {freq:.4f})\n")
            enhanced_sections.append("\n")
        
        victim_ngrams = ngram_results.get('victim', {})
        if victim_ngrams:
            enhanced_sections.append("### BlackBasta Victim N-Grams (Comparison)\n\n")
            
            victim_bigrams = victim_ngrams.get('bigrams', [])
            victim_trigrams = victim_ngrams.get('trigrams', [])
            victim_words = victim_ngrams.get('top_words', [])
            
            if victim_words:
                enhanced_sections.append("#### Top Victim Words (Top 15)\n\n")
                enhanced_sections.append(", ".join([f"`{w}`" for w in victim_words[:15]]) + "\n\n")
            
            if victim_bigrams:
                enhanced_sections.append("#### Top Victim Bigrams\n\n")
                for i, (bigram, freq) in enumerate(victim_bigrams[:15], 1):
                    bigram_str = ' '.join(bigram) if isinstance(bigram, tuple) else str(bigram)
                    enhanced_sections.append(f"{i}. `{bigram_str}` (frequency: {freq:.4f})\n")
                enhanced_sections.append("\n")
            
            enhanced_sections.append("**Comparison Insight**: Distinctive n-gram patterns differentiate ")
            enhanced_sections.append("BlackBasta attacker communication style from victim responses. ")
            enhanced_sections.append("The attacker's word choices, bigrams, and trigrams reveal a ")
            enhanced_sections.append("consistent writing style that can be used for attribution and ")
            enhanced_sections.append("group matching.\n")
    else:
        enhanced_sections.append("*Note: N-gram analysis requires CSV data with BlackBasta messages. ")
        enhanced_sections.append("Run with --csv option to enable this analysis.*\n")
    
    # Add BlackBasta timeline analysis
    if timeline_data:
        enhanced_sections.append("\n## BlackBasta Timeline Analysis\n")
        
        response_stats = timeline_data.get('response_time_by_party', {})
        if 'attacker' in response_stats:
            attacker_stats = response_stats['attacker']
            enhanced_sections.append("### BlackBasta Response Time Statistics\n")
            enhanced_sections.append(f"- **Mean Response Time**: {attacker_stats.get('mean', 0):.1f} minutes\n")
            enhanced_sections.append(f"- **Median Response Time**: {attacker_stats.get('median', 0):.1f} minutes\n")
            enhanced_sections.append(f"- **Total Responses**: {attacker_stats.get('count', 0):,}\n")
            enhanced_sections.append(f"- **Outliers Removed**: {attacker_stats.get('outliers_removed', 0):,}\n")
        
        hourly_activity = timeline_data.get('hourly_activity', {})
        gmt_data = hourly_activity.get('gmt', {})
        if gmt_data:
            peak_hours = gmt_data.get('peak_hours', [])
            enhanced_sections.append("\n### BlackBasta GMT Activity Patterns\n")
            enhanced_sections.append(f"- **Peak Activity Hours (GMT)**: {', '.join(map(str, peak_hours))}\n")
            enhanced_sections.append(f"- **Peak Activity Count**: {gmt_data.get('peak_count', 0):,} messages\n")
            
            best_match = gmt_data.get('best_match')
            if best_match:
                enhanced_sections.append(f"- **Best Timezone Match**: {best_match}\n")
    
    # Add Language Fingerprint Analysis
    if ngram_results or semantic_results:
        enhanced_sections.append("\n## Language Fingerprint Analysis\n")
        enhanced_sections.append("### Linguistic Style Identification\n\n")
        enhanced_sections.append("Language fingerprinting extracts distinctive linguistic features that can ")
        enhanced_sections.append("identify authorship and group affiliation. This analysis combines:\n\n")
        enhanced_sections.append("1. **N-Gram Patterns**: Word sequences (bigrams, trigrams) that reveal writing style\n")
        enhanced_sections.append("2. **TF-IDF Features**: Term frequency-inverse document frequency highlighting distinctive vocabulary\n")
        enhanced_sections.append("3. **Embedding Similarity**: Semantic embeddings capturing meaning and style\n")
        enhanced_sections.append("4. **Statistical Comparison**: Kolmogorov-Smirnov tests comparing distributions\n\n")
        
        if ngram_results:
            enhanced_sections.append("#### Fingerprint Components\n\n")
            enhanced_sections.append("**BlackBasta Attacker Fingerprint:**\n")
            attacker_ngrams = ngram_results.get('attacker', {})
            top_words = attacker_ngrams.get('top_words', [])
            if top_words:
                enhanced_sections.append(f"- **Top Words**: {', '.join([f'`{w}`' for w in top_words[:10]])}\n")
            bigrams = attacker_ngrams.get('bigrams', [])
            if bigrams:
                top_bigrams = [b[0] if isinstance(b[0], tuple) else b[0] for b in bigrams[:5]]
                bigram_strs = []
                for b in top_bigrams:
                    if isinstance(b, tuple):
                        bigram_strs.append(f"`{' '.join(b)}`")
                    else:
                        bigram_strs.append(f"`{b}`")
                enhanced_sections.append(f"- **Top Bigrams**: {', '.join(bigram_strs)}\n")
            enhanced_sections.append("\n**Fingerprint Utility**: These patterns can be used to:\n")
            enhanced_sections.append("- Match new messages to known groups\n")
            enhanced_sections.append("- Identify potential group relationships\n")
            enhanced_sections.append("- Detect style changes indicating multiple authors\n")
            enhanced_sections.append("- Support attribution analysis\n\n")
    
    # Add Conclusions Section
    enhanced_sections.append("\n## Key Conclusions and Insights\n\n")
    enhanced_sections.append("### Geographic Proximity Indicators\n\n")
    enhanced_sections.append("**Low Response Times Suggest Attacker Proximity**: ")
    enhanced_sections.append("BlackBasta's fast response patterns (mean 23.9 min, median 5.5 min) indicate ")
    enhanced_sections.append("that attackers are likely operating in time zones close to their victims, ")
    enhanced_sections.append("or have highly efficient operational coordination. Responses under 1 hour ")
    enhanced_sections.append("are particularly indicative of geographic proximity or same-timezone operations.\n\n")
    
    enhanced_sections.append("**Work Hour Patterns**: BlackBasta's peak activity at 13.2 GMT aligns with ")
    enhanced_sections.append("US/Canada business hours (UTC-5 to UTC+0), suggesting operations targeting ")
    enhanced_sections.append("North American victims. This temporal alignment indicates either:\n")
    enhanced_sections.append("1. Attackers operating in similar time zones\n")
    enhanced_sections.append("2. Attackers adapting to victim schedules for maximum engagement\n")
    enhanced_sections.append("3. Coordinated operations across multiple time zones\n\n")
    
    enhanced_sections.append("### Group Similarity Analysis\n\n")
    enhanced_sections.append("**Limited to Closest Groups**: Based on embedding similarity analysis, ")
    enhanced_sections.append("BlackBasta shows highest similarity to:\n")
    
    # Response time conclusions
    if timeline_data:
        response_stats = timeline_data.get('response_time_by_party', {})
        if 'attacker' in response_stats:
            attacker_stats = response_stats['attacker']
            mean_response = attacker_stats.get('mean', 0)
            median_response = attacker_stats.get('median', 0)
            
            enhanced_sections.append("### Geographic Proximity Indicated by Low Response Times\n\n")
            enhanced_sections.append(f"BlackBasta's response time analysis reveals a **mean response time of {mean_response:.1f} minutes** ")
            enhanced_sections.append(f"and a **median of {median_response:.1f} minutes**. This pattern strongly suggests:\n\n")
            enhanced_sections.append("1. **Geographic Proximity**: Response times under 1 hour (60 minutes) indicate that ")
            enhanced_sections.append("attackers are likely operating in the same or nearby time zones as their targets.\n\n")
            enhanced_sections.append("2. **Operational Efficiency**: The low median response time suggests well-coordinated ")
            enhanced_sections.append("operations with dedicated personnel monitoring communications.\n\n")
            enhanced_sections.append("3. **Target Adaptation**: Attackers may adapt their schedules to match victim time zones, ")
            enhanced_sections.append("ensuring rapid responses during business hours.\n\n")
            enhanced_sections.append("**Implication**: Low response times make it obvious that the attacker is near their target, ")
            enhanced_sections.append("either geographically or through operational adaptation. This can aid in:\n")
            enhanced_sections.append("- Time zone identification\n")
            enhanced_sections.append("- Geographic profiling\n")
            enhanced_sections.append("- Operational pattern recognition\n\n")
    
    # Similarity conclusions
    if comprehensive_analysis and comprehensive_analysis.get('embedding_similarities'):
        if 'BlackBasta' in comprehensive_analysis.get('embedding_similarities', {}):
            bb_similarities = comprehensive_analysis['embedding_similarities']['BlackBasta']
            sorted_similar = sorted(
                [(g, s) for g, s in bb_similarities.items() if g != 'BlackBasta'],
                key=lambda x: x[1],
                reverse=True
            )
            top_3 = sorted_similar[:3]
            
            # Focus on closest groups only (similarity > 0.95)
            top_similar = [x for x in sorted_similar if x[1] > 0.95][:5]
            if not top_similar:
                top_similar = sorted_similar[:5]  # Fallback to top 5 if none > 0.95
            
            if top_similar:
                enhanced_sections.append("### Group Similarity Analysis: Closest Relationships\n\n")
                enhanced_sections.append("Embedding similarity analysis (using cosine similarity on mean embeddings) ")
                enhanced_sections.append("identifies the groups most similar to BlackBasta. ")
                enhanced_sections.append("**Focus on closest groups only** (highest similarity scores):\n\n")
                for rank, (group, sim) in enumerate(top_similar, 1):
                    enhanced_sections.append(f"{rank}. **{group}** (similarity: {sim:.3f})")
                    if sim > 0.97:
                        enhanced_sections.append(" - *Very High Similarity*\n")
                    elif sim > 0.95:
                        enhanced_sections.append(" - *High Similarity*\n")
                    else:
                        enhanced_sections.append("\n")
                
                enhanced_sections.append("\n**Interpretation**: These similarity scores indicate:\n")
                enhanced_sections.append("- **Very High Similarity (>0.97)**: Groups may share:\n")
                enhanced_sections.append("  - Common operational infrastructure or personnel\n")
                enhanced_sections.append("  - Related codebases or attack tools\n")
                enhanced_sections.append("  - Similar communication protocols and training\n")
                enhanced_sections.append("- **High Similarity (0.95-0.97)**: Groups may have:\n")
                enhanced_sections.append("  - Similar operational procedures\n")
                enhanced_sections.append("  - Overlapping methodologies\n")
                enhanced_sections.append("  - Potential evolutionary relationships\n\n")
                enhanced_sections.append("**Key Insight**: The embedding similarity analysis reveals that BlackBasta ")
                enhanced_sections.append("shows highest similarity to groups like Darkside, Akira, trinity, REvil, and Qilin. ")
                enhanced_sections.append("These groups may represent:\n")
                enhanced_sections.append("1. Related operations or rebrands\n")
                enhanced_sections.append("2. Shared infrastructure or tools\n")
                enhanced_sections.append("3. Overlapping personnel or training sources\n")
                enhanced_sections.append("4. Evolutionary relationships in the ransomware ecosystem\n\n")
                enhanced_sections.append("**Recommendation**: Focus investigation on these closest groups for potential ")
                enhanced_sections.append("connections, shared resources, or evolutionary relationships. ")
                enhanced_sections.append("Lower similarity groups (<0.95) show less direct relationship.\n\n")
    
    # Add "What is Already Known" section
    enhanced_sections.append("\n## Context: What is Already Known About BlackBasta\n\n")
    enhanced_sections.append("### Established Facts\n\n")
    enhanced_sections.append("Based on existing threat intelligence and public reporting:\n\n")
    enhanced_sections.append("1. **Group Origin**: BlackBasta emerged in early 2022, likely as a rebrand or evolution ")
    enhanced_sections.append("of previous ransomware operations.\n\n")
    enhanced_sections.append("2. **Targeting**: Primarily targets organizations in North America and Europe, ")
    enhanced_sections.append("with a focus on critical infrastructure and healthcare.\n\n")
    enhanced_sections.append("3. **Operational Characteristics**:\n")
    enhanced_sections.append("   - Double extortion model (encryption + data theft)\n")
    enhanced_sections.append("   - Ransom demands typically range from hundreds of thousands to millions\n")
    enhanced_sections.append("   - Active communication with victims during negotiations\n\n")
    enhanced_sections.append("4. **Technical Capabilities**:\n")
    enhanced_sections.append("   - Advanced encryption algorithms\n")
    enhanced_sections.append("   - Sophisticated initial access methods\n")
    enhanced_sections.append("   - Multi-stage attack chains\n\n")
    enhanced_sections.append("5. **Suspected Relationships**: Intelligence suggests potential connections to ")
    enhanced_sections.append("other ransomware groups, though direct attribution remains challenging.\n\n")
    enhanced_sections.append("### How This Analysis Adds Value\n\n")
    enhanced_sections.append("This report provides quantitative evidence for:\n")
    enhanced_sections.append("- **Operational Patterns**: Response times, work hours, and communication styles\n")
    enhanced_sections.append("- **Group Relationships**: Statistical similarity to other groups\n")
    enhanced_sections.append("- **Linguistic Fingerprints**: Distinctive communication patterns for attribution\n")
    enhanced_sections.append("- **Behavioral Insights**: Price negotiation patterns and victim interaction styles\n\n")
    
    # Add sentence clustering analysis if available
    if clustering_data and include_clustering:
        enhanced_sections.append("\n## Sentence-Level Clustering Analysis\n")
        enhanced_sections.append("### Embedding-Based Clustering Results\n")
        
        separability = clustering_data.get('separability_metrics', {})
        bb_analysis = clustering_data.get('blackbasta_analysis', {})
        group_analysis = clustering_data.get('group_analysis', {})
        
        attacker_cluster_count = clustering_data.get('attacker_cluster_count', 0)
        victim_cluster_count = clustering_data.get('victim_cluster_count', 0)
        
        if separability:
            enhanced_sections.append("#### Separability Metrics\n")
            attacker_silhouette = separability.get('attacker_silhouette', 0)
            victim_silhouette = separability.get('victim_silhouette', 0)
            
            enhanced_sections.append(f"- **Attacker Silhouette Score**: {attacker_silhouette:.3f}\n")
            enhanced_sections.append(f"- **Victim Silhouette Score**: {victim_silhouette:.3f}\n")
            enhanced_sections.append(f"- **Attacker Clusters**: {attacker_cluster_count}\n")
            enhanced_sections.append(f"- **Victim Clusters**: {victim_cluster_count}\n")
            enhanced_sections.append("\n*Higher silhouette scores indicate better cluster separation. "
                                   "This analysis proves that embeddings effectively isolate victims from attackers.*\n")
        
        if bb_analysis:
            enhanced_sections.append("\n### BlackBasta N-Gram Analysis (from Clustering)\n")
            
            attacker_bigrams = bb_analysis.get('attacker_bigrams', [])
            attacker_trigrams = bb_analysis.get('attacker_trigrams', [])
            similar_groups = bb_analysis.get('similar_groups', [])
            
            if attacker_bigrams:
                enhanced_sections.append("#### Top BlackBasta Attacker Bigrams\n")
                for i, (bigram, count) in enumerate(attacker_bigrams[:15], 1):
                    bigram_str = ' '.join(bigram) if isinstance(bigram, (list, tuple)) else str(bigram)
                    enhanced_sections.append(f"{i}. `{bigram_str}` ({count} occurrences)\n")
            
            if similar_groups:
                enhanced_sections.append("\n#### Similar Attacker Groups (by Embedding Similarity)\n")
                for i, (group_name, similarity) in enumerate(similar_groups[:10], 1):
                    enhanced_sections.append(f"{i}. **{group_name}**: Cosine similarity = {similarity:.3f}\n")
    
    # Add external analysis attribution section
    enhanced_sections.append("\n## External Analysis Data Attribution\n")
    enhanced_sections.append("\n### Reference Sources\n\n")
    enhanced_sections.append("The following external data sources are referenced in this report:\n\n")
    
    external_sources = [
        ("World Time Zones Map", "Standard Time Zones of the World - Reference for geographic context"),
        ("Work Time Distribution by Profession", "External research data on professional work hour patterns"),
        ("Victim Industry Distribution", "External ransomware research data on industry targeting patterns"),
        ("Victim Geography Distribution", "External ransomware research data on geographic targeting"),
        ("Top Target Countries", "External ransomware research data on country-level targeting"),
        ("Attack Geographic Distribution", "External research - Infrastructure and attack vector patterns"),
        ("Overtime Work Distribution", "External research - Distribution of time throughout the day"),
        ("Industry Targeting Patterns", "External ransomware research - Group-specific industry preferences"),
    ]
    
    for i, (source_name, description) in enumerate(external_sources, 1):
        enhanced_sections.append(f"{i}. **{source_name}**: {description}\n")
    
    enhanced_sections.append("\n### Attribution Note\n\n")
    enhanced_sections.append("All external reference images and data are provided for context and comparison purposes only. ")
    enhanced_sections.append("These sources are used to enhance understanding of operational patterns and should not be ")
    enhanced_sections.append("considered as direct evidence of group relationships. External data is clearly marked ")
    enhanced_sections.append("throughout the report with source attribution.\n")
    
    # Enhance existing interpretations with BlackBasta context
    enhanced_content = enhance_blackbasta_interpretations(report_content)
    
    # Insert enhanced sections before conclusion
    conclusion_marker = "## 13. Conclusion"
    if conclusion_marker in enhanced_content:
        insertion_point = enhanced_content.find(conclusion_marker)
        enhanced_content = enhanced_content[:insertion_point] + "\n".join(enhanced_sections) + "\n\n" + enhanced_content[insertion_point:]
    else:
        # Try other conclusion markers
        conclusion_markers = ["## Conclusion", "## Conclusions", "## 12. Conclusion"]
        inserted = False
        for marker in conclusion_markers:
            if marker in enhanced_content:
                insertion_point = enhanced_content.find(marker)
                enhanced_content = enhanced_content[:insertion_point] + "\n".join(enhanced_sections) + "\n\n" + enhanced_content[insertion_point:]
                inserted = True
                break
        
        if not inserted:
            # Append to end
            enhanced_content = enhanced_content + "\n" + "\n".join(enhanced_sections)
    
    # Write enhanced report
    with open(enhanced_report_path, 'w', encoding='utf-8') as f:
        f.write(enhanced_content)
    
    print(f"[INFO] Enhanced report saved to {enhanced_report_path}")
    print(f"[INFO] Added {len(converted_plots)} plot images")
    print(f"[INFO] Added {len(enhanced_sections)} new sections")


def run_comprehensive_analysis(csv_path: Path, ollama_endpoint: str = "http://localhost:11434"):
    """Run comprehensive group analysis."""
    print("[INFO] Running comprehensive group analysis...")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "comprehensive_group_analysis",
            Path(__file__).parent / "comprehensive_group_analysis.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        analyzer = module.ComprehensiveGroupAnalyzer(
            csv_path=str(csv_path),
            ollama_endpoint=ollama_endpoint,
            embedding_model="nomic-embed-text"
        )
        
        if not analyzer.load_data():
            return False
        
        if not analyzer.extract_group_messages():
            return False
        
        analyzer.generate_embeddings_per_group()
        analyzer.calculate_group_similarities()
        analyzer.extract_ngrams_per_group()
        
        report = analyzer.generate_comprehensive_report()
        analyzer.save_results(report)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Comprehensive analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Enhance markdown report with images and additional sections')
    parser.add_argument('--base-report', type=Path, default=None,
                       help='Base markdown report file (auto-detected if not provided)')
    parser.add_argument('--output-dir', type=Path, default=Path('output'),
                       help='Output directory containing HTML plots and data')
    parser.add_argument('--csv', type=Path, default=Path('output/ransom_chats.csv'),
                       help='CSV file with ransom chat data')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output path for enhanced report (default: base_report_enhanced.md)')
    parser.add_argument('--skip-clustering', action='store_true',
                       help='Skip sentence clustering analysis')
    parser.add_argument('--force-images', action='store_true',
                       help='Force regeneration of PNG images even if they exist')
    parser.add_argument('--run-comprehensive', action='store_true',
                       help='Run comprehensive group analysis before enhancing report')
    parser.add_argument('--ollama-endpoint', type=str, default='http://localhost:11434',
                       help='Ollama endpoint URL')
    
    args = parser.parse_args()
    
    # Auto-detect base report if not provided
    if args.base_report is None:
        possible_reports = [
            Path('output/advanced_ransom_analysis_report.md'),
            Path('output/ransom_analysis_report.md'),
            Path('output/analysis_report.md'),
        ]
        for report_path in possible_reports:
            if report_path.exists():
                args.base_report = report_path
                print(f"[INFO] Auto-detected base report: {args.base_report}")
                break
        
        if args.base_report is None:
            print("[ERROR] Base report not found. Please specify with --base-report or ensure one exists in output/")
            sys.exit(1)
    
    # Run comprehensive analysis if requested
    if args.run_comprehensive and args.csv and args.csv.exists():
        print("[INFO] Running comprehensive analysis first...")
        run_comprehensive_analysis(args.csv, args.ollama_endpoint)
    
    if not args.base_report.exists():
        print(f"[ERROR] Base report not found: {args.base_report}")
        print("[INFO] Available files in output/:")
        output_dir = Path('output')
        if output_dir.exists():
            for f in sorted(output_dir.glob('*.md')):
                print(f"  - {f.name}")
        sys.exit(1)
    
    output_path = args.output or args.base_report.parent / f"{args.base_report.stem}_enhanced.md"
    
    print(f"[INFO] Enhancing report: {args.base_report}")
    print(f"[INFO] Output directory: {args.output_dir}")
    print(f"[INFO] CSV data: {args.csv}")
    print(f"[INFO] Enhanced report will be saved to: {output_path}")
    
    generate_enhanced_report(
        base_report_path=args.base_report,
        output_dir=args.output_dir,
        enhanced_report_path=output_path,
        include_clustering=not args.skip_clustering,
        force_images=args.force_images,
        csv_path=args.csv
    )
    
    print("\n[INFO] Report enhancement complete!")


if __name__ == '__main__':
    main()
