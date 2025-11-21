#!/usr/bin/env python3
"""
Create comprehensive visualizations: UMAP 2D, clustering, box plots for prices and response times.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Plotting
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("[WARNING] plotly not installed")

# UMAP
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[WARNING] umap-learn not installed")

# ML
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_comprehensive_analysis(output_dir: Path) -> dict:
    """Load comprehensive analysis results."""
    analysis_file = output_dir / 'comprehensive_group_analysis.json'
    if not analysis_file.exists():
        print(f"[ERROR] Comprehensive analysis file not found: {analysis_file}")
        return {}
    
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load comprehensive analysis: {e}")
        return {}


def create_umap_2d_visualization(analysis_data: dict, output_dir: Path, plots_dir: Path) -> Path:
    """Create 2D UMAP visualization of group embeddings."""
    if not HAS_UMAP or not HAS_PLOTLY:
        print("[WARNING] UMAP or Plotly not available, skipping 2D visualization")
        return None
    
    print("[INFO] Creating 2D UMAP visualization...")
    
    try:
        # Load embeddings from analysis
        # We need to regenerate embeddings or load them from a saved file
        # For now, we'll create a visualization based on group centroids
        
        # Check if we have embedding data
        if 'group_embeddings' not in analysis_data or not analysis_data['group_embeddings']:
            print("[WARNING] No embedding data available for UMAP visualization")
            return None
        
        # Get group names and their embeddings
        groups = []
        embeddings_list = []
        
        for group_name, emb_data in analysis_data['group_embeddings'].items():
            if isinstance(emb_data, dict) and 'embeddings' in emb_data:
                emb_array = np.array(emb_data['embeddings'])
                if len(emb_array) > 0:
                    # Use mean embedding for each group
                    mean_emb = np.mean(emb_array, axis=0)
                    groups.append(group_name)
                    embeddings_list.append(mean_emb)
        
        if len(embeddings_list) < 3:
            print("[WARNING] Not enough groups for UMAP visualization")
            return None
        
        embeddings_array = np.array(embeddings_list)
        
        # Create 2D UMAP
        umap_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(5, len(groups)-1))
        coords_2d = umap_2d.fit_transform(embeddings_array)
        
        # Get clustering assignments if available
        cluster_assignments = analysis_data.get('clustering_results', {}).get('group_to_cluster', {})
        colors = []
        for group in groups:
            cluster_id = cluster_assignments.get(group, 0)
            colors.append(cluster_id)
        
        # Create plot
        fig = go.Figure()
        
        # Color by cluster
        unique_clusters = sorted(set(colors))
        color_palette = px.colors.qualitative.Set3
        
        for cluster_id in unique_clusters:
            cluster_groups = [g for g, c in zip(groups, colors) if c == cluster_id]
            cluster_indices = [i for i, c in enumerate(colors) if c == cluster_id]
            
            x_coords = [coords_2d[i, 0] for i in cluster_indices]
            y_coords = [coords_2d[i, 1] for i in cluster_indices]
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers+text',
                name=f'Cluster {cluster_id}',
                text=cluster_groups,
                textposition='top center',
                marker=dict(
                    size=15,
                    color=color_palette[cluster_id % len(color_palette)],
                    opacity=0.7,
                    line=dict(width=2, color='black')
                ),
                hovertemplate='<b>%{text}</b><br>Cluster: ' + str(cluster_id) + '<extra></extra>'
            ))
        
        # Highlight BlackBasta
        if 'BlackBasta' in groups:
            bb_idx = groups.index('BlackBasta')
            fig.add_trace(go.Scatter(
                x=[coords_2d[bb_idx, 0]],
                y=[coords_2d[bb_idx, 1]],
                mode='markers',
                name='BlackBasta (Highlight)',
                marker=dict(
                    size=25,
                    color='red',
                    symbol='star',
                    line=dict(width=3, color='darkred')
                ),
                showlegend=True
            ))
        
        fig.update_layout(
            title='2D UMAP Projection: Group Embedding Similarity',
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            height=800,
            hovermode='closest'
        )
        
        # Save as PNG
        png_path = plots_dir / 'umap_2d_groups.png'
        try:
            import plotly.io as pio
            img_bytes = pio.to_image(fig, format='png', width=1200, height=800)
            with open(png_path, 'wb') as f:
                f.write(img_bytes)
            print(f"[INFO] Created {png_path.name}")
            return png_path
        except Exception as e:
            print(f"[WARNING] Could not export to PNG: {e}")
            html_path = plots_dir / 'umap_2d_groups.html'
            fig.write_html(str(html_path))
            return html_path
            
    except Exception as e:
        print(f"[ERROR] Failed to create UMAP visualization: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_price_boxplots(csv_path: Path, output_dir: Path, plots_dir: Path) -> dict:
    """Create box plots for price distribution per group."""
    if not HAS_PLOTLY:
        return {}
    
    print("[INFO] Creating price distribution box plots...")
    
    try:
        # Read CSV with error handling
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin-1')
        
        # Filter valid prices
        if 'price_amount' not in df.columns:
            print("[WARNING] price_amount column not found")
            return {}
        
        df = df[df['price_amount'].notna() & (df['price_amount'] > 0)]
        
        if len(df) == 0:
            print("[WARNING] No valid price data found")
            return {}
        
        # Filter by group_name
        if 'group_name' not in df.columns:
            print("[WARNING] group_name column not found")
            return {}
        
        # Get groups with price data
        groups_with_prices = []
        groups = df['group_name'].dropna().unique()
        groups = sorted([g for g in groups if str(g).strip()])
        
        for group in groups:
            group_data = df[df['group_name'] == group]['price_amount']
            if len(group_data) > 0:
                groups_with_prices.append((group, group_data))
        
        if len(groups_with_prices) == 0:
            print("[WARNING] No groups with price data found")
            return {}
        
        results = {}
        
        # If more than 12 groups, split into 4 panels (2x2)
        if len(groups_with_prices) > 12:
            print(f"[INFO] Splitting {len(groups_with_prices)} groups into 4 panels (2x2)...")
            groups_per_panel = (len(groups_with_prices) + 3) // 4  # Round up
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Panel 1', 'Panel 2', 'Panel 3', 'Panel 4'),
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            for panel_idx in range(4):
                row = (panel_idx // 2) + 1
                col = (panel_idx % 2) + 1
                
                start_idx = panel_idx * groups_per_panel
                end_idx = min(start_idx + groups_per_panel, len(groups_with_prices))
                panel_groups = groups_with_prices[start_idx:end_idx]
                
                for group, group_data in panel_groups:
                    fig.add_trace(
                        go.Box(
                            y=group_data,
                            name=str(group),
                            boxmean='sd',
                            showlegend=False
                        ),
                        row=row, col=col
                    )
            
            fig.update_layout(
                title='Price Distribution by Group (Box Plots - Boîte à Moustache - 4 Panels)',
                height=1200,
                showlegend=False
            )
            
            fig.update_yaxes(title_text="Price Amount (USD)", row=1, col=1)
            fig.update_yaxes(title_text="Price Amount (USD)", row=1, col=2)
            fig.update_yaxes(title_text="Price Amount (USD)", row=2, col=1)
            fig.update_yaxes(title_text="Price Amount (USD)", row=2, col=2)
            
        else:
            # Single panel for <= 12 groups
            fig = go.Figure()
            
            for group, group_data in groups_with_prices:
                fig.add_trace(go.Box(
                    y=group_data,
                    name=str(group),
                    boxmean='sd'
                ))
            
            fig.update_layout(
                title='Price Distribution by Group (Box Plot - Boîte à Moustache)',
                yaxis_title='Price Amount (USD)',
                xaxis_title='Group',
                height=600,
                showlegend=False
            )
        
        # Save
        png_path = plots_dir / 'price_distribution_boxplot.png'
        try:
            import plotly.io as pio
            img_bytes = pio.to_image(fig, format='png', width=1400, height=600)
            with open(png_path, 'wb') as f:
                f.write(img_bytes)
            print(f"[INFO] Created {png_path.name}")
        except Exception as e:
            print(f"[WARNING] Could not export to PNG: {e}")
            html_path = plots_dir / 'price_distribution_boxplot.html'
            fig.write_html(str(html_path))
            png_path = html_path
        
        # BlackBasta-specific box plot
        bb_df = df[df['group_name'].str.contains('BlackBasta', case=False, na=False)]
        if len(bb_df) > 0:
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Box(
                y=bb_df['price_amount'],
                name='BlackBasta',
                boxmean='sd',
                marker_color='rgba(239, 68, 68, 0.7)'
            ))
            
            fig_bb.update_layout(
                title='BlackBasta Price Distribution (Box Plot)',
                yaxis_title='Price Amount (USD)',
                height=500,
                showlegend=False
            )
            
            bb_png_path = plots_dir / 'blackbasta_prices_boxplot.png'
            try:
                import plotly.io as pio
                img_bytes = pio.to_image(fig_bb, format='png', width=1000, height=500)
                with open(bb_png_path, 'wb') as f:
                    f.write(img_bytes)
                print(f"[INFO] Created {bb_png_path.name}")
            except Exception as e:
                html_path = plots_dir / 'blackbasta_prices_boxplot.html'
                fig_bb.write_html(str(html_path))
                bb_png_path = html_path
            
            return {'all_groups': png_path, 'blackbasta': bb_png_path}
        
        return {'all_groups': png_path}
        
    except Exception as e:
        print(f"[ERROR] Failed to create price box plots: {e}")
        import traceback
        traceback.print_exc()
        return {}


def create_response_time_distributions(csv_path: Path, output_dir: Path, plots_dir: Path) -> dict:
    """Create response time distribution plots, focusing on BlackBasta with 3-hour max."""
    if not HAS_PLOTLY:
        return {}
    
    print("[INFO] Creating response time distribution plots...")
    
    try:
        # Read CSV with error handling
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin-1')
        
        if 'response_time_minutes' not in df.columns:
            print("[WARNING] response_time_minutes column not found")
            return {}
        
        # Filter valid response times
        df = df[df['response_time_minutes'].notna() & (df['response_time_minutes'] >= 0)]
        
        # Filter to 3 hours max (180 minutes)
        df_3h = df[df['response_time_minutes'] <= 180].copy()
        
        if len(df_3h) == 0:
            print("[WARNING] No valid response time data found")
            return {}
        
        results = {}
        
        # BlackBasta-specific histogram (3 hours max)
        bb_df = df_3h[df_3h['group_name'].str.contains('BlackBasta', case=False, na=False)]
        if len(bb_df) > 0:
            fig_bb = go.Figure()
            
            # Create histogram with 60 bins
            fig_bb.add_trace(go.Histogram(
                x=bb_df['response_time_minutes'],
                nbinsx=60,
                name='BlackBasta',
                marker_color='rgba(239, 68, 68, 0.7)',
                opacity=0.7
            ))
            
            fig_bb.update_layout(
                title='BlackBasta Response Time Distribution (0-3 hours, 60 bins)',
                xaxis_title='Response Time (minutes)',
                yaxis_title='Frequency',
                height=500,
                bargap=0.1
            )
            
            bb_png_path = plots_dir / 'blackbasta_response_times_3h.png'
            try:
                import plotly.io as pio
                img_bytes = pio.to_image(fig_bb, format='png', width=1200, height=500)
                with open(bb_png_path, 'wb') as f:
                    f.write(img_bytes)
                print(f"[INFO] Created {bb_png_path.name}")
                results['blackbasta_3h'] = bb_png_path
            except Exception as e:
                html_path = plots_dir / 'blackbasta_response_times_3h.html'
                fig_bb.write_html(str(html_path))
                results['blackbasta_3h'] = html_path
        
        # Comparison: BlackBasta vs other groups
        if 'group_name' in df_3h.columns:
            # Get top groups by message count
            top_groups = df_3h['group_name'].value_counts().head(6).index.tolist()
            
            if 'BlackBasta' in top_groups or any('BlackBasta' in str(g) for g in top_groups):
                fig_comp = go.Figure()
                
                # Add BlackBasta
                bb_data = df_3h[df_3h['group_name'].str.contains('BlackBasta', case=False, na=False)]['response_time_minutes']
                if len(bb_data) > 0:
                    fig_comp.add_trace(go.Histogram(
                        x=bb_data,
                        nbinsx=60,
                        name='BlackBasta',
                        marker_color='rgba(239, 68, 68, 0.7)',
                        opacity=0.7
                    ))
                
                # Add other groups
                colors = px.colors.qualitative.Set3
                for i, group in enumerate(top_groups[:5]):
                    if 'BlackBasta' not in str(group):
                        group_data = df_3h[df_3h['group_name'] == group]['response_time_minutes']
                        if len(group_data) > 0:
                            fig_comp.add_trace(go.Histogram(
                                x=group_data,
                                nbinsx=60,
                                name=str(group),
                                marker_color=colors[i % len(colors)],
                                opacity=0.6
                            ))
                
                fig_comp.update_layout(
                    title='Response Time Distribution Comparison (0-3 hours, 60 bins)',
                    xaxis_title='Response Time (minutes)',
                    yaxis_title='Frequency',
                    height=600,
                    barmode='overlay',
                    bargap=0.1
                )
                
                comp_png_path = plots_dir / 'response_time_comparison_3h.png'
                try:
                    import plotly.io as pio
                    img_bytes = pio.to_image(fig_comp, format='png', width=1400, height=600)
                    with open(comp_png_path, 'wb') as f:
                        f.write(img_bytes)
                    print(f"[INFO] Created {comp_png_path.name}")
                    results['comparison_3h'] = comp_png_path
                except Exception as e:
                    html_path = plots_dir / 'response_time_comparison_3h.html'
                    fig_comp.write_html(str(html_path))
                    results['comparison_3h'] = html_path
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Failed to create response time plots: {e}")
        import traceback
        traceback.print_exc()
        return {}


def create_work_hour_distributions(csv_path: Path, output_dir: Path, plots_dir: Path) -> dict:
    """Create work hour distribution plots, ensuring first graph is displayed."""
    if not HAS_PLOTLY:
        return {}
    
    print("[INFO] Creating work hour distribution plots...")
    
    try:
        # Read CSV with error handling
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin-1')
        
        if 'timestamp' not in df.columns:
            print("[WARNING] timestamp column not found")
            return {}
        
        # Parse timestamps and extract hour (GMT)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df[df['timestamp'].notna()]
        df['hour_gmt'] = df['timestamp'].dt.hour
        
        if len(df) == 0:
            print("[WARNING] No valid timestamp data found")
            return {}
        
        results = {}
        
        # Overall distribution (first graph)
        fig_overall = go.Figure()
        fig_overall.add_trace(go.Histogram(
            x=df['hour_gmt'],
            nbinsx=24,
            name='All Groups',
            marker_color='rgba(59, 130, 246, 0.7)',
            opacity=0.7
        ))
        
        fig_overall.update_layout(
            title='Overall Hourly Activity Distribution (GMT) - All Groups',
            xaxis_title='Hour of Day (GMT)',
            yaxis_title='Message Count',
            height=500,
            bargap=0.1
        )
        
        overall_png_path = plots_dir / 'work_hours_overall.png'
        try:
            import plotly.io as pio
            img_bytes = pio.to_image(fig_overall, format='png', width=1200, height=500)
            with open(overall_png_path, 'wb') as f:
                f.write(img_bytes)
            print(f"[INFO] Created {overall_png_path.name}")
            results['overall'] = overall_png_path
        except Exception as e:
            html_path = plots_dir / 'work_hours_overall.html'
            fig_overall.write_html(str(html_path))
            results['overall'] = html_path
        
        # BlackBasta-specific
        if 'group_name' in df.columns:
            bb_df = df[df['group_name'].str.contains('BlackBasta', case=False, na=False)]
            if len(bb_df) > 0:
                fig_bb = go.Figure()
                fig_bb.add_trace(go.Histogram(
                    x=bb_df['hour_gmt'],
                    nbinsx=24,
                    name='BlackBasta',
                    marker_color='rgba(239, 68, 68, 0.7)',
                    opacity=0.7
                ))
                
                fig_bb.update_layout(
                    title='BlackBasta Hourly Activity Distribution (GMT)',
                    xaxis_title='Hour of Day (GMT)',
                    yaxis_title='Message Count',
                    height=500,
                    bargap=0.1
                )
                
                bb_png_path = plots_dir / 'blackbasta_work_hours.png'
                try:
                    import plotly.io as pio
                    img_bytes = pio.to_image(fig_bb, format='png', width=1200, height=500)
                    with open(bb_png_path, 'wb') as f:
                        f.write(img_bytes)
                    print(f"[INFO] Created {bb_png_path.name}")
                    results['blackbasta'] = bb_png_path
                except Exception as e:
                    html_path = plots_dir / 'blackbasta_work_hours.html'
                    fig_bb.write_html(str(html_path))
                    results['blackbasta'] = html_path
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Failed to create work hour plots: {e}")
        import traceback
        traceback.print_exc()
        return {}


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create comprehensive visualizations')
    parser.add_argument('--csv', type=Path, default=Path('output/ransom_chats.csv'),
                       help='Path to CSV file')
    parser.add_argument('--output-dir', type=Path, default=Path('output'),
                       help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    plots_dir = output_dir / 'report_plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Load comprehensive analysis
    analysis_data = load_comprehensive_analysis(output_dir)
    
    # Create visualizations
    print("[INFO] Creating comprehensive visualizations...")
    
    # 1. UMAP 2D
    umap_path = create_umap_2d_visualization(analysis_data, output_dir, plots_dir)
    
    # 2. Price box plots
    price_plots = create_price_boxplots(args.csv, output_dir, plots_dir)
    
    # 3. Response time distributions
    response_plots = create_response_time_distributions(args.csv, output_dir, plots_dir)
    
    # 4. Work hour distributions
    work_hour_plots = create_work_hour_distributions(args.csv, output_dir, plots_dir)
    
    print("\n[SUCCESS] Visualization creation complete!")
    print(f"[INFO] Created {len([p for p in [umap_path] + list(price_plots.values()) + list(response_plots.values()) + list(work_hour_plots.values()) if p])} plots")


if __name__ == '__main__':
    main()

