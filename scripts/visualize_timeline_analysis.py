#!/usr/bin/env python3
"""Generate visualizations from timeline analysis JSON."""

import json
import sys
from pathlib import Path
import argparse

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    import numpy as np
    from scipy.stats import norm
except ImportError:
    print("[ERROR] plotly, numpy, and scipy are required. Install with: pip install plotly numpy scipy")
    sys.exit(1)


def load_analysis(json_path: Path) -> dict:
    """Load analysis JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_response_time_histogram(data: dict, output_path: Path):
    """Create histogram of response times by party (outliers removed, 0-5 hours, 60 bins)."""
    response_stats = data.get('response_time_by_party', {})
    
    if not response_stats:
        print("[WARNING] No response time data found")
        return
    
    fig = go.Figure()
    
    colors = {'attacker': 'rgba(239, 68, 68, 0.7)', 'victim': 'rgba(59, 130, 246, 0.7)'}
    
    for party, stats in response_stats.items():
        values = stats.get('values', [])  # Already filtered (outliers removed)
        if not values:
            continue
        
        outliers_removed = stats.get('outliers_removed', 0)
        
        # Convert to hours and filter to 0-5 hours range
        values_hours = [v / 60.0 for v in values if 0 <= v / 60.0 <= 5]
        
        if not values_hours:
            continue
        
        fig.add_trace(go.Histogram(
            x=values_hours,
            name=f"{party.capitalize()} (outliers removed: {outliers_removed})",
            opacity=0.7,
            marker_color=colors.get(party, 'rgba(128, 128, 128, 0.7)'),
            nbinsx=60,
            xbins=dict(start=0, end=5, size=5/60)  # 60 bins from 0 to 5 hours
        ))
    
    # Calculate statistics for subtitle
    total_outliers = sum(stats.get('outliers_removed', 0) for stats in response_stats.values())
    total_original = sum(stats.get('count_original', 0) for stats in response_stats.values())
    
    fig.update_layout(
        title={
            'text': 'Response Time Distribution: Attacker vs Victim<br><sub>0-5 Hour Window | 60 Bins | Z-Score Outlier Removal (σ=3.0) | ' +
                    f'{total_outliers} outliers removed from {total_original} total responses</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title='Response Time (hours)',
        yaxis_title='Message Frequency',
        barmode='overlay',
        height=550,
        showlegend=True,
        xaxis=dict(range=[0, 5], dtick=0.5),
        yaxis=dict(title_font={'size': 12}),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=11)
    )
    
    html_path = output_path.parent / f"{output_path.stem}_response_times.html"
    fig.write_html(str(html_path))
    print(f"[INFO] Response time histogram saved to {html_path}")


def create_timestamp_distribution(data: dict, output_path: Path):
    """Create histogram of messages over time."""
    timestamp_data = data.get('timestamp_distribution', {})
    
    if not timestamp_data or not timestamp_data.get('counts'):
        print("[WARNING] No timestamp distribution data found")
        return
    
    bin_labels = timestamp_data.get('bin_labels', [])
    counts = timestamp_data.get('counts', [])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=bin_labels,
        y=counts,
        marker_color='rgba(99, 102, 241, 0.7)',
        marker_line_color='rgba(99, 102, 241, 1)',
        marker_line_width=1
    ))
    
    outliers_removed = timestamp_data.get('outliers_removed', 0)
    total_original = timestamp_data.get('total_original', 0)
    
    fig.update_layout(
        title={
            'text': f'Message Distribution Over Time<br><sub>30 Bins | Temporal Analysis | ' +
                    f'{outliers_removed} timestamp outliers removed from {total_original} total | Null timestamps excluded</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title='Time Period',
        yaxis_title='Number of Messages',
        height=550,
        xaxis=dict(tickangle=-45, title_font={'size': 12}),
        yaxis=dict(title_font={'size': 12}),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=11)
    )
    
    html_path = output_path.parent / f"{output_path.stem}_timestamp_distribution.html"
    fig.write_html(str(html_path))
    print(f"[INFO] Timestamp distribution saved to {html_path}")


def create_hourly_activity_chart(data: dict, output_path: Path):
    """Create hourly activity chart with attacker vs victim breakdown and group filtering."""
    hourly_data = data.get('hourly_activity', {})
    
    if not hourly_data or not hourly_data.get('hours'):
        print("[WARNING] No hourly activity data found")
        return
    
    # Create comprehensive chart with multiple subplots
    from plotly.subplots import make_subplots
    
    # Create 4-row layout: Total Local, Total GMT, Attacker vs Victim Local, Attacker vs Victim GMT
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'Total Activity: Local Time',
            'Total Activity: GMT (UTC)',
            'Attacker vs Victim: Local Time',
            'Attacker vs Victim: GMT (UTC)'
        ),
        vertical_spacing=0.12,
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )
    
    # Overall activity (local)
    hours = hourly_data.get('hours', [])
    counts = hourly_data.get('counts', [])
    
    fig.add_trace(go.Bar(
        x=hours,
        y=counts,
        name='Total (Local)',
        marker_color='rgba(99, 102, 241, 0.7)',
        showlegend=False
    ), row=1, col=1)
    
    # GMT activity
    gmt_data = hourly_data.get('gmt', {})
    if gmt_data:
        gmt_hours = gmt_data.get('hours', [])
        gmt_counts = gmt_data.get('counts', [])
        
        fig.add_trace(go.Bar(
            x=gmt_hours,
            y=gmt_counts,
            name='Total (GMT)',
            marker_color='rgba(239, 68, 68, 0.7)',
            showlegend=False
        ), row=2, col=1)
    
    # Attacker vs Victim breakdown (Local)
    party_hourly = hourly_data.get('by_party', {})
    attacker_hours = []
    attacker_counts = []
    victim_hours = []
    victim_counts = []
    
    # Aggregate all attacker parties
    attacker_parties = []
    victim_parties = []
    
    for party, party_data in party_hourly.items():
        party_lower = party.lower().strip()
        if 'victim' in party_lower or 'client' in party_lower:
            victim_parties.append(party)
        else:
            attacker_parties.append(party)
    
    # Aggregate attacker activity
    attacker_hourly_combined = {}
    for party in attacker_parties:
        if party in party_hourly:
            party_hours = party_hourly[party].get('hours', [])
            party_counts = party_hourly[party].get('counts', [])
            for h, c in zip(party_hours, party_counts):
                attacker_hourly_combined[h] = attacker_hourly_combined.get(h, 0) + c
    
    # Aggregate victim activity
    victim_hourly_combined = {}
    for party in victim_parties:
        if party in party_hourly:
            party_hours = party_hourly[party].get('hours', [])
            party_counts = party_hourly[party].get('counts', [])
            for h, c in zip(party_hours, party_counts):
                victim_hourly_combined[h] = victim_hourly_combined.get(h, 0) + c
    
    # Convert to sorted lists
    attacker_hours = sorted(attacker_hourly_combined.keys())
    attacker_counts = [attacker_hourly_combined[h] for h in attacker_hours]
    
    victim_hours = sorted(victim_hourly_combined.keys())
    victim_counts = [victim_hourly_combined[h] for h in victim_hours]
    
    # Add attacker vs victim traces (Local)
    if attacker_hours:
        fig.add_trace(go.Bar(
            x=attacker_hours,
            y=attacker_counts,
            name='Attacker',
            marker_color='rgba(239, 68, 68, 0.8)',
            legendgroup='attacker'
        ), row=3, col=1)
    
    if victim_hours:
        fig.add_trace(go.Bar(
            x=victim_hours,
            y=victim_counts,
            name='Victim',
            marker_color='rgba(34, 197, 94, 0.8)',
            legendgroup='victim'
        ), row=3, col=1)
    
    # Attacker vs Victim breakdown (GMT)
    gmt_party_hourly = gmt_data.get('by_party', {}) if gmt_data else {}
    
    gmt_attacker_hourly_combined = {}
    for party in attacker_parties:
        if party in gmt_party_hourly:
            party_hours = gmt_party_hourly[party].get('hours', [])
            party_counts = gmt_party_hourly[party].get('counts', [])
            for h, c in zip(party_hours, party_counts):
                gmt_attacker_hourly_combined[h] = gmt_attacker_hourly_combined.get(h, 0) + c
    
    gmt_victim_hourly_combined = {}
    for party in victim_parties:
        if party in gmt_party_hourly:
            party_hours = gmt_party_hourly[party].get('hours', [])
            party_counts = gmt_party_hourly[party].get('counts', [])
            for h, c in zip(party_hours, party_counts):
                gmt_victim_hourly_combined[h] = gmt_victim_hourly_combined.get(h, 0) + c
    
    gmt_attacker_hours = sorted(gmt_attacker_hourly_combined.keys())
    gmt_attacker_counts = [gmt_attacker_hourly_combined[h] for h in gmt_attacker_hours]
    
    gmt_victim_hours = sorted(gmt_victim_hourly_combined.keys())
    gmt_victim_counts = [gmt_victim_hourly_combined[h] for h in gmt_victim_hours]
    
    if gmt_attacker_hours:
        fig.add_trace(go.Bar(
            x=gmt_attacker_hours,
            y=gmt_attacker_counts,
            name='Attacker (GMT)',
            marker_color='rgba(239, 68, 68, 0.8)',
            legendgroup='attacker',
            showlegend=False
        ), row=4, col=1)
    
    if gmt_victim_hours:
        fig.add_trace(go.Bar(
            x=gmt_victim_hours,
            y=gmt_victim_counts,
            name='Victim (GMT)',
            marker_color='rgba(34, 197, 94, 0.8)',
            legendgroup='victim',
            showlegend=False
        ), row=4, col=1)
    
    # Get timezone match info for subtitle
    timezone_matches = gmt_data.get('timezone_matches', {}) if gmt_data else {}
    top_tz = max(timezone_matches.items(), key=lambda x: x[1])[0] if timezone_matches else None
    tz_info = f" | Top Match: {top_tz}" if top_tz else ""
    
    fig.update_layout(
        title={
            'text': f'Hourly Activity Pattern: Total & Attacker vs Victim Breakdown<br><sub>Peak Activity Analysis | Geographic Indicators | Null timestamps excluded{tz_info}</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.02,
            xanchor="center",
            x=0.5
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=11)
    )
    
    fig.update_xaxes(title_text="Hour of Day (Local)", row=1, col=1, tickmode='linear', tick0=0, dtick=2)
    fig.update_xaxes(title_text="Hour of Day (GMT/UTC)", row=2, col=1, tickmode='linear', tick0=0, dtick=2)
    fig.update_xaxes(title_text="Hour of Day (Local)", row=3, col=1, tickmode='linear', tick0=0, dtick=2)
    fig.update_xaxes(title_text="Hour of Day (GMT/UTC)", row=4, col=1, tickmode='linear', tick0=0, dtick=2)
    
    fig.update_yaxes(title_text="Messages", row=1, col=1)
    fig.update_yaxes(title_text="Messages", row=2, col=1)
    fig.update_yaxes(title_text="Messages", row=3, col=1)
    fig.update_yaxes(title_text="Messages", row=4, col=1)
    
    html_path = output_path.parent / f"{output_path.stem}_hourly_activity.html"
    fig.write_html(str(html_path))
    print(f"[INFO] Hourly activity chart saved to {html_path}")
    
    # Create separate chart with group filtering
    create_hourly_activity_by_group_chart_inline(data, output_path)
    
    # Create comparison chart with expected normal distribution
    create_distribution_comparison_chart(data, output_path)


def create_hourly_activity_by_group_chart_inline(data: dict, output_path: Path):
    """Create hourly activity chart filtered by group with interactive dropdown."""
    hourly_data = data.get('hourly_activity', {})
    party_hourly = hourly_data.get('by_party', {})
    gmt_data = hourly_data.get('gmt', {})
    gmt_party_hourly = gmt_data.get('by_party', {}) if gmt_data else {}
    
    if not party_hourly:
        return
    
    # Group parties by type
    attacker_parties = []
    victim_parties = []
    
    for party in party_hourly.keys():
        party_lower = party.lower().strip()
        if 'victim' in party_lower or 'client' in party_lower:
            victim_parties.append(party)
        else:
            attacker_parties.append(party)
    
    # Calculate totals for each attacker group
    attacker_totals = {}
    for party in attacker_parties:
        if party in party_hourly:
            party_data = party_hourly[party]
            total = sum(party_data.get('counts', []))
            attacker_totals[party] = total
    
    top_attackers = sorted(attacker_totals.items(), key=lambda x: x[1], reverse=True)[:10]
    
    if not top_attackers:
        return
    
    # Create interactive chart with dropdown
    fig = go.Figure()
    
    buttons = []
    
    for idx, (party, total) in enumerate(top_attackers):
        # Local time data
        party_data = party_hourly[party]
        hours = party_data.get('hours', [])
        counts = party_data.get('counts', [])
        
        # GMT data
        gmt_hours = []
        gmt_counts = []
        if party in gmt_party_hourly:
            gmt_party_data = gmt_party_hourly[party]
            gmt_hours = gmt_party_data.get('hours', [])
            gmt_counts = gmt_party_data.get('counts', [])
        
        # Create visibility array (all False except first group)
        is_visible = idx == 0
        trace_start = len(fig.data)
        
        # Add local time trace
        fig.add_trace(go.Bar(
            x=hours,
            y=counts,
            name=f'{party} - Local',
            marker_color='rgba(239, 68, 68, 0.7)',
            visible=is_visible,
            legendgroup=party
        ))
        
        # Add GMT time trace
        if gmt_hours:
            fig.add_trace(go.Bar(
                x=gmt_hours,
                y=gmt_counts,
                name=f'{party} - GMT',
                marker_color='rgba(99, 102, 241, 0.7)',
                visible=is_visible,
                legendgroup=party
            ))
        
        trace_end = len(fig.data)
        
        # Create button for this group
        visibility = [False] * len(fig.data)
        for i in range(trace_start, trace_end):
            visibility[i] = True
        
        buttons.append(
            dict(
                label=f'{party} ({total:,})',
                method="update",
                args=[{"visible": visibility},
                      {"title": f"Hourly Activity: {party} ({total:,} messages)"}]
            )
        )
    
    # Add "All Groups" button
    all_visibility = [True] * len(fig.data)
    buttons.insert(0, dict(
        label="All Top Groups",
        method="update",
        args=[{"visible": all_visibility},
              {"title": "Hourly Activity: Top Attacker Groups Combined"}]
    ))
    
    fig.update_layout(
        title={
            'text': 'Hourly Activity by Attacker Group: All Top Groups Combined<br><sub>Use dropdown to filter by specific group</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title="Hour of Day",
        yaxis_title="Number of Messages",
        height=700,
        showlegend=True,
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.02,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=11),
        xaxis=dict(tickmode='linear', tick0=0, dtick=2),
        barmode='group'
    )
    
    html_path = output_path.parent / f"{output_path.stem}_hourly_activity_by_group.html"
    fig.write_html(str(html_path))
    print(f"[INFO] Hourly activity by group chart saved to {html_path}")


def create_conversation_metrics_chart(data: dict, output_path: Path):
    """Create charts for conversation metrics with outlier removal and filtering."""
    conv_metrics = data.get('conversation_metrics', {})
    per_conv = conv_metrics.get('per_conversation', [])
    
    if not per_conv:
        print("[WARNING] No conversation metrics found")
        return
    
    # Messages per conversation distribution (filter to max 500, remove outliers)
    msg_counts_all = [c['total_messages'] for c in per_conv]
    msg_counts = [m for m in msg_counts_all if m <= 500]  # Logical bound: 500 messages max
    
    # Remove outliers using IQR
    if len(msg_counts) > 4:
        q1 = np.percentile(msg_counts, 25)
        q3 = np.percentile(msg_counts, 75)
        iqr = q3 - q1
        lower_bound = max(0, q1 - 1.5 * iqr)
        upper_bound = q3 + 1.5 * iqr
        msg_counts_filtered = [m for m in msg_counts if lower_bound <= m <= upper_bound]
    else:
        msg_counts_filtered = msg_counts
    
    # Duration distribution (filter by attacker and chat_id, remove outliers)
    durations_all = [(c['duration_minutes'] / 60.0, c.get('attacker', 'unknown'), c['chat_id']) 
                     for c in per_conv if c.get('duration_minutes')]
    durations = [d[0] for d in durations_all]  # Extract just durations
    
    # Remove outliers
    if len(durations) > 4:
        q1 = np.percentile(durations, 25)
        q3 = np.percentile(durations, 75)
        iqr = q3 - q1
        lower_bound = max(0, q1 - 1.5 * iqr)
        upper_bound = q3 + 1.5 * iqr
        durations_filtered = [d for d in durations if lower_bound <= d <= upper_bound]
    else:
        durations_filtered = durations
    
    # Response times (filtered)
    response_times_all = [c['avg_response_time'] / 60.0 for c in per_conv if c.get('avg_response_time')]
    response_times = [rt for rt in response_times_all if rt <= 100]  # Logical bound: 100 hours max
    
    # Remove outliers
    if len(response_times) > 4:
        q1 = np.percentile(response_times, 25)
        q3 = np.percentile(response_times, 75)
        iqr = q3 - q1
        lower_bound = max(0, q1 - 1.5 * iqr)
        upper_bound = q3 + 1.5 * iqr
        response_times_filtered = [rt for rt in response_times if lower_bound <= rt <= upper_bound]
    else:
        response_times_filtered = response_times
    
    # Exchanges
    exchanges_all = [c['num_exchanges'] for c in per_conv if c.get('num_exchanges')]
    exchanges = exchanges_all
    
    # Remove outliers
    if len(exchanges) > 4:
        q1 = np.percentile(exchanges, 25)
        q3 = np.percentile(exchanges, 75)
        iqr = q3 - q1
        lower_bound = max(0, q1 - 1.5 * iqr)
        upper_bound = q3 + 1.5 * iqr
        exchanges_filtered = [e for e in exchanges if lower_bound <= e <= upper_bound]
    else:
        exchanges_filtered = exchanges
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Messages per Conversation (≤500, outliers removed)', 
                       'Conversation Duration (hours, outliers removed)', 
                       'Response Times per Conversation (≤100h, outliers removed)', 
                       'Number of Exchanges (outliers removed)'),
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "histogram"}, {"type": "histogram"}]]
    )
    
    # Messages per conversation (60 bins)
    if msg_counts_filtered:
        fig.add_trace(
            go.Histogram(x=msg_counts_filtered, name='Messages', nbinsx=60, 
                        marker_color='rgba(99, 102, 241, 0.7)'),
            row=1, col=1
        )
    
    # Duration distribution (60 bins)
    if durations_filtered:
        fig.add_trace(
            go.Histogram(x=durations_filtered, name='Duration', nbinsx=60, 
                        marker_color='rgba(239, 68, 68, 0.7)'),
            row=1, col=2
        )
    
    # Response times (60 bins)
    if response_times_filtered:
        fig.add_trace(
            go.Histogram(x=response_times_filtered, name='Response Time', nbinsx=60, 
                        marker_color='rgba(34, 197, 94, 0.7)'),
            row=2, col=1
        )
    
    # Exchanges (60 bins)
    if exchanges_filtered:
        fig.add_trace(
            go.Histogram(x=exchanges_filtered, name='Exchanges', nbinsx=60, 
                        marker_color='rgba(251, 191, 36, 0.7)'),
            row=2, col=2
        )
    
    # Add global stats as annotations
    aggregate = conv_metrics.get('aggregate', {})
    stats_text = f"Global Stats: {aggregate.get('total_conversations', 0)} conversations, "
    stats_text += f"Avg {aggregate.get('avg_messages_per_conversation', 0):.1f} msgs/conv"
    
    # Calculate outlier removal stats
    outliers_removed = {
        'messages': len(msg_counts_all) - len(msg_counts_filtered) if msg_counts_filtered else 0,
        'durations': len(durations_all) - len(durations_filtered) if durations_filtered else 0,
        'response_times': len(response_times_all) - len(response_times_filtered) if response_times_filtered else 0,
        'exchanges': len(exchanges_all) - len(exchanges_filtered) if exchanges_filtered else 0
    }
    
    fig.update_layout(
        title={
            'text': f'Conversation Metrics Distribution<br><sub>60 Bins | IQR Outlier Removal (1.5×IQR) | ' +
                    f'Logical Bounds Applied | {stats_text}</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=850,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=10)
    )
    
    fig.update_xaxes(title_text="Messages", row=1, col=1)
    fig.update_xaxes(title_text="Hours", row=1, col=2)
    fig.update_xaxes(title_text="Hours", row=2, col=1)
    fig.update_xaxes(title_text="Exchanges", row=2, col=2)
    
    html_path = output_path.parent / f"{output_path.stem}_conversation_metrics.html"
    fig.write_html(str(html_path))
    print(f"[INFO] Conversation metrics charts saved to {html_path}")
    
    # Create separate charts filtered by distinct attacker/party
    create_attacker_filtered_charts(data, output_path)


def create_distribution_comparison_chart(data: dict, output_path: Path):
    """Create chart comparing observed GMT distribution to expected normal business hours."""
    hourly_data = data.get('hourly_activity', {})
    gmt_data = hourly_data.get('gmt', {})
    distribution_comparison = gmt_data.get('distribution_comparison', {})
    
    if not distribution_comparison or not distribution_comparison.get('observed_distribution'):
        print("[WARNING] No distribution comparison data found")
        return
    
    # Create expected normal distribution (centered at 12pm, std=3)
    hours_24 = list(range(24))
    expected_normal = []
    for hour in hours_24:
        prob = norm.pdf(hour, loc=12, scale=3)
        expected_normal.append(prob)
    expected_normal = np.array(expected_normal)
    expected_normal = expected_normal / expected_normal.sum()
    
    observed = np.array(distribution_comparison.get('observed_distribution', [0]*24))
    expected = np.array(distribution_comparison.get('expected_distribution', expected_normal.tolist()))
    
    # Ensure both are length 24
    if len(observed) < 24:
        observed_padded = np.zeros(24)
        gmt_hours = gmt_data.get('hours', [])
        gmt_counts = gmt_data.get('counts', [])
        for h, c in zip(gmt_hours, gmt_counts):
            if 0 <= h < 24:
                observed_padded[h] = c
        observed_padded = observed_padded / observed_padded.sum() if observed_padded.sum() > 0 else observed_padded
        observed = observed_padded
    
    fig = go.Figure()
    
    # Expected normal distribution
    fig.add_trace(go.Scatter(
        x=hours_24,
        y=expected * 100,  # Convert to percentage
        mode='lines+markers',
        name='Expected Normal (6am-6pm, centered at 12pm)',
        line=dict(color='rgba(34, 197, 94, 0.8)', width=3, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Observed GMT distribution
    fig.add_trace(go.Bar(
        x=hours_24,
        y=observed * 100,  # Convert to percentage
        name='Observed GMT Activity',
        marker_color='rgba(239, 68, 68, 0.7)',
        marker_line_color='rgba(239, 68, 68, 1)',
        marker_line_width=1
    ))
    
    # Add statistical significance info
    ks_pvalue = distribution_comparison.get('ks_pvalue', 1.0)
    chi2_pvalue = distribution_comparison.get('chi2_pvalue', 1.0)
    ks_significant = distribution_comparison.get('ks_significant', False)
    
    significance_text = f"KS p-value: {ks_pvalue:.4f} | "
    significance_text += f"{'Significantly different' if ks_significant else 'Similar to'} normal business hours"
    
    fig.update_layout(
        title={
            'text': f'GMT Activity Distribution vs Expected Normal Business Hours<br><sub>{significance_text}</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title='Hour of Day (GMT/UTC)',
        yaxis_title='Activity Percentage (%)',
        height=550,
        showlegend=True,
        xaxis=dict(tickmode='linear', tick0=0, dtick=2, title_font={'size': 12}),
        yaxis=dict(title_font={'size': 12}),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=11)
    )
    
    html_path = output_path.parent / f"{output_path.stem}_distribution_comparison.html"
    fig.write_html(str(html_path))
    print(f"[INFO] Distribution comparison chart saved to {html_path}")


def create_attacker_filtered_charts(data: dict, output_path: Path):
    """Create charts filtered by distinct attacker/party."""
    conv_metrics = data.get('conversation_metrics', {})
    per_conv = conv_metrics.get('per_conversation', [])
    by_attacker = conv_metrics.get('by_attacker', {})
    
    if not per_conv or not by_attacker:
        print("[WARNING] No attacker data found for filtering")
        return
    
    # Get top attackers by conversation count
    attacker_list = sorted(by_attacker.items(), key=lambda x: x[1].get('conversation_count', 0), reverse=True)[:10]
    
    if not attacker_list:
        return
    
    # Create subplot for each metric, filtered by attacker
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Messages per Conversation by Attacker', 
                       'Conversation Duration by Attacker', 
                       'Response Times by Attacker', 
                       'Exchanges by Attacker'),
        specs=[[{"type": "box"}, {"type": "box"}],
               [{"type": "box"}, {"type": "box"}]]
    )
    
    colors = ['rgba(99, 102, 241, 0.7)', 'rgba(239, 68, 68, 0.7)', 'rgba(34, 197, 94, 0.7)', 
              'rgba(251, 191, 36, 0.7)', 'rgba(168, 85, 247, 0.7)', 'rgba(236, 72, 153, 0.7)',
              'rgba(20, 184, 166, 0.7)', 'rgba(245, 158, 11, 0.7)', 'rgba(59, 130, 246, 0.7)', 
              'rgba(249, 115, 22, 0.7)']
    
    row_col_map = [(1, 1), (1, 2), (2, 1), (2, 2)]
    metrics = [
        ('total_messages', 'Messages', 500),
        ('duration_minutes', 'Duration (hours)', None),
        ('avg_response_time', 'Response Time (hours)', 100),
        ('num_exchanges', 'Exchanges', None)
    ]
    
    for metric_idx, (metric_key, metric_label, max_val) in enumerate(metrics):
        row, col = row_col_map[metric_idx]
        
        for idx, (attacker, stats) in enumerate(attacker_list[:6]):  # Top 6 attackers
            # Get data for this attacker
            attacker_data = []
            for conv in per_conv:
                if conv.get('attacker') == attacker:
                    val = conv.get(metric_key)
                    if val is not None:
                        # Convert to hours if needed
                        if metric_key == 'duration_minutes':
                            val = val / 60.0
                        elif metric_key == 'avg_response_time':
                            val = val / 60.0
                        
                        # Apply logical bounds
                        if max_val and val > max_val:
                            continue
                        
                        attacker_data.append(val)
            
            if attacker_data:
                # Remove outliers using IQR
                if len(attacker_data) > 4:
                    q1 = np.percentile(attacker_data, 25)
                    q3 = np.percentile(attacker_data, 75)
                    iqr = q3 - q1
                    if iqr > 0:
                        lower_bound = max(0, q1 - 1.5 * iqr) if metric_key != 'duration_minutes' else q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        attacker_data = [v for v in attacker_data if lower_bound <= v <= upper_bound]
                
                fig.add_trace(
                    go.Box(
                        y=attacker_data,
                        name=attacker[:20],  # Truncate long names
                        marker_color=colors[idx % len(colors)],
                        boxmean='sd'
                    ),
                    row=row, col=col
                )
    
    fig.update_layout(
        title={
            'text': 'Conversation Metrics by Attacker/Party<br><sub>Filtered by Distinct Attacker | IQR Outlier Removal | Top 6 Attackers Shown</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=900,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=10)
    )
    
    html_path = output_path.parent / f"{output_path.stem}_by_attacker.html"
    fig.write_html(str(html_path))
    print(f"[INFO] Attacker-filtered charts saved to {html_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Visualize timeline analysis')
    parser.add_argument('json_file', type=Path, help='Input JSON analysis file')
    parser.add_argument('-o', '--output', type=Path, help='Output directory for HTML files', default=None)
    
    args = parser.parse_args()
    
    if not args.json_file.exists():
        print(f"[ERROR] JSON file does not exist: {args.json_file}")
        sys.exit(1)
    
    output_dir = args.output or args.json_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Loading analysis from {args.json_file}")
    data = load_analysis(args.json_file)
    
    print("[INFO] Generating visualizations...")
    create_response_time_histogram(data, args.json_file)
    create_timestamp_distribution(data, args.json_file)
    create_hourly_activity_chart(data, args.json_file)
    create_conversation_metrics_chart(data, args.json_file)
    
    print("\n[INFO] All visualizations generated successfully!")


if __name__ == '__main__':
    main()
