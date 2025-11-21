"""Create hourly activity chart with group filtering."""

import json
from pathlib import Path
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_hourly_activity_by_group_chart(data: dict, output_path: Path):
    """Create hourly activity chart filtered by group with interactive filtering."""
    hourly_data = data.get('hourly_activity', {})
    
    if not hourly_data or not hourly_data.get('hours'):
        print("[WARNING] No hourly activity data found")
        return
    
    # Load full data to get group information
    csv_path = output_path.parent.parent / "output" / "ransom_chats.csv"
    if not csv_path.exists():
        print(f"[WARNING] CSV file not found: {csv_path}")
        return
    
    import csv
    groups_data = defaultdict(lambda: {'attacker': defaultdict(int), 'victim': defaultdict(int), 
                                      'gmt_attacker': defaultdict(int), 'gmt_victim': defaultdict(int)})
    
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                group_name = row.get('group_name', 'Unknown')
                party = row.get('party', '').lower().strip()
                timestamp = row.get('timestamp', '')
                
                if not timestamp or timestamp.lower() in ['null', 'none', '']:
                    continue
                
                try:
                    from datetime import datetime
                    from semantic_detector.web.app.core.timestamp_analyzer import TimestampAnalyzer
                    analyzer = TimestampAnalyzer()
                    dt = analyzer.parse_timestamp(timestamp)
                    if dt:
                        hour = dt.hour
                        
                        # Determine if attacker or victim
                        is_victim = 'victim' in party or 'client' in party
                        party_type = 'victim' if is_victim else 'attacker'
                        
                        # Local time
                        groups_data[group_name][party_type][hour] += 1
                        
                        # GMT time
                        if dt.tzinfo is not None:
                            from datetime import timezone
                            gmt_dt = dt.astimezone(timezone.utc)
                            gmt_hour = gmt_dt.hour
                        else:
                            gmt_hour = dt.hour
                        
                        groups_data[group_name][f'gmt_{party_type}'][gmt_hour] += 1
                except Exception:
                    continue
    except Exception as e:
        print(f"[WARNING] Error loading group data: {e}")
        return
    
    # Get top groups by message count
    group_totals = {}
    for group, data_dict in groups_data.items():
        total = sum(data_dict['attacker'].values()) + sum(data_dict['victim'].values())
        group_totals[group] = total
    
    top_groups = sorted(group_totals.items(), key=lambda x: x[1], reverse=True)[:10]
    
    if not top_groups:
        print("[WARNING] No group data found")
        return
    
    # Create interactive chart with dropdown
    fig = go.Figure()
    
    # Create traces for each group (initially hidden)
    buttons = []
    visible_traces = []
    
    for idx, (group_name, total) in enumerate(top_groups):
        group_data = groups_data[group_name]
        
        # Local attacker hours
        attacker_hours = sorted(group_data['attacker'].keys())
        attacker_counts = [group_data['attacker'][h] for h in attacker_hours]
        
        # Local victim hours
        victim_hours = sorted(group_data['victim'].keys())
        victim_counts = [group_data['victim'][h] for h in victim_hours]
        
        # GMT attacker hours
        gmt_attacker_hours = sorted(group_data['gmt_attacker'].keys())
        gmt_attacker_counts = [group_data['gmt_attacker'][h] for h in gmt_attacker_hours]
        
        # GMT victim hours
        gmt_victim_hours = sorted(group_data['gmt_victim'].keys())
        gmt_victim_counts = [group_data['gmt_victim'][h] for h in gmt_victim_hours]
        
        # Create visibility array (all False except first group)
        is_visible = idx == 0
        
        # Add traces for this group
        trace_start = len(fig.data)
        
        if attacker_hours:
            fig.add_trace(go.Bar(
                x=attacker_hours,
                y=attacker_counts,
                name=f'{group_name} - Attacker (Local)',
                marker_color='rgba(239, 68, 68, 0.7)',
                visible=is_visible,
                legendgroup=group_name
            ))
        
        if victim_hours:
            fig.add_trace(go.Bar(
                x=victim_hours,
                y=victim_counts,
                name=f'{group_name} - Victim (Local)',
                marker_color='rgba(34, 197, 94, 0.7)',
                visible=is_visible,
                legendgroup=group_name
            ))
        
        if gmt_attacker_hours:
            fig.add_trace(go.Bar(
                x=gmt_attacker_hours,
                y=gmt_attacker_counts,
                name=f'{group_name} - Attacker (GMT)',
                marker_color='rgba(239, 68, 68, 0.5)',
                visible=is_visible,
                legendgroup=group_name
            ))
        
        if gmt_victim_hours:
            fig.add_trace(go.Bar(
                x=gmt_victim_hours,
                y=gmt_victim_counts,
                name=f'{group_name} - Victim (GMT)',
                marker_color='rgba(34, 197, 94, 0.5)',
                visible=is_visible,
                legendgroup=group_name
            ))
        
        trace_end = len(fig.data)
        
        # Create button for this group
        visibility = [False] * len(fig.data)
        for i in range(trace_start, trace_end):
            visibility[i] = True
        
        buttons.append(
            dict(
                label=group_name,
                method="update",
                args=[{"visible": visibility},
                      {"title": f"Hourly Activity by Group: {group_name} ({total:,} messages)"}]
            )
        )
    
    # Add "All Groups" button
    all_visibility = [True] * len(fig.data)
    buttons.insert(0, dict(
        label="All Groups",
        method="update",
        args=[{"visible": all_visibility},
              {"title": "Hourly Activity by Group: All Groups Combined"}]
    ))
    
    fig.update_layout(
        title={
            'text': 'Hourly Activity by Group: All Groups Combined<br><sub>Use dropdown to filter by specific group</sub>',
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
        xaxis=dict(tickmode='linear', tick0=0, dtick=2)
    )
    
    html_path = output_path.parent / f"{output_path.stem}_hourly_activity_by_group.html"
    fig.write_html(str(html_path))
    print(f"[INFO] Hourly activity by group chart saved to {html_path}")

