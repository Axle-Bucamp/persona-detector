#!/usr/bin/env python3
"""Generate comprehensive one-page HTML report with PDF export capability."""

import json
import sys
from pathlib import Path
from datetime import datetime
import argparse


def load_analysis(json_path: Path) -> dict:
    """Load analysis JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_time(minutes: float) -> str:
    """Format minutes into readable time string."""
    if minutes < 60:
        return f"{minutes:.1f} minutes"
    elif minutes < 1440:
        return f"{minutes/60:.1f} hours"
    else:
        return f"{minutes/1440:.1f} days"


def generate_html_report(data: dict, output_path: Path):
    """Generate comprehensive HTML report."""
    
    summary = data.get('summary', {})
    response_stats = data.get('response_time_by_party', {})
    conv_metrics = data.get('conversation_metrics', {})
    hourly_activity = data.get('hourly_activity', {})
    timestamp_dist = data.get('timestamp_distribution', {})
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ransom Chat Timeline Analysis - Comprehensive Report</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        
        .report-container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        
        .header {{
            text-align: center;
            border-bottom: 3px solid #1e40af;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            color: #1e40af;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            color: #666;
            font-size: 1.1em;
        }}
        
        .header .meta {{
            margin-top: 15px;
            font-size: 0.9em;
            color: #888;
        }}
        
        .export-buttons {{
            text-align: right;
            margin-bottom: 20px;
        }}
        
        .btn {{
            background: #1e40af;
            color: white;
            border: none;
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 5px;
            font-size: 14px;
            margin-left: 10px;
        }}
        
        .btn:hover {{
            background: #1e3a8a;
        }}
        
        .section {{
            margin-bottom: 40px;
            page-break-inside: avoid;
        }}
        
        .section-title {{
            color: #1e40af;
            font-size: 1.8em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e5e7eb;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .stat-card {{
            background: #f8fafc;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #1e40af;
        }}
        
        .stat-card h3 {{
            color: #1e40af;
            font-size: 1.1em;
            margin-bottom: 10px;
        }}
        
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #1e3a8a;
        }}
        
        .stat-card .label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        .table-container {{
            overflow-x: auto;
            margin: 20px 0;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th {{
            background: #1e40af;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        
        td {{
            padding: 10px;
            border-bottom: 1px solid #e5e7eb;
        }}
        
        tr:hover {{
            background: #f8fafc;
        }}
        
        .chart-container {{
            margin: 30px 0;
            height: 400px;
        }}
        
        .insight-box {{
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        
        .insight-box h4 {{
            color: #1e40af;
            margin-bottom: 10px;
        }}
        
        .insight-box p {{
            color: #374151;
            line-height: 1.8;
        }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #e5e7eb;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .export-buttons {{
                display: none;
            }}
            .report-container {{
                box-shadow: none;
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="report-container" id="report-content">
        <div class="export-buttons">
            <button class="btn" onclick="exportToPDF()">Export to PDF</button>
            <button class="btn" onclick="window.print()">Print</button>
        </div>
        
        <div class="header">
            <h1>🔍 Ransom Chat Timeline Intelligence Analysis</h1>
            <div class="subtitle">Comprehensive Communication Pattern & Behavioral Intelligence Report</div>
            <div class="meta">
                📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
                📊 Analysis Period: {timestamp_dist.get('min_timestamp', 'N/A')[:10]} to {timestamp_dist.get('max_timestamp', 'N/A')[:10]} |
                💬 {summary.get('total_messages', 0):,} Messages | {summary.get('total_conversations', 0):,} Conversations
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">Executive Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Messages</h3>
                    <div class="value">{summary.get('total_messages', 0):,}</div>
                    <div class="label">across all conversations</div>
                </div>
                <div class="stat-card">
                    <h3>Total Conversations</h3>
                    <div class="value">{summary.get('total_conversations', 0):,}</div>
                    <div class="label">unique chat sessions</div>
                </div>
                <div class="stat-card">
                    <h3>Date Range</h3>
                    <div class="value">{timestamp_dist.get('min_timestamp', 'N/A')[:10]}</div>
                    <div class="label">to {timestamp_dist.get('max_timestamp', 'N/A')[:10]}</div>
                </div>
            </div>
            
            <div class="insight-box">
                <h4>🎯 Key Findings & Methodology</h4>
                <p>
                    This comprehensive intelligence analysis examines <strong>{summary.get('total_messages', 0):,} messages</strong> across 
                    <strong>{summary.get('total_conversations', 0):,} ransom chat conversations</strong>, revealing critical insights 
                    into communication patterns, response behaviors, and operational timing. The analysis employs 
                    <strong>advanced outlier detection techniques</strong>:
                </p>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li><strong>Z-Score Method (σ=3.0)</strong>: Applied to response times for robust outlier detection</li>
                    <li><strong>IQR Method (1.5×IQR)</strong>: Used for conversation metrics - ideal for skewed distributions and robust to extreme values</li>
                    <li><strong>Logical Bounds</strong>: 500 messages max per conversation, 100 hours max response time</li>
                    <li><strong>60-Bin Histograms</strong>: Granular analysis with 0-5 hour focus window for response times</li>
                    <li><strong>GMT Analysis</strong>: Geographic indicator identification through timezone matching</li>
                </ul>
                <p style="margin-top: 10px;">
                    These techniques ensure accurate representation of <em>normal operational patterns</em> while filtering 
                    exceptional circumstances and data anomalies that could skew intelligence assessments.
                </p>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">Response Time Analysis</h2>
"""
    
    # Add response time statistics
    if response_stats:
        html_content += """
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Party</th>
                            <th>Count</th>
                            <th>Mean</th>
                            <th>Median</th>
                            <th>Min</th>
                            <th>Max</th>
                            <th>Outliers Removed</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        for party, stats in response_stats.items():
            html_content += f"""
                        <tr>
                            <td><strong>{party.capitalize()}</strong></td>
                            <td>{stats.get('count', 0):,}</td>
                            <td>{format_time(stats.get('mean', 0))}</td>
                            <td>{format_time(stats.get('median', 0))}</td>
                            <td>{format_time(stats.get('min', 0))}</td>
                            <td>{format_time(stats.get('max', 0))}</td>
                            <td>{stats.get('outliers_removed', 0):,}</td>
                        </tr>
"""
        html_content += """
                    </tbody>
                </table>
            </div>
"""
        
        attacker_stats = response_stats.get('attacker', {})
        victim_stats = response_stats.get('victim', {})
        
        if attacker_stats and victim_stats:
            html_content += f"""
            <div class="insight-box">
                <h4>Response Time Insights</h4>
                <p>
                    Attackers demonstrate a median response time of <strong>{format_time(attacker_stats.get('median', 0))}</strong>, 
                    indicating relatively quick operational responses. However, the mean response time of 
                    <strong>{format_time(attacker_stats.get('mean', 0))}</strong> reveals significant variability. 
                    Victims show a median response time of <strong>{format_time(victim_stats.get('median', 0))}</strong> 
                    with a mean of <strong>{format_time(victim_stats.get('mean', 0))}</strong>, suggesting victims 
                    often require more time for internal coordination and decision-making processes. The analysis 
                    filtered {attacker_stats.get('outliers_removed', 0) + victim_stats.get('outliers_removed', 0)} 
                    extreme outliers to focus on normal operational patterns.
                </p>
            </div>
"""
        
        # Add response time chart
        html_content += """
            <div class="chart-container" id="response-time-chart"></div>
"""
    
    # Conversation Metrics
    html_content += """
        <div class="section">
            <h2 class="section-title">Conversation Metrics</h2>
"""
    
    aggregate = conv_metrics.get('aggregate', {})
    if aggregate:
        html_content += f"""
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Avg Messages/Conversation</h3>
                    <div class="value">{aggregate.get('avg_messages_per_conversation', 0):.1f}</div>
                    <div class="label">median: {aggregate.get('median_messages_per_conversation', 0):.1f}</div>
                </div>
                <div class="stat-card">
                    <h3>Avg Duration</h3>
                    <div class="value">{format_time(aggregate.get('avg_duration_minutes', 0))}</div>
                    <div class="label">per conversation</div>
                </div>
                <div class="stat-card">
                    <h3>Avg Response Time</h3>
                    <div class="value">{format_time(aggregate.get('avg_response_time_across_conversations', 0))}</div>
                    <div class="label">across conversations</div>
                </div>
            </div>
"""
        
        html_content += f"""
            <div class="insight-box">
                <h4>Conversation Characteristics</h4>
                <p>
                    The analysis reveals an average of <strong>{aggregate.get('avg_messages_per_conversation', 0):.1f}</strong> 
                    messages per conversation, with a median of <strong>{aggregate.get('median_messages_per_conversation', 0):.1f}</strong>. 
                    Conversations average <strong>{format_time(aggregate.get('avg_duration_minutes', 0))}</strong> in duration, 
                    indicating that ransomware negotiations are often protracted processes requiring sustained engagement. 
                    Metrics are filtered with logical bounds (500 messages max) and strengthened outlier removal using 
                    IQR methodology to ensure accurate representation of normal patterns.
                </p>
            </div>
"""
        
        # Add conversation metrics chart
        html_content += """
            <div class="chart-container" id="conversation-metrics-chart"></div>
"""
    
    # Temporal Analysis
    html_content += """
        <div class="section">
            <h2 class="section-title">Temporal Activity Analysis</h2>
"""
    
    if hourly_activity:
        peak_hours = hourly_activity.get('peak_hours', [])
        peak_count = hourly_activity.get('peak_count', 0)
        gmt_data = hourly_activity.get('gmt', {})
        distribution_comparison = gmt_data.get('distribution_comparison', {})
        best_match = gmt_data.get('best_match')
        best_match_score = gmt_data.get('best_match_score', {})
        timezone_scores = gmt_data.get('timezone_scores', {})
        
        html_content += f"""
            <div class="insight-box">
                <h4>Activity Patterns & Statistical Analysis</h4>
                <p>
                    Peak activity occurs at hours <strong>{', '.join(map(str, peak_hours))}</strong> (local time) 
                    with <strong>{peak_count:,}</strong> messages. GMT (UTC) analysis reveals peak activity at hours 
                    <strong>{', '.join(map(str, gmt_data.get('peak_hours', [])))}</strong>.
                </p>
"""
        
        # Statistical significance analysis
        if distribution_comparison:
            ks_pvalue = distribution_comparison.get('ks_pvalue', 1.0)
            chi2_pvalue = distribution_comparison.get('chi2_pvalue', 1.0)
            ks_significant = distribution_comparison.get('ks_significant', False)
            chi2_significant = distribution_comparison.get('chi2_significant', False)
            
            html_content += f"""
                <p><strong>Statistical Comparison to Normal Business Hours (6am-6pm):</strong></p>
                <ul>
                    <li><strong>Kolmogorov-Smirnov Test:</strong> p-value = {ks_pvalue:.4f} 
                        {'(Significant - distribution differs from normal business hours)' if ks_significant else '(Not significant - similar to normal business hours)'}</li>
                    <li><strong>Chi-Square Test:</strong> p-value = {chi2_pvalue:.4f} 
                        {'(Significant - distribution differs)' if chi2_significant else '(Not significant - similar distribution)'}</li>
                </ul>
                <p>
                    <strong>Interpretation:</strong> A <strong>non-significant p-value (p > 0.05)</strong> indicates that 
                    the observed GMT activity distribution is <em>statistically similar</em> to a normal business hours 
                    distribution (6am-6pm, centered at 12pm). This suggests attackers may be operating during standard 
                    business hours in their timezone. Conversely, a <strong>significant p-value (p < 0.05)</strong> would 
                    indicate non-standard work patterns, such as night shifts or adaptation to target schedules.
                </p>
                <p>
                    <em>Important Note: Attackers may work at night, adapt to their target's schedule, or use VPNs to 
                    mask their true location. The statistical analysis provides indicators but should be interpreted 
                    cautiously in conjunction with other intelligence sources.</em>
                </p>
"""
        
        # Best timezone match
        if best_match and best_match_score:
            correlation = best_match_score.get('correlation', 0)
            html_content += f"""
                <p><strong>Best GMT Timezone Match:</strong> <strong>{best_match}</strong></p>
                <ul>
                    <li>Distribution Correlation: {correlation:.3f}</li>
                    <li>Overall Match Score: {best_match_score.get('score', 0):.3f}</li>
                </ul>
"""
        
        # Top timezone matches
        if timezone_scores:
            html_content += """
                <p><strong>Top Timezone Matches (by distribution correlation):</strong></p>
                <ul>
"""
            sorted_tz = sorted(timezone_scores.items(), key=lambda x: x[1].get('correlation', 0), reverse=True)[:5]
            for tz_name, score_data in sorted_tz:
                corr = score_data.get('correlation', 0)
                html_content += f"<li><strong>{tz_name}:</strong> Correlation = {corr:.3f}, Score = {score_data.get('score', 0):.3f}</li>"
            html_content += """
                </ul>
"""
        
        html_content += """
            </div>
"""
        
        # Add hourly activity chart
        html_content += """
            <div class="chart-container" id="hourly-activity-chart"></div>
"""
        
        # Add distribution comparison chart if available
        if distribution_comparison:
            html_content += """
            <div class="section">
                <h2 class="section-title">Distribution Comparison: Observed vs Expected Normal Business Hours</h2>
                <div class="chart-container" id="distribution-comparison-chart"></div>
            </div>
"""
        
        # Add distribution comparison chart
        if gmt_data.get('distribution_comparison'):
            html_content += """
            <div class="chart-container" id="distribution-comparison-chart"></div>
"""
    
    # Timestamp Distribution
    if timestamp_dist:
        outliers_removed = timestamp_dist.get('outliers_removed', 0)
        html_content += f"""
        <div class="section">
            <h2 class="section-title">Message Distribution Over Time</h2>
            <div class="insight-box">
                <h4>Temporal Distribution</h4>
                <p>
                    The temporal distribution shows message activity patterns across the analyzed period. 
                    <strong>{outliers_removed:,}</strong> extreme timestamp outliers were removed to focus 
                    on normal operational patterns. The distribution uses 30 bins for granular time-based analysis.
                </p>
            </div>
            <div class="chart-container" id="timestamp-distribution-chart"></div>
        </div>
"""
    
    html_content += """
        <div class="footer">
            <p>Generated by Ransom Chat Timeline Analysis System</p>
            <p>This report contains sensitive intelligence data - handle accordingly</p>
        </div>
    </div>
    
    <script>
        // Load and render charts
        function loadCharts() {
            // Response time chart (0-5 hours, 60 bins)
"""
    
    # Add response time chart data
    if response_stats:
        html_content += """
            const responseData = {
"""
        for party, stats in response_stats.items():
            values = stats.get('values', [])
            values_hours = [v / 60.0 for v in values if 0 <= v / 60.0 <= 5]
            html_content += f"""
                '{party}': {json.dumps(values_hours)},
"""
        html_content += """
            };
            
            const responseTraces = [];
            const colors = {'attacker': 'rgba(239, 68, 68, 0.7)', 'victim': 'rgba(59, 130, 246, 0.7)'};
            for (const [party, values] of Object.entries(responseData)) {
                responseTraces.push({
                    x: values,
                    type: 'histogram',
                    name: party.charAt(0).toUpperCase() + party.slice(1),
                    opacity: 0.7,
                    marker: {color: colors[party] || 'rgba(128, 128, 128, 0.7)'},
                    nbinsx: 60,
                    xbins: {start: 0, end: 5, size: 5/60}
                });
            }
            
            Plotly.newPlot('response-time-chart', responseTraces, {
                title: 'Response Time Distribution (0-5 hours, 60 bins)',
                xaxis: {title: 'Response Time (hours)', range: [0, 5]},
                yaxis: {title: 'Frequency'},
                barmode: 'overlay',
                height: 400
            });
"""
    
    # Add conversation metrics chart
    if conv_metrics:
        per_conv = conv_metrics.get('per_conversation', [])
        msg_counts = [c['total_messages'] for c in per_conv if c['total_messages'] <= 500]
        durations = [c['duration_minutes'] / 60.0 for c in per_conv if c.get('duration_minutes')]
        response_times = [c['avg_response_time'] / 60.0 for c in per_conv if c.get('avg_response_time') and c['avg_response_time'] / 60.0 <= 100]
        exchanges = [c['num_exchanges'] for c in per_conv if c.get('num_exchanges')]
        
        html_content += f"""
            // Conversation metrics chart
            const convTraces = [
                {{
                    x: {json.dumps(msg_counts)},
                    type: 'histogram',
                    name: 'Messages',
                    marker: {{color: 'rgba(99, 102, 241, 0.7)'}},
                    nbinsx: 60
                }},
                {{
                    x: {json.dumps(durations)},
                    type: 'histogram',
                    name: 'Duration',
                    marker: {{color: 'rgba(239, 68, 68, 0.7)'}},
                    nbinsx: 60
                }},
                {{
                    x: {json.dumps(response_times)},
                    type: 'histogram',
                    name: 'Response Time',
                    marker: {{color: 'rgba(34, 197, 94, 0.7)'}},
                    nbinsx: 60
                }},
                {{
                    x: {json.dumps(exchanges)},
                    type: 'histogram',
                    name: 'Exchanges',
                    marker: {{color: 'rgba(251, 191, 36, 0.7)'}},
                    nbinsx: 60
                }}
            ];
            
            Plotly.newPlot('conversation-metrics-chart', convTraces, {{
                title: 'Conversation Metrics Distribution (60 bins, Outliers Removed)',
                height: 400,
                showlegend: false
            }});
"""
    
    # Add hourly activity chart
    if hourly_activity:
        hours = hourly_activity.get('hours', [])
        counts = hourly_activity.get('counts', [])
        gmt_data = hourly_activity.get('gmt', {})
        gmt_hours = gmt_data.get('hours', [])
        gmt_counts = gmt_data.get('counts', [])
        
        html_content += f"""
            // Hourly activity chart
            const hourlyTraces = [
                {{
                    x: {json.dumps(hours)},
                    y: {json.dumps(counts)},
                    type: 'bar',
                    name: 'Local Time',
                    marker: {{color: 'rgba(99, 102, 241, 0.7)'}}
                }},
                {{
                    x: {json.dumps(gmt_hours)},
                    y: {json.dumps(gmt_counts)},
                    type: 'bar',
                    name: 'GMT (UTC)',
                    marker: {{color: 'rgba(239, 68, 68, 0.7)'}}
                }}
            ];
            
            Plotly.newPlot('hourly-activity-chart', hourlyTraces, {{
                title: 'Hourly Activity: Local vs GMT',
                xaxis: {{title: 'Hour of Day'}},
                yaxis: {{title: 'Number of Messages'}},
                barmode: 'group',
                height: 400
            }});
"""
        
        # Add distribution comparison chart
        distribution_comparison = gmt_data.get('distribution_comparison', {})
        if distribution_comparison:
            observed_dist = distribution_comparison.get('observed_distribution', [])
            expected_dist = distribution_comparison.get('expected_distribution', [])
            ks_pvalue = distribution_comparison.get('ks_pvalue', 1.0)
            ks_significant = distribution_comparison.get('ks_significant', False)
            
            # Convert boolean to string for JavaScript
            ks_sig_js = 'true' if ks_significant else 'false'
            
            html_content += f"""
            // Distribution comparison chart
            const observedDist = {json.dumps(observed_dist)};
            const expectedDist = {json.dumps(expected_dist)};
            const ksPValue = {ks_pvalue};
            const ksSignificant = {ks_sig_js};
            
            if (observedDist && observedDist.length === 24 && expectedDist && expectedDist.length === 24) {{
                const hours24 = Array.from({{length: 24}}, (_, i) => i);
                
                const comparisonTraces = [
                    {{
                        x: hours24,
                        y: expectedDist.map(v => v * 100),
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: 'Expected Normal (6am-6pm, centered at 12pm)',
                        line: {{color: 'rgba(34, 197, 94, 0.8)', width: 3, dash: 'dash'}},
                        marker: {{size: 6}}
                    }},
                    {{
                        x: hours24,
                        y: observedDist.map(v => v * 100),
                        type: 'bar',
                        name: 'Observed GMT Activity',
                        marker: {{color: 'rgba(239, 68, 68, 0.7)'}},
                        marker_line: {{color: 'rgba(239, 68, 68, 1)', width: 1}}
                    }}
                ];
                
                const significanceText = `KS p-value: ${{ksPValue.toFixed(4)}} | ${{ksSignificant ? 'Significantly different' : 'Similar to'}} normal business hours`;
                
                Plotly.newPlot('distribution-comparison-chart', comparisonTraces, {{
                    title: `GMT Activity Distribution vs Expected Normal Business Hours<br><sub>${{significanceText}}</sub>`,
                    xaxis: {{title: 'Hour of Day (GMT/UTC)', tickmode: 'linear', tick0: 0, dtick: 2}},
                    yaxis: {{title: 'Activity Percentage (%)'}},
                    height: 400,
                    barmode: 'overlay',
                    showlegend: true
                }});
            }}
"""
    
    # Add timestamp distribution chart
    if timestamp_dist:
        bin_labels = timestamp_dist.get('bin_labels', [])
        counts = timestamp_dist.get('counts', [])
        html_content += f"""
            // Timestamp distribution chart
            Plotly.newPlot('timestamp-distribution-chart', [{{
                x: {json.dumps(bin_labels)},
                y: {json.dumps(counts)},
                type: 'bar',
                marker: {{color: 'rgba(99, 102, 241, 0.7)'}}
            }}], {{
                title: 'Message Distribution Over Time (30 bins)',
                xaxis: {{title: 'Time Period', tickangle: -45}},
                yaxis: {{title: 'Number of Messages'}},
                height: 400
            }});
"""
    
    html_content += """
        }
        
        // PDF Export function
        async function exportToPDF() {
            const { jsPDF } = window.jspdf;
            const doc = new jsPDF('p', 'mm', 'a4');
            const element = document.getElementById('report-content');
            
            try {
                const canvas = await html2canvas(element, {
                    scale: 2,
                    useCORS: true,
                    logging: false
                });
                
                const imgData = canvas.toDataURL('image/png');
                const imgWidth = 210; // A4 width in mm
                const pageHeight = 297; // A4 height in mm
                const imgHeight = (canvas.height * imgWidth) / canvas.width;
                let heightLeft = imgHeight;
                let position = 0;
                
                doc.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
                heightLeft -= pageHeight;
                
                while (heightLeft > 0) {
                    position = heightLeft - imgHeight;
                    doc.addPage();
                    doc.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
                    heightLeft -= pageHeight;
                }
                
                doc.save('ransom_chat_timeline_analysis_report.pdf');
            } catch (error) {
                alert('PDF export failed. Please try printing instead.');
                console.error(error);
            }
        }
        
        // Load charts when page is ready
        window.addEventListener('load', loadCharts);
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"[INFO] HTML report saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate HTML report from timeline analysis')
    parser.add_argument('json_file', type=Path, help='Input JSON analysis file')
    parser.add_argument('-o', '--output', type=Path, help='Output HTML file path', default=None)
    
    args = parser.parse_args()
    
    if not args.json_file.exists():
        print(f"[ERROR] JSON file does not exist: {args.json_file}")
        sys.exit(1)
    
    output_path = args.output or args.json_file.parent / f"{args.json_file.stem}_full_report.html"
    
    print(f"[INFO] Loading analysis from {args.json_file}")
    data = load_analysis(args.json_file)
    
    print("[INFO] Generating HTML report...")
    generate_html_report(data, output_path)


if __name__ == '__main__':
    main()

