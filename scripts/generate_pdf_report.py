#!/usr/bin/env python3
"""Generate PDF report from timeline analysis with storytelling."""

import json
import sys
from pathlib import Path
from datetime import datetime
import argparse

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
except ImportError:
    print("[ERROR] reportlab is required. Install with: pip install reportlab")
    sys.exit(1)


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


def create_pdf_report(data: dict, output_path: Path):
    """Generate PDF report with storytelling."""
    doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY
    )
    
    # Title page
    story.append(Paragraph("Ransom Chat Timeline Analysis Report", title_style))
    story.append(Spacer(1, 0.5*inch))
    
    summary = data.get('summary', {})
    date_range = summary.get('date_range', {})
    
    story.append(Paragraph(f"<b>Analysis Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"<b>Total Messages Analyzed:</b> {summary.get('total_messages', 0):,}", styles['Normal']))
    story.append(Paragraph(f"<b>Total Conversations:</b> {summary.get('total_conversations', 0):,}", styles['Normal']))
    if date_range.get('min') and date_range.get('max'):
        story.append(Paragraph(f"<b>Date Range:</b> {date_range['min']} to {date_range['max']}", styles['Normal']))
    
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(
        "This report presents a comprehensive analysis of ransom chat conversations, examining "
        "communication patterns, response times, and temporal behaviors across multiple ransomware groups. "
        "The analysis reveals critical insights into negotiation dynamics, operational patterns, and potential "
        "geographic indicators based on activity timing.",
        body_style
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # Response Time Analysis
    story.append(Paragraph("Response Time Analysis", heading_style))
    
    response_stats = data.get('response_time_by_party', {})
    if response_stats:
        story.append(Paragraph(
            "Response times between attackers and victims reveal distinct communication patterns. "
            "These metrics help understand negotiation dynamics and operational efficiency.",
            body_style
        ))
        story.append(Spacer(1, 0.1*inch))
        
        # Create table for response times
        table_data = [['Party', 'Count', 'Mean', 'Median', 'Min', 'Max', 'Outliers Removed']]
        
        for party, stats in response_stats.items():
            table_data.append([
                party.capitalize(),
                f"{stats.get('count', 0):,}",
                format_time(stats.get('mean', 0)),
                format_time(stats.get('median', 0)),
                format_time(stats.get('min', 0)),
                format_time(stats.get('max', 0)),
                f"{stats.get('outliers_removed', 0):,}"
            ])
        
        table = Table(table_data, colWidths=[1.2*inch, 0.8*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.2*inch))
        
        # Storytelling
        attacker_stats = response_stats.get('attacker', {})
        victim_stats = response_stats.get('victim', {})
        
        if attacker_stats and victim_stats:
            story.append(Paragraph(
                f"<b>Key Findings:</b> Attackers show a median response time of {format_time(attacker_stats.get('median', 0))}, "
                f"indicating relatively quick operational responses. However, the mean response time of "
                f"{format_time(attacker_stats.get('mean', 0))} suggests significant variability, with some "
                f"conversations experiencing extended delays. Victims, on the other hand, demonstrate a median "
                f"response time of {format_time(victim_stats.get('median', 0))}, with a mean of "
                f"{format_time(victim_stats.get('mean', 0))}. This pattern suggests that victims often require "
                f"more time to coordinate responses, potentially due to internal decision-making processes or "
                f"security team consultations.",
                body_style
            ))
    
    story.append(PageBreak())
    
    # Conversation Metrics
    story.append(Paragraph("Conversation Metrics", heading_style))
    
    conv_metrics = data.get('conversation_metrics', {})
    aggregate = conv_metrics.get('aggregate', {})
    
    if aggregate:
        story.append(Paragraph(
            "Understanding conversation characteristics helps identify negotiation patterns and operational "
            "efficiency across different ransomware groups.",
            body_style
        ))
        story.append(Spacer(1, 0.1*inch))
        
        metrics_data = [
            ['Metric', 'Value'],
            ['Total Conversations', f"{aggregate.get('total_conversations', 0):,}"],
            ['Avg Messages/Conversation', f"{aggregate.get('avg_messages_per_conversation', 0):.1f}"],
            ['Median Messages/Conversation', f"{aggregate.get('median_messages_per_conversation', 0):.1f}"],
        ]
        
        if aggregate.get('avg_duration_minutes'):
            metrics_data.append(['Avg Duration', format_time(aggregate['avg_duration_minutes'])])
        if aggregate.get('avg_response_time_across_conversations'):
            metrics_data.append(['Avg Response Time', format_time(aggregate['avg_response_time_across_conversations'])])
        
        metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph(
            f"The analysis reveals an average of {aggregate.get('avg_messages_per_conversation', 0):.1f} messages per "
            f"conversation, with a median of {aggregate.get('median_messages_per_conversation', 0):.1f} messages. "
            f"This indicates that while some negotiations are brief, others involve extensive back-and-forth "
            f"communication. The average conversation duration of {format_time(aggregate.get('avg_duration_minutes', 0))} "
            f"suggests that ransomware negotiations can span significant periods, requiring sustained engagement "
            f"from both parties.",
            body_style
        ))
    
    story.append(PageBreak())
    
    # Temporal Analysis
    story.append(Paragraph("Temporal Activity Analysis", heading_style))
    
    hourly_activity = data.get('hourly_activity', {})
    
    if hourly_activity:
        story.append(Paragraph(
            "Analyzing activity patterns by hour provides insights into operational schedules and potential "
            "geographic indicators based on GMT timing.",
            body_style
        ))
        story.append(Spacer(1, 0.1*inch))
        
        # Local time analysis
        peak_hours = hourly_activity.get('peak_hours', [])
        peak_count = hourly_activity.get('peak_count', 0)
        
        if peak_hours:
            story.append(Paragraph(
                f"<b>Peak Activity Hours (Local Time):</b> The analysis identifies peak activity at hours "
                f"{', '.join(map(str, peak_hours))} with {peak_count:,} messages. This suggests coordinated "
                f"operational windows when ransomware groups are most active.",
                body_style
            ))
            story.append(Spacer(1, 0.1*inch))
        
        # GMT analysis
        gmt_data = hourly_activity.get('gmt', {})
        if gmt_data:
            gmt_peak_hours = gmt_data.get('peak_hours', [])
            gmt_peak_count = gmt_data.get('peak_count', 0)
            timezone_matches = gmt_data.get('timezone_matches', {})
            
            story.append(Paragraph(
                f"<b>GMT Activity Analysis:</b> When analyzed in GMT (UTC), peak activity occurs at hours "
                f"{', '.join(map(str, gmt_peak_hours))} with {gmt_peak_count:,} messages.",
                body_style
            ))
            story.append(Spacer(1, 0.1*inch))
            
            if timezone_matches:
                story.append(Paragraph(
                    "<b>Potential Geographic Indicators:</b> Based on GMT activity patterns, the following "
                    "timezones show alignment with peak activity hours:",
                    body_style
                ))
                story.append(Spacer(1, 0.1*inch))
                
                tz_data = [['Timezone', 'Business Hours Overlap']]
                for tz_name, overlap in sorted(timezone_matches.items(), key=lambda x: x[1], reverse=True):
                    tz_data.append([tz_name, f"{overlap} hours"])
                
                tz_table = Table(tz_data, colWidths=[3*inch, 2*inch])
                tz_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                ]))
                story.append(tz_table)
                story.append(Spacer(1, 0.2*inch))
                
                story.append(Paragraph(
                    "This analysis suggests potential geographic locations where ransomware operators may be "
                    "based, though it should be interpreted cautiously as operators may use VPNs or operate "
                    "across multiple timezones.",
                    body_style
                ))
    
    story.append(PageBreak())
    
    # Timestamp Distribution
    story.append(Paragraph("Message Distribution Over Time", heading_style))
    
    timestamp_dist = data.get('timestamp_distribution', {})
    if timestamp_dist:
        outliers_removed = timestamp_dist.get('outliers_removed', 0)
        total_original = timestamp_dist.get('total_original', 0)
        
        story.append(Paragraph(
            f"The temporal distribution of messages shows activity patterns across the analyzed time period. "
            f"Outlier removal was applied to focus on normal operational patterns, removing {outliers_removed:,} "
            f"extreme values from {total_original:,} total timestamps.",
            body_style
        ))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph(
            f"<b>Analysis Period:</b> {timestamp_dist.get('min_timestamp', 'N/A')} to "
            f"{timestamp_dist.get('max_timestamp', 'N/A')}",
            styles['Normal']
        ))
    
    story.append(PageBreak())
    
    # Conclusions
    story.append(Paragraph("Conclusions and Recommendations", heading_style))
    
    story.append(Paragraph(
        "<b>Key Insights:</b>",
        styles['Heading3']
    ))
    
    insights = [
        "Response time patterns reveal distinct operational behaviors between attackers and victims, "
        "with attackers generally responding faster but showing high variability.",
        
        "Conversation metrics indicate that ransomware negotiations are often protracted processes, "
        "requiring sustained engagement over extended periods.",
        
        "Temporal analysis suggests potential geographic indicators based on GMT activity patterns, "
        "though these should be interpreted with caution given the use of VPNs and distributed operations.",
        
        "Outlier removal techniques help identify normal operational patterns by filtering extreme "
        "values that may represent exceptional circumstances or data anomalies."
    ]
    
    for i, insight in enumerate(insights, 1):
        story.append(Paragraph(f"{i}. {insight}", body_style))
        story.append(Spacer(1, 0.1*inch))
    
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph(
        "<b>Recommendations:</b>",
        styles['Heading3']
    ))
    
    recommendations = [
        "Monitor response time patterns to identify potential negotiation strategies and operational changes.",
        
        "Use temporal analysis to predict peak activity windows and allocate resources accordingly.",
        
        "Combine GMT analysis with other intelligence sources to build a more complete picture of "
        "ransomware group operations.",
        
        "Continuously update analysis as new data becomes available to track evolving patterns."
    ]
    
    for i, rec in enumerate(recommendations, 1):
        story.append(Paragraph(f"{i}. {rec}", body_style))
        story.append(Spacer(1, 0.1*inch))
    
    # Build PDF
    doc.build(story)
    print(f"[INFO] PDF report saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate PDF report from timeline analysis')
    parser.add_argument('json_file', type=Path, help='Input JSON analysis file')
    parser.add_argument('-o', '--output', type=Path, help='Output PDF file path', default=None)
    
    args = parser.parse_args()
    
    if not args.json_file.exists():
        print(f"[ERROR] JSON file does not exist: {args.json_file}")
        sys.exit(1)
    
    output_path = args.output or args.json_file.parent / f"{args.json_file.stem}_report.pdf"
    
    print(f"[INFO] Loading analysis from {args.json_file}")
    data = load_analysis(args.json_file)
    
    print("[INFO] Generating PDF report...")
    create_pdf_report(data, output_path)


if __name__ == '__main__':
    main()

