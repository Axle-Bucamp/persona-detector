#!/usr/bin/env python3
"""Analyze ransom chat timeline metrics and response time distributions."""

import csv
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from semantic_detector.web.app.core.timestamp_analyzer import TimestampAnalyzer
from semantic_detector.web.app.core.text_utils import (
    sanitize_text, parse_raw_text, parse_identifier, parse_json_string
)
import numpy as np


def sanitize_json_for_output(obj: Union[dict, list, str, int, float, None]) -> Union[dict, list, str, int, float, None]:
    """Recursively sanitize JSON-serializable objects."""
    if isinstance(obj, dict):
        return {k: sanitize_json_for_output(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json_for_output(item) for item in obj]
    elif isinstance(obj, str):
        return parse_json_string(obj)  # Use JSON-safe parser
    elif isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    else:
        # Convert to string and sanitize
        try:
            return sanitize_text(str(obj), preserve_newlines=True)
        except Exception:
            return ''
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats
from scipy.stats import norm, ks_2samp, chi2_contingency


def load_csv_data(csv_path: Path) -> List[Dict[str, Any]]:
    """Load CSV data into list of dictionaries with safe text parsing."""
    rows = []
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Sanitize all text fields using parse_raw_text
                for key, value in row.items():
                    if isinstance(value, str):
                        row[key] = parse_raw_text(value)  # Removes weird characters
                
                # Convert response_time_minutes to float if present
                if row.get('response_time_minutes'):
                    try:
                        row['response_time_minutes'] = float(row['response_time_minutes'])
                    except ValueError:
                        row['response_time_minutes'] = None
                else:
                    row['response_time_minutes'] = None
                
                # Convert price_amount to float if present
                if row.get('price_amount'):
                    try:
                        row['price_amount'] = float(row['price_amount'])
                    except ValueError:
                        row['price_amount'] = None
                else:
                    row['price_amount'] = None
                
                rows.append(row)
    except UnicodeDecodeError as e:
        print(f"[ERROR] Encoding error reading CSV: {e}", file=sys.stderr)
        # Try with latin-1 as fallback
        try:
            with open(csv_path, 'r', encoding='latin-1', errors='replace') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key, value in row.items():
                        if isinstance(value, str):
                            row[key] = parse_raw_text(value)  # Removes weird characters
                    rows.append(row)
        except Exception as e2:
            print(f"[ERROR] Failed to read CSV even with latin-1: {e2}", file=sys.stderr)
            raise
    except Exception as e:
        print(f"[ERROR] Unexpected error reading CSV: {e}", file=sys.stderr)
        raise
    
    return rows


def group_by_conversation(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group messages by chat_id (conversation)."""
    conversations = defaultdict(list)
    for row in rows:
        chat_id = row.get('chat_id', 'unknown')
        conversations[chat_id].append(row)
    
    # Sort messages within each conversation by message_index
    for chat_id in conversations:
        conversations[chat_id].sort(key=lambda x: int(x.get('message_index', 0)))
    
    return dict(conversations)


def remove_outliers_iqr(values: List[float]) -> Tuple[List[float], List[float]]:
    """Remove outliers using IQR method."""
    if len(values) < 4:
        return values, []
    
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    filtered = [v for v in values if lower_bound <= v <= upper_bound]
    outliers = [v for v in values if v < lower_bound or v > upper_bound]
    
    return filtered, outliers


def remove_outliers_zscore(values: List[float], threshold: float = 3.0) -> Tuple[List[float], List[float]]:
    """Remove outliers using Z-score method."""
    if len(values) < 3:
        return values, []
    
    mean = np.mean(values)
    std = np.std(values)
    
    if std == 0:
        return values, []
    
    z_scores = np.abs([(v - mean) / std for v in values])
    filtered = [v for v, z in zip(values, z_scores) if z <= threshold]
    outliers = [v for v, z in zip(values, z_scores) if z > threshold]
    
    return filtered, outliers


def calculate_response_times_per_party(conversations: Dict[str, List[Dict[str, Any]]], 
                                       timestamp_analyzer: TimestampAnalyzer,
                                       remove_outliers: bool = True) -> Dict[str, Dict[str, Any]]:
    """Calculate response time statistics per party (attacker vs victim)."""
    attacker_times = []
    victim_times = []
    
    for chat_id, messages in conversations.items():
        for i in range(1, len(messages)):
            prev_msg = messages[i - 1]
            curr_msg = messages[i]
            
            # Only calculate if parties are different (actual response)
            if prev_msg.get('party') != curr_msg.get('party'):
                rt = curr_msg.get('response_time_minutes')
                if rt is not None and rt > 0:
                    party = curr_msg.get('party', '').lower()
                    if 'victim' in party or 'victim' in prev_msg.get('party', '').lower():
                        victim_times.append(rt)
                    else:
                        attacker_times.append(rt)
    
    stats = {}
    
    if attacker_times:
        # Remove outliers before statistics
        if remove_outliers:
            attacker_times_filtered, attacker_outliers = remove_outliers_zscore(attacker_times)
        else:
            attacker_times_filtered, attacker_outliers = attacker_times, []
        
        attacker_array = np.array(attacker_times_filtered)
        stats['attacker'] = {
            'count': len(attacker_times_filtered),
            'count_original': len(attacker_times),
            'outliers_removed': len(attacker_outliers),
            'mean': float(np.mean(attacker_array)),
            'median': float(np.median(attacker_array)),
            'std': float(np.std(attacker_array)),
            'min': float(np.min(attacker_array)),
            'max': float(np.max(attacker_array)),
            'values': attacker_times_filtered,
            'outliers': attacker_outliers
        }
    
    if victim_times:
        # Remove outliers before statistics
        if remove_outliers:
            victim_times_filtered, victim_outliers = remove_outliers_zscore(victim_times)
        else:
            victim_times_filtered, victim_outliers = victim_times, []
        
        victim_array = np.array(victim_times_filtered)
        stats['victim'] = {
            'count': len(victim_times_filtered),
            'count_original': len(victim_times),
            'outliers_removed': len(victim_outliers),
            'mean': float(np.mean(victim_array)),
            'median': float(np.median(victim_array)),
            'std': float(np.std(victim_array)),
            'min': float(np.min(victim_array)),
            'max': float(np.max(victim_array)),
            'values': victim_times_filtered,
            'outliers': victim_outliers
        }
    
    return stats


def create_timestamp_histogram(rows: List[Dict[str, Any]], n_bins: int = 30) -> Dict[str, Any]:
    """Create histogram of messages binned by timestamp."""
    timestamps = []
    timestamp_analyzer = TimestampAnalyzer()
    
    for row in rows:
        ts_str = row.get('timestamp')
        # Explicitly exclude null, empty, or invalid timestamps
        if ts_str and ts_str.strip() and ts_str.lower() not in ['null', 'none', '']:
            dt = timestamp_analyzer.parse_timestamp(ts_str)
            if dt:
                timestamps.append(dt)
    
    if not timestamps:
        return {'bin_edges': [], 'counts': [], 'bin_labels': []}
    
    # Convert to numeric (seconds since epoch)
    timestamps_numeric = [ts.timestamp() for ts in timestamps]
    min_ts = min(timestamps_numeric)
    max_ts = max(timestamps_numeric)
    
    # Create bins
    bin_edges = np.linspace(min_ts, max_ts, n_bins + 1)
    counts, bin_edges_actual = np.histogram(timestamps_numeric, bins=bin_edges)
    
    # Create labels
    bin_labels = []
    for i in range(len(bin_edges_actual) - 1):
        center = (bin_edges_actual[i] + bin_edges_actual[i + 1]) / 2
        dt_center = datetime.fromtimestamp(center)
        bin_labels.append(dt_center.strftime('%Y-%m-%d %H:%M'))
    
    return {
        'bin_edges': bin_edges_actual.tolist(),
        'counts': counts.tolist(),
        'bin_labels': bin_labels,
        'min_timestamp': datetime.fromtimestamp(min_ts).isoformat(),
        'max_timestamp': datetime.fromtimestamp(max_ts).isoformat()
    }


def analyze_hourly_activity(rows: List[Dict[str, Any]], 
                            timestamp_analyzer: TimestampAnalyzer) -> Dict[str, Any]:
    """Analyze message activity by hour of day."""
    hourly_counts = defaultdict(int)
    hourly_by_party = defaultdict(lambda: defaultdict(int))
    gmt_hourly_counts = defaultdict(int)
    gmt_hourly_by_party = defaultdict(lambda: defaultdict(int))
    
    for row in rows:
        ts_str = row.get('timestamp')
        # Explicitly exclude null, empty, or invalid timestamps
        if ts_str and ts_str.strip() and ts_str.lower() not in ['null', 'none', '']:
            dt = timestamp_analyzer.parse_timestamp(ts_str)
            if dt:
                # Local hour
                hour = dt.hour
                hourly_counts[hour] += 1
                party = row.get('party', 'unknown')
                hourly_by_party[party][hour] += 1
                
                # GMT hour (UTC)
                if dt.tzinfo is not None:
                    # Convert to UTC
                    from datetime import timezone
                    gmt_dt = dt.astimezone(timezone.utc)
                    gmt_hour = gmt_dt.hour
                else:
                    # Assume UTC if no timezone info
                    gmt_hour = dt.hour
                
                gmt_hourly_counts[gmt_hour] += 1
                gmt_hourly_by_party[party][gmt_hour] += 1
    
    # Convert to sorted lists
    hours = sorted(hourly_counts.keys())
    counts = [hourly_counts[h] for h in hours]
    
    gmt_hours = sorted(gmt_hourly_counts.keys())
    gmt_counts = [gmt_hourly_counts[h] for h in gmt_hours]
    
    # Find peak hours
    if counts:
        max_count = max(counts)
        peak_hours = [hours[i] for i, c in enumerate(counts) if c == max_count]
    else:
        peak_hours = []
    
    if gmt_counts:
        max_gmt_count = max(gmt_counts)
        peak_gmt_hours = [gmt_hours[i] for i, c in enumerate(gmt_counts) if c == max_gmt_count]
    else:
        peak_gmt_hours = []
    
    # Party breakdown
    party_hourly = {}
    for party, party_counts in hourly_by_party.items():
        party_hours = sorted(party_counts.keys())
        party_hourly[party] = {
            'hours': party_hours,
            'counts': [party_counts[h] for h in party_hours]
        }
    
    gmt_party_hourly = {}
    for party, party_counts in gmt_hourly_by_party.items():
        party_hours = sorted(party_counts.keys())
        gmt_party_hourly[party] = {
            'hours': party_hours,
            'counts': [party_counts[h] for h in party_hours]
        }
    
    # Map GMT hours to potential timezones/countries
    # Common business hours by GMT offset (6am-6pm local time = 12 hours)
    timezone_mapping = {
        'GMT+0 (UK/Portugal)': list(range(6, 19)),  # 6am-6pm local = 6-18 GMT
        'GMT+1 (Central Europe)': list(range(5, 18)),  # 6am-6pm local = 5-17 GMT
        'GMT+2 (Eastern Europe)': list(range(4, 17)),  # 6am-6pm local = 4-16 GMT
        'GMT+3 (Russia/Moscow)': list(range(3, 16)),  # 6am-6pm local = 3-15 GMT
        'GMT+5 (Pakistan)': list(range(1, 14)),  # 6am-6pm local = 1-13 GMT (wraps)
        'GMT+8 (China/Singapore)': list(range(22, 24)) + list(range(0, 11)),  # 6am-6pm local = 22-10 GMT (wraps)
        'GMT+9 (Japan/Korea)': list(range(21, 24)) + list(range(0, 10)),  # 6am-6pm local = 21-9 GMT (wraps)
        'GMT-5 (US Eastern)': list(range(11, 24)),  # 6am-6pm local = 11-23 GMT
        'GMT-8 (US Pacific)': list(range(14, 24)) + list(range(0, 3)),  # 6am-6pm local = 14-2 GMT (wraps)
    }
    
    # Create normal distribution for business hours (6am-6pm, centered at 12pm)
    business_hours_normal = list(range(6, 19))  # 6am to 6pm (18:00)
    # Create expected distribution (normal curve centered at 12pm/noon)
    expected_distribution = []
    for hour in range(24):
        # Normal distribution centered at 12 (noon), std dev of 3 hours
        prob = norm.pdf(hour, loc=12, scale=3)
        expected_distribution.append(prob)
    # Normalize to sum to 1
    expected_distribution = np.array(expected_distribution)
    expected_distribution = expected_distribution / expected_distribution.sum()
    
    # Compare GMT distribution to expected normal business hours
    gmt_distribution_comparison = {}
    if gmt_counts and len(gmt_counts) == len(gmt_hours):
        # Create observed distribution (normalized)
        observed_gmt = np.zeros(24)
        for hour, count in zip(gmt_hours, gmt_counts):
            observed_gmt[hour] = count
        observed_gmt_normalized = observed_gmt / observed_gmt.sum() if observed_gmt.sum() > 0 else observed_gmt
        
        # Kolmogorov-Smirnov test for distribution comparison
        # Create cumulative distributions
        expected_cdf = np.cumsum(expected_distribution)
        observed_cdf = np.cumsum(observed_gmt_normalized)
        
        # KS test
        ks_statistic, ks_pvalue = ks_2samp(expected_cdf, observed_cdf)
        
        # Chi-square test
        # Create frequency arrays
        observed_freq = observed_gmt
        expected_freq = expected_distribution * observed_gmt.sum()
        # Remove zeros for chi-square (need at least 1 expected frequency)
        mask = expected_freq > 0.1  # Keep bins with meaningful expected frequency
        if mask.sum() > 1:
            obs_masked = observed_freq[mask]
            exp_masked = expected_freq[mask]
            # Chi-square goodness of fit
            chi2_stat = np.sum((obs_masked - exp_masked) ** 2 / exp_masked)
            # Degrees of freedom
            df = len(obs_masked) - 1
            # P-value from chi-square distribution
            from scipy.stats import chi2
            chi2_pvalue = 1 - chi2.cdf(chi2_stat, df) if df > 0 else 1.0
        else:
            chi2_stat, chi2_pvalue = 0, 1.0
        
        gmt_distribution_comparison = {
            'ks_statistic': float(ks_statistic),
            'ks_pvalue': float(ks_pvalue),
            'ks_significant': ks_pvalue < 0.05,
            'chi2_statistic': float(chi2_stat) if 'chi2_stat' in locals() else 0,
            'chi2_pvalue': float(chi2_pvalue) if 'chi2_pvalue' in locals() else 1.0,
            'chi2_significant': chi2_pvalue < 0.05 if 'chi2_pvalue' in locals() else False,
            'observed_distribution': observed_gmt_normalized.tolist(),
            'expected_distribution': expected_distribution.tolist()
        }
    
    # Find best matching timezone based on peak activity and distribution similarity
    timezone_matches = {}
    timezone_scores = {}
    
    for tz_name, business_hours in timezone_mapping.items():
        # Calculate overlap with peak hours
        overlap = sum(1 for h in peak_gmt_hours if h in business_hours)
        
        # Calculate distribution similarity (if we have GMT data)
        if gmt_distribution_comparison:
            # Shift expected distribution to match timezone
            tz_offset = int(tz_name.split('+')[1].split()[0]) if '+' in tz_name else -int(tz_name.split('-')[1].split()[0])
            shifted_expected = np.roll(expected_distribution, -tz_offset)
            shifted_expected_normalized = shifted_expected / shifted_expected.sum()
            
            # Calculate correlation
            if len(observed_gmt_normalized) == 24:
                correlation = np.corrcoef(observed_gmt_normalized, shifted_expected_normalized)[0, 1]
                timezone_scores[tz_name] = {
                    'overlap': overlap,
                    'correlation': float(correlation) if not np.isnan(correlation) else 0,
                    'score': overlap * 0.3 + abs(correlation) * 0.7  # Weighted score
                }
        
        if overlap > 0:
            timezone_matches[tz_name] = overlap
    
    # Find best match
    best_match = None
    if timezone_scores:
        best_match = max(timezone_scores.items(), key=lambda x: x[1]['score'])
    
    return {
        'hours': hours,
        'counts': counts,
        'peak_hours': peak_hours,
        'peak_count': max_count if counts else 0,
        'by_party': party_hourly,
        'gmt': {
            'hours': gmt_hours,
            'counts': gmt_counts,
            'peak_hours': peak_gmt_hours,
            'peak_count': max_gmt_count if gmt_counts else 0,
            'by_party': gmt_party_hourly,
            'timezone_matches': timezone_matches,
            'timezone_scores': timezone_scores,
            'best_match': best_match[0] if best_match else None,
            'best_match_score': best_match[1] if best_match else None,
            'distribution_comparison': gmt_distribution_comparison
        }
    }


def calculate_conversation_metrics(conversations: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Calculate metrics per conversation with filtering by attacker/chat_id."""
    conversation_stats = []
    attacker_stats = defaultdict(lambda: {'conversations': [], 'messages': [], 'durations': [], 'response_times': []})
    
    for chat_id, messages in conversations.items():
        # Count messages by party
        party_counts = defaultdict(int)
        timestamps = []
        response_times = []
        timestamp_analyzer = TimestampAnalyzer()
        
        # Identify attacker (non-victim party)
        attacker_name = None
        for msg in messages:
            party = msg.get('party', 'unknown')
            party_counts[party] += 1
            if party.lower() not in ['victim', 'client'] and not attacker_name:
                attacker_name = party
        
        for msg in messages:
            ts_str = msg.get('timestamp')
            # Explicitly exclude null, empty, or invalid timestamps
            if ts_str and ts_str.strip() and ts_str.lower() not in ['null', 'none', '']:
                dt = timestamp_analyzer.parse_timestamp(ts_str)
                if dt:
                    timestamps.append(dt)
            
            rt = msg.get('response_time_minutes')
            if rt is not None and rt > 0:
                response_times.append(rt)
        
        # Calculate conversation duration
        duration_minutes = None
        if len(timestamps) >= 2:
            # Normalize timestamps (remove timezone info for comparison)
            normalized_ts = []
            for ts in timestamps:
                if ts.tzinfo is not None:
                    # Convert to naive datetime
                    normalized_ts.append(ts.replace(tzinfo=None))
                else:
                    normalized_ts.append(ts)
            duration = max(normalized_ts) - min(normalized_ts)
            duration_minutes = duration.total_seconds() / 60.0
        
        stats = {
            'chat_id': chat_id,
            'attacker': attacker_name or 'unknown',
            'total_messages': len(messages),
            'party_counts': dict(party_counts),
            'duration_minutes': duration_minutes,
            'avg_response_time': float(np.mean(response_times)) if response_times else None,
            'median_response_time': float(np.median(response_times)) if response_times else None,
            'num_exchanges': len(response_times)
        }
        
        conversation_stats.append(stats)
        
        # Group by attacker
        if attacker_name:
            attacker_stats[attacker_name]['conversations'].append(chat_id)
            attacker_stats[attacker_name]['messages'].append(len(messages))
            if duration_minutes:
                attacker_stats[attacker_name]['durations'].append(duration_minutes)
            if response_times:
                attacker_stats[attacker_name]['response_times'].extend(response_times)
    
    # Aggregate statistics (global)
    total_messages = [s['total_messages'] for s in conversation_stats]
    durations = [s['duration_minutes'] for s in conversation_stats if s['duration_minutes'] is not None]
    avg_response_times = [s['avg_response_time'] for s in conversation_stats if s['avg_response_time'] is not None]
    
    # Aggregate by attacker
    attacker_aggregates = {}
    for attacker, data in attacker_stats.items():
        if data['messages']:
            attacker_aggregates[attacker] = {
                'conversation_count': len(data['conversations']),
                'avg_messages': float(np.mean(data['messages'])),
                'median_messages': float(np.median(data['messages'])),
                'avg_duration': float(np.mean(data['durations'])) if data['durations'] else None,
                'median_duration': float(np.median(data['durations'])) if data['durations'] else None,
                'avg_response_time': float(np.mean(data['response_times'])) if data['response_times'] else None
            }
    
    return {
        'per_conversation': conversation_stats,
        'aggregate': {
            'total_conversations': len(conversations),
            'avg_messages_per_conversation': float(np.mean(total_messages)) if total_messages else 0,
            'median_messages_per_conversation': float(np.median(total_messages)) if total_messages else 0,
            'avg_duration_minutes': float(np.mean(durations)) if durations else None,
            'median_duration_minutes': float(np.median(durations)) if durations else None,
            'avg_response_time_across_conversations': float(np.mean(avg_response_times)) if avg_response_times else None
        },
        'by_attacker': attacker_aggregates
    }


def generate_report(csv_path: Path, output_path: Optional[Path] = None):
    """Generate comprehensive timeline analysis report."""
    print(f"[INFO] Loading data from {csv_path}")
    rows = load_csv_data(csv_path)
    print(f"[INFO] Loaded {len(rows)} messages")
    
    # Group by conversation
    conversations = group_by_conversation(rows)
    print(f"[INFO] Found {len(conversations)} conversations")
    
    timestamp_analyzer = TimestampAnalyzer()
    
    # Calculate response time statistics per party
    print("[INFO] Calculating response time statistics per party...")
    response_time_stats = calculate_response_times_per_party(conversations, timestamp_analyzer)
    
    # Create timestamp histogram
    print("[INFO] Creating timestamp histogram...")
    timestamp_histogram = create_timestamp_histogram(rows, n_bins=30)
    
    # Analyze hourly activity
    print("[INFO] Analyzing hourly activity patterns...")
    hourly_activity = analyze_hourly_activity(rows, timestamp_analyzer)
    
    # Calculate conversation metrics
    print("[INFO] Calculating conversation metrics...")
    conversation_metrics = calculate_conversation_metrics(conversations)
    
    # Compile report
    report = {
        'summary': {
            'total_messages': len(rows),
            'total_conversations': len(conversations),
            'date_range': {
                'min': timestamp_histogram.get('min_timestamp'),
                'max': timestamp_histogram.get('max_timestamp')
            }
        },
        'response_time_by_party': response_time_stats,
        'timestamp_distribution': timestamp_histogram,
        'hourly_activity': hourly_activity,
        'conversation_metrics': conversation_metrics
    }
    
    # Save report
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8', errors='replace') as f:
            # Sanitize text fields in report before writing
            sanitized_report = sanitize_json_for_output(report)
            json.dump(sanitized_report, f, indent=2, default=str, ensure_ascii=False)
        print(f"[INFO] Report saved to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("TIMELINE ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nTotal Messages: {len(rows)}")
    print(f"Total Conversations: {len(conversations)}")
    
    if response_time_stats:
        print("\n--- Response Time Statistics ---")
        for party, stats in response_time_stats.items():
            print(f"\n{party.upper()}:")
            print(f"  Count: {stats['count']}")
            print(f"  Mean: {stats['mean']:.2f} minutes ({stats['mean']/60:.2f} hours)")
            print(f"  Median: {stats['median']:.2f} minutes ({stats['median']/60:.2f} hours)")
            print(f"  Min: {stats['min']:.2f} minutes")
            print(f"  Max: {stats['max']:.2f} minutes ({stats['max']/60:.2f} hours)")
    
    if conversation_metrics['aggregate']:
        agg = conversation_metrics['aggregate']
        print("\n--- Conversation Metrics ---")
        print(f"Average messages per conversation: {agg['avg_messages_per_conversation']:.2f}")
        print(f"Median messages per conversation: {agg['median_messages_per_conversation']:.2f}")
        if agg['avg_duration_minutes']:
            print(f"Average conversation duration: {agg['avg_duration_minutes']:.2f} minutes ({agg['avg_duration_minutes']/60:.2f} hours)")
        if agg['avg_response_time_across_conversations']:
            print(f"Average response time across conversations: {agg['avg_response_time_across_conversations']:.2f} minutes")
    
    if hourly_activity['peak_hours']:
        print("\n--- Hourly Activity ---")
        print(f"Peak activity hours: {hourly_activity['peak_hours']} ({hourly_activity['peak_count']} messages)")
        print("\nActivity by party:")
        for party, data in hourly_activity['by_party'].items():
            if data['hours']:
                peak_idx = data['counts'].index(max(data['counts']))
                peak_hour = data['hours'][peak_idx]
                print(f"  {party}: Peak at hour {peak_hour} ({data['counts'][peak_idx]} messages)")
    
    print("\n" + "="*80)
    
    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze ransom chat timeline metrics')
    parser.add_argument('csv_file', type=Path, help='Input CSV file path')
    parser.add_argument('-o', '--output', type=Path, help='Output JSON report path', default=None)
    
    args = parser.parse_args()
    
    if not args.csv_file.exists():
        print(f"[ERROR] CSV file does not exist: {args.csv_file}")
        sys.exit(1)
    
    output_path = args.output or args.csv_file.parent / f"{args.csv_file.stem}_timeline_analysis.json"
    
    generate_report(args.csv_file, output_path)


if __name__ == '__main__':
    main()

