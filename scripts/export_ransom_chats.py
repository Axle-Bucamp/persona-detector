#!/usr/bin/env python3
"""Export ransom chat JSON files to unified CSV format."""

import json
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from semantic_detector.web.app.core.price_extractor import PriceExtractor
from semantic_detector.web.app.core.timestamp_analyzer import TimestampAnalyzer
from semantic_detector.web.app.core.text_utils import (
    sanitize_text, clean_csv_value, parse_raw_text, 
    parse_identifier, parse_json_string
)


def find_json_files(directory: Path) -> List[Path]:
    """Find all JSON files recursively."""
    json_files = []
    for json_file in directory.rglob("*.json"):
        # Skip parsers directory
        if "parsers" not in str(json_file):
            json_files.append(json_file)
    return sorted(json_files)


def extract_group_name(file_path: Path, base_dir: Path) -> str:
    """Extract group name from directory structure."""
    relative = file_path.relative_to(base_dir)
    parts = relative.parts
    # Group name is typically the parent directory of the JSON file
    if len(parts) > 1:
        return parts[0]  # First directory after base
    return "unknown"


def process_json_file(file_path: Path, base_dir: Path, price_extractor: PriceExtractor, 
                     timestamp_analyzer: TimestampAnalyzer) -> List[Dict[str, Any]]:
    """Process a single JSON file and return rows."""
    rows = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chat_id = data.get('chat_id', file_path.stem)
        messages = data.get('messages', [])
        group_name = extract_group_name(file_path, base_dir)
        
        if not messages:
            return rows
        
        # Calculate response times for this chat
        response_times = timestamp_analyzer.calculate_response_times(messages)
        
        for idx, message in enumerate(messages):
            # Safely extract and sanitize content using parse_raw_text
            content = message.get('content', '')
            content = parse_raw_text(content)  # Removes all weird characters
            
            party = parse_identifier(message.get('party', 'Unknown'), allow_unicode=True)
            timestamp = parse_raw_text(message.get('timestamp', ''))
            
            # Extract price (on sanitized content)
            price_info = price_extractor.extract_first_price(content)
            price_amount = price_info['amount'] if price_info else None
            price_currency = parse_identifier(price_info['currency'], allow_unicode=True) if price_info else None
            
            # Get response time
            response_time_minutes = response_times[idx] if idx < len(response_times) else None
            
            # Clean all values for CSV
            row = {
                'chat_id': clean_csv_value(chat_id),
                'message_index': idx,
                'party': clean_csv_value(party),
                'content': clean_csv_value(content),
                'timestamp': clean_csv_value(timestamp),
                'response_time_minutes': response_time_minutes if response_time_minutes is not None else '',
                'price_amount': price_amount if price_amount else '',
                'price_currency': clean_csv_value(price_currency) if price_currency else '',
                'group_name': clean_csv_value(group_name)
            }
            
            rows.append(row)
    
    except json.JSONDecodeError as e:
        print(f"[WARNING] Invalid JSON in {file_path}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"[WARNING] Error processing {file_path}: {e}", file=sys.stderr)
    
    return rows


def export_to_csv(input_dir: Path, output_file: Path):
    """Export all JSON files to CSV."""
    print(f"[INFO] Scanning directory: {input_dir}")
    
    json_files = find_json_files(input_dir)
    print(f"[INFO] Found {len(json_files)} JSON files")
    
    if not json_files:
        print("[ERROR] No JSON files found!")
        return
    
    price_extractor = PriceExtractor()
    timestamp_analyzer = TimestampAnalyzer()
    
    # CSV columns
    columns = [
        'chat_id', 'message_index', 'party', 'content', 'timestamp',
        'response_time_minutes', 'price_amount', 'price_currency', 'group_name'
    ]
    
    total_rows = 0
    
    with open(output_file, 'w', newline='', encoding='utf-8', errors='replace') as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        
        for i, json_file in enumerate(json_files, 1):
            rows = process_json_file(json_file, input_dir, price_extractor, timestamp_analyzer)
            
            for row in rows:
                try:
                    # Ensure all values are clean and safe
                    clean_row = {k: clean_csv_value(v) for k, v in row.items()}
                    writer.writerow(clean_row)
                except Exception as e:
                    print(f"[WARNING] Error writing row from {json_file}: {e}", file=sys.stderr)
                    # Try to write with even more aggressive cleaning
                    try:
                        ultra_clean_row = {}
                        for k, v in row.items():
                            if isinstance(v, (int, float)):
                                ultra_clean_row[k] = v
                            else:
                                ultra_clean_row[k] = clean_csv_value(str(v) if v else '')
                        writer.writerow(ultra_clean_row)
                    except Exception as e2:
                        print(f"[ERROR] Failed to write row even after aggressive cleaning: {e2}", file=sys.stderr)
                        continue
            
            total_rows += len(rows)
            
            if i % 10 == 0:
                print(f"[INFO] Processed {i}/{len(json_files)} files ({total_rows} rows)...")
    
    print(f"[INFO] Export complete: {total_rows} rows written to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Export ransom chat JSON files to CSV')
    parser.add_argument('input_dir', type=Path, help='Input directory containing JSON files')
    parser.add_argument('output_file', type=Path, help='Output CSV file path')
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        print(f"[ERROR] Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Create output directory if needed
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    export_to_csv(args.input_dir, args.output_file)


if __name__ == '__main__':
    main()

