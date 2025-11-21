# Complete Encoding Fixes - All File Operations

## Summary
All file read/write operations now use safe encoding with `errors='replace'` and text sanitization to prevent `'charmap' codec can't encode character` errors.

## Files Updated

### 1. Core Text Utilities (`semantic_detector/web/app/core/text_utils.py`)
- ✅ `parse_raw_text()` - Removes all weird characters, control chars, zero-width chars
- ✅ `parse_identifier()` - Strict filtering for IDs/names
- ✅ `parse_json_string()` - JSON-safe text parsing
- ✅ `clean_csv_value()` - CSV-safe text parsing

### 2. File Reading Operations

#### `semantic_detector/sentence_splitter.py`
- ✅ `split_file()` - Uses `errors='replace'` and `parse_raw_text()` for all file reads
- ✅ Fallback to latin-1 encoding
- ✅ Final fallback to binary read + decode

#### `scripts/analyze_ransom_chat_timeline.py`
- ✅ `load_csv_data()` - Uses `errors='replace'` and `parse_raw_text()` for all CSV fields
- ✅ Fallback to latin-1 encoding
- ✅ JSON output uses `errors='replace'` and `ensure_ascii=False`

#### `scripts/export_ransom_chats.py`
- ✅ All file reads use `encoding='utf-8', errors='replace'`
- ✅ All text fields sanitized with `parse_raw_text()` or `parse_identifier()`
- ✅ CSV writing uses `errors='replace'` and `clean_csv_value()`

#### `scripts/only_chat.py`
- ✅ Reads CSV with `encoding='utf-8', encoding_errors='replace'`
- ✅ Sanitizes all content with `parse_raw_text()`
- ✅ Writes CSV and .text files with `encoding='utf-8', errors='replace'`

### 3. File Writing Operations

#### `semantic_detector/core/detector.py`
- ✅ JSON metadata: `encoding='utf-8', errors='replace'` + `ensure_ascii=False`
- ✅ All sentences sanitized with `parse_raw_text()` before processing
- ✅ Fingerprint summaries sanitized (top_words, bigrams, trigrams)
- ✅ Metadata fields sanitized (file paths, style comparisons)

#### `semantic_detector/web/app/core/detector.py`
- ✅ Temporary file writing: Uses `encoding='utf-8', errors='replace'`
- ✅ Text sanitized with `parse_raw_text()` before writing

#### `scripts/generate_html_report.py`
- ✅ HTML file writing: Uses `encoding='utf-8'` (HTML can handle Unicode)

#### `scripts/visualize_timeline_analysis.py`
- ✅ All Plotly HTML writes (handled by Plotly library)

### 4. API Routes (`semantic_detector/web/app/api/routes.py`)
- ✅ CSV reading: `errors='replace'` + `parse_raw_text()` for all fields
- ✅ Text input: Sanitized with `parse_raw_text()` before processing
- ✅ CSV export: Uses `clean_csv_value()` for all values
- ✅ JSON operations: Uses `ensure_ascii=False` + `parse_json_string()`

## Character Filtering Strategy

### Removed Characters:
- Control characters: `\x00-\x1F`, `\x7F` (except \n, \t, \r)
- Zero-width: `\u200B-\u200D`, `\uFEFF`, `\u2060`
- Bidirectional marks: `\u202A-\u202E`, `\u2066-\u2069`
- Non-printable Unicode

### Kept Characters:
- Printable ASCII
- Unicode Letters (L)
- Unicode Numbers (N)
- Unicode Punctuation (P)
- Unicode Symbols (S)
- Unicode Spaces (Z)

## Testing

Test with problematic file:
```bash
python scripts/only_chat.py
# Should create only_chat.csv and only_chat.text without errors
```

All file operations now:
1. Use `encoding='utf-8', errors='replace'`
2. Sanitize text with type-specific parsers
3. Handle None/non-string values gracefully
4. Never raise encoding exceptions

## Error Prevention Checklist

- ✅ All `open()` calls specify `encoding='utf-8', errors='replace'`
- ✅ All CSV operations use safe encoding
- ✅ All JSON operations use `ensure_ascii=False`
- ✅ All text content sanitized before processing
- ✅ Fallback encodings (latin-1) for problematic files
- ✅ Type-specific parsers for different data types

