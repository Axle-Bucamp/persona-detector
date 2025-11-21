# Encoding and Text Parsing Safety Fixes

## Problem
The application was encountering `'charmap' codec can't encode character` errors when processing text containing special characters (e.g., Cyrillic characters like '\u0421').

## Solution
Added comprehensive text sanitization and encoding safety throughout the application.

## Changes Made

### 1. New Text Utilities Module (`semantic_detector/web/app/core/text_utils.py`)
Created a new module with safe text parsing functions:

- **`sanitize_text()`**: Normalizes Unicode, removes control characters, handles encoding errors
- **`safe_encode()`**: Safely encodes text to bytes with error handling
- **`safe_decode()`**: Safely decodes bytes to string with error handling  
- **`clean_csv_value()`**: Cleans values specifically for CSV export

### 2. Updated Export Script (`scripts/export_ransom_chats.py`)
- Added text sanitization for all text fields before CSV writing
- Uses `errors='replace'` when opening files
- Sanitizes content, party, timestamp, and group_name fields
- Added error handling for problematic rows

### 3. Updated Analysis Script (`scripts/analyze_ransom_chat_timeline.py`)
- Added safe CSV reading with `errors='replace'`
- Sanitizes all text fields during CSV loading
- Added `sanitize_json_for_output()` for JSON serialization
- Fallback to latin-1 encoding if UTF-8 fails

### 4. Updated API Routes (`semantic_detector/web/app/api/routes.py`)
- Safe CSV reading with UTF-8/latin-1 fallback
- Text sanitization for all user-provided content
- Safe CSV export with encoding error handling
- Proper Content-Type headers with charset specification

## Key Features

### Unicode Normalization
- Uses NFKC normalization to handle composed/decomposed characters
- Removes zero-width characters and problematic control chars

### Encoding Fallbacks
- Primary: UTF-8 with `errors='replace'`
- Fallback: latin-1 with `errors='replace'`
- Ensures no encoding errors crash the application

### CSV Safety
- Removes newlines from CSV values (replaced with spaces)
- Handles special characters that might break CSV parsing
- Aggressive cleaning for problematic rows

### Error Handling
- Try/except blocks around all encoding operations
- Graceful degradation when encoding fails
- Logs warnings but continues processing

## Usage

All text processing now automatically uses safe encoding:

```python
from semantic_detector.web.app.core.text_utils import sanitize_text, clean_csv_value

# Sanitize text
clean_text = sanitize_text(user_input, preserve_newlines=False)

# Clean CSV value
csv_safe = clean_csv_value(value)
```

## Testing

The export script successfully processed 11,089 rows with various special characters including Cyrillic text without errors.

## Files Modified

1. `semantic_detector/web/app/core/text_utils.py` (new)
2. `scripts/export_ransom_chats.py`
3. `scripts/analyze_ransom_chat_timeline.py`
4. `semantic_detector/web/app/api/routes.py`

## Prevention

All file operations now:
- Explicitly specify UTF-8 encoding
- Use `errors='replace'` or `errors='ignore'` for safety
- Sanitize text before processing
- Handle encoding exceptions gracefully

