"""Text utilities for safe parsing and encoding."""

import re
import unicodedata
import string
from typing import Optional, Literal

# Define safe character sets
PRINTABLE_ASCII = set(string.printable)
SAFE_UNICODE_CATEGORIES = {
    'L',  # Letter
    'N',  # Number
    'P',  # Punctuation
    'S',  # Symbol
    'Z',  # Separator (space)
}


def parse_raw_text(text: Optional[str], max_length: Optional[int] = None) -> str:
    """
    Parse raw text content, keeping only safe printable characters.
    Removes control characters, zero-width characters, and problematic Unicode.
    
    Args:
        text: Input text (can be None)
        max_length: Maximum length to truncate (None for no limit)
        
    Returns:
        Cleaned text string safe for storage and display
    """
    if text is None:
        return ''
    
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ''
    
    # Normalize Unicode (NFKC: compatibility decomposition + composition)
    try:
        text = unicodedata.normalize('NFKC', text)
    except Exception:
        # If normalization fails, try basic encoding/decoding
        try:
            text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        except Exception:
            return ''
    
    # Remove control characters (except newline, tab, carriage return)
    # Keep: \n (0x0A), \t (0x09), \r (0x0D)
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    
    # Remove zero-width characters
    text = re.sub(r'[\u200B-\u200D\uFEFF\u2060]', '', text)
    
    # Remove bidirectional marks and other invisible characters
    text = re.sub(r'[\u202A-\u202E\u2066-\u2069]', '', text)
    
    # Keep only characters that are printable or common Unicode categories
    cleaned_chars = []
    for char in text:
        # Allow printable ASCII
        if char in PRINTABLE_ASCII:
            cleaned_chars.append(char)
        # Allow common Unicode letters, numbers, punctuation, spaces
        elif unicodedata.category(char)[0] in SAFE_UNICODE_CATEGORIES:
            # Additional check: must be printable
            try:
                if char.isprintable() or char.isspace():
                    cleaned_chars.append(char)
            except Exception:
                continue
        # Skip everything else (control chars, marks, etc.)
    
    text = ''.join(cleaned_chars)
    
    # Normalize whitespace (replace multiple spaces/tabs with single space)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Truncate if needed
    if max_length and len(text) > max_length:
        text = text[:max_length].rstrip() + '...'
    
    return text.strip()


def sanitize_text(text: Optional[str], max_length: Optional[int] = None, 
                 preserve_newlines: bool = True) -> str:
    """
    Sanitize text for safe storage and display.
    Uses parse_raw_text as base and optionally preserves newlines.
    
    Args:
        text: Input text (can be None)
        max_length: Maximum length to truncate (None for no limit)
        preserve_newlines: Whether to preserve newline characters
        
    Returns:
        Sanitized text string
    """
    if text is None:
        return ''
    
    # Use raw text parser as base
    text = parse_raw_text(text, max_length=None)
    
    # If not preserving newlines, remove them
    if not preserve_newlines:
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        # Normalize multiple spaces
        text = re.sub(r' +', ' ', text)
    
    # Apply max_length after newline handling
    if max_length and len(text) > max_length:
        text = text[:max_length].rstrip() + '...'
    
    return text.strip()


def safe_encode(text: str, encoding: str = 'utf-8', errors: str = 'replace') -> bytes:
    """
    Safely encode text to bytes.
    
    Args:
        text: Input text
        encoding: Target encoding (default: utf-8)
        errors: Error handling strategy ('replace', 'ignore', 'xmlcharrefreplace')
        
    Returns:
        Encoded bytes
    """
    if not isinstance(text, str):
        text = str(text)
    
    try:
        return text.encode(encoding, errors=errors)
    except (UnicodeEncodeError, AttributeError):
        # Fallback: replace problematic characters
        return text.encode(encoding, errors='replace')


def safe_decode(data: bytes, encoding: str = 'utf-8', errors: str = 'replace') -> str:
    """
    Safely decode bytes to string.
    
    Args:
        data: Input bytes
        encoding: Source encoding (default: utf-8)
        errors: Error handling strategy
        
    Returns:
        Decoded string
    """
    if isinstance(data, str):
        return data
    
    try:
        return data.decode(encoding, errors=errors)
    except (UnicodeDecodeError, AttributeError):
        # Fallback: replace problematic bytes
        return data.decode(encoding, errors='replace')


def clean_csv_value(value: any) -> str:
    """
    Clean a value for CSV export - removes all problematic characters.
    
    Args:
        value: Value to clean (can be any type)
        
    Returns:
        Cleaned string value safe for CSV export
    """
    if value is None:
        return ''
    
    # Convert to string
    if not isinstance(value, str):
        try:
            value = str(value)
        except Exception:
            return ''
    
    # Use raw text parser (no newlines, strict filtering)
    value = parse_raw_text(value, max_length=None)
    
    # Remove all newlines and line breaks for CSV
    value = value.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    # Remove CSV delimiter characters that could break parsing (if not quoted)
    # Keep them but ensure proper CSV quoting will handle them
    # Just normalize whitespace
    value = re.sub(r'\s+', ' ', value)
    
    # Final UTF-8 validation
    try:
        value.encode('utf-8', errors='strict')
    except UnicodeEncodeError:
        # Replace any remaining problematic characters
        value = value.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    
    return value.strip()


def parse_json_string(text: Optional[str]) -> str:
    """
    Parse text that will be used in JSON - allows more characters than CSV.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text safe for JSON encoding
    """
    if text is None:
        return ''
    
    # Use raw text parser but allow newlines (JSON can handle them)
    text = parse_raw_text(text, max_length=None)
    
    # JSON-safe: keep newlines but escape them properly
    # The JSON encoder will handle escaping
    
    return text.strip()


def parse_identifier(text: Optional[str], allow_unicode: bool = False) -> str:
    """
    Parse identifier-like text (usernames, IDs, etc.) - very strict.
    
    Args:
        text: Input text
        allow_unicode: Whether to allow Unicode letters (default: ASCII only)
        
    Returns:
        Cleaned identifier string
    """
    if text is None:
        return ''
    
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ''
    
    # Normalize
    try:
        text = unicodedata.normalize('NFKC', text)
    except Exception:
        text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    
    # Remove all non-printable characters
    if allow_unicode:
        # Allow Unicode letters and numbers
        text = ''.join(c for c in text if c.isalnum() or c in '_-')
    else:
        # ASCII alphanumeric and underscore/hyphen only
        text = ''.join(c for c in text if c.isalnum() or c in '_-')
        # Ensure ASCII only
        text = text.encode('ascii', errors='ignore').decode('ascii')
    
    return text.strip()

