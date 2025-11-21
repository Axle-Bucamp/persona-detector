import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import text utils directly to avoid full module import
import re
import unicodedata
import string

PRINTABLE_ASCII = set(string.printable)
SAFE_UNICODE_CATEGORIES = {'L', 'N', 'P', 'S', 'Z'}

def parse_raw_text(text, max_length=None):
    """Parse raw text content, keeping only safe printable characters."""
    if text is None:
        return ''
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ''
    try:
        text = unicodedata.normalize('NFKC', text)
    except Exception:
        try:
            text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        except Exception:
            return ''
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    text = re.sub(r'[\u200B-\u200D\uFEFF\u2060]', '', text)
    text = re.sub(r'[\u202A-\u202E\u2066-\u2069]', '', text)
    cleaned_chars = []
    for char in text:
        if char in PRINTABLE_ASCII:
            cleaned_chars.append(char)
        elif unicodedata.category(char)[0] in SAFE_UNICODE_CATEGORIES:
            try:
                if char.isprintable() or char.isspace():
                    cleaned_chars.append(char)
            except Exception:
                continue
    text = ''.join(cleaned_chars)
    text = re.sub(r'[ \t]+', ' ', text)
    if max_length and len(text) > max_length:
        text = text[:max_length].rstrip() + '...'
    return text.strip()

# Read CSV with safe encoding
df = pd.read_csv(
    r'C:\Users\dying\Documents\persona-detector\output\ransom_chats.csv',
    encoding='utf-8',
    encoding_errors='replace'
)

df = df["content"]

# Sanitize all content before writing
if hasattr(df, 'apply'):
    df_clean = df.apply(lambda x: parse_raw_text(str(x)) if pd.notna(x) else '')
else:
    df_clean = pd.Series([parse_raw_text(str(x)) if pd.notna(x) else '' for x in df])

# Convert back to DataFrame for CSV export
df_output = pd.DataFrame({'content': df_clean})

# Write CSV with safe encoding
df_output.to_csv(
    r'C:\Users\dying\Documents\persona-detector\output\only_chat.csv',
    index=False,
    encoding='utf-8',
    errors='replace'
)

# Also write as .text file with safe encoding
text_file = r'C:\Users\dying\Documents\persona-detector\output\only_chat.text'
with open(text_file, 'w', encoding='utf-8', errors='replace') as f:
    for content in df_clean:
        if content and str(content).strip():
            # Ensure content is safe for text file
            safe_content = parse_raw_text(str(content))
            f.write(safe_content + '\n')

print(f"[INFO] Created only_chat.csv and only_chat.text with {len(df_clean)} entries")