"""
Sentence splitter for large text files.
Splits text into sentences using period-based detection with context awareness.
"""

import re
from typing import List, Optional
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class SentenceSplitter:
    """Splits text into sentences with various strategies."""
    
    def __init__(self, min_sentence_length: int = 10, max_sentence_length: int = 500):
        """
        Initialize sentence splitter.
        
        Args:
            min_sentence_length: Minimum characters for a valid sentence
            max_sentence_length: Maximum characters for a sentence
        """
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
    
    def split_by_period(self, text: str) -> List[str]:
        """
        Split text by periods with context awareness.
        Handles abbreviations, decimals, URLs, etc.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        # Clean text
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Use NLTK for better sentence tokenization
        sentences = sent_tokenize(text)
        
        # Filter and clean sentences
        filtered_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= self.min_sentence_length:
                if len(sentence) > self.max_sentence_length:
                    # Split very long sentences further
                    parts = self._split_long_sentence(sentence)
                    filtered_sentences.extend(parts)
                else:
                    filtered_sentences.append(sentence)
        
        return filtered_sentences
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split sentences that exceed max length."""
        # Split on common delimiters
        parts = re.split(r'[;:]', sentence)
        result = []
        for part in parts:
            part = part.strip()
            if len(part) >= self.min_sentence_length:
                if len(part) > self.max_sentence_length:
                    # Further split on commas if still too long
                    subparts = re.split(r',\s+', part)
                    current = ""
                    for subpart in subparts:
                        if len(current + subpart) < self.max_sentence_length:
                            current += (", " if current else "") + subpart
                        else:
                            if current:
                                result.append(current.strip())
                            current = subpart
                    if current:
                        result.append(current.strip())
                else:
                    result.append(part)
        return result if result else [sentence]
    
    def split_file(self, file_path: str, encoding: str = 'utf-8') -> List[str]:
        """
        Split a text file into sentences with safe encoding.
        
        Args:
            file_path: Path to text file
            encoding: File encoding
            
        Returns:
            List of sentences
        """
        try:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                text = f.read()
            # Sanitize text before processing
            from semantic_detector.web.app.core.text_utils import parse_raw_text
            text = parse_raw_text(text)
            return self.split_by_period(text)
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1', errors='replace') as f:
                text = f.read()
            from semantic_detector.web.app.core.text_utils import parse_raw_text
            text = parse_raw_text(text)
            return self.split_by_period(text)
        except Exception as e:
            # Final fallback: read as bytes and decode
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                text = content.decode('utf-8', errors='replace')
                from semantic_detector.web.app.core.text_utils import parse_raw_text
                text = parse_raw_text(text)
                return self.split_by_period(text)
            except Exception:
                raise ValueError(f"Could not read file {file_path}: {e}")

