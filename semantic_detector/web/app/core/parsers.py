"""Parsers for CSV, JSON, regex-delimited, markdown, and PDF data."""

import csv
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from io import StringIO, BytesIO

try:
    import jsonpath_ng
except ImportError:
    jsonpath_ng = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


class CSVParser:
    """Parser for CSV files."""
    
    def __init__(self):
        """Initialize CSV parser."""
        self.module_name = "CSV Parser"
    
    def parse(self, file_content: str, text_column: str) -> Dict[str, Any]:
        """
        Parse CSV file and extract text from specified column.
        
        Args:
            file_content: CSV file content as string
            text_column: Name of column containing text to analyze
            
        Returns:
            Dictionary with parsed data and metadata
        """
        reader = csv.DictReader(StringIO(file_content))
        rows = list(reader)
        
        if not rows:
            raise ValueError("CSV file is empty")
        
        # Get all column names
        column_names = list(rows[0].keys())
        
        if text_column not in column_names:
            raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {', '.join(column_names)}")
        
        # Extract text from specified column
        texts = []
        metadata_list = []
        
        for idx, row in enumerate(rows):
            text = row.get(text_column, '').strip()
            if text:  # Only include non-empty texts
                texts.append(text)
                metadata_list.append({
                    'row_index': idx,
                    'column_name': text_column,
                    'all_columns': {k: v for k, v in row.items()}
                })
        
        if not texts:
            raise ValueError(f"No text found in column '{text_column}'")
        
        return {
            'texts': texts,
            'metadata': {
                'parser_type': 'csv',
                'module_name': self.module_name,
                'text_column': text_column,
                'all_columns': column_names,
                'total_rows': len(rows),
                'text_rows': len(texts),
                'row_metadata': metadata_list
            }
        }


class JSONParser:
    """Parser for JSON files using JSONPath expressions."""
    
    def __init__(self):
        """Initialize JSON parser."""
        self.module_name = "JSON Parser"
    
    def parse(self, file_content: str, jsonpath: str) -> Dict[str, Any]:
        """
        Parse JSON file and extract text using JSONPath expression.
        
        Args:
            file_content: JSON file content as string
            jsonpath: JSONPath expression (e.g., "$.messages[].text")
            
        Returns:
            Dictionary with parsed data and metadata
        """
        if jsonpath_ng is None:
            raise ValueError("jsonpath-ng package is required for JSON parsing. Install it with: pip install jsonpath-ng")
        
        try:
            data = json.loads(file_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {str(e)}")
        
        # Parse JSONPath expression
        try:
            jsonpath_expr = jsonpath_ng.parse(jsonpath)
            matches = [match.value for match in jsonpath_expr.find(data)]
        except Exception as e:
            raise ValueError(f"Invalid JSONPath expression '{jsonpath}': {str(e)}")
        
        if not matches:
            raise ValueError(f"No matches found for JSONPath '{jsonpath}'")
        
        # Extract texts (handle both strings and objects)
        texts = []
        metadata_list = []
        
        for idx, match in enumerate(matches):
            if isinstance(match, str):
                text = match.strip()
            elif isinstance(match, dict):
                # If match is a dict, try to find text field or convert to string
                text = match.get('text', match.get('content', str(match))).strip()
            else:
                text = str(match).strip()
            
            if text:
                texts.append(text)
                metadata_list.append({
                    'row_index': idx,
                    'jsonpath': jsonpath,
                    'original_value': match
                })
        
        if not texts:
            raise ValueError(f"No text found using JSONPath '{jsonpath}'")
        
        return {
            'texts': texts,
            'metadata': {
                'parser_type': 'json',
                'module_name': self.module_name,
                'jsonpath': jsonpath,
                'total_matches': len(matches),
                'text_rows': len(texts),
                'row_metadata': metadata_list
            }
        }


class RegexParser:
    """Parser for regex-delimited text."""
    
    def __init__(self):
        """Initialize regex parser."""
        self.module_name = "Regex Parser"
    
    def parse(self, file_content: str, pattern: str, use_capture_group: bool = False) -> Dict[str, Any]:
        """
        Parse text using regex pattern.
        
        Args:
            file_content: Text content as string
            pattern: Regex pattern for splitting or matching
            use_capture_group: If True, extract capture groups; if False, split by pattern
            
        Returns:
            Dictionary with parsed data and metadata
        """
        try:
            compiled_pattern = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {str(e)}")
        
        texts = []
        metadata_list = []
        
        if use_capture_group:
            # Extract capture groups
            matches = compiled_pattern.finditer(file_content)
            for idx, match in enumerate(matches):
                if match.groups():
                    # Use first capture group
                    text = match.group(1).strip()
                else:
                    # Use entire match
                    text = match.group(0).strip()
                
                if text:
                    texts.append(text)
                    metadata_list.append({
                        'row_index': idx,
                        'match_start': match.start(),
                        'match_end': match.end()
                    })
        else:
            # Split by pattern
            parts = compiled_pattern.split(file_content)
            for idx, part in enumerate(parts):
                text = part.strip()
                if text:
                    texts.append(text)
                    metadata_list.append({
                        'row_index': idx,
                        'split_position': idx
                    })
        
        if not texts:
            raise ValueError(f"No text extracted using pattern '{pattern}'")
        
        # Determine module name based on pattern
        pattern_name = self._get_pattern_name(pattern)
        
        return {
            'texts': texts,
            'metadata': {
                'parser_type': 'regex',
                'module_name': f"{self.module_name} ({pattern_name})",
                'pattern': pattern,
                'pattern_display': pattern,
                'use_capture_group': use_capture_group,
                'total_rows': len(texts),
                'row_metadata': metadata_list
            }
        }
    
    def _get_pattern_name(self, pattern: str) -> str:
        """Get human-readable name for common patterns."""
        pattern_lower = pattern.lower().strip()
        
        # Common patterns
        if pattern == ',' or pattern == r',':
            return "CSV Delimiter"
        elif pattern == r'\.\s+' or pattern == r'\.':
            return "Sentence Splitter"
        elif pattern.startswith('^#+'):
            return "Markdown Parser"
        elif pattern == r'\n\n+':
            return "Paragraph Splitter"
        elif pattern == r'\s+':
            return "Whitespace Splitter"
        else:
            return "Custom Pattern"


class MarkdownParser:
    """Parser for Markdown files using regex patterns."""
    
    def __init__(self):
        """Initialize markdown parser."""
        self.module_name = "Markdown Parser"
    
    def parse(self, file_content: str, extract_mode: str = "sections") -> Dict[str, Any]:
        """
        Parse markdown file and extract text sections.
        
        Args:
            file_content: Markdown content as string
            extract_mode: Extraction mode - "sections" (by headers), "paragraphs", or "sentences"
            
        Returns:
            Dictionary with parsed data and metadata
        """
        texts = []
        metadata_list = []
        
        if extract_mode == "sections":
            # Extract sections by markdown headers (##, ###, etc.)
            # Pattern: ^#+\s+(.+)$ captures header text
            # Then capture content until next header or end
            pattern = r'^(#{1,6})\s+(.+)$'
            lines = file_content.split('\n')
            current_section = []
            current_header = None
            current_header_level = 0
            
            for idx, line in enumerate(lines):
                header_match = re.match(pattern, line)
                if header_match:
                    # Save previous section
                    if current_section and current_header:
                        section_text = '\n'.join(current_section).strip()
                        if section_text:
                            texts.append(section_text)
                            metadata_list.append({
                                'row_index': len(texts) - 1,
                                'header_level': current_header_level,
                                'header_text': current_header,
                                'line_number': idx - len(current_section)
                            })
                    
                    # Start new section
                    current_header = header_match.group(2)
                    current_header_level = len(header_match.group(1))
                    current_section = [line]
                else:
                    current_section.append(line)
            
            # Don't forget last section
            if current_section and current_header:
                section_text = '\n'.join(current_section).strip()
                if section_text:
                    texts.append(section_text)
                    metadata_list.append({
                        'row_index': len(texts) - 1,
                        'header_level': current_header_level,
                        'header_text': current_header,
                        'line_number': len(lines) - len(current_section)
                    })
        
        elif extract_mode == "paragraphs":
            # Extract paragraphs (separated by blank lines)
            paragraphs = re.split(r'\n\s*\n', file_content)
            for idx, para in enumerate(paragraphs):
                text = para.strip()
                # Remove markdown formatting but keep text
                text = re.sub(r'^#+\s+', '', text)  # Remove headers
                text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Remove bold
                text = re.sub(r'\*(.+?)\*', r'\1', text)  # Remove italic
                text = re.sub(r'`(.+?)`', r'\1', text)  # Remove code
                text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)  # Remove links
                
                if text and len(text) > 10:  # Filter very short paragraphs
                    texts.append(text)
                    metadata_list.append({
                        'row_index': idx,
                        'extract_mode': 'paragraphs'
                    })
        
        else:  # sentences
            # Extract sentences (split by periods, exclamation, question marks)
            sentences = re.split(r'[.!?]+\s+', file_content)
            for idx, sent in enumerate(sentences):
                text = sent.strip()
                # Clean markdown
                text = re.sub(r'^#+\s+', '', text)
                text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
                text = re.sub(r'\*(.+?)\*', r'\1', text)
                text = re.sub(r'`(.+?)`', r'\1', text)
                text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
                
                if text and len(text) > 5:
                    texts.append(text)
                    metadata_list.append({
                        'row_index': idx,
                        'extract_mode': 'sentences'
                    })
        
        if not texts:
            raise ValueError(f"No text extracted from markdown using mode '{extract_mode}'")
        
        return {
            'texts': texts,
            'metadata': {
                'parser_type': 'markdown',
                'module_name': self.module_name,
                'extract_mode': extract_mode,
                'total_rows': len(texts),
                'row_metadata': metadata_list
            }
        }


class PDFParser:
    """Parser for PDF files - converts to markdown then parses."""
    
    def __init__(self):
        """Initialize PDF parser."""
        self.module_name = "PDF Parser (via Markdown)"
    
    def parse(self, file_content: bytes, extract_mode: str = "sections") -> Dict[str, Any]:
        """
        Parse PDF file by converting to markdown-like text, then parsing as markdown.
        
        Args:
            file_content: PDF file content as bytes
            extract_mode: Extraction mode - "sections", "paragraphs", or "sentences"
            
        Returns:
            Dictionary with parsed data and metadata
        """
        if fitz is None:
            raise ValueError(
                "PyMuPDF (fitz) package is required for PDF parsing. "
                "Install it with: pip install pymupdf"
            )
        
        # Convert PDF to text/markdown
        try:
            pdf_doc = fitz.open(stream=file_content, filetype="pdf")
            markdown_text = []
            
            for page_num, page in enumerate(pdf_doc):
                # Extract text from page
                page_text = page.get_text()
                
                # Try to preserve some structure
                # Detect potential headers (all caps, short lines, etc.)
                lines = page_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Detect potential headers
                    if len(line) < 100 and (line.isupper() or re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', line)):
                        # Likely a header
                        markdown_text.append(f"\n## {line}\n")
                    else:
                        markdown_text.append(line)
                
                # Add page break
                if page_num < len(pdf_doc) - 1:
                    markdown_text.append("\n\n---\n\n")
            
            pdf_doc.close()
            markdown_content = '\n'.join(markdown_text)
            
        except Exception as e:
            raise ValueError(f"Error reading PDF: {str(e)}")
        
        # Now parse as markdown
        markdown_parser = MarkdownParser()
        result = markdown_parser.parse(markdown_content, extract_mode)
        
        # Update metadata to indicate PDF source
        result['metadata']['parser_type'] = 'pdf'
        result['metadata']['module_name'] = self.module_name
        result['metadata']['pdf_warning'] = True
        result['metadata']['original_format'] = 'pdf'
        
        return result


class UnifiedDataFormatter:
    """Format parsed data into unified structure."""
    
    def __init__(self):
        """Initialize formatter."""
        pass
    
    def format_for_analysis(
        self,
        texts: List[str],
        cluster_ids: Optional[List[int]] = None,
        word_frequencies: Optional[List[Dict[str, float]]] = None,
        tfidf_features: Optional[List[List[Tuple[str, float]]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Format data into unified structure for table display.
        
        Args:
            texts: List of raw text strings
            cluster_ids: List of cluster IDs (same length as texts)
            word_frequencies: List of word frequency dicts per text
            tfidf_features: List of top TF-IDF features per text
            metadata: List of metadata dicts per text
            
        Returns:
            List of unified data rows
        """
        unified_rows = []
        
        for idx, text in enumerate(texts):
            row = {
                'sentence_index': idx,
                'raw_text': text,
                'cluster_id': cluster_ids[idx] if cluster_ids and idx < len(cluster_ids) else None
            }
            
            # Add word frequencies (top 10)
            if word_frequencies and idx < len(word_frequencies):
                word_freq = word_frequencies[idx]
                sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                for i, (word, freq) in enumerate(sorted_words, 1):
                    row[f'word_freq_{i}'] = word
                    row[f'word_freq_{i}_value'] = float(freq)
                # Pad if less than 10
                for i in range(len(sorted_words) + 1, 11):
                    row[f'word_freq_{i}'] = None
                    row[f'word_freq_{i}_value'] = 0.0
            else:
                for i in range(1, 11):
                    row[f'word_freq_{i}'] = None
                    row[f'word_freq_{i}_value'] = 0.0
            
            # Add TF-IDF features (top 10)
            if tfidf_features and idx < len(tfidf_features):
                tfidf = tfidf_features[idx]
                for i, (feature, score) in enumerate(tfidf[:10], 1):
                    row[f'tfidf_feature_{i}'] = feature
                    row[f'tfidf_feature_{i}_value'] = float(score)
                # Pad if less than 10
                for i in range(len(tfidf[:10]) + 1, 11):
                    row[f'tfidf_feature_{i}'] = None
                    row[f'tfidf_feature_{i}_value'] = 0.0
            else:
                for i in range(1, 11):
                    row[f'tfidf_feature_{i}'] = None
                    row[f'tfidf_feature_{i}_value'] = 0.0
            
            # Add metadata
            if metadata and idx < len(metadata):
                row['source_metadata'] = metadata[idx]
            else:
                row['source_metadata'] = {}
            
            unified_rows.append(row)
        
        return unified_rows

