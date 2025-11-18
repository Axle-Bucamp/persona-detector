"""
Language fingerprinting module.
Extracts linguistic features for speaker/context identification.
Inspired by stylometric analysis used in the Unabomber case.
"""

import re
from typing import List, Dict, Tuple, Any
from collections import Counter, defaultdict
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    stopwords.words('english')
except LookupError:
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)


class LanguageFingerprint:
    """Extract linguistic features for fingerprinting."""
    
    def __init__(self):
        """Initialize language fingerprint extractor."""
        self.stop_words = set(stopwords.words('english'))
        self.feature_names = []  # Store feature names for interpretability
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract comprehensive linguistic features.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of feature values
        """
        features = {}
        
        # Basic statistics
        features.update(self._basic_stats(text))
        
        # Word-level features
        features.update(self._word_features(text))
        
        # Character-level features
        features.update(self._character_features(text))
        
        # N-gram features
        features.update(self._ngram_features(text, n=2))  # Bigrams
        features.update(self._ngram_features(text, n=3))  # Trigrams
        
        # Punctuation features
        features.update(self._punctuation_features(text))
        
        # Vocabulary richness
        features.update(self._vocabulary_features(text))
        
        return features
    
    def _basic_stats(self, text: str) -> Dict[str, float]:
        """Extract basic text statistics."""
        words = word_tokenize(text.lower())
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences) if sentences else 1,
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'char_word_ratio': len(text) / len(words) if words else 0,
        }
    
    def _word_features(self, text: str) -> Dict[str, float]:
        """Extract word-level features."""
        words = word_tokenize(text.lower())
        words_no_stop = [w for w in words if w not in self.stop_words and w.isalpha()]
        
        if not words:
            return {}
        
        # Word length distribution
        word_lengths = [len(w) for w in words if w.isalpha()]
        
        return {
            'stopword_ratio': (len(words) - len(words_no_stop)) / len(words),
            'unique_word_ratio': len(set(words)) / len(words),
            'avg_word_length': np.mean(word_lengths) if word_lengths else 0,
            'max_word_length': max(word_lengths) if word_lengths else 0,
            'min_word_length': min(word_lengths) if word_lengths else 0,
        }
    
    def _character_features(self, text: str) -> Dict[str, float]:
        """Extract character-level features."""
        if not text:
            return {}
        
        chars = text.lower()
        total_chars = len(chars)
        
        if total_chars == 0:
            return {}
        
        return {
            'digit_ratio': sum(c.isdigit() for c in chars) / total_chars,
            'uppercase_ratio': sum(c.isupper() for c in text) / len(text) if text else 0,
            'lowercase_ratio': sum(c.islower() for c in text) / len(text) if text else 0,
            'space_ratio': sum(c.isspace() for c in chars) / total_chars,
            'vowel_ratio': sum(c in 'aeiou' for c in chars) / total_chars,
            'consonant_ratio': sum(c.isalpha() and c not in 'aeiou' for c in chars) / total_chars,
        }
    
    def _ngram_features(self, text: str, n: int) -> Dict[str, float]:
        """Extract n-gram frequency features."""
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha()]
        
        if len(words) < n:
            return {}
        
        # Generate n-grams
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams.append(ngram)
        
        if not ngrams:
            return {}
        
        ngram_counts = Counter(ngrams)
        total_ngrams = len(ngrams)
        
        # Top n-gram frequencies
        top_ngrams = ngram_counts.most_common(10)
        features = {}
        
        for i, (ngram, count) in enumerate(top_ngrams):
            features[f'{n}gram_freq_{i}'] = count / total_ngrams
        
        # Diversity metrics
        features[f'{n}gram_diversity'] = len(ngram_counts) / total_ngrams if total_ngrams > 0 else 0
        
        return features
    
    def _punctuation_features(self, text: str) -> Dict[str, float]:
        """Extract punctuation usage features."""
        if not text:
            return {}
        
        total_chars = len(text)
        if total_chars == 0:
            return {}
        
        punct_chars = '.,!?;:-\'"()[]{}'
        punct_counts = {char: text.count(char) for char in punct_chars}
        
        features = {}
        for char, count in punct_counts.items():
            features[f'punct_{char}'] = count / total_chars
        
        features['punct_total_ratio'] = sum(punct_counts.values()) / total_chars
        
        return features
    
    def _vocabulary_features(self, text: str) -> Dict[str, float]:
        """Extract vocabulary richness features."""
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha()]
        
        if not words:
            return {}
        
        word_freq = Counter(words)
        
        # Type-token ratio
        ttr = len(word_freq) / len(words)
        
        # Hapax legomena (words that appear only once)
        hapax = sum(1 for count in word_freq.values() if count == 1)
        hapax_ratio = hapax / len(word_freq) if word_freq else 0
        
        return {
            'type_token_ratio': ttr,
            'hapax_ratio': hapax_ratio,
            'vocab_size': len(word_freq),
        }
    
    def extract_fingerprint_vector(self, text: str) -> np.ndarray:
        """
        Extract fingerprint as a feature vector.
        
        Args:
            text: Input text
            
        Returns:
            Feature vector as numpy array
        """
        features = self.extract_features(text)
        return np.array(list(features.values()))
    
    def extract_fingerprints_batch(self, texts: List[str]) -> np.ndarray:
        """
        Extract fingerprints for multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        fingerprints = []
        for text in texts:
            fp = self.extract_fingerprint_vector(text)
            fingerprints.append(fp)
        
        # Pad to same length
        max_len = max(len(fp) for fp in fingerprints)
        padded = []
        for fp in fingerprints:
            if len(fp) < max_len:
                fp = np.pad(fp, (0, max_len - len(fp)), 'constant')
            padded.append(fp)
        
        return np.array(padded)
    
    def extract_word_distribution(self, text: str, top_n: int = 50) -> Dict[str, float]:
        """
        Extract word frequency distribution for fingerprint representation.
        
        Args:
            text: Input text
            top_n: Number of top words to return
            
        Returns:
            Dictionary mapping words to their normalized frequencies
        """
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha() and w not in self.stop_words]
        
        if not words:
            return {}
        
        word_counts = Counter(words)
        total_words = len(words)
        
        # Normalize frequencies
        distribution = {word: count / total_words for word, count in word_counts.items()}
        
        # Return top N words
        top_words = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return dict(top_words)
    
    def extract_ngram_distribution(self, text: str, n: int = 2, top_n: int = 30) -> Dict[Tuple[str, ...], float]:
        """
        Extract n-gram frequency distribution.
        
        Args:
            text: Input text
            n: N-gram size (2 for bigrams, 3 for trigrams)
            top_n: Number of top n-grams to return
            
        Returns:
            Dictionary mapping n-grams to their normalized frequencies
        """
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha()]
        
        if len(words) < n:
            return {}
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams.append(ngram)
        
        if not ngrams:
            return {}
        
        ngram_counts = Counter(ngrams)
        total_ngrams = len(ngrams)
        
        # Normalize frequencies
        distribution = {ngram: count / total_ngrams for ngram, count in ngram_counts.items()}
        
        # Return top N n-grams
        top_ngrams = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return dict(top_ngrams)
    
    def extract_contextual_features(self, text: str) -> Dict[str, Any]:
        """
        Extract contextual features including word distributions and n-grams.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with comprehensive fingerprint representation
        """
        return {
            'word_distribution': self.extract_word_distribution(text),
            'bigram_distribution': self.extract_ngram_distribution(text, n=2),
            'trigram_distribution': self.extract_ngram_distribution(text, n=3),
            'features': self.extract_features(text),
        }
    
    def get_fingerprint_summary(self, text: str) -> Dict[str, Any]:
        """
        Get a human-readable summary of the fingerprint.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with fingerprint summary
        """
        contextual = self.extract_contextual_features(text)
        features = contextual['features']
        
        return {
            'top_words': list(contextual['word_distribution'].keys())[:10],
            'top_bigrams': [' '.join(ngram) for ngram in list(contextual['bigram_distribution'].keys())[:10]],
            'top_trigrams': [' '.join(ngram) for ngram in list(contextual['trigram_distribution'].keys())[:10]],
            'avg_sentence_length': features.get('avg_sentence_length', 0),
            'vocab_richness': features.get('type_token_ratio', 0),
            'punctuation_style': {k: v for k, v in features.items() if k.startswith('punct_') and v > 0},
        }

