"""Sentiment analysis for sentences."""

from typing import List, Dict
import numpy as np
from collections import Counter
import re


class SentimentAnalyzer:
    """Simple sentiment analyzer using word-based approach."""
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        # Positive and negative word lists (simplified)
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'delighted',
            'positive', 'optimistic', 'hopeful', 'confident', 'successful',
            'beautiful', 'perfect', 'brilliant', 'outstanding', 'superb'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike',
            'sad', 'angry', 'disappointed', 'frustrated', 'worried', 'anxious',
            'negative', 'pessimistic', 'hopeless', 'failed', 'failure',
            'ugly', 'broken', 'wrong', 'problem', 'issue', 'difficult'
        }
    
    def analyze_sentence(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a sentence.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        words = re.findall(r'\b\w+\b', text.lower())
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        total_words = len(words)
        
        if total_words == 0:
            return {
                'sentiment_score': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'sentiment_label': 'neutral'
            }
        
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        
        # Sentiment score: -1 (very negative) to +1 (very positive)
        sentiment_score = (positive_ratio - negative_ratio) * 2
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        # Determine label
        if sentiment_score > 0.1:
            label = 'positive'
        elif sentiment_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'sentiment_score': float(sentiment_score),
            'positive_ratio': float(positive_ratio),
            'negative_ratio': float(negative_ratio),
            'sentiment_label': label
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment for multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of sentiment analysis results
        """
        return [self.analyze_sentence(text) for text in texts]
    
    def get_average_sentiment(self, texts: List[str]) -> Dict[str, float]:
        """
        Get average sentiment across multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Average sentiment metrics
        """
        results = self.analyze_batch(texts)
        
        avg_score = np.mean([r['sentiment_score'] for r in results])
        avg_positive = np.mean([r['positive_ratio'] for r in results])
        avg_negative = np.mean([r['negative_ratio'] for r in results])
        
        # Count labels
        labels = [r['sentiment_label'] for r in results]
        label_counts = Counter(labels)
        total = len(labels)
        
        return {
            'average_sentiment_score': float(avg_score),
            'average_positive_ratio': float(avg_positive),
            'average_negative_ratio': float(avg_negative),
            'positive_count': label_counts.get('positive', 0),
            'negative_count': label_counts.get('negative', 0),
            'neutral_count': label_counts.get('neutral', 0),
            'positive_percentage': float(label_counts.get('positive', 0) / total * 100),
            'negative_percentage': float(label_counts.get('negative', 0) / total * 100),
            'neutral_percentage': float(label_counts.get('neutral', 0) / total * 100)
        }
    
    def get_person_score(self, texts: List[str]) -> float:
        """
        Calculate average person score (sentiment-based).
        Higher score = more positive/optimistic person.
        
        Args:
            texts: List of texts from a person
            
        Returns:
            Person score (0-100)
        """
        avg_sentiment = self.get_average_sentiment(texts)
        # Convert -1 to +1 scale to 0-100 scale
        person_score = (avg_sentiment['average_sentiment_score'] + 1) * 50
        return float(max(0, min(100, person_score)))

