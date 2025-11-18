"""
Semantic Detector - Language Fingerprinting & Embedding Analysis System.

A comprehensive system for analyzing text embeddings, language fingerprinting,
and identifying speakers/contexts using linguistic features.
"""

__version__ = "0.1.0"
__author__ = "Semantic Detector Team"

from semantic_detector.core.detector import SemanticDetector
from semantic_detector.core.config import settings

__all__ = [
    "SemanticDetector",
    "settings",
    "__version__",
]
