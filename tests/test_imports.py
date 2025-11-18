#!/usr/bin/env python3
"""Test imports without requiring all dependencies."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing imports...")

try:
    print("1. Testing sentence_splitter...")
    from semantic_detector.sentence_splitter import SentenceSplitter
    print("   [OK] SentenceSplitter imported")
except Exception as e:
    print(f"   [ERROR] {e}")

try:
    print("2. Testing language_fingerprint...")
    from semantic_detector.language_fingerprint import LanguageFingerprint
    print("   [OK] LanguageFingerprint imported")
except Exception as e:
    print(f"   [ERROR] {e}")

try:
    print("3. Testing clustering...")
    from semantic_detector.clustering import SpeakerClusterer
    print("   [OK] SpeakerClusterer imported")
except Exception as e:
    print(f"   [ERROR] {e}")

try:
    print("4. Testing style_finder...")
    from semantic_detector.style_finder import StyleFinder
    print("   [OK] StyleFinder imported")
except Exception as e:
    print(f"   [ERROR] {e}")

try:
    print("5. Testing embedding_generator...")
    from semantic_detector.embedding_generator import EmbeddingGenerator
    print("   [OK] EmbeddingGenerator imported")
except Exception as e:
    print(f"   [ERROR] {e}")

try:
    print("6. Testing core.detector...")
    from semantic_detector.core.detector import SemanticDetector
    print("   [OK] SemanticDetector imported")
except Exception as e:
    print(f"   [ERROR] {e}")

try:
    print("7. Testing main package import...")
    from semantic_detector import SemanticDetector
    print("   [OK] Main package import works")
except Exception as e:
    print(f"   [ERROR] {e}")

try:
    print("8. Testing CLI import...")
    from semantic_detector.cli.main import main
    print("   [OK] CLI main imported")
except Exception as e:
    print(f"   [ERROR] {e}")

try:
    print("9. Testing web app imports...")
    from semantic_detector.web.app.core.config import settings
    print("   [OK] Web app config imported")
except Exception as e:
    print(f"   [ERROR] {e}")

print("\nImport test complete!")
