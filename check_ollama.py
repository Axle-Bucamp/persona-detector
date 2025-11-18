#!/usr/bin/env python3
"""Check Ollama connection and available models."""

import requests
import sys

endpoint = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"

print(f"Checking Ollama at {endpoint}...")

try:
    # Check if Ollama is running
    response = requests.get(f"{endpoint}/api/tags", timeout=5)
    response.raise_for_status()
    data = response.json()
    
    print(f"[OK] Ollama is running")
    print(f"\nAvailable models:")
    models = data.get('models', [])
    if models:
        for model in models:
            name = model.get('name', 'unknown')
            print(f"  - {name}")
    else:
        print("  (no models found)")
    
    # Check if nomic-embed-text exists
    model_names = [m.get('name', '') for m in models]
    if 'nomic-embed-text' not in model_names:
        print(f"\n[WARNING] 'nomic-embed-text' not found!")
        print(f"[INFO] Install it with: ollama pull nomic-embed-text")
    else:
        print(f"\n[OK] 'nomic-embed-text' is available")
    
    # Test embeddings endpoint
    print(f"\nTesting embeddings endpoint...")
    test_response = requests.post(
        f"{endpoint}/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": "test"
        },
        timeout=10
    )
    
    if test_response.status_code == 200:
        print(f"[OK] Embeddings endpoint works!")
        data = test_response.json()
        embedding = data.get('embedding', [])
        print(f"  Embedding dimension: {len(embedding)}")
    elif test_response.status_code == 404:
        print(f"[ERROR] Embeddings endpoint returned 404")
        print(f"  This might mean:")
        print(f"  1. The model doesn't exist (run: ollama pull nomic-embed-text)")
        print(f"  2. The API endpoint format has changed")
    else:
        print(f"[ERROR] Unexpected status code: {test_response.status_code}")
        print(f"  Response: {test_response.text[:200]}")
        
except requests.exceptions.ConnectionError:
    print(f"[ERROR] Cannot connect to Ollama at {endpoint}")
    print(f"[INFO] Make sure Ollama is running: ollama serve")
except Exception as e:
    print(f"[ERROR] {e}")

