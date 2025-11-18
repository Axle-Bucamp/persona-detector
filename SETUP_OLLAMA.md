# Setting Up Ollama for Semantic Detector

## Quick Setup

1. **Install Ollama** (if not already installed):
   - Download from: https://ollama.com
   - Or use: `curl -fsSL https://ollama.com/install.sh | sh`

2. **Pull the embedding model**:
   ```bash
   ollama pull nomic-embed-text
   ```

3. **Verify installation**:
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434
   
   # List available models
   ollama list
   
   # Test the model
   python check_ollama.py
   ```

## Alternative Embedding Models

If `nomic-embed-text` is not available, you can use other embedding models:

```bash
# Option 1: nomic-embed-text (recommended)
ollama pull nomic-embed-text

# Option 2: mxbai-embed-large
ollama pull mxbai-embed-large

# Option 3: all-minilm (smaller, faster)
ollama pull all-minilm
```

Then update your code to use the model:
```python
detector = SemanticDetector(embedding_model="mxbai-embed-large")
```

## Troubleshooting

### 404 Error on `/api/embeddings`
- **Cause**: Model not installed or endpoint changed
- **Solution**: 
  1. Pull the model: `ollama pull nomic-embed-text`
  2. Verify: `ollama list | grep nomic`

### Connection Refused
- **Cause**: Ollama not running
- **Solution**: Start Ollama service
  ```bash
  ollama serve
  # Or on Windows/Mac, Ollama should auto-start
  ```

### Model Not Found
- **Cause**: Model name incorrect or not available
- **Solution**: Check available models and use correct name
  ```bash
  ollama list
  ollama pull <correct-model-name>
  ```

## Testing

Run the check script:
```bash
python check_ollama.py
```

This will verify:
- Ollama is running
- Models are available
- Embeddings endpoint works

