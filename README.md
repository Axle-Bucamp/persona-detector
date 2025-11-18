# Semantic Detector

Language fingerprinting and embedding analysis system with web interface.

## Quick Start

### Prerequisites

1. **Install Ollama** (for embeddings):
   ```bash
   # Download from https://ollama.com or:
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Pull the embedding model**:
   ```bash
   ollama pull nomic-embed-text
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

### Using UV (Recommended)

```bash
# Run CLI
uv run python main.py data/sample_discussion.txt

# Run web app
uv run python run_web.py
```

### Using Docker

```bash
# Build and run with Docker Compose (includes Ollama)
docker-compose up

# Or build manually
docker build -t semantic-detector .
docker run -p 8000:8000 semantic-detector
```

## Project Structure

```
semantic-detector/
├── semantic_detector/      # Main package
│   ├── core/              # Core modules
│   ├── cli/               # CLI interface
│   └── web/               # Web application
├── data/                  # Sample data
├── tests/                 # Test suite
├── main.py                # CLI entry point
├── run_web.py             # Web app entry point
├── pyproject.toml         # UV/Pip dependencies
└── Dockerfile             # Docker configuration
```

## Features

- **Language Fingerprinting**: TF-IDF, n-grams, stylometric features
- **3D Visualization**: Interactive UMAP projections
- **Sentiment Analysis**: Per-sentence and cluster-level sentiment
- **Style Identification**: Find similar writing styles
- **Web Interface**: FastAPI with Jinja2 templates
- **CLI Interface**: Command-line tool for batch processing

## Usage

### CLI

```bash
python main.py <file> [options]
python main.py data/sample_discussion.txt --clusters 10
```

### Web App

```bash
python run_web.py
# Or: python -m semantic_detector.web
# Open http://localhost:8000
```

### Python Module

```python
from semantic_detector import SemanticDetector

detector = SemanticDetector()
result = detector.process_text_file("data/sample_discussion.txt")
```

## Troubleshooting

### 404 Error on Embeddings

If you see `404 Client Error: Not Found`:
1. **Install the model**: `ollama pull nomic-embed-text`
2. **Verify**: `ollama list | grep nomic`
3. **Test**: `python check_ollama.py`

See [SETUP_OLLAMA.md](SETUP_OLLAMA.md) for detailed setup instructions.

## Requirements

- Python 3.9+
- Ollama (optional, for embeddings)
- See `pyproject.toml` for dependencies

## License

MIT License
