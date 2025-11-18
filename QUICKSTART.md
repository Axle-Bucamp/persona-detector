# Quick Start Guide

## Installation

```bash
# Install UV (if not already installed)
# Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Download NLTK data
python -m nltk.downloader punkt punkt_tab stopwords
```

## Usage

### CLI Mode

```bash
# Analyze sample discussion (10 people, different styles)
uv run python main.py data/sample_discussion.txt --clusters 10

# With custom options
uv run python main.py data/sample_discussion.txt --clusters 10 --method dbscan
```

### Web App Mode

```bash
# Start web server
uv run python run_web.py

# Or use custom port
uv run python run_web.py 8080

# Open browser to http://localhost:8000
```

### Docker Mode

```bash
# Build and run with Docker Compose (includes Ollama)
docker-compose up

# Or build manually
docker build -t semantic-detector .
docker run -p 8000:8000 semantic-detector
```

### Python Module

```python
from semantic_detector import SemanticDetector

detector = SemanticDetector()
result = detector.process_text_file("data/sample_discussion.txt")

# Access results
sentences = result['sentences']
labels = result['labels']
coords_3d = result['coords_3d']
```

## Sample Data

The `data/sample_discussion.txt` file contains a simulated discussion between 10 people with distinct writing styles:
- Person 1: Formal Academic
- Person 2: Casual Conversational  
- Person 3: Technical Professional
- Person 4: Creative Writer
- Person 5: Business Executive
- Person 6: Poetic Expressive
- Person 7: Scientific Analytical
- Person 8: Friendly Informal
- Person 9: Philosophical Reflective
- Person 10: Concise Direct

This is perfect for testing the clustering and style identification features!

