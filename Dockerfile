FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install uv

# Copy project files
COPY pyproject.toml ./
COPY semantic_detector/ ./semantic_detector/
COPY data/ ./data/
COPY semantic_detector/web/public/ ./semantic_detector/web/public/ 2>/dev/null || true

# Install dependencies
RUN uv sync --frozen

# Download NLTK data
RUN python -m nltk.downloader punkt punkt_tab stopwords -q

# Expose port for web app
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "run_web.py"]

