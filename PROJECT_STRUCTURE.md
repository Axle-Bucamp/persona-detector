# Project Structure

## Clean Module Layout

```
semantic-detector/                    # Root (minimal)
├── main.py                          # CLI entry point
├── run_web.py                       # Web app entry point
├── pyproject.toml                   # UV/Pip dependencies
├── Dockerfile                       # Docker configuration
├── docker-compose.yml               # Docker Compose setup
├── Makefile                         # Convenience commands
├── README.md                        # Main documentation
├── CHANGELOG.md                     # Version history
│
├── semantic_detector/               # Main Python package
│   ├── __init__.py                  # Package exports
│   │
│   ├── core/                        # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py                # Configuration
│   │   └── detector.py              # Main SemanticDetector class
│   │
│   ├── cli/                         # CLI interface
│   │   ├── __init__.py
│   │   ├── __main__.py              # Module entry point
│   │   └── main.py                  # CLI implementation
│   │
│   ├── web/                         # Web application
│   │   ├── __init__.py
│   │   ├── __main__.py              # Module entry point
│   │   ├── static/                  # Static assets
│   │   │   ├── css/style.css
│   │   │   └── js/
│   │   │       ├── app.js
│   │   │       └── visualizations.js
│   │   ├── templates/               # Jinja2 templates
│   │   │   └── index.html
│   │   └── app/                     # FastAPI app
│   │       ├── __init__.py
│   │       ├── main.py              # FastAPI app
│   │       ├── api/
│   │       │   └── routes.py        # API endpoints
│   │       ├── core/
│   │       │   ├── config.py
│   │       │   ├── detector.py      # Web service wrapper
│   │       │   ├── tfidf_analyzer.py
│   │       │   └── sentiment_analyzer.py
│   │       └── models/
│   │           └── schemas.py       # Pydantic models
│   │
│   └── [core modules]              # Shared modules
│       ├── sentence_splitter.py
│       ├── embedding_generator.py
│       ├── language_fingerprint.py
│       ├── clustering.py
│       ├── style_finder.py
│       └── visualization.py
│
├── data/                            # Sample data
│   └── sample_discussion.txt        # 10-person discussion
│
└── tests/                           # Test suite
    ├── test_semantic_detector.py
    └── test_app_imports.py
```

## Entry Points

- **CLI**: `python main.py` or `python -m semantic_detector.cli`
- **Web**: `python run_web.py` or `python -m semantic_detector.web`
- **Module**: `from semantic_detector import SemanticDetector`

## Key Features

- ✅ Clean module structure
- ✅ Minimal root directory
- ✅ Proper Python package
- ✅ Docker support
- ✅ UV environment
- ✅ Enhanced test data

