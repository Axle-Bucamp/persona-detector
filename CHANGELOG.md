# Changelog

## Version 0.1.0

### Added
- Complete module reorganization
- Docker support with docker-compose
- Enhanced test data with 10-person discussion
- TF-IDF vectorization for style identification
- Sentiment analysis with person scores
- 3D Plotly visualizations in web UI
- Histogram distributions for words and features
- Improved error handling for Ollama connections

### Changed
- Project restructured as proper Python module
- Root directory minimal (main.py, pyproject.toml, Dockerfile)
- All core modules in `semantic_detector/` package
- Web app in `semantic_detector/web/`
- CLI in `semantic_detector/cli/`
- Tests in `tests/` directory
- Sample data in `data/` directory

### Fixed
- Import paths updated for module structure
- Web app static/template paths corrected
- Better error messages and warnings

