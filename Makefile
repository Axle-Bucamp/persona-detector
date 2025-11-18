.PHONY: install test run-web run-cli clean docker-build docker-up

install:
	uv sync

test:
	uv run python -m pytest tests/

run-web:
	uv run python run_web.py

run-cli:
	uv run python main.py data/sample_discussion.txt

docker-build:
	docker build -t semantic-detector .

docker-up:
	docker-compose up

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.html" -delete
	find . -type f -name "*_metadata.json" -delete

