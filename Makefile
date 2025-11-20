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

docker-up-detached:
	docker-compose up -d

docker-down:
	docker-compose down

cloudflare:
	@echo "[INFO] Starting Cloudflare tunnel..."
	@bash start-cloudflare.sh

cloudflare-quick:
	@echo "[INFO] Starting quick Cloudflare tunnel (no config needed)..."
	cloudflared tunnel --url http://localhost:8000

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.html" -delete
	find . -type f -name "*_metadata.json" -delete

