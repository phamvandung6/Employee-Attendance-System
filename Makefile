.PHONY: install dev run test lint format clean migrate upgrade downgrade docker-up docker-down

# Install dependencies
install:
	uv sync

# Run development server
dev:
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run production server
run:
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run tests
test:
	uv run pytest tests/ -v --cov=app --cov-report=html

# Lint code
lint:
	uv run ruff check app tests

# Format code
format:
	uv run black app tests
	uv run ruff check --fix app tests

# Clean cache and temp files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache

# Database migrations
migrate:
	uv run alembic revision --autogenerate -m "$(msg)"

upgrade:
	uv run alembic upgrade head

downgrade:
	uv run alembic downgrade -1

# Docker commands
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Initialize database
init-db:
	uv run python scripts/init_db.py

# Seed data
seed:
	uv run python scripts/seed_data.py

