# Makefile for Alpaca Trading Bot

.PHONY: help install install-dev test test-cov lint format clean run run-paper validate docker-build docker-run setup

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code"
	@echo "  clean        - Clean cache and temporary files"
	@echo "  run          - Run the trading bot"
	@echo "  run-paper    - Run the trading bot in paper mode"
	@echo "  validate     - Validate configuration"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run in Docker"
	@echo "  setup        - Complete development setup"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Testing
test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ --cov=. --cov-report=html --cov-report=term-missing

test-unit:
	python -m pytest tests/ -v -m "unit"

test-integration:
	python -m pytest tests/ -v -m "integration"

# Code quality
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	mypy . --ignore-missing-imports

format:
	black .
	isort .

format-check:
	black --check .
	isort --check-only .

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Running
run:
	python main.py

run-paper:
	python main.py --paper

run-debug:
	python main.py --log-level DEBUG

run-symbols:
	python main.py --symbols TQQQ SQQQ QQQ SPY

validate:
	python main.py --validate-only

# Docker
docker-build:
	docker build -t alpaca-trading-bot .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f trading-bot

# Development setup
setup: clean install-dev
	@echo "Setting up development environment..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env file from template"; fi
	@echo "Please edit .env file with your API keys"
	@echo "Setup complete!"

# Database
db-migrate:
	alembic upgrade head

db-reset:
	rm -f trading_bot.db
	alembic upgrade head

# Monitoring
logs:
	tail -f logs/trading_bot.log

# Release
build:
	python setup.py sdist bdist_wheel

upload-test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*

# Documentation
docs:
	sphinx-build -b html docs/ docs/_build/

docs-clean:
	rm -rf docs/_build/

# Security
security-check:
	pip-audit
	bandit -r . -x tests/

# Performance
profile:
	python -m cProfile -o profile.stats main.py --validate-only
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# CI/CD helpers
ci-install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e ".[dev]"

ci-test:
	python -m pytest tests/ --cov=. --cov-report=xml --junitxml=junit.xml

ci-lint:
	flake8 . --format=github --tee --output-file=flake8-report.txt
	mypy . --junit-xml=mypy-report.xml --ignore-missing-imports
