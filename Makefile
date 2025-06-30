.PHONY: format check install help run run-dev run-fastapi run-fastapi-dev

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies using Poetry
	poetry lock
	poetry install --no-root

dev_install: ## Install dependencies using Poetry including dev dependencies
	poetry lock
	poetry install --no-root --extras dev

format: ## Format code using ruff
	poetry run ruff format .
	poetry run ruff check --fix .

check: ## Check code formatting and linting without fixing
	poetry run ruff format --check .
	poetry run ruff check .