SHELL := /bin/bash 
export PYTHONPATH = .
WORKDIR := .

export LANG = en_US.UTF-8
export LC_ALL = en_US.UTF-8
export LC_CTYPE = en_US.UTF-8

.PHONY: help
help:
	@echo "Makefile for python projects"
	@echo "  TODO: add new help text for your project"
	@echo ""
	@echo "Initial setup - install local dependencies:"
	@echo "  make setup"
	@echo ""
	@echo "Linting and formatting:"
	@echo "  make lint"
	@echo "  make format"
	@echo ""
	@echo "Run tests:"
	@echo "  make test"
	@echo ""
	@echo "Remove build atrifacts:"
	@echo "  make clean"

PROJECT_ROOT := $(shell echo `git rev-parse --show-toplevel`)

.PHONY: setup
setup:
	@echo "### Start environment setup"
	@echo $(PROJECT_ROOT)
	poetry install --with dev,test
	@echo "### Environment setup complete"

.PHONY: setup-githooks
setup-githooks:
	@echo "### Installing git hooks"
	./githooks/setup.sh

.PHONY: clean
clean:
	@echo "### Cleaning up: __pycache__"
	find . -name __pycache__ -type d | xargs rm -r
	@echo "### Cleaning up: poetry"
	poetry env remove --all

.PHONY: lint
lint:
	poetry run pylint -j 4 eiq

.PHONY: test
test:
	poetry run pytest --cov-config=.coveragerc --cov=eiq --cov-fail-under=80 --cov-branch

.PHONY: format
format: setup
	@echo "### Running isort to PEP-8 compatible sort order"
	poetry run isort .
	@echo "### Running black for PEP-8 compatible files"
	poetry run black .

.PHONY: format-ci
format-ci:
	poetry run black . --exclude="venv|.pyenv" --check
	poetry run isort . --check-only

.PHONY: build
build:
	@echo "### Building docker images"
	docker compose build

.PHONY: start
up: build
	@echo "### Starting services"
	docker compose up -d
	@echo "### Services are running in the background"
	@echo "### TODO: add a helpful message for where to find the services(s) for your project"

.PHONY: down
down:
	@echo "### Stopping sericves and removing containers"
	docker compose down

.PHONY: stop
stop:
	@echo "### Stopping services"
	docker compose stop

.PHONY: coverage
coverage: setup
	@echo "### Stopping services"
	poetry run coverage run -m pytest
	poetry run coverage report -m
