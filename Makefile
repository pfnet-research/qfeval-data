PROJECT_NAME := qfeval_data
RUN := uv run

.PHONY: check
check: test lint

.PHONY: install
install:
	uv sync --dev

.PHONY: test
test: doctest pytest

.PHONY: doctest
doctest:
	@echo "To be implemented"
#	$(RUN) pytest --doctest-modules $(PROJECT_NAME)

.PHONY: pytest
pytest:
	$(RUN) pytest tests -m "not gpu"

.PHONY: pytest-gpu
pytest-gpu:
	$(RUN) pytest tests

.PHONY: test-cov
test-cov:
	$(RUN) pytest --cov=$(PROJECT_NAME) --cov-report=xml -m "not gpu"

.PHONY: lint
lint: lint-black lint-isort flake8 mypy

.PHONY: lint-black
lint-black:
	$(RUN) black --check --diff --quiet .

.PHONY: lint-isort
lint-isort:
	$(RUN) isort --check --quiet .

.PHONY: mypy
mypy:
	$(RUN) mypy $(PROJECT_NAME)

.PHONY: flake8
flake8:
	$(RUN) pflake8 $(PROJECT_NAME)

.PHONY: format
format: format-black format-isort

.PHONY: format-black
format-black:
	$(RUN) black --quiet .

.PHONY: format-isort
format-isort:
	$(RUN) isort --quiet .
