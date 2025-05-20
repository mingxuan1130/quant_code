.PHONY: setup demo test clean lint format docs

# Python version
PYTHON := python3
PIP := pip3

# Default parameters
TRAIN_START := 2020-01-01
TRAIN_END := 2023-12-31
TEST_START := 2024-01-01
TEST_END := 2024-03-31

setup:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PYTHON) -c "from config import FEATURE_STORE, RAW_DATA_DIR, MODELS_DIR, LOGS_DIR, OUTPUT_DIR; print('Created directories:\n' + '\n'.join(str(d) for d in [FEATURE_STORE, RAW_DATA_DIR, MODELS_DIR, LOGS_DIR, OUTPUT_DIR]))"

demo:
	$(PYTHON) strategy/topN_tree.py \
		--train_start $(TRAIN_START) \
		--train_end $(TRAIN_END) \
		--test_start $(TEST_START) \
		--test_end $(TEST_END)

test:
	pytest tests/ -v --cov=. --cov-report=term-missing

lint:
	flake8 .
	black . --check
	isort . --check-only

format:
	black .
	isort .

docs:
	sphinx-build -b html docs/ docs/_build/html

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".coverage" -exec rm -r {} +
	find . -type d -name "htmlcov" -exec rm -r {} +
	find . -type d -name "dist" -exec rm -r {} +
	find . -type d -name "build" -exec rm -r {} +
	find . -type d -name ".tox" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	find . -type d -name ".ruff_cache" -exec rm -r {} + 