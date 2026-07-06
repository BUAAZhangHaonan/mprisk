PYTHON ?= python3.11
PYTEST_ENV ?= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

.PHONY: test lint compile verify clean

test:
	$(PYTEST_ENV) $(PYTHON) -m pytest -q

lint:
	ruff check src scripts tests

compile:
	$(PYTHON) -m compileall -q src scripts

verify: compile test

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
