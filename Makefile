PYTHON ?= python
PYTEST_ENV ?= PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

.PHONY: test lint compile verify clean

test:
	$(PYTEST_ENV) $(PYTHON) -m pytest -q

lint:
	ruff check src scripts tests

compile:
	$(PYTHON) -m compileall -q src scripts curation

verify: compile test

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
