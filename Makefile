.PHONY: test lint compile verify clean

test:
	pytest -q

lint:
	ruff check src scripts tests

compile:
	python -m compileall -q src scripts

verify: compile test

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
