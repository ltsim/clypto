.PHONY: clean clean-build clean-cython clean-pyc uv-lock uv-sync uv-test docs-serve docs-build

clean: clean-build clean-cython clean-pyc

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .eggs/
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -exec rm -f {} +

clean-cython:
	find clypto -name '*.c' -exec rm -f {} +
	find clypto -name '*.so' -exec rm -f {} +
	find clypto -name '*.pyd' -exec rm -f {} +
	find clypto -name '*.html' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

uv-lock:
	uv lock

uv-sync:
	uv sync --extra dev

uv-test:
	uv run pytest tests/

docs-serve:
	uv sync --extra docs --no-install-project
	uv run --no-sync mkdocs serve

docs-build:
	uv sync --extra docs --no-install-project
	uv run --no-sync mkdocs build --strict