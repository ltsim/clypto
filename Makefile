.PHONY: clean clean-build clean-cython clean-pyc install compile all

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

compile:
	python setup.py build_ext --inplace

install:
	python -m pip install -e .

all: clean compile install
