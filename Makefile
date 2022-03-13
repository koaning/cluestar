install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"
	pre-commit install
