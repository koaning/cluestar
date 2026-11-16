black:
	black --target-version py310 cluestar tests

flake:
	flake8 cluestar tests

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"
	python -m pip install black flake8 isort interrogate twine wheel

interrogate:
	interrogate -vv --ignore-nested-functions --ignore-semiprivate --ignore-private --ignore-magic --ignore-module --ignore-init-method --fail-under 100 cluestar

pypi:
	python -m pip install --upgrade build twine
	python -m build
	twine upload dist/*

clean:
	rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache

style: clean black flake interrogate clean

check: clean black flake interrogate clean