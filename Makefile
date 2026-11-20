install:
	uv sync

run:
	uv run python3 -m src --functions_definition data/input/functions_definition.json  --input data/input/function_calling_tests.json  --output data/output/function_calling_results.json

debug:
	uv run python3 pdb main.py

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache build dist *.egg-info

lint:
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src
	uv run mypy src --strict
