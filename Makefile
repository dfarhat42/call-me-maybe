install:
	uv sync

run:
	uv run main.py

debug:
	uv run python3 pdb main.py

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache build dist *.egg-info

lint:
	uv run flake8 main.py src
	uv run mypy main.py src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src main.py
	uv run mypy src main.py --strict
