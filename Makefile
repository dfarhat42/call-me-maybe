install:
	uv sync

run:
	uv run python3 -m src --functions_definition /home/dfarhat/sgoinfre/call_me_maybe/data/input/functions_definition.json  --input /home/dfarhat/sgoinfre/call_me_maybe/data/input/function_calling_tests.json  --output /home/dfarhat/sgoinfre/call_me_maybe/data/output/function_calling_results.json

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
