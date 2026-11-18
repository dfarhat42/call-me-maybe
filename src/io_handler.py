"""Command-line parsing and JSON file input/output."""

import argparse
import json
from pathlib import Path
from typing import Any

import pydantic as pyd

from .models import Function, FunctionCall, Prompts

DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calls.json"


class CliError(Exception):
    """A user-facing error: bad arguments, files or contents."""


def parse_args() -> dict[str, str]:
    """Parse and check the command-line arguments.

    Returns:
        Dict with the keys functions_definition, input and output.

    Raises:
        CliError: If an input file is missing or the output path is
            a directory.
    """
    parser = argparse.ArgumentParser(
        description="Translate prompts into function calls."
    )
    parser.add_argument(
        "--functions_definition", type=str, default=DEFAULT_FUNCTIONS
    )
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    for label, value in (
        ("functions_definition", args.functions_definition),
        ("input", args.input),
    ):
        if not Path(value).is_file():
            raise CliError(f"File for --{label} not found: {value}")
    if Path(args.output).is_dir():
        raise CliError(f"--output is a directory: {args.output}")

    return {
        "functions_definition": args.functions_definition,
        "input": args.input,
        "output": args.output,
    }


def _read_json(path: str, label: str) -> Any:
    """Read a JSON file, turning every failure into a CliError."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as error:
        raise CliError(f"{label} file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise CliError(
            f"{label} file is not valid JSON ({path}): {error}"
        ) from error
    except (OSError, UnicodeDecodeError) as error:
        raise CliError(f"Cannot read {label} file {path}: {error}") from error


def load_and_validate_json(
    args_dict: dict[str, str],
) -> tuple[list[Function], list[Prompts]]:
    """Load both input files and validate them with pydantic.

    Args:
        args_dict: The dict returned by parse_args.

    Returns:
        (functions, prompts).

    Raises:
        CliError: If a file is unreadable, malformed or empty of
            function definitions.
    """
    raw_funcs = _read_json(args_dict["functions_definition"], "Functions")
    raw_prompts = _read_json(args_dict["input"], "Input")

    for label, raw in (("Functions", raw_funcs), ("Input", raw_prompts)):
        if not isinstance(raw, list):
            raise CliError(f"{label} file must contain a JSON array")

    try:
        functions = [Function.model_validate(item) for item in raw_funcs]
        prompts = [Prompts.model_validate(item) for item in raw_prompts]
    except pyd.ValidationError as error:
        raise CliError(f"Invalid file content: {error}") from error

    if not functions:
        raise CliError("No function definitions provided")
    return functions, prompts


def write_output(out_file: str, results: list[FunctionCall]) -> None:
    """Write the results as a JSON array, creating the directory.

    Args:
        out_file: Destination path.
        results: Validated calls to write.

    Raises:
        CliError: If the file cannot be written.
    """
    path = Path(out_file)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                [call.model_dump() for call in results], file, indent=4
            )
    except OSError as error:
        raise CliError(f"Cannot write output {out_file}: {error}") from error
