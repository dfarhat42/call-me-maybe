from .token_table import load_special_ids, get_logit_size, build_token_table
from llm_sdk.llm_sdk import Small_LLM_Model
from pathlib import Path
from .models import Function
from .validators import is_string_terminator
from .generation import (append_forced,
                         generate_name,
                         build_number_masks,
                         generate_number,
                         build_string_masks,
                         generate_string,
                         generate_boolean)
from .io_handler import (
    CliError, load_and_validate_json, parse_args, write_output,
)
from .models import FunctionCall, validate_call
import sys
import numpy as np


INSTRUCTIONS = (
    "You convert a request into one function call written as JSON.\n"
    "Rules:\n"
    "- Pick the function whose description matches the request.\n"
    "- Copy numbers exactly as written in the request.\n"
    "- Copy text values exactly, without surrounding quotes.\n"
    "- For regex parameters, write a general pattern that matches the "
    "kind of text described, never a copy of one specific match. "
    "Use [0-9]+ for numbers, [aeiouAEIOU] for vowels, \\w+ for words.\n"
    "- Output only the JSON.\n"
)

EXAMPLES = (
    "Examples (these functions are NOT available; they only show the "
    "format):\n"
    "Functions:\n"
    "- fn_multiply(a: number, b: number): Multiply two numbers.\n"
    "- fn_shout(text: string): Convert text to upper case.\n"
    "- fn_remove_pattern(source: string, regex: string): "
    "Remove everything matching a regex from a string.\n\n"
    "Request: What is 7.5 times -2?\n"
    '{"name": "fn_multiply", "parameters": {"a": 7.5, "b": -2}}\n\n'
    "Request: Shout the phrase 'good morning'\n"
    '{"name": "fn_shout", "parameters": {"text": "good morning"}}\n\n'
    'Request: Remove all digits from "room 42 on floor 7"\n'
    '{"name": "fn_remove_pattern", "parameters": '
    '{"source": "room 42 on floor 7", "regex": "[0-9]+"}}\n\n'
    "Now the real task.\n"
)


def build_prompt(functions: list[Function], request: str) -> str:
    """Build the chat prompt that precedes the generated JSON.

    Args:
        functions: Available function definitions.
        request: The user's natural-language request.

    Returns:
        Prompt text ending right where the assistant answer begins.
    """
    lines = []
    for func in functions:
        params = ", ".join(
            f"{name}: {param.type}" for name, param in func.parameters.items()
        )
        lines.append(f"- {func.name}({params}): {func.description}")
    catalog = "\n".join(lines)
    return (
        "<|im_start|>user\n"
        f"{INSTRUCTIONS}\n{EXAMPLES}"
        f"Functions:\n{catalog}\n\n"
        f"Request: {request}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


NUMERIC_TYPES = ("number", "integer")


def main() -> None:
    """Run the pipeline and write the output file."""
    try:
        args = parse_args()
        functions, prompts = load_and_validate_json(args)
    except CliError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    names = [func.name for func in functions]
    by_name = {func.name: func for func in functions}

    try:
        model = Small_LLM_Model()
        special_ids = load_special_ids(
            Path(model.get_path_to_tokenizer_file())
        )
        texts, allowed = build_token_table(
            model, get_logit_size(model), special_ids
        )
    except Exception as error:
        print(f"Error: cannot load the model: {error}", file=sys.stderr)
        sys.exit(1)

    quote_mask = np.array(
        [bool(t) and is_string_terminator(t) for t in texts],
        dtype=np.bool_,
    ) & allowed
    chars_mask, stop_mask = build_number_masks(texts, allowed)
    string_mask = build_string_masks(texts, allowed)

    results: list[FunctionCall] = []
    failed = 0

    for item in prompts:
        input_ids = model.encode(
            build_prompt(functions, item.prompt)
        ).tolist()[0]
        append_forced(model, input_ids, '{"name": "')
        try:
            name = generate_name(
                model, input_ids, names, texts, allowed, quote_mask
            )
            func = by_name[name]

            values: dict[str, float | int | str | bool] = {}
            prev_was_string = False

            if not func.parameters:
                append_forced(model, input_ids, '", "parameters": {')

            for index, (key, param) in enumerate(func.parameters.items()):
                is_string = param.type == "string"

                if index == 0:
                    forced = '", "parameters": {'
                else:
                    forced = ('"' if prev_was_string else "") + ", "
                forced += f'"{key}": ' + ('"' if is_string else "")
                append_forced(model, input_ids, forced)

                if is_string:
                    values[key] = generate_string(
                        model, input_ids, texts, string_mask, quote_mask
                    )
                elif param.type == "boolean":
                    values[key] = generate_boolean(
                        model, input_ids, texts, allowed, quote_mask
                    )
                elif param.type in ("number", "integer"):
                    is_int = param.type == "integer"
                    text = generate_number(
                        model, input_ids, texts, chars_mask, stop_mask,
                        integer_only=is_int,
                    )
                    values[key] = int(text) if is_int else float(text)
                else:
                    raise ValueError(
                        f"Unsupported parameter type: {param.type!r}"
                    )
                prev_was_string = is_string

            call = FunctionCall(
                prompt=item.prompt, name=name, parameters=values
            )
            validate_call(call, func)
            results.append(call)
        except Exception as error:
            failed += 1
            print(f"Skipped {item.prompt!r}: {error}", file=sys.stderr)

    try:
        write_output(args["output"], results)
    except CliError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {len(results)} calls to {args['output']}"
          f" ({failed} skipped)")


if __name__ == "__main__":
    main()
