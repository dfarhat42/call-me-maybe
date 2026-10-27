import argparse
from pathlib import Path
import json
from models import Function, Prompts


def parse_args() -> dict[str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--functions_definition',
        type=str,
        required=True
        )
    parser.add_argument(
        '--input',
        type=str,
        required=True
        )
    parser.add_argument(
        '--output',
        type=str,
        required=True
        )
    args = parser.parse_args()
    func_path = Path(args.functions_definition)
    input_path = Path(args.input)
    if not func_path.exists() or not func_path.is_file():
        raise FileNotFoundError("Path provided "
                                "for functions_definition not found")
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError("Path provided "
                                "for input not found")
    args_dict = {
        "functions_definition": args.functions_definition,
        "input": args.input,
        "output": args.output
    }
    return args_dict


def load_and_validate_json(args_dict: dict[str, str]) -> tuple[list[Function],
                                                               list[Prompts]]:
    with open(args_dict["functions_definition"], 'r') as file:
        func_defs = json.load(file)
    with open(args_dict["input"], 'r') as file:
        input_data = json.load(file)
    func_defs_list: list[Function] = []
    for func in func_defs:
        func_defs_list.append(Function.model_validate(func))
    prompt_list: list[Prompts] = []
    for prompt in input_data:
        prompt_list.append(Prompts.model_validate(prompt))
    return (func_defs_list, prompt_list)


if __name__ == "__main__":
    diction = parse_args()
    load_and_validate_json(diction)
