import json
from pathlib import Path


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_token_mapping(
    vocab: dict[str, int],
    added_tokens: list[dict]
) -> tuple[dict[int, str], list[str]]:
    """Build an ID-to-token mapping and report conflicts."""
    token_mapping: dict[int, str] = {}
    conflicts: list[str] = []

    # Add ordinary vocabulary tokens.
    for token, token_id in vocab.items():
        if token_id in token_mapping:
            conflicts.append(
                f"ID {token_id}: {token_mapping[token_id]!r} "
                f"and {token!r}"
            )
        else:
            token_mapping[token_id] = token

    # Add tokens defined in tokenizer.json.
    for entry in added_tokens:
        token = entry["content"]
        token_id = entry["id"]

        if token_id in token_mapping:
            if token_mapping[token_id] != token:
                conflicts.append(
                    f"ID {token_id}: {token_mapping[token_id]!r} "
                    f"and {token!r}"
                )
        else:
            token_mapping[token_id] = token

    return token_mapping, conflicts
