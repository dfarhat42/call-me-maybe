"""Build the token-ID -> text table used by constrained decoding."""

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from llm_sdk.llm_sdk import Small_LLM_Model

REPLACEMENT_CHAR = "\ufffd"


def load_special_ids(tokenizer_path: Path) -> set[int]:
    """Return the IDs of tokens flagged as special in tokenizer.json.

    Args:
        tokenizer_path: Path to the model's tokenizer.json file.

    Returns:
        Set of token IDs that must never appear in generated values.
    """
    with tokenizer_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return {
        entry["id"]
        for entry in data.get("added_tokens", [])
        if entry.get("special", False)
    }


def get_logit_size(model: Small_LLM_Model) -> int:
    """Measure how many logit positions the model outputs.

    Args:
        model: The loaded model wrapper.

    Returns:
        The length of the logits vector (the number of token IDs the
        model can emit, including unused padding positions).
    """
    return len(model.get_logits_from_input_ids([0]))


def build_token_table(
    model: Small_LLM_Model,
    logit_size: int,
    special_ids: set[int],
) -> tuple[list[str], NDArray[np.bool_]]:
    """Decode every token ID once and record which ones are usable.

    Args:
        model: The loaded model wrapper (its public decode is used).
        logit_size: Number of logit positions to cover.
        special_ids: IDs that must be excluded (special tokens).

    Returns:
        A pair (texts, allowed). texts[i] is the text produced by
        token i ("" if unusable). allowed[i] is True only if token i
        may be used in a constrained segment.
    """
    texts: list[str] = [""] * logit_size
    allowed = np.zeros(logit_size, dtype=np.bool_)

    for token_id in range(logit_size):
        if token_id in special_ids:
            continue
        try:
            text = model.decode([token_id])
        except Exception:
            continue  # unmapped / padding ID
        if not text or REPLACEMENT_CHAR in text:
            continue  # empty or partial UTF-8 token
        texts[token_id] = text
        allowed[token_id] = True

    return texts, allowed
