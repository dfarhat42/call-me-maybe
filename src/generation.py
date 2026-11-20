"""Shared helpers for constrained token selection."""

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from .validators import has_longer_name, is_name_complete, is_name_prefix
from llm_sdk.llm_sdk import Small_LLM_Model
from .validators import is_nb_complete, is_nb_prefix


NUMBER_CHARS = set("0123456789-.")
BOOLEAN_VALUES = ["true", "false"]


def next_logits(
    model: Small_LLM_Model, input_ids: list[int]
) -> NDArray[np.float32]:
    """Run the model once and return the next-token logits.

    Args:
        model: The loaded model wrapper.
        input_ids: Token IDs of the context so far.

    Returns:
        A 1-D float array with one logit per token ID.
    """
    return np.asarray(
        model.get_logits_from_input_ids(input_ids), dtype=np.float32
    )


def best_allowed(
    logits: NDArray[np.float32],
    mask: NDArray[np.bool_],
    texts: list[str],
    predicate: Callable[[str], bool] | None = None,
) -> tuple[int, float] | None:
    """Return the highest-scoring token that passes all constraints.

    Args:
        logits: Next-token logits from the model.
        mask: Boolean array; only True positions are candidates.
        texts: Token ID -> text table.
        predicate: Optional extra check on a candidate's text.

    Returns:
        (token_id, logit) of the best valid token, or None if no
        candidate passes.
    """
    masked = np.where(mask, logits, -np.inf)
    for token_id in np.argsort(masked)[::-1]:
        if masked[token_id] == -np.inf:
            return None  # ran out of candidates
        tid = int(token_id)
        if predicate is None or predicate(texts[tid]):
            return tid, float(masked[tid])
    return None


def append_forced(
    model: Small_LLM_Model, input_ids: list[int], text: str
) -> None:
    """Append forced text to the context without calling the model.

    Args:
        model: The loaded model wrapper (its encode is used).
        input_ids: Context to extend in place.
        text: Literal text to add, e.g. '", "parameters": {'.
    """
    input_ids.extend(model.encode(text).tolist()[0])


def generate_name(
    model: Small_LLM_Model,
    input_ids: list[int],
    names: list[str],
    texts: list[str],
    allowed: NDArray[np.bool_],
    quote_mask: NDArray[np.bool_]
) -> str:
    """Let the model choose one function name, token by token.

    Only tokens that keep the buffer a prefix of a real name are
    considered. Chosen tokens are appended to input_ids in place.

    Args:
        model: The loaded model wrapper.
        input_ids: Context so far; must end where the name begins.
        names: All valid function names.
        texts: Token ID -> text table.
        allowed: Mask of usable tokens.
        quote_mask: Mask of tokens whose text starts with a quote.

    Returns:
        The chosen function name.

    Raises:
        ValueError: If no function names were given.
        RuntimeError: If no valid token exists at some step.
    """
    if not names:
        raise ValueError("No function names available")

    buffer = ""
    max_steps = max(len(name) for name in names)  # >= 1 char per token

    for _ in range(max_steps):
        complete = is_name_complete(buffer, names)
        longer = has_longer_name(buffer, names)

        if complete and not longer:
            break  # nothing left for the model to decide

        logits = next_logits(model, input_ids)
        current = buffer

        def is_valid(text: str) -> bool:
            return is_name_prefix(current, text, names)

        cont = best_allowed(logits, allowed, texts, is_valid)

        if complete:  # a longer name exists: stopping is optional
            stop = best_allowed(logits, allowed & quote_mask, texts)
            if stop is not None and (cont is None or stop[1] > cont[1]):
                break

        if cont is None:
            raise RuntimeError(
                f"No valid token to continue function name {buffer!r}"
            )

        token_id = cont[0]
        input_ids.append(token_id)
        buffer += texts[token_id]

    if not is_name_complete(buffer, names):
        raise RuntimeError(f"Incomplete function name: {buffer!r}")
    return buffer


def build_number_masks(
    texts: list[str], allowed: NDArray[np.bool_]
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Precompute the token masks used by number generation.

    Args:
        texts: Token ID -> text table.
        allowed: Mask of usable tokens.

    Returns:
        (chars_mask, stop_mask). chars_mask marks tokens made only of
        digits, '-' and '.'. stop_mask marks tokens whose text, after
        leading whitespace, starts with ',' or '}'.
    """
    chars = np.array(
        [bool(t) and set(t) <= NUMBER_CHARS for t in texts],
        dtype=np.bool_,
    )
    stop = np.array(
        [t.lstrip().startswith((",", "}")) for t in texts],
        dtype=np.bool_,
    )
    return allowed & chars, allowed & stop


def generate_number(
    model: Small_LLM_Model,
    input_ids: list[int],
    texts: list[str],
    chars_mask: NDArray[np.bool_],
    stop_mask: NDArray[np.bool_],
    integer_only: bool = False,
    max_len: int = 20,
) -> str:
    """Let the model write one number, token by token.

    Chosen tokens are appended to input_ids in place. The stop token
    is discarded, so the caller appends the forced separator.

    Args:
        model: The loaded model wrapper.
        input_ids: Context so far; must end right before the value.
        texts: Token ID -> text table.
        chars_mask: Tokens made only of number characters.
        stop_mask: Tokens that start with ',' or '}'.
        integer_only: Forbid a decimal point.
        max_len: Maximum number of characters in the value.

    Returns:
        The number as text (convert with float() or int() later).

    Raises:
        RuntimeError: If no valid token exists or the value is
            incomplete when the length cap is hit.
    """
    buffer = ""

    for _ in range(max_len):
        logits = next_logits(model, input_ids)
        current = buffer

        def fits(text: str) -> bool:
            return (
                len(current) + len(text) <= max_len
                and is_nb_prefix(current + text, integer_only)
            )

        cont = best_allowed(logits, chars_mask, texts, fits)

        if is_nb_complete(buffer, integer_only):
            stop = best_allowed(logits, stop_mask, texts)
            if cont is None or (stop is not None and stop[1] > cont[1]):
                return buffer

        if cont is None:
            raise RuntimeError(f"No valid token to continue {buffer!r}")

        input_ids.append(cont[0])
        buffer += texts[cont[0]]

    if not is_nb_complete(buffer, integer_only):
        raise RuntimeError(f"Incomplete number: {buffer!r}")
    return buffer


def build_string_masks(
    texts: list[str], allowed: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    """Precompute the mask of tokens usable inside a string value.

    A token is usable if it contains no double quote and no control
    character (code point below 0x20). Backslashes are allowed, since
    Python's json module escapes them when the result is written.

    Args:
        texts: Token ID -> text table.
        allowed: Mask of usable tokens.

    Returns:
        Boolean mask of tokens that may appear inside a string.
    """
    safe = np.array(
        [
            bool(t) and '"' not in t and all(ord(c) >= 0x20 for c in t)
            for t in texts
        ],
        dtype=np.bool_,
    )
    return allowed & safe


def generate_string(
    model: Small_LLM_Model,
    input_ids: list[int],
    texts: list[str],
    string_mask: NDArray[np.bool_],
    quote_mask: NDArray[np.bool_],
    max_len: int = 200,
) -> str:
    """Let the model write one string value, token by token.

    Chosen tokens are appended to input_ids in place. The closing
    quote token is discarded, so the caller appends the forced text
    that starts with '"'.

    Args:
        model: The loaded model wrapper.
        input_ids: Context so far; must end right after the opening
            quote of the value.
        texts: Token ID -> text table.
        string_mask: Tokens usable inside a string.
        quote_mask: Tokens whose text starts with a quote.
        max_len: Maximum number of characters in the value.

    Returns:
        The raw string value, with surrounding whitespace stripped.
    """
    buffer = ""

    while len(buffer) < max_len:
        logits = next_logits(model, input_ids)
        current = buffer

        def fits(text: str) -> bool:
            return len(current) + len(text) <= max_len

        cont = best_allowed(logits, string_mask, texts, fits)
        stop = best_allowed(logits, quote_mask, texts)

        if cont is None or (stop is not None and stop[1] > cont[1]):
            break

        input_ids.append(cont[0])
        buffer += texts[cont[0]]

    return buffer.strip()


def generate_boolean(
    model: Small_LLM_Model,
    input_ids: list[int],
    texts: list[str],
    allowed: NDArray[np.bool_],
    quote_mask: NDArray[np.bool_],
) -> bool:
    """Let the model choose between true and false.

    Reuses the name-selection logic with the two literals as the
    only valid "names". Chosen tokens are appended to input_ids.

    Args:
        model: The loaded model wrapper.
        input_ids: Context so far; must end right before the value.
        texts: Token ID -> text table.
        allowed: Mask of usable tokens.
        quote_mask: Mask of tokens starting with a quote (unused in
            practice, since neither literal is a prefix of the other).

    Returns:
        The chosen boolean.
    """
    choice = generate_name(
        model, input_ids, BOOLEAN_VALUES, texts, allowed, quote_mask
    )
    return choice == "true"
