"""Validators for numbers, strings and function names."""


import regex as re


def is_nb_complete(text: str, integer_only: bool) -> bool:
    """Check whether text is a finished, valid number.

    Follows the JSON number grammar: an optional minus sign, an integer
    part without leading zeros, and (unless integer_only) an optional
    decimal part with at least one digit.

    Args:
        text: The number text built so far.
        integer_only: If True, a decimal point is not allowed.

    Returns:
        True if text is a complete number, e.g. "0", "-12", "1.5".
        False for "", "-", "1." or "007".
    """
    if integer_only:
        return re.fullmatch(r"-?(0|[1-9]\d*)", text) is not None
    return re.fullmatch(r"-?(0|[1-9]\d*)(\.\d+)?", text) is not None


def is_nb_prefix(text: str, integer_only: bool) -> bool:
    """Check whether text can still grow into a valid number.

    Looser than is_nb_complete: the empty string, a lone "-" and a
    trailing "." (e.g. "1.") are legal prefixes, but a leading ".",
    leading zeros and any other characters are not.

    Args:
        text: The candidate number text (buffer plus new token).
        integer_only: If True, a decimal point is not allowed.

    Returns:
        True if text is a legal prefix of a number.
    """
    if integer_only:
        return re.fullmatch(r"(|-|-?(0|[1-9]\d*))", text) is not None
    return re.fullmatch(r"(|-|-?(0|[1-9]\d*)(\.\d*)?)", text) is not None


def is_string_terminator(token_text: str) -> bool:
    """Check whether a token signals the end of a string value.

    A token counts as a terminator if its text starts with a double
    quote after leading whitespace is removed. The token itself is
    discarded by the caller.

    Args:
        token_text: The text of the candidate token.

    Returns:
        True if the token starts with a closing quote.
    """
    return token_text.lstrip().startswith('"')


def is_name_prefix(buffer: str, token_text: str, names: list[str]) -> bool:
    """Check whether buffer plus token is a prefix of a valid name.

    Args:
        buffer: The name text built so far.
        token_text: The text of the candidate token.
        names: All valid names (function names, or "true"/"false").

    Returns:
        True if at least one name starts with buffer + token_text.
    """
    for name in names:
        if name.startswith(buffer + token_text):
            return True
    return False


def is_name_complete(buffer: str, names: list[str]) -> bool:
    """Check whether buffer exactly equals one of the valid names.

    Args:
        buffer: The name text built so far.
        names: All valid names.

    Returns:
        True if buffer is a full name.
    """
    for name in names:
        if buffer == name:
            return True
    return False


def has_longer_name(buffer: str, names: list[str]) -> bool:
    """Check whether some name extends buffer with more characters.

    Used to decide if stopping is optional: when buffer is a complete
    name that is also a prefix of a longer one (e.g. "fn_add" and
    "fn_add_numbers"), the model must choose between stopping and
    continuing.

    Args:
        buffer: The name text built so far.
        names: All valid names.

    Returns:
        True if a name starts with buffer and is strictly longer.
    """
    for name in names:
        if name.startswith(buffer) and len(name) > len(buffer):
            return True
    return False
