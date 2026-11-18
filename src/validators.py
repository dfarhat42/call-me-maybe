import regex as re


def is_nb_complete(text: str, integer_only: bool) -> bool:
    if integer_only:
        return re.fullmatch(r"-?(0|[1-9]\d*)", text) is not None
    return re.fullmatch(r"-?(0|[1-9]\d*)(\.\d+)?", text) is not None


def is_nb_prefix(text: str, integer_only: bool) -> bool:
    if integer_only:
        return re.fullmatch(r"(|-|-?(0|[1-9]\d*))", text) is not None
    return re.fullmatch(r"(|-|-?(0|[1-9]\d*)(\.\d*)?)", text) is not None


def is_string_continuation(buffer: str,
                           token_text: str, max_len: int) -> bool:
    if '"' in token_text:
        return False
    if any(ord(char) < 0x20 for char in token_text):
        return False
    if "\\" in token_text:
        return False
    if len(buffer) + len(token_text) > max_len:
        return False
    return True


def is_string_terminator(token_text: str) -> bool:
    return token_text.lstrip().startswith('"')


def is_name_prefix(buffer: str, token_text: str, names: list[str]) -> bool:
    for name in names:
        if name.startswith(buffer + token_text):
            return True
    return False


def is_name_complete(buffer: str, names: list[str]) -> bool:
    for name in names:
        if buffer == name:
            return True
    return False


def has_longer_name(buffer: str, names: list[str]) -> bool:
    for name in names:
        if name.startswith(buffer) and len(name) > len(buffer):
            return True
    return False
