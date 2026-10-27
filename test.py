"""Phase 2 exploration: tokenizer and vocabulary layer.

Run from the repo root with:   uv run python test_phase2.py
Quick mode (oracle check on a random sample only):
                               uv run python test_phase2.py --sample

Only PUBLIC methods of Small_LLM_Model are used and torch is never
imported. The byte<->unicode mapping below is a REFERENCE used to verify
your understanding: re-derive it yourself before writing your own module.
"""

import json
import random
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

try:
    from llm_sdk import llm_sdk
    ModelClass = llm_sdk.Small_LLM_Model
except (ImportError, AttributeError):
    from llm_sdk import Small_LLM_Model as ModelClass  # type: ignore

SAMPLE_MODE = "--sample" in sys.argv
NOTES: Dict[str, Any] = {}
S: Dict[str, Any] = {}  # shared state between sections


def banner(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_section(name: str, func: Callable[[], None]) -> None:
    """Run one section, catching errors so the rest still runs."""
    banner(name)
    try:
        func()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAILED] {type(exc).__name__}: {exc}")
        traceback.print_exc()


def to_int_list(tensor: Any) -> List[int]:
    """Convert encode()'s tensor to a flat list of ints (no torch)."""
    data = tensor.tolist() if hasattr(tensor, "tolist") else tensor
    while data and isinstance(data[0], list):
        data = data[0]
    return [int(x) for x in data]


def bytes_to_unicode() -> Dict[int, str]:
    """Reference GPT-2 byte -> printable unicode mapping."""
    keep = (list(range(ord("!"), ord("~") + 1))
            + list(range(0xA1, 0xAC + 1))
            + list(range(0xAE, 0xFF + 1)))
    mapping: Dict[int, str] = {}
    shift = 0
    for byte in range(256):
        if byte in keep:
            mapping[byte] = chr(byte)
        else:
            mapping[byte] = chr(256 + shift)
            shift += 1
    return mapping


# --------------------------------------------------------------------------
def t2_0_setup() -> None:
    """Load the model, the vocab file and the logits length."""
    model = ModelClass()
    S["model"] = model
    path = model.get_path_to_vocab_file()
    with open(path, "r", encoding="utf-8") as handle:
        vocab: Dict[str, int] = json.load(handle)
    S["vocab"] = vocab
    probe = model.get_logits_from_input_ids(
        to_int_list(model.encode("hello")))
    S["n"] = len(probe)
    print(f"Vocab entries: {len(vocab)}   logits length: {S['n']}")
    print(f"Vocab path: {path}")


def t2_1_inspect_vocab() -> None:
    """T2.1: compare vocab form with decoded form."""
    model, vocab = S["model"], S["vocab"]
    by_id = {i: t for t, i in vocab.items()}
    print(f"{'id':>7}  {'vocab form':<14} {'decode()':<14}")
    for i in (220, 374, 17, 3838, 6388, 13, 198, 1, 330):
        print(f"{i:>7}  {by_id.get(i, '?')!r:<14} "
              f"{model.decode([i])!r:<14}")
    shifted = [t for t in vocab if any(0x100 <= ord(c) <= 0x143
                                       for c in t)]
    print(f"\nTokens containing chars in U+0100..U+0143: {len(shifted)}")
    for mark, name in (("\u0120", "G-dot (space)"),
                       ("\u010a", "C-dot (newline)"),
                       ("\u0109", "c-dot (tab)")):
        count = sum(1 for t in vocab if mark in t)
        print(f"  tokens containing {name}: {count}")
    NOTES["tokens_with_shifted_chars"] = len(shifted)


def t2_2_reverse_table() -> None:
    """T2.2: ID -> vocab string table sized to the logits length."""
    vocab, n = S["vocab"], S["n"]
    table: List[Optional[str]] = [None] * n
    for tok, idx in vocab.items():
        if idx >= n:
            raise ValueError(f"id {idx} outside logits range {n}")
        table[idx] = tok
    S["vocab_form"] = table
    missing = [i for i, t in enumerate(table) if t is None]
    print(f"Table length: {len(table)}")
    print(f"Indices without a vocab string: {len(missing)} "
          f"(first {missing[0]}, last {missing[-1]})")
    NOTES["no_string_count"] = len(missing)
    NOTES["no_string_range"] = [missing[0], missing[-1]]


def t2_3_byte_mapping() -> None:
    """T2.3: verify the byte<->unicode mapping."""
    fwd = bytes_to_unicode()
    inv = {c: b for b, c in fwd.items()}
    S["fwd"], S["inv"] = fwd, inv
    print(f"Mapping size: {len(fwd)}, distinct chars: "
          f"{len(set(fwd.values()))}, inverse size: {len(inv)}")
    print(f"Bijection: {len(inv) == 256}")
    identity = sum(1 for b, c in fwd.items() if ord(c) == b)
    print(f"Bytes mapping to themselves: {identity}")
    print(f"Shifted bytes: {256 - identity}")
    for byte, name in ((0x20, "space"), (0x0A, "newline"),
                       (0x09, "tab"), (0x00, "NUL")):
        print(f"  byte {byte:#04x} ({name}) -> {fwd[byte]!r} "
              f"U+{ord(fwd[byte]):04X}")
    NOTES["identity_bytes"] = identity


def t2_4_real_text() -> None:
    """T2.4: convert every token to bytes/real text, check vs decode()."""
    model, inv, table = S["model"], S["inv"], S["vocab_form"]
    n = S["n"]
    raw: List[Optional[bytes]] = [None] * n
    text: List[Optional[str]] = [None] * n
    undecodable: Set[int] = set()
    unmapped: Set[int] = set()
    for i, tok in enumerate(table):
        if tok is None:
            continue
        try:
            raw_bytes = bytes(inv[c] for c in tok)
        except KeyError:
            unmapped.add(i)
            continue
        raw[i] = raw_bytes
        try:
            text[i] = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            undecodable.add(i)
    S.update(raw=raw, text=text, undecodable=undecodable)
    print(f"Converted to text:           "
          f"{sum(1 for t in text if t is not None)}")
    print(f"Chars not in inverse mapping: {len(unmapped)}")
    print(f"Partial / invalid UTF-8:      {len(undecodable)}")
    sample = sorted(undecodable)[:5]
    print(f"  sample undecodable ids: {sample}")
    print(f"  their bytes: {[raw[i] for i in sample]}")

    ids = [i for i in range(n) if table[i] is not None]
    if SAMPLE_MODE:
        random.seed(0)
        ids = random.sample(ids, 5000)
        print("(--sample mode: checking 5000 random ids)")
    start = time.perf_counter()
    bad_text, bad_undec = [], []
    for i in ids:
        oracle = model.decode([i])
        if i in undecodable:
            if "\ufffd" not in oracle:
                bad_undec.append(i)
        elif oracle != text[i]:
            bad_text.append(i)
    print(f"Oracle check over {len(ids)} ids "
          f"({time.perf_counter() - start:.1f}s)")
    print(f"  mismatches on decodable tokens:   {len(bad_text)}")
    print(f"  undecodable but oracle had no U+FFFD: {len(bad_undec)}")
    for i in bad_text[:5]:
        print(f"    id {i}: mine={text[i]!r} oracle={model.decode([i])!r}")
    NOTES["undecodable_count"] = len(undecodable)
    NOTES["oracle_mismatches"] = len(bad_text)


def t2_5_special_tokens() -> None:
    """T2.5: inspect tokenizer file, build the forbidden set."""
    model, n = S["model"], S["n"]
    path = model.get_path_to_tokenizer_file()
    print(f"Tokenizer file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        tok = json.load(handle)
    print(f"Top-level keys: {list(tok.keys())}")
    added = tok.get("added_tokens", [])
    print(f"added_tokens: {len(added)}")
    for entry in added[:12]:
        print(f"  id={entry.get('id')} special={entry.get('special')} "
              f"content={entry.get('content')!r}")
    if len(added) > 12:
        print(f"  ... {len(added) - 12} more")
    ids = [e["id"] for e in added]
    if ids:
        print(f"Added id range: {min(ids)}..{max(ids)}")
    for name in ("<|endoftext|>", "<|im_end|>", "<|im_start|>"):
        match = [e["id"] for e in added if e.get("content") == name]
        print(f"  {name}: {match}")
    merges_path = model.get_path_to_merges_file()
    with open(merges_path, "r", encoding="utf-8") as handle:
        merges = handle.readlines()
    print(f"Merges file: {len(merges)} lines (bonus: tokenizer recode)")

    no_string = {i for i, t in enumerate(S["vocab_form"]) if t is None}
    forbidden = no_string | set(ids) | S["undecodable"]
    usable = set(range(n)) - forbidden
    S["forbidden"], S["usable"] = forbidden, usable
    print(f"\nForbidden: {len(forbidden)} "
          f"(no string {len(no_string)}, added {len(set(ids))}, "
          f"undecodable {len(S['undecodable'])})")
    print(f"Usable:    {len(usable)}")
    print(f"Partition covers logits range: "
          f"{len(forbidden) + len(usable) == n and not forbidden & usable}")
    NOTES["added_token_count"] = len(ids)
    NOTES["usable_count"] = len(usable)
    NOTES["eos_candidates"] = {
        e["content"]: e["id"] for e in added
        if e.get("content") in ("<|endoftext|>", "<|im_end|>")}


def t2_6_classification() -> None:
    """T2.6: precompute classification arrays."""
    text, n = S["text"], S["n"]
    digits = np.zeros(n, dtype=bool)
    numchars = np.zeros(n, dtype=bool)
    has_quote = np.zeros(n, dtype=bool)
    has_bslash = np.zeros(n, dtype=bool)
    has_ctrl = np.zeros(n, dtype=bool)
    allowed_num = set("0123456789.-+eE")
    for i in S["usable"]:
        s = text[i]
        if not s:
            continue
        digits[i] = s.isascii() and s.isdigit()
        numchars[i] = all(c in allowed_num for c in s)
        has_quote[i] = '"' in s
        has_bslash[i] = "\\" in s
        has_ctrl[i] = any(ord(c) < 0x20 for c in s)
    S["cls"] = dict(digits=digits, numchars=numchars, quote=has_quote,
                    bslash=has_bslash, ctrl=has_ctrl)
    for name, arr in S["cls"].items():
        print(f"{name:>8}: {int(arr.sum()):>7} tokens")
    multi = [text[i] for i in np.flatnonzero(digits) if len(text[i]) > 1]
    print(f"\nMulti-digit tokens: {len(multi)} "
          f"(Qwen splits digits one by one -> expect 0)")
    mixed = [text[i] for i in S["usable"]
             if text[i] and text[i][0].isdigit() and not digits[i]][:10]
    print(f"Tokens starting with a digit but not all digits: {mixed}")
    straddle = [text[i] for i in np.flatnonzero(has_quote)][:12]
    print(f"Tokens containing a quote (string boundary risk): {straddle}")
    NOTES["digit_tokens"] = int(digits.sum())
    NOTES["multi_digit_tokens"] = len(multi)


def t2_7_scaffold() -> None:
    """T2.7: scaffold text -> token IDs."""
    model = S["model"]
    for scaffold in ('{"name": "', '", "parameters": {"', '": ',
                     ', "', '}}'):
        ids = to_int_list(model.encode(scaffold))
        pieces = [S["text"][i] for i in ids]
        print(f"{scaffold!r:24} -> {ids}")
        print(f"{'':24}    {pieces}")
        print(f"{'':24}    decodes back: {model.decode(ids) == scaffold}")
    a = to_int_list(model.encode('{"name": "'))
    b = to_int_list(model.encode("fn_add_numbers"))
    whole = to_int_list(model.encode('{"name": "fn_add_numbers'))
    print(f"\nencode(a)+encode(b) == encode(a+b): {a + b == whole}")
    print(f"  concatenated: {a + b}")
    print(f"  whole:        {whole}")


def t2_8_roundtrip() -> None:
    """T2.8: round-trip tests with my tables."""
    model, raw, text = S["model"], S["raw"], S["text"]
    cases = ["What is the sum of 2 and 3?", " leading space",
             "line1\nline2", "tab\there", 'quote " and \\ slash',
             "caf\u00e9", "\u4f60\u597d", "emoji \U0001F600 end",
             "\U0001F9D1\u200d\U0001F4BB", "", "   ", '{"a": 40.0}']
    all_ok = True
    for case in cases:
        ids = to_int_list(model.encode(case))
        if any(i in S["forbidden"] and i not in S["undecodable"]
               for i in ids):
            print(f"{case!r}: uses a forbidden non-partial id!")
            all_ok = False
            continue
        joined = b"".join(raw[i] for i in ids)  # type: ignore[misc]
        ok_bytes = joined.decode("utf-8") == case
        partial = [i for i in ids if i in S["undecodable"]]
        per_token = None
        if not partial:
            per_token = "".join(text[i] for i in ids) == case  # type: ignore
        print(f"{case!r:32} tokens={len(ids):2d} "
              f"bytes-roundtrip={ok_bytes} "
              f"per-token-text={per_token} partial={len(partial)}")
        all_ok = all_ok and ok_bytes
    print(f"\nAll byte-level round trips OK: {all_ok}")
    print("Note: where partial>0, token-by-token text decoding is "
          "impossible; only the joined bytes recover the string.")
    NOTES["roundtrip_ok"] = all_ok


def t2_9_notes() -> None:
    """Dump findings."""
    print(json.dumps(NOTES, indent=2, default=str))
    with open("phase2_notes.json", "w", encoding="utf-8") as handle:
        json.dump(NOTES, handle, indent=2, default=str)
    print("\nSaved to phase2_notes.json (add to .gitignore)")


def main() -> int:
    """Run all sections."""
    run_section("T2.0 Setup", t2_0_setup)
    if "vocab" not in S:
        return 1
    run_section("T2.1 Vocab form vs decoded form", t2_1_inspect_vocab)
    run_section("T2.2 ID -> vocab string table", t2_2_reverse_table)
    run_section("T2.3 Byte <-> unicode mapping", t2_3_byte_mapping)
    run_section("T2.4 Real text + oracle check", t2_4_real_text)
    run_section("T2.5 Special tokens + forbidden set", t2_5_special_tokens)
    run_section("T2.6 Token classification", t2_6_classification)
    run_section("T2.7 Scaffold tokenization", t2_7_scaffold)
    run_section("T2.8 Round trips", t2_8_roundtrip)
    run_section("T2.9 Notes", t2_9_notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
