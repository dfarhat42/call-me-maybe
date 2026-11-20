*This project has been created as part of the 42 curriculum by dfarhat.*

# call me maybe

Function calling with a small LLM (Qwen3-0.6B) using constrained decoding.

## Description

Given a list of function definitions and a list of natural-language prompts, the program does not answer the prompts. It translates each one into a structured function call: the name of the function to call and its arguments, with the right types.

For example, "What is the sum of 40 and 2?" becomes:

```json
{
    "prompt": "What is the sum of 40 and 2?",
    "name": "fn_add_numbers",
    "parameters": {"a": 40.0, "b": 2.0}
}
```

A 0.6B model asked for JSON in a plain prompt is unreliable. Here the model is never trusted to produce structure. The program drives generation token by token and only lets the model choose among tokens that keep the output valid and schema-compliant. The output is valid JSON by construction.

Only `numpy`, `pydantic` and the provided `llm_sdk` are used. No `transformers`, `outlines`, `dspy` or similar library is imported by the project code.

## Instructions

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # install dependencies from uv.lock
uv run python -m src         # run with the default paths
```

Default paths:

| Argument | Default |
|---|---|
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calls.json` |

The output directory is created if it does not exist. The first run downloads the Qwen3-0.6B weights (about 1.5 GB) into the Hugging Face cache, so it is much slower than later runs.

Makefile targets:

| Target | Action |
|---|---|
| `make install` | `uv sync` |
| `make run` | run the program |
| `make debug` | run under `pdb` |
| `make clean` | remove `__pycache__` and `.mypy_cache` |
| `make lint` | `flake8 .` and `mypy .` with the flags required by the subject |
| `make lint-strict` | `flake8 .` and `mypy . --strict` |

### Example usage

```bash
# Defaults
uv run python -m src

# Custom paths
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json
```

On success the program prints a summary such as `Wrote 11 calls to data/output/function_calls.json (0 skipped)`. Errors (missing or invalid files, model loading failures) are printed to stderr as one clean message with exit code 1.

Example output file:

```json
[
    {
        "prompt": "Greet shrek",
        "name": "fn_greet",
        "parameters": {"name": "shrek"}
    },
    {
        "prompt": "What is the square root of 144",
        "name": "fn_get_square_root",
        "parameters": {"a": 144.0}
    }
]
```

## Algorithm explanation

### Overview

Generation is not one free-running loop. The expected output is fully known ahead of time, so it is split into segments, and the model is consulted only where a real decision exists.

```
{"name": "<NAME>", "parameters": {"<key>": <VALUE>, ... }}
 forced     model     forced        forced   model
```

| Segment | Who writes it | Constraint |
|---|---|---|
| `prompt` field | Python (copied from the input) | none, never generated |
| Structural text (braces, keys, colons, commas, quotes) | Python (forced) | appended to the context without a model call |
| Function name | the model | must stay a prefix of a real function name |
| Parameter value | the model | must follow the declared type (number, integer, string, boolean) |

The function to call is always chosen by the LLM, never by heuristics. Parameter keys come from the function definition, in definition order. The final JSON object is built in Python from the chosen name and the generated values, and written with `json.dump`.

### Token table

At startup every token ID is decoded once with `decode([id])` into a text table (`texts[id]`) and a boolean mask (`allowed[id]`). A token is excluded if it is special (flagged in `tokenizer.json`), if its ID is unused padding (the logit vector is longer than the vocabulary), if it decodes to an empty string, or if it decodes to a partial UTF-8 character (contains U+FFFD). The table is built once per run.

### Masking the logits

At each step the model returns one logit per token. Invalid tokens are masked out (set to `-inf` in effect) and the best remaining token is chosen greedily:

1. A state-independent mask removes most tokens up front (for example, for numbers only tokens made of digits, `-` and `.` remain).
2. The remaining candidates are walked in descending logit order, and the first one that passes the state-dependent check is selected (for example, "buffer + token is still a legal number prefix").

### Function name

The candidate set is the list of real function names. A token is valid if `buffer + token` is a prefix of at least one name. When the buffer equals a name and no longer name extends it, generation stops without calling the model. If the buffer is a complete name that is also a prefix of a longer one, the model's logit for the best continuation is compared against the best quote-starting token, and the higher one wins.

### Numbers

A number is validated with two regular expressions:

- **Prefix** (can still become a number): the empty string, `-`, or an integer part followed by an optional `.` and zero or more digits.
- **Complete**: `-?(0|[1-9]\d*)(\.\d+)?`, or without the decimal part for `integer` parameters.

This guarantees no leading zeros, no leading `.`, no `+`, no exponent, and that every sign and decimal point is followed by digits. A value can only end when it is complete. At each step the best continuation token is compared against the best terminator token (a token whose text starts with `,` or `}` after leading whitespace). If the terminator scores higher, or no continuation exists, the value ends. The terminator token is discarded and Python appends the forced separator. Numbers are converted with `float()` (or `int()` for `integer` parameters), so `2` is written as `2.0`.

### Strings

The model writes the raw text of the value, not its JSON-escaped form. Tokens containing a double quote or a control character are excluded. Backslashes are allowed, because Python's `json` module escapes them when the file is written (this is what makes regexes like `\d+` possible). The closing quote is the terminator: a token whose text starts with `"` ends the value, and is discarded. Every string, including the empty one, is complete, so the terminator is always offered. If a string is not closed within 200 characters, the prompt is reported as an error instead of writing truncated text.

### Booleans

A boolean is a choice between the two literals `true` and `false`, so it reuses the function-name selection with those two strings as the only valid names.

### Discarded terminators

When a number or string ends, the token that signalled the end is not appended to the context. Only tokens that were chosen and kept enter the context, and the closing quote or separator is appended as forced text. This avoids merged tokens such as `",` or `"}` breaking the structure.

### Validation of the result

Each generated call is checked against its function definition before it is kept: the name must match, the set of parameter keys must be identical, and each value must have the declared type (with `bool` explicitly excluded from `number` and `integer`). Results are modelled with pydantic.

## Design decisions

- **Segments instead of one JSON automaton.** A first version used a character-level JSON automaton that let the model write the whole object. It was replaced because most of the output is not a decision (the prompt copy, keys, punctuation), and forcing it saves forward passes and removes whole classes of errors.
- **Greedy decoding over masked logits.** The chosen token is always the highest-scoring valid one. This is deterministic and easy to reason about.
- **Text table from `decode`.** The mandatory path uses the SDK's public `decode`. No private members of `llm_sdk` are used.
- **First-character terminators.** Reading "the model wants to stop" from any token starting with the terminator character captures the model's intent better than requiring an exact token, because merged tokens such as `",` carry the same signal.
- **Python escapes the output.** Strings are generated raw and serialized with `json`, so there is no escape grammar to constrain and the file is always valid JSON.
- **`Parameter.type` is a free string.** An unknown type in the definitions file only fails the prompts that use that function (with a clear error), not the whole run.
- **Failed prompts are skipped, not guessed.** A prompt that raises an error is logged to stderr and omitted from the output, rather than writing a wrong entry. The output file is always written, even if empty (`[]`).
- **Prompt design.** The prompt lists the functions with their parameter types and descriptions, adds short rules, and includes two examples using functions that are explicitly marked as not available. The examples show the expected layout without hardcoding answers to the test prompts. The assistant turn starts with an empty think block to avoid reasoning output.

### Known limitations

- Double quotes cannot appear inside generated string values (the quote is the terminator).
- A string value cannot end with trailing whitespace (the value is stripped).
- Tokens that are only part of a multi-byte character are excluded, so some non-ASCII text may not be reproducible.
- Numbers are plain decimals: no exponents (`1e5`), no `+` sign, no leading zeros.
- Complex nested arguments (objects, arrays) are not supported.

## Performance analysis

Measured on the 11 provided example prompts with Qwen3-0.6B:

| Metric | Result |
|---|---|
| JSON validity | 100% by construction, output is built and dumped by Python |
| Function selection | 11/11 correct |
| Numbers and simple strings | all correct (`2.0`, `265.0`, `16.0`, `144.0`, `shrek`, `john`, `hello`, `world`) |
| Regex prompts | the numbers pattern (`[0-9]+`) and the cat/dog substitution are correct; the vowels prompt is weak (see below) |
| Speed | about 3 s of setup (model load plus token table, once the model is cached) and about 150 s for the 11 prompts, roughly 13 s per prompt |

The 5-minute limit is met for 11 prompts with a margin of about 2x. The main cost is that the SDK has no KV cache, so every generated token is a full forward pass over the whole context. Total time grows with (tokens generated) times (context length), which is why a longer prompt slows every step. Skipping forward passes where only one token is possible (for example the shared `fn_` name prefix) is the first optimization to apply if timing becomes tight.

Accuracy is limited by the prompt and the model, not by the constraints: the constraints guarantee a valid value, but not the right one. On the vowels prompt the model produced `aeiou` instead of a character class and `asterisk` instead of `*`.

## Challenges faced

- **Token strings are not text.** The raw vocabulary uses byte-level BPE symbols (`Ġ` for a space). The token table is built with `decode` so the validators see real text. Some tokens are only half of a multi-byte character and are excluded.
- **Merged tokens.** A single token can contain a quote and a newline (`"\n`). Terminators are therefore detected by their first character and discarded, instead of being appended.
- **A string value that never stops.** A regex repeated `|([0-9]+)` until the length cap. The fix was twofold: a prompt rule asking for one short pattern, and treating hitting the length cap as an error instead of returning truncated text.
- **Regexes and backslashes.** Forbidding backslashes made regexes impossible, so they are allowed in raw values and escaped by Python at write time.
- **Prompt length versus speed.** A longer prompt with more examples improved some answers but slowed every forward pass roughly threefold. The prompt was shortened and tested against the measured time.
- **Type checking.** `mypy --strict` cannot infer lambdas with default arguments, so predicates are written as small named inner functions. The SDK is configured to be excluded from strict checking.
- **Lockfile.** `uv.lock` must be committed, otherwise a fresh clone re-resolves and re-downloads every package.

## Testing strategy

- **Unit-level checks of the validators** with a table of expected results: numbers (`""`, `-`, `0`, `00`, `-0`, `-007`, `12`, `1.`, `1.5`, `.5`, `1.2.3`, `+1`, `1e5`, `1a`, in both float and integer mode), string continuations and terminators, and name prefixes (including a name that is a prefix of another).
- **Token table inspection:** printing the table entries for a space-prefixed word, digits, a quote, a newline and a special token, and checking the allowed count.
- **End-to-end runs** on the provided example prompts, comparing names, values and types by hand, including a function with an integer and a boolean parameter (`fn_is_even`).
- **Failure paths:** missing files, invalid JSON, a wrong top-level shape, a prompt entry without a `prompt` key, and an unwritable output path each produce one clean error line. A forced string length cap checks that a failing prompt is skipped without stopping the run.
- **Static checks:** `flake8` and `mypy` through `make lint`.

Edge cases worth re-running before a review: negative numbers, decimals, empty strings, strings with apostrophes, requests containing two quoted strings, functions with no parameters, and ambiguous prompts.

## Resources

- Qwen3 model card: https://huggingface.co/Qwen/Qwen3-0.6B
- JSON specification (number and string grammar): https://www.json.org
- Python `re` documentation: https://docs.python.org/3/library/re.html
- pydantic documentation: https://docs.pydantic.dev
- uv documentation: https://docs.astral.sh/uv/

### How AI was used

An AI assistant (Claude) was used as a design and review partner:

- Discussing the overall approach (segments instead of a single JSON automaton, the two-question number validator, terminator handling) and explaining concepts such as byte-level BPE.
- Reviewing code I wrote and pointing out bugs and type errors.
- Interpreting test output, mypy errors and uv cache logs.

All generated code was run and checked against test cases, and I can explain each part of the algorithm.
