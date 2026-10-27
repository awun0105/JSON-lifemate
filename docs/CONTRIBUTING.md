# Code conventions

Read `docs/SPEC.md` for what to build; this file is how to write it. SPEC wins on any conflict.

## Tooling

- Python 3.11, `uv`. `make check` (ruff + `mypy --strict` on `lifemate/`) and `make test` must pass before a task is done.
- Tests run offline on CPU. Tests that need the data package use the `data` marker and skip when `data/train+public_test/` is absent; tests that need a HF tokenizer use the `tokenizer` marker and skip with a reason.
- `vllm`, `torch`, `transformers` are imported only inside the class that needs them (optional extras), with a comment naming the extra. Everything else imports at the top of the file.

## Style

- One module, one job (SPEC 4.0). Dependencies are passed through constructors; no globals, no singletons.
- Swappable parts are `typing.Protocol`s in `lifemate/core/types.py`; objects are wired only in `lifemate/run/factory.py`.
- Value objects are `@dataclass(frozen=True, slots=True)`. Enums for stop reasons, error codes, guard verdicts; serialise with `.value`.
- Every threshold (seconds, chars, steps, retries) lives in `Config`, never inline.
- JSON that reaches a prompt, a schema or an output file goes through `lifemate.core.jsonutil.canonical_json`.
- Files: `with open(...)` or `Path.read_text`. Never read `*_environment/` or `state.json` outside `lifemate/io/zaic.py`.
- Exceptions: only `Runner` catches `Exception` (per-sample boundary). Elsewhere catch the specific type.
- Logging: `structlog`, event names `lowercase_with_underscores`, values as kwargs, no f-strings in events.
- Public functions have type hints and a one-line docstring. Line length 120.
- No hard-coded paths (`/kaggle`, `/content`, `/home`, `train+public_test`) inside `lifemate/`; paths come from env profiles.
- Code copied from elsewhere keeps a provenance line: `# Adapted from <repo> <path>@<commit>`.

## Workflow

- One task per commit, message `<area>: <what>` (e.g. `agent: budget with predictive deadline`).
- When implementation deviates from SPEC for a good reason, change the code and update the matching SPEC section in the same commit.

## Package layout (SPEC 4.1)

`lifemate/` is split by role: `core` (types, config, paths; imports nothing else), `agent` (the loop; imports
`core` only), `backends` and `io` (models, files; import `core`), `run` (wiring, CLI, report; imports all),
`eval` and `train` (scripts and fine-tune; import `core`, `io`, `run`). No upward imports. Fixed inputs live in
`configs/` (prompts, splits, env profiles), operational glue in `ops/`, generated things in `artifacts/`,
competition data in `data/`.
