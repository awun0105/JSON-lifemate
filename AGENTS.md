# For coding agents and new contributors

Read in this order before touching code:

1. `docs/overview.md`: the task as the organisers state it (scoring, limits, data layout).
2. `docs/SPEC.md`: the source of truth for implementation; it describes the current state only. Section 1.3
   says what is verified and what is not; section 11 lists open questions. Appendices A–F hold the pipeline,
   evaluation and splits, measurement infra, submission rules, model list with licenses and fine-tune plan
   (E.5), and the timeline. Code and SPEC must stay in step: when you change behaviour, update SPEC
   in the same commit.
3. `docs/decisions.md` (decision log: date, decision, reason, what it supersedes), `docs/experiments.md`
   (one row per run with scores), `docs/rationale.md` (first-principles reasoning behind the fine-tune plan).
4. `docs/CONTRIBUTING.md`: code conventions (ruff, mypy --strict, test docstring format, logging).
5. `docs/docker.md`, `docs/runpod.md` for the GPU/Docker side; `docs/eda.md` for data facts.

Hard rules (competition and safety):

- Never open or read any `*_environment/state.json` outside `lifemate/io/zaic.py`; tests enforce this.
- `data/train+public_test/` is private competition data: never copy it into git, docs, notebooks or logs.
- Never log or write `HF_TOKEN` or any key; it is read at run time only (`hf_token_source`).
- Do not run model inference, training or Docker builds on the dev machine without a GPU; those steps are
  for Kaggle, Colab or a rented pod (SPEC 4.16).
- One `ToolEnv` per sample; always `env.validate` before `env.call` (every call, even rejected, is scored).
- Split id lists in `configs/splits/` are git-ignored; run `make splits` after cloning (needs the data package).
- Data for SFT comes only from `configs/splits/train.txt` (plus synthetic data built from the train catalog);
  never from `val_*` or `public_test`.

Working style: `make check` and `make test` must pass before a commit; one commit per task with a message that
names the SPEC section; record any deviation from SPEC in SPEC itself.

Documentation log rules:

- Every run (agent, ASR, train), including failed ones, adds a row to `docs/experiments.md`.
- Every change of direction adds a row to `docs/decisions.md`, naming the decision it supersedes.
- `docs/SPEC.md` always describes the current state; history goes to the two logs, not into SPEC.
