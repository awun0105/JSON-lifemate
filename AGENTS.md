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

- Labels derived from the data (hand-checked word lists, id lists) may be kept as positions only
  (sample id, turn, word index, label), never as copied text, and stay git-ignored like `configs/splits/`.
  "Can't go on git" is not by itself a reason to skip hand labelling.

Naming (ruff passing is not enough; PEP 8 fixes the case style, not whether a name is understandable):

- A name says what the thing does or holds, readable without the docstring: `find_foreign_words`,
  `match_words_in_transcript`, `ENGLISH_SPELLED_LIKE_VIETNAMESE`, not `hits`, `spelled_in`, `EN_LIKE_VI`.
- No one- or two-letter names or private abbreviations (`pk`, `sk`, `gk`, `rep`, `cands`, `per`, `boot`) outside
  loop indices (`i`, `j`) and very short comprehensions. Module names name their job (`foreign_words.py`, not
  `en_mixed.py`). JSON keys follow the same rule (`wer_errors`, not `wer_err`).
- Constants and data structures get a comment saying their shape and content (e.g. "word -> other spellings").
- After a rename, re-run the affected pipeline on stored results and check the output is unchanged.

Reports (`docs/reports/`), written for a reader who does not read code:

- Every section goes problem → idea → solution → algorithm → reason → limits; each experiment then says which
  of the prepared tools it uses. Never open a section with the method before stating the problem it solves,
  what it runs on and where its output is used.
- Explain every term on first use or in the glossary. Whenever a file, function, constant or config key is
  named, say where it is (`path › name`), what it does (input → output) and, for a variable, its data
  structure and what it stores. Keep a code lookup appendix in sync with renames.
- Reasons must be true and checked; when a stated reason turns out wrong, correct it in the report
  ("Đính chính") instead of silently replacing it.

Working style: `make check` and `make test` must pass before a commit; one commit per task with a message that
names the SPEC section; record any deviation from SPEC in SPEC itself.

Documentation log rules:

- Every run (agent, ASR, train), including failed ones, adds a row to `docs/experiments.md`.
- Every change of direction adds a row to `docs/decisions.md`, naming the decision it supersedes.
- `docs/SPEC.md` always describes the current state; history goes to the two logs, not into SPEC.
