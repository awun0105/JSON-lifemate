# LifeMate agent (Zalo AI Challenge 2026)

Agent that turns a Vietnamese spoken request into tool calls on the organisers' `zaic_tools` simulator.

- Design: `docs/SPEC.md` (source of truth for implementation)
- Rationale and experiment log: `docs/approach.md`
- Task statement: `docs/overview.md`; data notes: `docs/eda.md`
- Code conventions for contributors and coding agents: `docs/CONTRIBUTING.md`

## Quick start

```bash
make install          # uv sync + pre-commit (CPU; vLLM is an extra: make install-gpu)
make check            # ruff + mypy --strict
make test             # pytest, offline; data/tokenizer tests skip when absent
make run ENV=local SPLIT=train IDS=splits/smoke.txt
```

Competition data lives in `train+public_test/` (git-ignored). Runs are written under `runs/`.

## Running on Kaggle / Colab

1. Pack the code: `make pack` (or `scripts/pack_code.sh`). It prints `dist/lifemate-code-<sha>[-dirty].zip` and its sha256; the zip carries `VERSION` (commit hash) and `VERSION_STATUS` (clean/dirty).
2. Kaggle: upload the zip as a private Kaggle Dataset (e.g. `lifemate-code`), attach it and the competition data to a notebook built from `notebooks/kaggle_run.ipynb`, add the `HF_TOKEN` secret, and run all cells. The last cell saves `runs/` as a new version of a private dataset (create it once, see the cell comment).
3. Colab: copy the zip to `MyDrive/zac2026/` (or git-clone), open `notebooks/colab_run.ipynb`, set `HF_TOKEN` in Colab secrets and run all; runs go straight to Drive.
4. Fine-tuning: `notebooks/{kaggle,colab}_train_{llm,asr}.ipynb` install `requirements/train.txt` on top and run `lifemate.train.llm_pipeline` / `asr_pipeline` in one cell; a second session resumes from `pipeline_progress.json` (Kaggle: attach the `lifemate-runs` dataset saved by the last cell).
5. Notebooks only run commands (install, EnvInfo, quick tests, smoke, val_seen, val_unseen). Dependencies come from `requirements/notebook.txt`; regenerate it with `uv run python scripts/export_notebook_requirements.py` after changing `uv.lock`.
