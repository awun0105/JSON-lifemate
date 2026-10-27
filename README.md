# Voice-to-action agent

An offline agent that turns a spoken Vietnamese request into a sequence of tool calls on a simulated set of apps,
scored by the resulting app state. Small models only: one LLM under 5B parameters plus an ASR model, running
on a single GPU within a fixed time budget per request.

- Design and implementation spec: `docs/SPEC.md` (section 4.1 is the repo map)
- Rationale, decisions and experiment log: `docs/approach.md`
- Conventions for contributors and coding agents: `AGENTS.md`, `docs/CONTRIBUTING.md`
- GPU side: `docs/docker.md`, `docs/runpod.md`

## Quick start

```bash
make install          # uv sync + pre-commit (CPU; vLLM is an extra: make install-gpu)
make check            # ruff + mypy --strict
make test             # pytest, offline; data/tokenizer tests skip when absent
make run ENV=local SPLIT=train IDS=configs/splits/smoke.txt
```

Input data lives under `data/` and run outputs under `artifacts/`; both are git-ignored. `make report` lists
runs, `make report-train WORK=artifacts/runs/train/<name>` shows training curves.

## Notebooks and rented GPUs

`make pack` builds a code zip for `ops/notebooks/*_run.ipynb` (inference) and `*_train_{llm,asr}.ipynb`
(fine-tuning) on Kaggle or Colab. A rented pod is set up with one command via `ops/scripts/runpod_setup.sh`.
