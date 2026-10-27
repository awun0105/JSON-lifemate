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
