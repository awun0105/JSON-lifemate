# Adapted from ai-kit Makefile@ai-kit (target layout); no server or database targets here.
.DEFAULT_GOAL := help

ENV      ?= local
CONFIG   ?= configs/v0.toml
SPLIT    ?= train
IDS      ?= splits/smoke.txt
NAME     ?= $(shell date +%Y-%m-%d_%H%M)_$(notdir $(basename $(IDS)))

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:         ## Install deps (CPU) and pre-commit hooks
	uv sync --extra tokenizer
	uv run pre-commit install

install-gpu:     ## Install deps including vLLM
	uv sync --extra tokenizer --extra gpu

lint:            ## ruff check
	uv run ruff check .

format:          ## ruff format
	uv run ruff format .

typecheck:       ## mypy --strict on lifemate/
	uv run mypy

check: lint typecheck ## lint + typecheck

test:            ## pytest (offline; data/tokenizer tests skip when absent)
	uv run pytest -q

splits:          ## Regenerate splits/ (train, val_seen, val_unseen, smoke)
	uv run python eval/make_splits.py --smoke 30

run:             ## Run the agent: make run ENV=local SPLIT=train IDS=splits/smoke.txt
	uv run python -m lifemate.cli --config $(CONFIG) --env $(ENV) --split $(SPLIT) --ids $(IDS) --name $(NAME)

score:           ## Score a submission: make score SUB=runs/<name>/submission.jsonl IDS=splits/val_seen.txt
	uv run python eval/score.py $(SUB) --ids $(IDS)

pack:            ## Pack code for Kaggle/Colab into dist/
	scripts/pack_code.sh

.PHONY: help install install-gpu lint format typecheck check test splits run score pack
