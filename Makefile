# Adapted from ai-kit Makefile@ai-kit (target layout); no server or database targets here.
.DEFAULT_GOAL := help

ENV      ?= local
CONFIG   ?= configs/v0.toml
SPLIT    ?= train
IDS      ?= configs/splits/smoke.txt
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

splits:          ## Regenerate configs/splits/ (train, val_seen, val_unseen, smoke)
	uv run python -m lifemate.eval.make_splits --smoke 30

run:             ## Run the agent: make run ENV=local SPLIT=train IDS=configs/splits/smoke.txt
	uv run python -m lifemate.run.cli --config $(CONFIG) --env $(ENV) --split $(SPLIT) --ids $(IDS) --name $(NAME)

report:          ## Table of agent runs: make report [INDEX=artifacts/runs/index.csv]
	uv run python -m lifemate.run.report runs $(or $(INDEX),artifacts/runs/index.csv)

report-train:    ## Loss/eval table of a training work dir: make report-train WORK=artifacts/runs/train/<name>
	uv run python -m lifemate.run.report train $(WORK) --png $(WORK)/loss.png

train-deps:      ## Install the fine-tune stack (unsloth, trl, peft, ...) on a GPU box
	uv pip install -r ops/requirements/train.txt

score:           ## Score a submission: make score SUB=artifacts/runs/<name>/submission.jsonl IDS=configs/splits/val_seen.txt
	uv run python -m lifemate.eval.score $(SUB) --ids $(IDS)

pack:            ## Pack code for Kaggle/Colab into artifacts/dist/
	ops/scripts/pack_code.sh

.PHONY: help install install-gpu lint format typecheck check test splits run score pack report report-train train-deps

# Docker (SPEC 6.3). Build/run on a GPU box with docker + nvidia-container-toolkit.
DATA_SPLIT_DIR ?= data/train+public_test/public_test
IMAGE          ?= lifemate:dev

docker-build:    ## Build the inference image with weights from ./models
	docker build -f ops/docker/Dockerfile --build-arg MODEL_DIR=artifacts/models --build-arg CODE_VERSION=$(shell git rev-parse HEAD) -t $(IMAGE) .

docker-run:      ## Organisers-style run: make docker-run DATA_SPLIT_DIR=<split dir>
	mkdir -p result && chmod a+rwx result
	docker run --rm --gpus all --network none -v $(abspath $(DATA_SPLIT_DIR)):/data:ro -v $(CURDIR)/result:/app/result $(IMAGE)

.PHONY: docker-build docker-run
