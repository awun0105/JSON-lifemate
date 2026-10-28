# Phỏng theo Makefile@ai-kit (bố cục target); ở đây không có target server hay cơ sở dữ liệu.
#
# Nội dung: các lệnh tắt cho cài đặt, kiểm tra mã, kiểm thử, chạy agent, chấm điểm, đóng gói và Docker.
# Cách hoạt động: `make` không tham số chạy target help, in mọi target có chú thích `## ...`.
# Các biến bên dưới dùng `?=` nên có thể ghi đè trên dòng lệnh, ví dụ `make run ENV=kaggle`.
.DEFAULT_GOAL := help

# Tham số mặc định cho target run; NAME tự sinh từ thời gian và tên file IDS.
ENV      ?= local
CONFIG   ?= configs/v0.toml
SPLIT    ?= train
IDS      ?= configs/splits/smoke.txt
NAME     ?= $(shell date +%Y-%m-%d_%H%M)_$(notdir $(basename $(IDS)))

help:            ## Hiện trợ giúp này
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:         ## Cài thư viện (CPU) và pre-commit hook
	uv sync --extra tokenizer
	uv run pre-commit install

install-gpu:     ## Cài thư viện, kể cả vLLM
	uv sync --extra tokenizer --extra gpu

lint:            ## ruff check
	uv run ruff check .

format:          ## ruff format
	uv run ruff format .

typecheck:       ## mypy --strict trên lifemate/
	uv run mypy

check: lint typecheck ## lint + typecheck

test:            ## pytest (offline; bỏ qua test data/tokenizer khi thiếu)
	uv run pytest -q

splits:          ## Sinh lại configs/splits/ (train, val_seen, val_unseen, smoke, asr_tune, *_en_mixed)
	uv run python -m lifemate.eval.make_splits --smoke 30 --asr-tune 300 --en-mixed

run:             ## Chạy agent: make run ENV=local SPLIT=train IDS=configs/splits/smoke.txt
	uv run python -m lifemate.run.cli --config $(CONFIG) --env $(ENV) --split $(SPLIT) --ids $(IDS) --name $(NAME) $(EXTRA)

eval:            ## Baseline ASR trên val_seen + val_unseen: make eval ENV=colab TAG=base
	$(MAKE) run IDS=configs/splits/val_seen.txt NAME=$(or $(TAG),eval)_val_seen_asr EXTRA="--transcripts asr"
	$(MAKE) run IDS=configs/splits/val_unseen.txt NAME=$(or $(TAG),eval)_val_unseen_asr EXTRA="--transcripts asr"

report:          ## Bảng các lượt chạy agent: make report [INDEX=artifacts/runs/index.csv]
	uv run python -m lifemate.run.report runs $(or $(INDEX),artifacts/runs/index.csv)

report-train:    ## Bảng loss/eval của thư mục huấn luyện: make report-train WORK=artifacts/runs/train/<name>
	uv run python -m lifemate.run.report train $(WORK) --png $(WORK)/loss.png

train-deps:      ## Cài stack fine-tune (ms-swift, trl, peft, ...) trên máy GPU
	uv pip install -r ops/requirements/train.txt

score:           ## Chấm điểm file nộp: make score SUB=artifacts/runs/<name>/submission.jsonl IDS=configs/splits/val_seen.txt
	uv run python -m lifemate.eval.score $(SUB) --ids $(IDS)

pack:            ## Đóng gói mã cho Kaggle/Colab vào artifacts/dist/
	ops/scripts/pack_code.sh

.PHONY: help install install-gpu lint format typecheck check test splits run score pack report report-train train-deps

# Docker (SPEC 6.3). Build/chạy trên máy GPU có docker + nvidia-container-toolkit.
DATA_SPLIT_DIR ?= data/train+public_test/public_test
IMAGE          ?= lifemate:dev

docker-build:    ## Build image suy luận kèm trọng số từ ./models
	docker build -f ops/docker/Dockerfile --build-arg MODEL_DIR=artifacts/models --build-arg CODE_VERSION=$(shell git rev-parse HEAD) -t $(IMAGE) .

# chmod a+rwx: uid 10001 trong container phải ghi được vào result/.
docker-run:      ## Chạy theo cách của ban tổ chức: make docker-run DATA_SPLIT_DIR=<split dir>
	mkdir -p result && chmod a+rwx result
	docker run --rm --gpus all --network none -v $(abspath $(DATA_SPLIT_DIR)):/data:ro -v $(CURDIR)/result:/app/result $(IMAGE)

.PHONY: docker-build docker-run
