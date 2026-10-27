#!/usr/bin/env bash
# One-shot setup of a RunPod GPU pod (docs/runpod.md): tools, code, models, data check, smoke run.
#
#   curl -fsSL <raw url of this file> | bash -s -- --repo <git url> [--ref <branch>] [--no-smoke]
#   # or, with the code already in /workspace/lifemate:
#   bash ops/scripts/runpod_setup.sh --no-clone
#
# Expects: a pod with a CUDA image (e.g. runpod/pytorch), HF_TOKEN in the pod env when models are gated,
# the competition data uploaded to /workspace/zac2026/train+public_test (runpodctl send / rclone), never public.
set -euo pipefail

REPO_URL=""
REF="main"
CLONE=1
SMOKE=1
WORKSPACE="${WORKSPACE:-/workspace}"
CODE_DIR="${CODE_DIR:-$WORKSPACE/lifemate}"
DATA_DIR="$WORKSPACE/zac2026/train+public_test"
LLM_ID="${LLM_ID:-Qwen/Qwen3-4B-Instruct-2507}"
LLM_REV="${LLM_REV:-cdbee75f17c01a7cc42f958dc650907174af0554}"
ASR_ID="${ASR_ID:-Systran/faster-whisper-large-v3}"
ASR_REV="${ASR_REV:-edaa852ec7e145841d8ffdb056a99866b5f0a478}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) REPO_URL="$2"; shift 2 ;;
        --ref) REF="$2"; shift 2 ;;
        --no-clone) CLONE=0; shift ;;
        --no-smoke) SMOKE=0; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

log() { echo "[runpod_setup] $*" >&2; }

log "GPU:"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || { log "no GPU visible"; exit 1; }

# 1. uv (Python + locked deps), git, kaggle/hf CLIs.
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v git >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git; }

# 2. Code.
if [[ $CLONE -eq 1 ]]; then
    [[ -n "$REPO_URL" ]] || { log "--repo <git url> is required unless --no-clone"; exit 2; }
    if [[ -d "$CODE_DIR/.git" ]]; then
        git -C "$CODE_DIR" fetch -q && git -C "$CODE_DIR" checkout -q "$REF" && git -C "$CODE_DIR" pull -q --ff-only
    else
        git clone -q --branch "$REF" "$REPO_URL" "$CODE_DIR"
    fi
fi
cd "$CODE_DIR"
log "code: $(git rev-parse --short HEAD 2>/dev/null || echo unknown) in $CODE_DIR"

# 3. Dependencies, including vLLM (locked). Train stack is installed only when asked (make train-deps).
UV_HTTP_TIMEOUT=600 uv sync --frozen --extra tokenizer --extra gpu
uv run python -c "from pathlib import Path; from lifemate.core.envinfo import detect; print(detect(Path('.')))"

# 4. Model snapshots into artifacts/models/ (git-ignored). Same revisions as configs/v0.toml; Docker copies this folder.
mkdir -p models
export HF_HOME="$WORKSPACE/hf_cache"
uv run --with "huggingface_hub[cli]" hf download "$LLM_ID" --revision "$LLM_REV" --local-dir artifacts/models/llm >/dev/null
uv run --with "huggingface_hub[cli]" hf download "$ASR_ID" --revision "$ASR_REV" --local-dir artifacts/models/asr >/dev/null
log "models: $(du -sh artifacts/models/llm | cut -f1) llm, $(du -sh artifacts/models/asr | cut -f1) asr"

# 5. Data check (never downloaded by this script).
if [[ ! -d "$DATA_DIR/train" ]]; then
    log "data missing: upload the package to $DATA_DIR (runpodctl send <zip> / rclone copy), then rerun with --no-clone --no-smoke"
    exit 0
fi

# Split id lists are git-ignored; rebuild them from the data (deterministic).
uv run python -m lifemate.eval.make_splits --root "$DATA_DIR" --smoke 30

# 6. Offline tests, then the 30-sample smoke with the real LLM (gold transcripts).
uv run pytest -q -m "not data and not tokenizer"
if [[ $SMOKE -eq 1 ]]; then
    uv run python -m lifemate.run.cli --config configs/v0.toml --env runpod --split train --ids configs/splits/smoke.txt --name runpod_smoke
    uv run python -m lifemate.run.report runs "$WORKSPACE/zac2026/runs/index.csv" --limit 5
fi
log "done. Next: make run ENV=runpod IDS=configs/splits/val_seen.txt ; docker: ops/docker/smoke.sh (needs docker in the pod)"
