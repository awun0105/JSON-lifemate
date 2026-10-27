#!/usr/bin/env bash
# Cài đặt một lần cho pod GPU RunPod (docs/runpod.md): công cụ, mã nguồn, mô hình, kiểm tra dữ liệu, chạy smoke.
#
#   curl -fsSL <raw url của file này> | bash -s -- --repo <git url> [--ref <branch>] [--no-smoke]
#   # hoặc, khi mã nguồn đã nằm sẵn ở /workspace/lifemate:
#   bash ops/scripts/runpod_setup.sh --no-clone
#
# Yêu cầu: pod dùng image CUDA (ví dụ runpod/pytorch), HF_TOKEN trong môi trường pod khi mô hình bị giới hạn
# truy cập, dữ liệu cuộc thi đã tải lên /workspace/zac2026/train+public_test (runpodctl send / rclone),
# tuyệt đối không công khai.
#
# Cách hoạt động:
#   0. Đọc tham số dòng lệnh, kiểm tra có GPU.
#   1. Cài uv và git nếu thiếu.
#   2. Clone hoặc cập nhật mã nguồn (bỏ qua khi --no-clone).
#   3. Cài thư viện đã khoá, kể cả vLLM.
#   4. Tải snapshot mô hình LLM và ASR đúng revision vào artifacts/models/.
#   5. Kiểm tra dữ liệu; thiếu thì dừng sớm (script không bao giờ tải dữ liệu).
#   6. Dựng lại split, chạy kiểm thử offline, rồi chạy smoke 30 mẫu với LLM thật.
set -euo pipefail

# Giá trị mặc định; các biến dạng ${X:-...} có thể ghi đè qua môi trường.
REPO_URL=""
REF="main"
CLONE=1
SMOKE=1
WORKSPACE="${WORKSPACE:-/workspace}"
CODE_DIR="${CODE_DIR:-$WORKSPACE/lifemate}"
DATA_DIR="$WORKSPACE/zac2026/train+public_test"
# Mô hình và revision ghim cố định, trùng với configs/v0.toml.
LLM_ID="${LLM_ID:-Qwen/Qwen3-4B-Instruct-2507}"
LLM_REV="${LLM_REV:-cdbee75f17c01a7cc42f958dc650907174af0554}"
ASR_ID="${ASR_ID:-Systran/faster-whisper-large-v3}"
ASR_REV="${ASR_REV:-edaa852ec7e145841d8ffdb056a99866b5f0a478}"

# Đọc tham số dòng lệnh; tham số lạ thì báo lỗi và thoát mã 2.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) REPO_URL="$2"; shift 2 ;;
        --ref) REF="$2"; shift 2 ;;
        --no-clone) CLONE=0; shift ;;
        --no-smoke) SMOKE=0; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Ghi một dòng log có tiền tố [runpod_setup] ra stderr.
log() { echo "[runpod_setup] $*" >&2; }

# 0. Không thấy GPU thì dừng ngay.
log "GPU:"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || { log "no GPU visible"; exit 1; }

# 1. uv (Python + thư viện đã khoá), git, CLI kaggle/hf.
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Trình cài uv đặt file vào ~/.local/bin; thêm vào PATH cho phần còn lại của script.
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v git >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git; }

# 2. Mã nguồn: đã có repo thì fetch/checkout/pull fast-forward, chưa có thì clone.
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

# 3. Thư viện, kể cả vLLM (đã khoá). Stack huấn luyện chỉ cài khi được yêu cầu (make train-deps).
# UV_HTTP_TIMEOUT lớn vì các wheel GPU rất nặng, tải chậm dễ quá thời gian mặc định.
UV_HTTP_TIMEOUT=600 uv sync --frozen --extra tokenizer --extra gpu
# In thông tin môi trường mà LifeMate tự nhận diện.
uv run python -c "from pathlib import Path; from lifemate.core.envinfo import detect; print(detect(Path('.')))"

# 4. Snapshot mô hình vào artifacts/models/ (git bỏ qua). Cùng revision với configs/v0.toml; Docker sao chép thư mục này.
# HF_HOME đặt trong /workspace (ổ bền của pod) để cache không mất khi khởi động lại.
mkdir -p models
export HF_HOME="$WORKSPACE/hf_cache"
uv run --with "huggingface_hub[cli]" hf download "$LLM_ID" --revision "$LLM_REV" --local-dir artifacts/models/llm >/dev/null
uv run --with "huggingface_hub[cli]" hf download "$ASR_ID" --revision "$ASR_REV" --local-dir artifacts/models/asr >/dev/null
log "models: $(du -sh artifacts/models/llm | cut -f1) llm, $(du -sh artifacts/models/asr | cut -f1) asr"

# 5. Kiểm tra dữ liệu (script này không bao giờ tải dữ liệu). Thiếu dữ liệu thì thoát 0 kèm hướng dẫn.
if [[ ! -d "$DATA_DIR/train" ]]; then
    log "data missing: upload the package to $DATA_DIR (runpodctl send <zip> / rclone copy), then rerun with --no-clone --no-smoke"
    exit 0
fi

# Danh sách id của các split bị git bỏ qua; dựng lại từ dữ liệu (tất định).
uv run python -m lifemate.eval.make_splits --root "$DATA_DIR" --smoke 30

# 6. Kiểm thử offline, sau đó smoke 30 mẫu với LLM thật (transcript gold).
uv run pytest -q -m "not data and not tokenizer"
if [[ $SMOKE -eq 1 ]]; then
    uv run python -m lifemate.run.cli --config configs/v0.toml --env runpod --split train --ids configs/splits/smoke.txt --name runpod_smoke
    uv run python -m lifemate.run.report runs "$WORKSPACE/zac2026/runs/index.csv" --limit 5
fi
log "done. Next: make run ENV=runpod IDS=configs/splits/val_seen.txt ; docker: ops/docker/smoke.sh (needs docker in the pod)"
