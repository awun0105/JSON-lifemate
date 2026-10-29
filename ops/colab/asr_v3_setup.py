"""Chuẩn bị máy Colab cho benchmark ASR lần 3 (chạy bằng `colab exec -s <phiên> -f ops/colab/asr_v3_setup.py`).

Cách hoạt động:
1. Clone mã ở nhánh `REF` vào `/content/lifemate` (xoá bản cũ nếu có).
2. Giải nén dữ liệu train đã tải lên (`/content/train.tar`) vào `/content/data/train+public_test/`.
3. Cài thư viện: `ops/requirements/notebook.txt`, faster-whisper, chunkformer (giữ torch có sẵn của Colab).
4. Ghi profile `configs/env/colabcli.toml` trỏ tới dữ liệu và thư mục chạy trên máy, dựng lại các split.

Lưu ý:
- Dữ liệu thi chỉ nằm trên máy Colab của chính tài khoản chạy, không lên git hay chỗ công khai.
- Các mô hình ASR đều công khai, không cần token Hugging Face.
"""

import subprocess
from pathlib import Path

REF = "asr-benchmark-v2"
REPO = "https://github.com/awun0105/JSON-lifemate.git"
CODE = Path("/content/lifemate")
DATA = Path("/content/data")


def sh(cmd: str, cwd: Path | None = None) -> None:
    """Chạy một lệnh shell, in lệnh và dừng khi lỗi."""
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd, executable="/bin/bash")


sh(f"rm -rf {CODE} && git clone --depth 1 --branch {REF} {REPO} {CODE}")
sh("git log --oneline -1", cwd=CODE)
if not (DATA / "train+public_test" / "train").is_dir():
    DATA.mkdir(parents=True, exist_ok=True)
    # `colab upload` chỉ nhận file nhỏ, nên dữ liệu được tải lên thành các phần `/content/p*` (tạo bằng `split`).
    if not Path("/content/train.tar").is_file():
        sh("cat /content/p* > /content/train.tar && rm -f /content/p*")
    sh(f"tar -xf /content/train.tar -C {DATA}")
sh("pip install -q -r ops/requirements/notebook.txt faster-whisper==1.2.1 chunkformer==1.2.2", cwd=CODE)
# CTranslate2 (lõi của faster-whisper) cần cuBLAS và cuDNN của CUDA 12; máy Colab chỉ có CUDA 13 đi kèm torch.
sh('pip install -q nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"', cwd=CODE)
(CODE / "configs/env/colabcli.toml").write_text(
    f'data_root = "{DATA}/train+public_test"\n'
    'runs_dir = "/content/runs"\n'
    'hf_cache = "/content/hf_cache"\n'
    'hf_token_source = "env:HF_TOKEN"\n'
    'persist = "local"\n',
    encoding="utf-8",
)
sh(
    f"python -m lifemate.eval.make_splits --root {DATA}/train+public_test --smoke 30 --asr-tune 300 --en-mixed",
    cwd=CODE,
)
sh(
    'python -c "import torch, faster_whisper, chunkformer; print(torch.__version__, torch.cuda.is_available())"',
    cwd=CODE,
)
print("SETUP OK", flush=True)
