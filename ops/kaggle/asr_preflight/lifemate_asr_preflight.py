"""Kernel Kaggle chạy thử trên CPU trước lần chọn config ASR (kernel riêng tư `quanlam/lifemate-asr-preflight`).

Cách dùng:
    kaggle kernels push -p ops/kaggle/asr_preflight   (không cần --accelerator)

Cách hoạt động:
1. Cài đúng các thư viện như kernel `asr_tune` (vLLM, faster-whisper, chunkformer, NeMo).
2. Chạy mọi lựa chọn engine hỗ trợ của từng model trên 3 mẫu (`asr_tune --preflight 3`).

Lưu ý:
- Bắt được lỗi cài đặt, lỗi nạp model và lỗi tham số mà không tốn GPU; chưa kiểm tra được CUDA và bộ nhớ GPU.
"""

import os
import subprocess

# Tham số truyền cho kịch bản smoke: chỉ chạy thử phần chọn config ASR, không chạy agent hay bước đo ASR riêng.
RUN = {
    "LIFEMATE_REF": "main",
    "RUN_GOLD": "0",
    "RUN_ASR": "0",
    "RUN_ASR_EVAL": "0",
    "ASR_TUNE": "chunkformer,parakeet",  # lần 2: chỉ hai model đã đổi cách cài hoặc engine sau lần 1
    "ASR_TUNE_PREFLIGHT": "3",
}
os.environ.update(RUN)
# Thư mục tạm chứa bản clone nông, chỉ dùng để lấy file kịch bản smoke.
src = "/kaggle/working/_launcher"
subprocess.check_call(
    [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        RUN["LIFEMATE_REF"],
        "https://github.com/awun0105/JSON-lifemate.git",
        src,
    ]
)
# Đọc mã kịch bản smoke vào bộ nhớ, rồi xoá bản clone tạm.
script = f"{src}/ops/kaggle/smoke/lifemate_smoke.py"
with open(script, encoding="utf-8") as f:
    code = f.read()
subprocess.check_call(["rm", "-rf", src])
# Chạy kịch bản smoke ngay trong tiến trình này; các biến môi trường ở trên đã có hiệu lực.
exec(compile(code, script, "exec"))
