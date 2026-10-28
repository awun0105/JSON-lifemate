"""Kernel Kaggle chạy thử trên CPU trước lần chọn cấu hình ASR (kernel riêng tư quanlam/lifemate-asr-preflight).

Cài đúng các thư viện như kernel asr_tune (vLLM, faster-whisper, chunkformer, NeMo), rồi chạy mọi lựa chọn engine
hỗ trợ của từng mô hình trên 3 mẫu (`asr_tune --preflight 3`).
Bắt lỗi cài đặt, nạp mô hình và tham số mà không tốn GPU; chưa kiểm được CUDA và bộ nhớ GPU.
Đẩy lên: kaggle kernels push -p ops/kaggle/asr_preflight (không cần --accelerator)
"""

import os
import subprocess

# Tham số truyền cho kịch bản smoke: chỉ chọn cấu hình ASR, không chạy agent hay bước đo ASR riêng.
RUN = {
    "LIFEMATE_REF": "main",
    "RUN_GOLD": "0",
    "RUN_ASR": "0",
    "RUN_ASR_EVAL": "0",
    "ASR_TUNE": "chunkformer,parakeet",  # lần 2: chỉ hai mô hình đổi cách cài/engine sau lần 1
    "ASR_TUNE_PREFLIGHT": "3",
}
os.environ.update(RUN)
# Thư mục tạm chứa bản clone dùng để lấy kịch bản smoke.
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
# Đọc mã kịch bản smoke vào bộ nhớ trước khi xoá bản clone tạm.
script = f"{src}/ops/kaggle/smoke/lifemate_smoke.py"
with open(script, encoding="utf-8") as f:
    code = f.read()
subprocess.check_call(["rm", "-rf", src])
# Chạy kịch bản smoke ngay trong tiến trình này; biến môi trường ở trên đã có hiệu lực.
exec(compile(code, script, "exec"))
