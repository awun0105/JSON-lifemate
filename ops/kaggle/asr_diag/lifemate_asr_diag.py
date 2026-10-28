"""Kernel Kaggle chẩn đoán một model ASR trên các mẫu hỏng của lần chạy trước (kernel riêng tư
`quanlam/lifemate-asr-diag`).

Cách dùng:
    kaggle kernels push -p ops/kaggle/asr_diag --accelerator NvidiaTeslaT4

Cách hoạt động:
1. Gắn output của kernel `quanlam/lifemate-asr-tune` (qua `kernel_sources`) để đọc file `hyps.jsonl` cũ.
2. Đặt `ASR_DIAG` (model), `ASR_DIAG_FROM` (file `hyps.jsonl` cũ) và `ASR_DIAG_SETS` (các bộ ghi đè config) vào biến
   môi trường, rồi chạy `ops/kaggle/smoke/lifemate_smoke.py` từ cùng git ref (mục 5c).
3. Mỗi bộ ghi đè cho một báo cáo `asr_diag/<model>/<i>.json`, kèm số câu chỉ gồm `⁇` (`n_unk`) và số cảnh báo
   `asr_nonfinite_output` (đầu ra NaN/inf) trong log.

Lưu ý:
- Kernel này không chạy agent.
"""

import os
import subprocess

# Bốn bộ ghi đè: dtype x batch. Bộ 0 giống lần chạy cũ (dtype auto = float16 trên T4, batch 8).
RUN = {
    "LIFEMATE_REF": "main",
    "RUN_GOLD": "0",
    "RUN_ASR": "0",
    "RUN_ASR_EVAL": "0",
    "ASR_DIAG": "parakeet",
    "ASR_DIAG_FROM": "/kaggle/input/**/parakeet__decode_greedy.hyps.jsonl",
    "ASR_DIAG_SETS": (
        "asr.dtype=auto,asr.batch_size=8;"
        "asr.dtype=float32,asr.batch_size=8;"
        "asr.dtype=auto,asr.batch_size=1;"
        "asr.dtype=float32,asr.batch_size=1"
    ),
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
