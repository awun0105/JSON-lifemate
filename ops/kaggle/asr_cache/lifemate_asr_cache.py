"""Kernel Kaggle dựng cache ASR cho dữ liệu SFT (kernel riêng tư `quanlam/lifemate-asr-cache`).

Cách dùng:
    kaggle kernels push -p ops/kaggle/asr_cache --accelerator NvidiaTeslaT4

Cách hoạt động:
1. Đặt `RUN_ASR_CACHE=1` cùng các tham số trong `RUN` vào biến môi trường.
2. Lấy `ops/kaggle/smoke/lifemate_smoke.py` từ cùng git ref rồi chạy.
3. Kết quả nằm ở `asr_cache/train_asr.json`; tải về `data/asr/train_asr.json` (không đưa lên git).

Lưu ý:
- Kernel này không chạy agent.
"""

import os
import subprocess

# Tham số truyền cho kịch bản smoke: chỉ dựng cache ASR; tắt nhánh gold, nhánh ASR của agent và bước đo ASR.
RUN = {
    "LIFEMATE_REF": "main",
    "RUN_GOLD": "0",
    "RUN_ASR": "0",
    "RUN_ASR_EVAL": "0",
    "RUN_ASR_CACHE": "1",
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
