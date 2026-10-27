"""Kernel Kaggle dựng cache ASR cho dữ liệu SFT (kernel riêng tư quanlam/lifemate-asr-cache).

Đặt RUN_ASR_CACHE=1 rồi lấy ops/kaggle/smoke/lifemate_smoke.py từ cùng git ref và chạy; kết quả ở
asr_cache/train_asr.json, tải về data/asr/train_asr.json (không đưa lên git). Không chạy agent.
Đẩy lên: kaggle kernels push -p ops/kaggle/asr_cache --accelerator NvidiaTeslaT4
"""

import os
import subprocess

# Tham số truyền cho kịch bản smoke: chỉ chạy nhánh ASR trên val_seen, bỏ nhánh gold và bước đo ASR.
RUN = {
    "LIFEMATE_REF": "main",
    "RUN_GOLD": "0",
    "RUN_ASR": "0",
    "RUN_ASR_EVAL": "0",
    "RUN_ASR_CACHE": "1",
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
