"""Kernel Kaggle chọn mô hình và cấu hình ASR công bằng (kernel riêng tư quanlam/lifemate-asr-tune).

Đặt ASR_TUNE rồi lấy ops/kaggle/smoke/lifemate_smoke.py từ cùng git ref và chạy; kịch bản đó gọi
`python -m lifemate.eval.asr_tune` (dò từng bước trên configs/splits/asr_tune.txt, cùng hàm mục tiêu cho mọi mô hình,
đo cấu hình thắng trên val_seen và val_unseen) và ghi kết quả vào asr_tune/. Không chạy agent.
Đẩy lên: kaggle kernels push -p ops/kaggle/asr_tune --accelerator NvidiaTeslaT4
"""

import os
import subprocess

# Tham số truyền cho kịch bản smoke: chỉ chọn cấu hình ASR, không chạy agent hay bước đo ASR riêng.
RUN = {
    "LIFEMATE_REF": "main",
    "RUN_GOLD": "0",
    "RUN_ASR": "0",
    "RUN_ASR_EVAL": "0",
    "ASR_TUNE": "large-v3,phowhisper,chunkformer,chunkformer-rnnt,parakeet",
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
