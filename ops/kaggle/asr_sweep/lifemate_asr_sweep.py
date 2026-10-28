"""Kernel Kaggle so các biến thể cấu hình ASR (kernel riêng tư quanlam/lifemate-asr-sweep).

Đặt ASR_SWEEP rồi lấy ops/kaggle/smoke/lifemate_smoke.py từ cùng git ref và chạy; mỗi biến thể ghi WER, tỷ lệ
nghe sai thực thể, số câu rỗng và từng câu nghe ra vào asr_sweep/. Không chạy agent. Danh sách biến thể ở
ASR_VARIANTS trong lifemate_smoke.py. Đẩy lên: kaggle kernels push -p ops/kaggle/asr_sweep --accelerator NvidiaTeslaT4
"""

import os
import subprocess

# Tham số truyền cho kịch bản smoke: chỉ chạy nhánh ASR trên val_seen, bỏ nhánh gold và bước đo ASR.
RUN = {
    "LIFEMATE_REF": "main",
    "RUN_GOLD": "0",
    "RUN_ASR": "0",
    "RUN_ASR_EVAL": "0",
    "ASR_SWEEP": "norep_noprompt,hotwords,short_prompt,short_prompt2",
    "ASR_EVAL_IDS": "configs/splits/val_seen.txt",
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
