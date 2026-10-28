"""Kernel Kaggle chọn model và config ASR một cách công bằng (kernel riêng tư `quanlam/lifemate-asr-tune`).

Cách dùng:
    kaggle kernels push -p ops/kaggle/asr_tune --accelerator NvidiaTeslaT4

Cách hoạt động:
1. Đặt `ASR_TUNE` (danh sách model) cùng các tham số trong `RUN` vào biến môi trường.
2. Lấy `ops/kaggle/smoke/lifemate_smoke.py` từ cùng git ref rồi chạy.
3. Kịch bản đó gọi `python -m lifemate.eval.asr_tune`: dò từng bước trên `configs/splits/asr_tune.txt`, cùng hàm
   mục tiêu cho mọi model, rồi đo config thắng trên `val_seen` và `val_unseen`. Kết quả ghi vào `asr_tune/`.

Lưu ý:
- Kernel này không chạy agent.
"""

import os
import subprocess

# Tham số truyền cho kịch bản smoke: chỉ chọn config ASR, không chạy agent hay bước đo ASR riêng.
RUN = {
    "LIFEMATE_REF": "main",
    "RUN_GOLD": "0",
    "RUN_ASR": "0",
    "RUN_ASR_EVAL": "0",
    # Đủ năm model: "large-v3,phowhisper,chunkformer,chunkformer-rnnt,parakeet". Lần chạy 29/10/2026 chỉ chạy lại
    # parakeet sau khi sửa engine NeMo (float16 cho đầu ra NaN); bốn model kia giữ kết quả lần chạy v3.
    "ASR_TUNE": "parakeet",
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
