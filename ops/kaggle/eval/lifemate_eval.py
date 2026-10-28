"""Kernel Kaggle chạy đánh giá agent cho vòng lặp tinh chỉnh prompt (kernel riêng tư `quanlam/lifemate-eval`).

Cách dùng:
    kaggle kernels push -p ops/kaggle/eval

Nội dung chính:
- `RUN`: tham số của lần chạy; sửa ở đây để đổi split hoặc tên lần chạy.

Cách hoạt động:
1. Ghi các tham số trong `RUN` vào biến môi trường.
2. Clone nông repo ở `LIFEMATE_REF` vào thư mục tạm.
3. Đọc mã của `lifemate_smoke.py`, xoá thư mục tạm, rồi chạy mã đó trong tiến trình hiện tại.
Kịch bản smoke chạy agent với transcript gold trên danh sách id và giữ `runs/`, `logs/` làm output. Nhờ vậy mỗi
vòng tinh chỉnh chỉ cần một commit trên main và một lệnh push kernel.

Lưu ý:
- File này chỉ là trình khởi chạy mỏng: đặt tham số rồi giao việc cho kịch bản smoke.
- Chạy agent trên chữ ASR thật cần GPU từ 24 GB (T4 15 GB không chứa nổi vLLM cùng whisper-large-v3). Khi đó dùng
  `make eval` trên Colab hoặc máy thuê.
"""

import os
import subprocess

# Tham số truyền cho kịch bản smoke: chạy agent với transcript gold trên val_seen và val_unseen; tắt nhánh ASR và
# bước đo ASR.
RUN = {
    "LIFEMATE_REF": "main",
    "RUN_IDS": "configs/splits/val_seen.txt,configs/splits/val_unseen.txt",
    "RUN_TAG": "base28",
    "RUN_GOLD": "1",
    "RUN_ASR": "0",
    "RUN_ASR_EVAL": "0",
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
