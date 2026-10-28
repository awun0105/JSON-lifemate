"""Chạy đánh giá trên Kaggle cho vòng lặp tinh chỉnh prompt (kernel riêng tư quanlam/lifemate-eval).

Nội dung: một trình khởi chạy mỏng, chỉ đặt tham số rồi giao việc cho kịch bản smoke.

Chạy agent với transcript PhoWhisper trên một danh sách id và giữ runs/ cùng logs/ làm output.
File lấy ops/kaggle/smoke/lifemate_smoke.py từ cùng git ref rồi exec với các tham số này,
nên mỗi vòng lặp chỉ cần một commit trên main và lệnh `kaggle kernels push -p ops/kaggle/eval`.
Sửa RUN bên dưới để đổi split hoặc tên lượt chạy.

Cách hoạt động:
  1. Ghi các tham số trong RUN vào biến môi trường.
  2. Clone nông repo ở LIFEMATE_REF vào thư mục tạm.
  3. Đọc mã của lifemate_smoke.py, xoá thư mục tạm, rồi exec mã đó trong tiến trình hiện tại.
"""

import os
import subprocess

# Tham số truyền cho kịch bản smoke: chỉ chạy nhánh ASR trên val_seen, bỏ nhánh gold và bước đo ASR.
RUN = {
    "LIFEMATE_REF": "main",
    "RUN_IDS": "configs/splits/val_seen.txt,configs/splits/val_unseen.txt",
    "RUN_TAG": "base28",
    "RUN_GOLD": "0",
    "RUN_ASR": "1",
    "RUN_ASR_EVAL": "0",
    "ASR_DTYPE": "int8_float16",
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
