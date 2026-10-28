"""Kernel Kaggle chọn model và config ASR một cách công bằng (kernel riêng tư `quanlam/lifemate-asr-tune`).

Cách dùng:
    kaggle kernels push -p ops/kaggle/asr_tune --accelerator NvidiaTeslaT4

Cách hoạt động:
1. Đặt `ASR_TUNE` (danh sách model) cùng các tham số trong `RUN` vào biến môi trường.
2. Lấy `ops/kaggle/smoke/lifemate_smoke.py` từ cùng git ref rồi chạy.
3. Kịch bản đó gọi `python -m lifemate.eval.asr_tune` (thêm `ASR_TUNE_ARGS`): dò từng bước trên tập dò, cùng hàm
   mục tiêu cho mọi model, rồi đo config thắng trên các tập báo cáo. Kết quả ghi vào `asr_tune/`.

Lưu ý:
- Kernel này không chạy agent.
"""

import os
import subprocess

# Tham số truyền cho kịch bản smoke: chỉ chọn config ASR, không chạy agent hay bước đo ASR riêng.
RUN = {
    "LIFEMATE_REF": "asr-benchmark-v2",  # nhánh của benchmark ASR lần 2; đổi lại "main" sau khi merge
    "RUN_GOLD": "0",
    "RUN_ASR": "0",
    "RUN_ASR_EVAL": "0",
    # Đủ năm model: "large-v3,phowhisper,chunkformer,chunkformer-rnnt,parakeet". Lần 1 (v3, v4) xem
    # `docs/reports/asr_benchmark.md`.
    # Lần 2 chạy hai phiên nối tiếp (tổng khoảng 7,5 giờ trên T4): v5a = "phowhisper,chunkformer-rnnt", v5b = "large-v3".
    "ASR_TUNE": "phowhisper,chunkformer-rnnt",
    # Benchmark lần 2 (câu chêm tiếng Anh, D40): chỉ dò bước câu mồi (13 tổ hợp câu mồi x lớp bảo vệ câu mồi), large-v3
    # bật VAD từ đầu, hàm mục tiêu có từ tiếng Anh. Dò trên `asr_tune` đầy đủ, đo trên `val_seen`, `val_unseen` đầy đủ
    # (mặc định); số của các tập `*_en_mixed` lấy bằng `rescore_asr` trên tập con. ChunkFormer RNNT không có lựa chọn
    # câu mồi nên chỉ chạy một config: lần chạy kiểm chứng số chấm offline từ kết quả lần 1.
    "ASR_TUNE_ARGS": "--stages prompt --start-option engine=vad --objective en_mixed",
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
