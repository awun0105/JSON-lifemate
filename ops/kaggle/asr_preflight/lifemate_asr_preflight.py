"""Kernel Kaggle chạy thử trên CPU trước lần chọn config ASR (kernel riêng tư `quanlam/lifemate-asr-preflight`).

Cách dùng:
    kaggle kernels push -p ops/kaggle/asr_preflight   (không cần --accelerator)

Cách hoạt động:
1. Cài đúng các thư viện như kernel `asr_tune` (vLLM, faster-whisper, chunkformer, NeMo).
2. Chạy mọi lựa chọn engine hỗ trợ của từng model trên 3 mẫu (`asr_tune --preflight 3`).

Lưu ý:
- Bắt được lỗi cài đặt, lỗi nạp model và lỗi tham số mà không tốn GPU; chưa kiểm tra được CUDA và bộ nhớ GPU.
"""

import os
import subprocess

# Tham số truyền cho kịch bản smoke: chỉ chạy thử phần chọn config ASR, không chạy agent hay bước đo ASR riêng.
RUN = {
    "LIFEMATE_REF": "asr-benchmark-v2",  # nhánh của benchmark ASR lần 2; đổi lại "main" sau khi merge
    "RUN_GOLD": "0",
    "RUN_ASR": "0",
    "RUN_ASR_EVAL": "0",
    "ASR_TUNE": "large-v3,phowhisper,chunkformer-rnnt",  # benchmark ASR lần 2 (câu chêm tiếng Anh)
    # Benchmark lần 2 (câu chêm tiếng Anh, D40): chỉ dò bước câu mồi (13 tổ hợp câu mồi x lớp bảo vệ câu mồi), large-v3
    # bật VAD từ đầu, hàm mục tiêu có từ tiếng Anh. Dò trên `asr_tune` đầy đủ, đo trên `val_seen`, `val_unseen` đầy đủ
    # (mặc định); số của các tập `*_en_mixed` lấy bằng `rescore_asr` trên tập con. ChunkFormer RNNT không có lựa chọn
    # câu mồi nên chỉ chạy một config: lần chạy kiểm chứng số chấm offline từ kết quả lần 1.
    "ASR_TUNE_ARGS": "--stages prompt --start-option engine=vad --objective en_mixed",
    "ASR_TUNE_PREFLIGHT": "3",
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
