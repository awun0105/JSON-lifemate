"""Kernel Kaggle của benchmark ASR lần 3 (kernel riêng tư `quanlam/lifemate-asr-v3`).

Báo cáo: `docs/reports/benchmark/asr_v3.md`.

Cách dùng:
    kaggle kernels push -p ops/kaggle/asr_v3 --accelerator NvidiaTeslaT4

Cách hoạt động:
1. Chọn phiên trong `SESSIONS` bằng `SESSION`, đặt các lệnh của phiên vào `ASR_V3_CMDS` (lệnh con của
   `python -m lifemate.eval.asr_v3`, kết quả dưới `/kaggle/working/asr_v3/`).
2. Lấy `ops/kaggle/smoke/lifemate_smoke.py` từ cùng git ref rồi chạy; kịch bản đó cài thư viện, dựng lại các split,
   chạy test offline rồi chạy các lệnh trên.

Các phiên:
- K1: ba biến thể câu mồi (E0, E1, E2) trên `asr_tune`, kèm RNNT, kiểm tra tất định 50 mẫu.
- K2: biến thể thắng của E trên tập học và hai tập val, kèm RNNT (`WINNER` đặt sau khi chọn E).
- K3: điểm âm thanh của các vùng cần quyết định trên tập học, `asr_tune` và hai tập val (thử nghiệm G); đọc chữ của K1,
  K2 qua `kernel_sources` (output của phiên trước gắn dưới `/kaggle/input`).
"""

import os
import subprocess

SESSION = "K1"
WINNER = "E0"  # biến thể câu mồi thắng của E, đặt sau phiên K1
SPLITS = "configs/splits"
SESSIONS = {
    "K1": [f"transcribe --ids {SPLITS}/asr_tune.txt --variants E0 E1 E2 --rnnt --rerun 50 --out asr_v3/k1"],
    "K2": [
        f"transcribe --ids {SPLITS}/{name}.txt --variants {WINNER} --rnnt --out asr_v3/k2_{name}"
        for name in ("asr_learn", "val_seen", "val_unseen")
    ],
    "K3": [
        f"score --ids {SPLITS}/{name}.txt --rnnt {src}/rnnt.hyps.jsonl --v3 {src}/v3_{WINNER}.hyps.jsonl "
        f"--rerun {50 if name == 'asr_tune' else 0} --out asr_v3/k3/scores_{name}"
        for name, src in (
            ("asr_tune", "/kaggle/input/lifemate-asr-v3-k1/asr_v3/k1"),
            ("asr_learn", "/kaggle/input/lifemate-asr-v3-k2/asr_v3/k2_asr_learn"),
            ("val_seen", "/kaggle/input/lifemate-asr-v3-k2/asr_v3/k2_val_seen"),
            ("val_unseen", "/kaggle/input/lifemate-asr-v3-k2/asr_v3/k2_val_unseen"),
        )
    ],
}
RUN = {
    "LIFEMATE_REF": "asr-benchmark-v2",
    "RUN_GOLD": "0",
    "RUN_ASR": "0",
    "RUN_ASR_EVAL": "0",
    "ASR_V3_CMDS": " ;; ".join(SESSIONS[SESSION]),
}
os.environ.update(RUN)
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
script = f"{src}/ops/kaggle/smoke/lifemate_smoke.py"
with open(script, encoding="utf-8") as f:
    code = f.read()
subprocess.check_call(["rm", "-rf", src])
exec(compile(code, script, "exec"))
