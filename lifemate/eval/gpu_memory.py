"""Đọc bộ nhớ GPU trong lúc chạy bằng `nvidia-smi` chạy nền (benchmark ASR lần 3, mục 2.4).

Nội dung chính:
- `GpuMemoryMonitor`: context manager chạy nền `nvidia-smi`, ghi `gpu_memory.csv`, trả bộ nhớ đỉnh.
- `parse_memory_csv`: đọc các giá trị MiB từ đầu ra của `nvidia-smi --format=csv,noheader,nounits`.

Lưu ý:
- whisper-large-v3 chạy trên CTranslate2, không phải PyTorch, nên `torch.cuda.max_memory_allocated` không thấy bộ nhớ
  của mô hình này. `nvidia-smi` đọc bộ nhớ của cả GPU, nên phiên đo không được có tiến trình GPU khác.
- Không có `nvidia-smi` (máy CPU) thì monitor không làm gì và bộ nhớ đỉnh là `None`.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import TracebackType


def parse_memory_csv(text: str) -> list[int]:
    """Các giá trị bộ nhớ đã dùng (MiB) trong đầu ra csv của `nvidia-smi`, mỗi dòng một giá trị; bỏ dòng lỗi."""
    values: list[int] = []
    for line in text.splitlines():
        first = line.split(",")[0].strip()
        if first.isdigit():
            values.append(int(first))
    return values


class GpuMemoryMonitor:
    """Chạy nền `nvidia-smi --query-gpu=memory.used -lms <chu kỳ>` trong khối `with`, ghi ra `out_csv`.

    Tham số:
        out_csv: file ghi các lần đọc.
        interval_ms: chu kỳ đọc, mili giây.

    Thuộc tính:
        baseline_mb: bộ nhớ đã dùng lúc bắt đầu (trước khi nạp mô hình).
        peak_mb: bộ nhớ lớn nhất trong khối trừ `baseline_mb`; `None` khi không có `nvidia-smi`.
    """

    def __init__(self, out_csv: Path, interval_ms: int = 200) -> None:
        self._out = Path(out_csv)
        self._interval = interval_ms
        self._proc: subprocess.Popen[bytes] | None = None
        self._file = None
        self.baseline_mb: int | None = None
        self.peak_mb: int | None = None

    def __enter__(self) -> GpuMemoryMonitor:
        exe = shutil.which("nvidia-smi")
        if exe is None:
            return self
        query = [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        first = parse_memory_csv(subprocess.run(query, capture_output=True, text=True, check=False).stdout)
        self.baseline_mb = first[0] if first else 0
        self._out.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._out.open("wb")
        self._proc = subprocess.Popen([*query, f"-lms={self._interval}"], stdout=self._file)
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        self._proc.wait(timeout=10)
        if self._file is not None:
            self._file.close()
        values = parse_memory_csv(self._out.read_text(encoding="utf-8", errors="replace"))
        self.peak_mb = (max(values) - (self.baseline_mb or 0)) if values else None
