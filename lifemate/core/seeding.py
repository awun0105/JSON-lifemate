"""Cố định seed ngẫu nhiên của Python, NumPy và PyTorch (thể lệ mục 8: cố định mọi random seed; SPEC 4.11, 6.5).

Nội dung chính:
- `seed_everything`: đặt cùng một seed cho `random`, NumPy và PyTorch (CPU và mọi GPU), tuỳ chọn bật chế độ tất
  định của PyTorch.

Dùng ở đâu:
- `lifemate.run.cli` › `run`: một lần trước khi nạp model, với `[llm].seed` và `deterministic=False`.
- `lifemate.train.common` (job train LLM, ASR): với seed của job và `deterministic=True`.

Lưu ý:
- NumPy và PyTorch là phụ thuộc tuỳ chọn (chỉ có trên máy GPU); thư viện chưa cài thì bỏ qua.
- `PYTHONHASHSEED` (thứ tự duyệt `set` theo hash chuỗi) chỉ có tác dụng khi đặt trước lúc Python khởi động. Hàm đặt
  biến này cho các tiến trình con; container chạy thi đặt sẵn trong `ops/docker/Dockerfile`.
- Lúc chạy thi không bật chế độ tất định của PyTorch: LLM chạy trên vLLM với seed và `batch_invariant` riêng, ASR
  giải mã không ngẫu nhiên, còn `use_deterministic_algorithms` có thể làm chậm hoặc cảnh báo ở thư viện ngoài.
"""

from __future__ import annotations

import importlib
import os
import random
from types import ModuleType

import structlog

log = structlog.get_logger(__name__)


def _optional_module(name: str) -> ModuleType | None:
    """Import gói tuỳ chọn (numpy, torch); không có thì trả về `None`."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def seed_everything(seed: int, *, deterministic: bool = True) -> list[str]:
    """Đặt seed cho Python, NumPy và PyTorch (CPU và CUDA).

    Tham số:
        seed: giá trị seed.
        deterministic: `True` để bật thuật toán tất định của PyTorch khi có thể (chậm hơn nhưng chạy lại ra cùng
            kết quả).

    Trả về:
        Tên các thư viện đã đặt seed, theo thứ tự `["random", "numpy", "torch"]` (bỏ thư viện chưa cài).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    seeded = ["random"]
    np = _optional_module("numpy")
    if np is not None:
        np.random.seed(seed)
        seeded.append("numpy")
    torch = _optional_module("torch")
    if torch is None:
        return seeded
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    seeded.append("torch")
    if deterministic:
        # cuBLAS cần biến này thì phép nhân ma trận mới tất định.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            # `warn_only=True`: phép tính không có bản tất định chỉ cảnh báo, không làm hỏng job.
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError, TypeError) as exc:
            log.warning("deterministic_algorithms_unavailable", error=str(exc))
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seeded
