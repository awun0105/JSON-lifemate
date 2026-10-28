"""Dò GPU, phiên bản mã nguồn và phiên bản thư viện của máy đang chạy, rồi chọn dtype phù hợp (SPEC 4.16).

Nội dung chính:
- `EnvInfo`: thông tin máy và mã nguồn của một lần chạy, ghi kèm kết quả để chạy lại được.
- `detect`: gom thông tin GPU (qua `torch`), phiên bản mã (git hoặc file `VERSION`) và phiên bản các gói chính.
- `code_version`: lấy commit hiện tại và cho biết cây làm việc có thay đổi chưa commit hay không.
- `resolve_dtype`: đổi dtype trong config (`auto`, `float16`, `bfloat16`) thành dtype thật theo GPU.

Cách hoạt động:
- Mọi bước dò đều chịu được lỗi. Thiếu `torch`, không có CUDA, không có git hay gói chưa cài thì trả về `None`
  hoặc bỏ qua, không làm hỏng lần chạy.

Lưu ý:
- bfloat16 chỉ chạy được trên GPU có compute capability từ 8.0 (Ampere) trở lên. `auto` chọn dtype theo ngưỡng
  này. Config ghi rõ bfloat16 mà GPU cũ hơn thì báo lỗi ngay từ đầu.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
from dataclasses import dataclass
from pathlib import Path

import structlog

from lifemate.core.config import ConfigError

log = structlog.get_logger(__name__)

_PACKAGES = ("vllm", "torch", "transformers", "lifemate")
# Compute capability thấp nhất hỗ trợ bfloat16 (sm80, đời Ampere).
_BF16_MIN_CC = (8, 0)


@dataclass(frozen=True, slots=True)
class EnvInfo:
    """Thông tin máy và mã nguồn của một lần chạy.

    Thuộc tính:
        gpu_name: tên GPU số 0; `None` khi không có GPU CUDA.
        compute_capability: cặp (major, minor) của GPU; `None` khi không có GPU CUDA.
        gpu_mem_gb: tổng bộ nhớ GPU, đơn vị GB; `None` khi không có GPU CUDA.
        code_version: commit git hoặc nội dung file `VERSION`; `"unknown"` khi không xác định được.
        code_dirty: `True` khi có thay đổi chưa commit; `None` khi không xác định được qua git.
        packages: tên gói tới phiên bản, chỉ gồm các gói đã cài.
    """

    gpu_name: str | None
    compute_capability: tuple[int, int] | None
    gpu_mem_gb: float | None
    code_version: str
    code_dirty: bool | None
    packages: dict[str, str]


def _gpu() -> tuple[str | None, tuple[int, int] | None, float | None]:
    """Trả về (tên, compute capability, bộ nhớ GB) của GPU số 0, hoặc ba giá trị `None` khi không có GPU."""
    try:
        import torch  # extra: gpu (đi kèm vllm)
    except ImportError:
        return None, None, None
    if not torch.cuda.is_available():
        return None, None, None
    props = torch.cuda.get_device_properties(0)
    return str(props.name), (int(props.major), int(props.minor)), round(props.total_memory / 2**30, 2)


def code_version(repo_root: Path) -> tuple[str, bool | None]:
    """Lấy phiên bản mã nguồn và trạng thái dirty của repo tại `repo_root`.

    Thứ tự thử: git, rồi file `VERSION`, cuối cùng trả về `("unknown", None)`. File `VERSION` dành cho bản sao
    không có `.git`, ví dụ mã được tải lên Kaggle dưới dạng dataset.

    Tham số:
        repo_root: thư mục gốc của repo.

    Trả về:
        Cặp `(phiên bản, dirty)`. `dirty` là `True` khi cây làm việc có thay đổi chưa commit, `None` khi phiên
        bản lấy từ file `VERSION` hoặc không xác định được.
    """
    if (repo_root / ".git").exists():
        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True
            ).stdout
            return head, bool(status.strip())
        except (OSError, subprocess.CalledProcessError):
            # Máy không có lệnh git hoặc repo hỏng: thử file `VERSION` bên dưới.
            pass
    version_file = repo_root / "VERSION"
    if version_file.is_file():
        return version_file.read_text(encoding="utf-8").strip(), None
    log.warning("code_version_unknown", repo_root=str(repo_root))
    return "unknown", None


def _packages() -> dict[str, str]:
    """Trả về phiên bản các gói trong `_PACKAGES`, bỏ qua gói chưa cài."""
    out: dict[str, str] = {}
    for name in _PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return out


def detect(repo_root: Path) -> EnvInfo:
    """Thu thập `EnvInfo` cho máy hiện tại và bản checkout tại `repo_root`."""
    name, cc, mem = _gpu()
    version, dirty = code_version(repo_root)
    return EnvInfo(name, cc, mem, version, dirty, _packages())


def resolve_dtype(cfg_dtype: str, info: EnvInfo) -> str:
    """Đổi dtype trong config thành dtype thật: `auto` thành bfloat16 trên GPU sm80 trở lên, còn lại là float16.

    Khi máy không có GPU, giá trị ghi rõ trong config được giữ nguyên vì không có gì để kiểm tra.

    Tham số:
        cfg_dtype: giá trị `llm.dtype` trong config.
        info: thông tin máy; chỉ dùng compute capability của GPU.

    Trả về:
        Tên dtype: `"bfloat16"`, `"float16"` hoặc giá trị gốc trong config.

    Lỗi:
        ConfigError: khi config yêu cầu bfloat16 trên GPU dưới sm80.
    """
    cc = info.compute_capability
    supports_bf16 = cc is not None and cc >= _BF16_MIN_CC
    if cfg_dtype == "auto":
        return "bfloat16" if supports_bf16 else "float16"
    if cfg_dtype == "bfloat16" and cc is not None and not supports_bf16:
        raise ConfigError(f"llm.dtype: bfloat16 needs compute capability >= 8.0, got {cc[0]}.{cc[1]}")
    return cfg_dtype
