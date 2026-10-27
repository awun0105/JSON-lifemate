"""Phát hiện GPU, phiên bản mã nguồn và phiên bản thư viện; chọn dtype (SPEC 4.16).

Nội dung file:
- `EnvInfo`: các thông tin về máy và mã nguồn của một lượt chạy, được ghi kèm kết quả để tái lập.
- `detect`: gom thông tin GPU (qua torch), phiên bản mã (git hoặc file VERSION) và phiên bản các gói chính.
- `resolve_dtype`: đổi dtype trong config (`auto`, `float16`, `bfloat16`) thành dtype thật theo GPU.

Cách xây dựng:
- Mọi phép dò đều chịu lỗi: thiếu torch, không có CUDA, không có git hay gói chưa cài đều trả về None hoặc bỏ qua,
  không làm hỏng lượt chạy.
- bfloat16 chỉ dùng được từ compute capability 8.0 (Ampere) trở lên; `auto` chọn theo ngưỡng này, còn yêu cầu
  bfloat16 tường minh trên GPU cũ thì báo lỗi sớm.
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
# Compute capability tối thiểu hỗ trợ bfloat16 (sm80, Ampere).
_BF16_MIN_CC = (8, 0)


@dataclass(frozen=True, slots=True)
class EnvInfo:
    """Thông tin về máy và mã nguồn của một lượt chạy.

    Các trường GPU là None khi không có GPU CUDA; `code_dirty` là None khi không xác định được qua git.
    """

    gpu_name: str | None
    compute_capability: tuple[int, int] | None
    gpu_mem_gb: float | None
    code_version: str
    code_dirty: bool | None
    packages: dict[str, str]


def _gpu() -> tuple[str | None, tuple[int, int] | None, float | None]:
    """Trả về (tên, compute capability, bộ nhớ GB) của GPU số 0, hoặc ba giá trị None khi không có GPU."""
    try:
        import torch  # extra: gpu (đi kèm vllm)
    except ImportError:
        return None, None, None
    if not torch.cuda.is_available():
        return None, None, None
    props = torch.cuda.get_device_properties(0)
    return str(props.name), (int(props.major), int(props.minor)), round(props.total_memory / 2**30, 2)


def code_version(repo_root: Path) -> tuple[str, bool | None]:
    """Trả về (phiên bản, dirty) lấy từ git, nếu không được thì từ file VERSION, cuối cùng là ("unknown", None).

    `dirty` là True khi cây làm việc có thay đổi chưa commit. File VERSION dùng cho bản sao không có `.git`
    (ví dụ mã được tải lên dưới dạng dataset), khi đó `dirty` là None.
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
            # Không có lệnh git hoặc repo hỏng: chuyển sang file VERSION.
            pass
    version_file = repo_root / "VERSION"
    if version_file.is_file():
        return version_file.read_text(encoding="utf-8").strip(), None
    log.warning("code_version_unknown", repo_root=str(repo_root))
    return "unknown", None


def _packages() -> dict[str, str]:
    """Phiên bản các gói trong `_PACKAGES`; gói chưa cài thì bỏ qua."""
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
    """Đổi dtype trong config thành dtype thật: auto → bfloat16 trên sm80+, ngược lại float16.

    Tham số:
        cfg_dtype: giá trị `llm.dtype` trong config.
        info: thông tin máy, dùng compute capability của GPU.

    Báo `ConfigError` khi yêu cầu bfloat16 trên GPU dưới sm80. Khi không có GPU, giá trị tường minh được giữ
    nguyên vì không có gì để kiểm tra.
    """
    cc = info.compute_capability
    supports_bf16 = cc is not None and cc >= _BF16_MIN_CC
    if cfg_dtype == "auto":
        return "bfloat16" if supports_bf16 else "float16"
    if cfg_dtype == "bfloat16" and cc is not None and not supports_bf16:
        raise ConfigError(f"llm.dtype: bfloat16 needs compute capability >= 8.0, got {cc[0]}.{cc[1]}")
    return cfg_dtype
