"""Detect GPU, code version and package versions; pick dtype (SPEC 4.16)."""

from __future__ import annotations

import importlib.metadata
import subprocess
from dataclasses import dataclass
from pathlib import Path

import structlog

from lifemate.core.config import ConfigError

log = structlog.get_logger(__name__)

_PACKAGES = ("vllm", "torch", "transformers", "lifemate")
_BF16_MIN_CC = (8, 0)


@dataclass(frozen=True, slots=True)
class EnvInfo:
    """Facts about the machine and code of one run."""

    gpu_name: str | None
    compute_capability: tuple[int, int] | None
    gpu_mem_gb: float | None
    code_version: str
    code_dirty: bool | None
    packages: dict[str, str]


def _gpu() -> tuple[str | None, tuple[int, int] | None, float | None]:
    try:
        import torch  # extra: gpu (comes with vllm)
    except ImportError:
        return None, None, None
    if not torch.cuda.is_available():
        return None, None, None
    props = torch.cuda.get_device_properties(0)
    return str(props.name), (int(props.major), int(props.minor)), round(props.total_memory / 2**30, 2)


def code_version(repo_root: Path) -> tuple[str, bool | None]:
    """Return (version, dirty) from git, else the VERSION file, else ("unknown", None)."""
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
            pass
    version_file = repo_root / "VERSION"
    if version_file.is_file():
        return version_file.read_text(encoding="utf-8").strip(), None
    log.warning("code_version_unknown", repo_root=str(repo_root))
    return "unknown", None


def _packages() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return out


def detect(repo_root: Path) -> EnvInfo:
    """Collect EnvInfo for this machine and checkout."""
    name, cc, mem = _gpu()
    version, dirty = code_version(repo_root)
    return EnvInfo(name, cc, mem, version, dirty, _packages())


def resolve_dtype(cfg_dtype: str, info: EnvInfo) -> str:
    """Map config dtype to the real one: auto → bfloat16 on sm80+, else float16."""
    cc = info.compute_capability
    supports_bf16 = cc is not None and cc >= _BF16_MIN_CC
    if cfg_dtype == "auto":
        return "bfloat16" if supports_bf16 else "float16"
    if cfg_dtype == "bfloat16" and cc is not None and not supports_bf16:
        raise ConfigError(f"llm.dtype: bfloat16 needs compute capability >= 8.0, got {cc[0]}.{cc[1]}")
    return cfg_dtype
