"""Cấu hình chạy và profile môi trường, đọc từ file TOML (SPEC 4.14, 4.16).

Nội dung file:
- Các dataclass bất biến cho từng mục của config chạy: `LLMConfig` ([llm]), `AgentConfig` ([agent]),
  `ASRConfig` ([asr]), gộp lại trong `Config`.
- `EnvProfile`: nơi đặt dữ liệu, thư mục run và cache trên một máy cụ thể (`configs/env/<tên>.toml`).
- `load_config` và `load_profile`: đọc TOML, kiểm tra khóa và giá trị, trả về dataclass.

Cách xây dựng:
- Config chạy (mô hình, tham số agent, ASR) tách khỏi profile môi trường (đường dẫn, nguồn token), để cùng một
  config chạy được trên Kaggle, Colab hay máy cá nhân chỉ bằng cách đổi profile.
- Kiểm tra chặt: khóa lạ hoặc thiếu khóa đều báo `ConfigError` ngay lúc nạp, thay vì lỗi muộn giữa lượt chạy.
  Chỉ các khóa trong tập optional (ví dụ `_ASR_OPTIONAL`) mới được bỏ trống và dùng giá trị mặc định.
- Dataclass `frozen=True, slots=True`: config không đổi sau khi nạp, mọi chỗ đọc đều thấy cùng một giá trị.
- Token HF không nằm trong file; profile chỉ ghi nguồn (`env:`, `kaggle_secret:`, `colab_userdata:`) và token
  được đọc lúc chạy.
"""

from __future__ import annotations

import dataclasses
import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar


class ConfigError(ValueError):
    """Khóa config sai giá trị, không được biết hoặc bị thiếu."""


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Mục [llm]: backend, mô hình và tham số sinh văn bản."""

    backend: str
    model: str
    revision: str
    dtype: str
    max_model_len: int
    gpu_memory_utilization: float
    seed: int
    max_tokens: int
    batch_invariant: bool
    warmup_runs: int


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Mục [agent]: giới hạn vòng lặp agent, prompt hệ thống và danh sách guard."""

    max_steps: int
    max_retries: int
    deadline_s: float  # số giây tính từ lúc bắt đầu mẫu (gồm cả ASR); bước LLM đang chạy bị dừng tại mốc này
    max_result_chars: int
    system_prompt: str
    guards: list[str]


@dataclass(frozen=True, slots=True)
class ASRConfig:
    """Mục [asr]: backend nhận dạng giọng nói; các trường có giá trị mặc định là tùy chọn trong TOML."""

    backend: str
    model: str
    revision: str
    engine: str = "hf_seq2seq"  # cách nạp `model` khi backend = "asr"; xem lifemate.backends.asr.ASR_ENGINES
    device: str = "auto"
    dtype: str = "auto"  # auto: bfloat16 trên GPU sm80+, float16 trên GPU cũ hơn, float32 trên CPU
    batch_size: int = 8
    generate_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    sample_rate: int = 16000
    sanitize: bool = False  # bỏ ảo giác kiểu phụ đề và đoạn lặp (lifemate.backends.asr_sanitizer)


ASR_ENGINE_NAMES = frozenset({"faster_whisper", "hf_seq2seq", "chunkformer", "viegem"})
ASR_DTYPES = frozenset({"auto", "float32", "float16", "bfloat16"})
# Phải khớp với các trường có giá trị mặc định của ASRConfig, nếu không `_build` sẽ đòi khóa đó trong TOML.
_ASR_OPTIONAL = frozenset({"engine", "device", "dtype", "batch_size", "generate_kwargs", "sample_rate", "sanitize"})


@dataclass(frozen=True, slots=True)
class Config:
    """Toàn bộ config của một lượt chạy: gồm ba mục llm, agent và asr."""

    llm: LLMConfig
    agent: AgentConfig
    asr: ASRConfig


@dataclass(frozen=True, slots=True)
class EnvProfile:
    """Nơi đặt dữ liệu, thư mục run và cache trên một máy."""

    data_root: str
    runs_dir: str
    hf_token_source: str
    persist: str
    hf_cache: str | None = None

    def with_overrides(self, root: str | None = None, out: str | None = None) -> EnvProfile:
        """Áp tham số dòng lệnh (--root, --out), ưu tiên hơn giá trị trong profile.

        Tham số:
            root: thư mục dữ liệu thay cho `data_root`; None thì giữ nguyên.
            out: thư mục run thay cho `runs_dir`; None thì giữ nguyên.

        Trả về một `EnvProfile` mới; profile gốc không đổi.
        """
        return dataclasses.replace(
            self,
            data_root=root if root is not None else self.data_root,
            runs_dir=out if out is not None else self.runs_dir,
        )

    def resolve_hf_token(self) -> str | None:
        """Đọc token HF từ nguồn đã khai báo lúc chạy; không bao giờ ghi token ra log.

        `hf_token_source` có dạng `<loại>:<tên>`, với loại là `env`, `kaggle_secret` hoặc `colab_userdata`.
        Trả về None khi loại không hỗ trợ, biến môi trường rỗng, hoặc thư viện của nền tảng không có.
        """
        kind, _, name = self.hf_token_source.partition(":")
        if kind == "env":
            # Biến môi trường rỗng được coi như không có token.
            return os.environ.get(name) or None
        if kind == "kaggle_secret":
            try:
                from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]  # Kaggle only
            except ImportError:
                return None
            return str(UserSecretsClient().get_secret(name))
        if kind == "colab_userdata":
            try:
                from google.colab import userdata  # type: ignore[import-not-found]  # Colab only
            except ImportError:
                return None
            return str(userdata.get(name))
        return None


_T = TypeVar("_T")


def _build(cls: type[_T], section: str, raw: Any, optional: frozenset[str] = frozenset()) -> _T:
    """Dựng dataclass `cls` từ bảng TOML `raw` của mục `section`, kiểm tra khóa trước.

    Báo `ConfigError` khi `raw` không phải bảng, có khóa không thuộc dataclass, hoặc thiếu khóa bắt buộc
    (mọi trường không nằm trong `optional`).
    """
    if not isinstance(raw, dict):
        raise ConfigError(f"[{section}] must be a table")
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    for key in raw:
        if key not in names:
            raise ConfigError(f"unknown key {section}.{key}")
    for name in names - optional:
        if name not in raw:
            raise ConfigError(f"missing key {section}.{name}")
    return cls(**raw)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def load_config(path: Path) -> Config:
    """Đọc và kiểm tra file TOML config chạy.

    Tham số:
        path: đường dẫn file TOML có đúng ba mục [llm], [agent], [asr].

    Trả về `Config` đã kiểm tra. Báo `ConfigError` khi thiếu hoặc thừa mục/khóa, hoặc khi giá trị nằm ngoài
    tập cho phép (backend, dtype, engine, device, batch_size...).
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    for key in raw:
        _check(key in {"llm", "agent", "asr"}, f"unknown key {key}")
    for key in ("llm", "agent", "asr"):
        _check(key in raw, f"missing key {key}")
    llm = _build(LLMConfig, "llm", raw["llm"])
    agent = _build(AgentConfig, "agent", raw["agent"])
    asr = _build(ASRConfig, "asr", raw["asr"], optional=_ASR_OPTIONAL)
    # Kiểm tra giá trị: backend "fake" và "gold" dùng cho test nên không cần tên mô hình.
    _check(llm.backend in {"vllm", "fake"}, f"llm.backend: {llm.backend!r} not in vllm|fake")
    _check(not (llm.backend == "vllm" and llm.model == ""), "llm.model: required when llm.backend = 'vllm'")
    _check(llm.dtype in {"auto", "float16", "bfloat16"}, f"llm.dtype: {llm.dtype!r} not in auto|float16|bfloat16")
    _check(asr.backend in {"gold", "asr"}, f"asr.backend: {asr.backend!r} not in gold|asr")
    _check(asr.engine in ASR_ENGINE_NAMES, f"asr.engine: {asr.engine!r} not in {'|'.join(sorted(ASR_ENGINE_NAMES))}")
    _check(asr.dtype in ASR_DTYPES, f"asr.dtype: {asr.dtype!r} not in {'|'.join(sorted(ASR_DTYPES))}")
    _check(not (asr.backend == "asr" and asr.model == ""), "asr.model: required when asr.backend = 'asr'")
    _check(asr.device in {"auto", "cpu", "cuda"}, f"asr.device: {asr.device!r} not in auto|cpu|cuda")
    # TOML không ép kiểu theo annotation của dataclass, nên kiểu của các trường số và bảng phải kiểm tra tay.
    _check(isinstance(asr.batch_size, int) and asr.batch_size >= 1, "asr.batch_size: must be an int >= 1")
    _check(isinstance(asr.generate_kwargs, dict), "asr.generate_kwargs: must be a table")
    _check(isinstance(asr.sample_rate, int) and asr.sample_rate > 0, "asr.sample_rate: must be a positive int")
    _check(
        isinstance(agent.guards, list) and all(isinstance(g, str) for g in agent.guards),
        "agent.guards: must be a list of str",
    )
    return Config(llm=llm, agent=agent, asr=asr)


def load_profile(name_or_path: str, repo_root: Path) -> EnvProfile:
    """Đọc profile môi trường theo tên (configs/env/<tên>.toml) hoặc theo đường dẫn.

    Tham số:
        name_or_path: tên profile, hoặc đường dẫn tới file có đuôi `.toml`.
        repo_root: gốc repo, dùng để tìm `configs/env/` khi chỉ có tên.

    Trả về `EnvProfile`; báo `ConfigError` khi khóa sai hoặc thiếu (`hf_cache` là tùy chọn).
    """
    path = Path(name_or_path)
    # Không có đuôi .toml thì hiểu là tên profile trong configs/env/.
    if path.suffix != ".toml":
        path = repo_root / "configs" / "env" / f"{name_or_path}.toml"
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return _build(EnvProfile, "env", raw, optional=frozenset({"hf_cache"}))
