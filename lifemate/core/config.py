"""Run config and environment profiles loaded from TOML (SPEC 4.14, 4.16)."""

from __future__ import annotations

import dataclasses
import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar


class ConfigError(ValueError):
    """Invalid, unknown or missing config key."""


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """[llm] section."""

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
    """[agent] section."""

    max_steps: int
    max_retries: int
    hard_limit_s: float
    initial_step_estimate_s: float
    max_result_chars: int
    system_prompt: str
    guards: list[str]


@dataclass(frozen=True, slots=True)
class ASRConfig:
    """[asr] section."""

    backend: str
    model: str
    revision: str
    engine: str = "hf_seq2seq"  # how `model` is loaded when backend = "asr"; see lifemate.backends.asr.ASR_ENGINES
    device: str = "auto"
    dtype: str = "auto"  # auto: bfloat16 on sm80+ GPU, float16 on older GPU, float32 on CPU
    batch_size: int = 8
    generate_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    sample_rate: int = 16000


ASR_ENGINE_NAMES = frozenset({"hf_seq2seq", "chunkformer", "viegem"})
ASR_DTYPES = frozenset({"auto", "float32", "float16", "bfloat16"})
_ASR_OPTIONAL = frozenset({"engine", "device", "dtype", "batch_size", "generate_kwargs", "sample_rate"})


@dataclass(frozen=True, slots=True)
class Config:
    """Whole run config."""

    llm: LLMConfig
    agent: AgentConfig
    asr: ASRConfig


@dataclass(frozen=True, slots=True)
class EnvProfile:
    """Where data, runs and caches live on one machine."""

    data_root: str
    runs_dir: str
    hf_token_source: str
    persist: str
    hf_cache: str | None = None

    def with_overrides(self, root: str | None = None, out: str | None = None) -> EnvProfile:
        """Apply CLI overrides (--root, --out), which win over the profile."""
        return dataclasses.replace(
            self,
            data_root=root if root is not None else self.data_root,
            runs_dir=out if out is not None else self.runs_dir,
        )

    def resolve_hf_token(self) -> str | None:
        """Read the HF token from its source at run time; never log it."""
        kind, _, name = self.hf_token_source.partition(":")
        if kind == "env":
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
    """Load and validate a run config TOML."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    for key in raw:
        _check(key in {"llm", "agent", "asr"}, f"unknown key {key}")
    for key in ("llm", "agent", "asr"):
        _check(key in raw, f"missing key {key}")
    llm = _build(LLMConfig, "llm", raw["llm"])
    agent = _build(AgentConfig, "agent", raw["agent"])
    asr = _build(ASRConfig, "asr", raw["asr"], optional=_ASR_OPTIONAL)
    _check(llm.backend in {"vllm", "fake"}, f"llm.backend: {llm.backend!r} not in vllm|fake")
    _check(not (llm.backend == "vllm" and llm.model == ""), "llm.model: required when llm.backend = 'vllm'")
    _check(llm.dtype in {"auto", "float16", "bfloat16"}, f"llm.dtype: {llm.dtype!r} not in auto|float16|bfloat16")
    _check(asr.backend in {"gold", "asr"}, f"asr.backend: {asr.backend!r} not in gold|asr")
    _check(asr.engine in ASR_ENGINE_NAMES, f"asr.engine: {asr.engine!r} not in {'|'.join(sorted(ASR_ENGINE_NAMES))}")
    _check(
        asr.dtype in ASR_DTYPES, f"asr.dtype: {asr.dtype!r} not in {'|'.join(sorted(ASR_DTYPES))}"
    )
    _check(not (asr.backend == "asr" and asr.model == ""), "asr.model: required when asr.backend = 'asr'")
    _check(asr.device in {"auto", "cpu", "cuda"}, f"asr.device: {asr.device!r} not in auto|cpu|cuda")
    _check(isinstance(asr.batch_size, int) and asr.batch_size >= 1, "asr.batch_size: must be an int >= 1")
    _check(isinstance(asr.generate_kwargs, dict), "asr.generate_kwargs: must be a table")
    _check(isinstance(asr.sample_rate, int) and asr.sample_rate > 0, "asr.sample_rate: must be a positive int")
    _check(
        isinstance(agent.guards, list) and all(isinstance(g, str) for g in agent.guards),
        "agent.guards: must be a list of str",
    )
    return Config(llm=llm, agent=agent, asr=asr)


def load_profile(name_or_path: str, repo_root: Path) -> EnvProfile:
    """Load an env profile by name (configs/env/<name>.toml) or by path."""
    path = Path(name_or_path)
    if path.suffix != ".toml":
        path = repo_root / "configs" / "env" / f"{name_or_path}.toml"
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return _build(EnvProfile, "env", raw, optional=frozenset({"hf_cache"}))
