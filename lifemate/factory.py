"""The only place that wires objects from Config (SPEC 4.0, Factory + registry)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from lifemate.agent import Agent
from lifemate.asr import ASRTranscriber, GoldTranscriber
from lifemate.config import Config, ConfigError, EnvProfile
from lifemate.data import load_split
from lifemate.envinfo import EnvInfo, resolve_dtype
from lifemate.guards import build_guard_chain
from lifemate.llm import FakeLLM, VLLMBackend
from lifemate.prompt import PromptBuilder
from lifemate.timing import TimedLLM, TimedTranscriber
from lifemate.tokens import HFTokenCounter, SimpleTokenCounter
from lifemate.tooldocs import CatalogIndex
from lifemate.types import AgentObserver, LLMBackend, Sample, TokenCounter, Transcriber
from lifemate.zaic import ZaicLoader, load_catalog

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Built:
    """Everything a run needs, wired."""

    agent: Agent
    loader: ZaicLoader
    index: CatalogIndex
    counter: TokenCounter
    llm: TimedLLM
    transcriber: TimedTranscriber
    dtype: str
    prompt_sha256: str


def _fake_llm(config: Config, profile: EnvProfile, samples: Sequence[Sample], env_info: EnvInfo, dtype: str) -> Any:
    if any(s.label is None for s in samples):
        raise ConfigError("llm.backend = 'fake' needs a labelled split")
    return FakeLLM(samples)


def _vllm(config: Config, profile: EnvProfile, samples: Sequence[Sample], env_info: EnvInfo, dtype: str) -> Any:
    return VLLMBackend(
        config.llm,
        dtype=dtype,
        hf_cache=profile.hf_cache,
        hf_token=profile.resolve_hf_token(),
        compute_capability=env_info.compute_capability,
    )


LLM_BACKENDS: dict[str, Callable[[Config, EnvProfile, Sequence[Sample], EnvInfo, str], LLMBackend]] = {
    "vllm": _vllm,
    "fake": _fake_llm,
}
TRANSCRIBERS: dict[str, Callable[[Config, Path, str], Transcriber]] = {
    "gold": lambda cfg, root, split: GoldTranscriber(),
    "asr": lambda cfg, root, split: ASRTranscriber(cfg.asr, data_root=root, split=split),
}


def _counter(config: Config) -> TokenCounter:
    if config.llm.backend == "fake":
        return SimpleTokenCounter()
    try:
        return HFTokenCounter(config.llm.model, config.llm.revision or None)
    except ImportError:
        log.warning("token_counter_fallback", reason="transformers not installed", counter="SimpleTokenCounter")
        return SimpleTokenCounter()


def _held_out(repo_root: Path) -> frozenset[str]:
    meta = repo_root / "splits" / "meta.json"
    if not meta.is_file():
        return frozenset()
    return frozenset(json.loads(meta.read_text(encoding="utf-8")).get("held_out_apps", []))


def build(
    config: Config,
    profile: EnvProfile,
    *,
    repo_root: Path,
    split: str,
    env_info: EnvInfo,
    samples: Sequence[Sample] | None = None,
    observer: AgentObserver | None = None,
) -> Built:
    """Wire LLM, transcriber, prompts, guards and agent for `split` under `profile.data_root`."""
    llm_factory = LLM_BACKENDS.get(config.llm.backend)
    if llm_factory is None:
        raise ConfigError(f"llm.backend: unknown backend {config.llm.backend!r}")
    asr_factory = TRANSCRIBERS.get(config.asr.backend)
    if asr_factory is None:
        raise ConfigError(f"asr.backend: unknown backend {config.asr.backend!r}")
    root = Path(profile.data_root)
    dtype = resolve_dtype(config.llm.dtype, env_info)
    index = CatalogIndex(load_catalog(root, split))
    counter = _counter(config)
    template_path = repo_root / config.agent.system_prompt
    template = template_path.read_text(encoding="utf-8")
    prompts = PromptBuilder(
        template,
        index,
        counter,
        max_model_len=config.llm.max_model_len,
        max_tokens=config.llm.max_tokens,
        max_result_chars=config.agent.max_result_chars,
        held_out_apps=_held_out(repo_root),
    )
    guards = build_guard_chain(config.agent.guards)
    fake_samples = samples
    if config.llm.backend == "fake" and fake_samples is None:
        fake_samples = load_split(root, split)
    llm = TimedLLM(llm_factory(config, profile, fake_samples or (), env_info, dtype))
    transcriber = TimedTranscriber(asr_factory(config, root, split))
    agent = Agent(llm, transcriber, prompts, guards, index, config.agent, config.llm, observer=observer)
    return Built(
        agent=agent,
        loader=ZaicLoader(root, split),
        index=index,
        counter=counter,
        llm=llm,
        transcriber=transcriber,
        dtype=dtype,
        prompt_sha256=hashlib.sha256(template.encode("utf-8")).hexdigest(),
    )
