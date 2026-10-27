"""Nơi duy nhất lắp ráp các đối tượng từ Config (SPEC 4.0, mẫu Factory + registry).

Cách xây dựng: backend LLM và transcriber được chọn theo tên qua hai registry `LLM_BACKENDS` và `TRANSCRIBERS`
(tên lạ gây ConfigError). `build` tạo lần lượt: dtype, chỉ mục catalog tool, bộ đếm token, `PromptBuilder` (từ file
system prompt, kèm danh sách app giữ lại khi train), chuỗi guard, rồi bọc LLM và transcriber trong lớp đo thời gian
(`TimedLLM`, `TimedTranscriber`) trước khi tạo `Agent`. Kết quả trả về trong `Built`, kèm SHA-256 của template prompt
để truy vết run.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from lifemate.agent.agent import Agent
from lifemate.agent.guards import build_guard_chain
from lifemate.agent.prompt import PromptBuilder
from lifemate.agent.tooldocs import CatalogIndex
from lifemate.backends.asr import ASRTranscriber, GoldTranscriber
from lifemate.backends.llm import FakeLLM, VLLMBackend
from lifemate.core.config import Config, ConfigError, EnvProfile
from lifemate.core.envinfo import EnvInfo, resolve_dtype
from lifemate.core.tokens import HFTokenCounter, SimpleTokenCounter
from lifemate.core.types import AgentObserver, LLMBackend, Sample, TokenCounter, Transcriber
from lifemate.io.data import load_split
from lifemate.io.zaic import ZaicLoader, load_catalog
from lifemate.run.timing import TimedLLM, TimedTranscriber

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Built:
    """Mọi thứ một lần chạy cần, đã lắp ráp xong."""

    agent: Agent
    loader: ZaicLoader
    index: CatalogIndex
    counter: TokenCounter
    llm: TimedLLM
    transcriber: TimedTranscriber
    dtype: str
    prompt_sha256: str


def _fake_llm(config: Config, profile: EnvProfile, samples: Sequence[Sample], env_info: EnvInfo, dtype: str) -> Any:
    """Tạo FakeLLM; cần split có nhãn vì FakeLLM phát lại lời gọi tool gốc."""
    if any(s.label is None for s in samples):
        raise ConfigError("llm.backend = 'fake' needs a labelled split")
    return FakeLLM(samples)


def _vllm(config: Config, profile: EnvProfile, samples: Sequence[Sample], env_info: EnvInfo, dtype: str) -> Any:
    """Tạo VLLMBackend với cache, token HF và compute capability lấy từ profile và thông tin môi trường."""
    return VLLMBackend(
        config.llm,
        dtype=dtype,
        hf_cache=profile.hf_cache,
        hf_token=profile.resolve_hf_token(),
        compute_capability=env_info.compute_capability,
    )


# Registry theo tên: `[llm].backend` và `[asr].backend` -> hàm tạo; hàm chỉ chạy khi tên được chọn
LLM_BACKENDS: dict[str, Callable[[Config, EnvProfile, Sequence[Sample], EnvInfo, str], LLMBackend]] = {
    "vllm": _vllm,
    "fake": _fake_llm,
}
TRANSCRIBERS: dict[str, Callable[[Config, Path, str], Transcriber]] = {
    "gold": lambda cfg, root, split: GoldTranscriber(),
    "asr": lambda cfg, root, split: ASRTranscriber(cfg.asr, data_root=root, split=split),
}


def _counter(config: Config) -> TokenCounter:
    """Bộ đếm token: tokenizer HF của mô hình; đếm đơn giản với backend fake hoặc khi thiếu transformers."""
    if config.llm.backend == "fake":
        return SimpleTokenCounter()
    try:
        return HFTokenCounter(config.llm.model, config.llm.revision or None)
    except ImportError:
        log.warning("token_counter_fallback", reason="transformers not installed", counter="SimpleTokenCounter")
        return SimpleTokenCounter()


def _held_out(repo_root: Path) -> frozenset[str]:
    """Tập app giữ lại (held-out) đọc từ `configs/splits/meta.json`; rỗng khi file không tồn tại."""
    meta = repo_root / "configs" / "splits" / "meta.json"
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
    """Lắp ráp LLM, transcriber, prompt, guard và agent cho `split` dưới `profile.data_root`.

    Kiểm tra tên backend trước khi tạo bất kỳ đối tượng nặng nào, để lỗi cấu hình hiện ra ngay. Với backend fake mà
    không truyền `samples`, toàn bộ split được nạp để lấy nhãn.
    """
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
