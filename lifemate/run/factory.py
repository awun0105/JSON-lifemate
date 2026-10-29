"""Lắp ráp mọi đối tượng của một lần chạy từ `Config`; đây là nơi duy nhất làm việc này (SPEC 4.0).

Nội dung chính:
- `build`: tạo agent và các thành phần đi kèm, trả về `Built`.
- `Built`: gói mọi đối tượng đã lắp ráp.
- `LLM_BACKENDS`, `TRANSCRIBERS`: registry chọn backend LLM và transcriber theo tên.

Cách hoạt động:
- Dùng mẫu Factory + registry: backend được chọn theo tên trong `[llm].backend` và `[asr].backend`; tên lạ gây
  `ConfigError`.
- `build` tạo lần lượt: dtype, chỉ mục catalog tool, bộ đếm token, `PromptBuilder` (từ file system prompt, kèm
  danh sách app held-out), chuỗi guard. Sau đó bọc LLM và transcriber trong lớp đo thời gian (`TimedLLM`,
  `TimedTranscriber`) rồi tạo `Agent`.
- Kết quả trả về trong `Built`, kèm SHA-256 của template prompt để truy vết run.
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
    """Mọi thứ một lần chạy cần, đã lắp ráp xong.

    Thuộc tính:
        agent: agent sẵn sàng chạy.
        loader: bộ nạp `zaic_tools` của split.
        index: chỉ mục catalog tool.
        counter: bộ đếm token.
        llm: backend LLM đã bọc đo thời gian.
        transcriber: transcriber đã bọc đo thời gian.
        dtype: dtype đã chọn cho LLM.
        prompt_sha256: SHA-256 của file template system prompt.
    """

    agent: Agent
    loader: ZaicLoader
    index: CatalogIndex
    counter: TokenCounter
    llm: TimedLLM
    transcriber: TimedTranscriber
    dtype: str
    prompt_sha256: str


def _fake_llm(config: Config, profile: EnvProfile, samples: Sequence[Sample], env_info: EnvInfo, dtype: str) -> Any:
    """Tạo `FakeLLM`; cần split có nhãn vì `FakeLLM` phát lại lời gọi tool gốc."""
    if any(s.label is None for s in samples):
        raise ConfigError("llm.backend = 'fake' needs a labelled split")
    return FakeLLM(samples)


def _vllm(config: Config, profile: EnvProfile, samples: Sequence[Sample], env_info: EnvInfo, dtype: str) -> Any:
    """Tạo `VLLMBackend` với cache, token HF và compute capability lấy từ profile và thông tin máy."""
    return VLLMBackend(
        config.llm,
        dtype=dtype,
        hf_cache=profile.hf_cache,
        hf_token=profile.resolve_hf_token(),
        compute_capability=env_info.compute_capability,
    )


# Registry theo tên: `[llm].backend` và `[asr].backend` -> hàm tạo; hàm chỉ chạy khi tên đó được chọn.
LLM_BACKENDS: dict[str, Callable[[Config, EnvProfile, Sequence[Sample], EnvInfo, str], LLMBackend]] = {
    "vllm": _vllm,
    "fake": _fake_llm,
}
TRANSCRIBERS: dict[str, Callable[[Config, Path, str], Transcriber]] = {
    "gold": lambda cfg, root, split: GoldTranscriber(),
    "asr": lambda cfg, root, split: ASRTranscriber(cfg.asr, data_root=root, split=split),
}


def _counter(config: Config) -> TokenCounter:
    """Tạo bộ đếm token: tokenizer HF của model; bộ đếm đơn giản khi backend là `fake` hoặc thiếu `transformers`."""
    if config.llm.backend == "fake":
        return SimpleTokenCounter()
    try:
        return HFTokenCounter(config.llm.model, config.llm.revision or None)
    except ImportError:
        log.warning("token_counter_fallback", reason="transformers not installed", counter="SimpleTokenCounter")
        return SimpleTokenCounter()


def _held_out(repo_root: Path) -> frozenset[str]:
    """Đọc tập app held-out từ `configs/splits/meta.json`; trả về tập rỗng khi file không tồn tại."""
    meta = repo_root / "configs" / "splits" / "meta.json"
    if not meta.is_file():
        return frozenset()
    return frozenset(json.loads(meta.read_text(encoding="utf-8")).get("held_out_apps", []))


def _template(config: Config, repo_root: Path) -> str:
    """Đọc file template system prompt khai báo ở `[agent].system_prompt` (đường dẫn tính từ gốc repo)."""
    return (repo_root / config.agent.system_prompt).read_text(encoding="utf-8")


def prompt_sha256(config: Config, repo_root: Path) -> str:
    """Trả về SHA-256 của file template system prompt; tính được trước khi nạp model, để CLI kiểm tra `--resume`."""
    return hashlib.sha256(_template(config, repo_root).encode("utf-8")).hexdigest()


def build(
    config: Config,
    profile: EnvProfile,
    *,
    repo_root: Path,
    split: str,
    env_info: EnvInfo,
    samples: Sequence[Sample] | None = None,
    observer: AgentObserver | None = None,
    transcriber: Transcriber | None = None,
) -> Built:
    """Lắp ráp LLM, transcriber, prompt, guard và agent cho `split` trong `profile.data_root`.

    Tên backend được kiểm tra trước khi tạo bất kỳ đối tượng nặng nào, để lỗi config lộ ra ngay.

    Tham số:
        config: config chạy.
        profile: profile môi trường; đường dẫn đã là tuyệt đối.
        repo_root: thư mục gốc repo.
        split: tên split.
        env_info: thông tin máy, dùng để chọn dtype.
        samples: các mẫu sẽ chạy. Với backend `fake` mà không truyền, cả split được nạp để lấy nhãn.
        observer: observer của agent; `None` là không ghi gì thêm.
        transcriber: transcriber có sẵn (ví dụ đọc lại chữ đã lưu); `None` thì tạo theo `config.asr.backend`.

    Trả về:
        `Built` chứa mọi đối tượng đã lắp ráp.

    Lỗi:
        ConfigError: khi tên backend không có trong registry, hoặc backend `fake` gặp mẫu không có nhãn.
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
    template = _template(config, repo_root)
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
    timed = TimedTranscriber(transcriber if transcriber is not None else asr_factory(config, root, split))
    agent = Agent(llm, timed, prompts, guards, index, config.agent, config.llm, observer=observer)
    return Built(
        agent=agent,
        loader=ZaicLoader(root, split),
        index=index,
        counter=counter,
        llm=llm,
        transcriber=timed,
        dtype=dtype,
        prompt_sha256=prompt_sha256(config, repo_root),
    )
