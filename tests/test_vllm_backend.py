"""Kiểm tra VLLMBackend mà không cần vLLM hay GPU thật.

Nội dung: biến môi trường, tham số sampling và structured output, fallback API cũ, warmup, và cơ chế hủy request khi
hết thời gian theo từng mẫu. Cách kiểm tra: module `vllm` giả cài vào `sys.modules` qua fixture `fake_vllm`, engine giả
(`FakeEngine`, `FakeLLMEngine`, `FakeTokenizer`) truyền qua `engine_factory`. Không cần dữ liệu cuộc thi hay tokenizer.
"""

import os
import sys
import types
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest

from lifemate.backends.llm import VLLMBackend, _structured_kwargs
from lifemate.core.config import ConfigError, LLMConfig

CFG = LLMConfig(
    backend="vllm",
    model="org/model",
    revision="abc123",
    dtype="float16",
    max_model_len=16384,
    gpu_memory_utilization=0.9,
    seed=7,
    max_tokens=512,
    batch_invariant=False,
    warmup_runs=2,
)
SCHEMA = {"type": "object"}


class _Params:
    """SamplingParams giả, chỉ lưu lại các đối số."""

    def __init__(self, **kw: Any) -> None:
        self.kw = kw


class _Structured:
    """StructuredOutputsParams giả (API vLLM mới), lưu schema JSON."""

    def __init__(self, *, json: Any) -> None:
        self.json = json


class _Guided(_Structured):
    """GuidedDecodingParams giả (API vLLM cũ)."""

    pass


def _fake_vllm(*, new_api: bool) -> tuple[types.ModuleType, types.ModuleType]:
    """Dựng module vllm giả, dùng API structured output mới hoặc API guided decoding cũ."""
    vllm = types.ModuleType("vllm")
    sp = types.ModuleType("vllm.sampling_params")
    vllm.SamplingParams = _Params  # type: ignore[attr-defined]
    if new_api:
        sp.StructuredOutputsParams = _Structured  # type: ignore[attr-defined]
    else:
        sp.GuidedDecodingParams = _Guided  # type: ignore[attr-defined]
    vllm.sampling_params = sp  # type: ignore[attr-defined]
    return vllm, sp


@pytest.fixture
def fake_vllm(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Cài vllm giả vào sys.modules và xóa VLLM_BATCH_INVARIANT, HF_TOKEN trong lúc chạy test."""
    vllm, sp = _fake_vllm(new_api=True)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sp)
    for var in ("VLLM_BATCH_INVARIANT", "HF_TOKEN"):
        monkeypatch.setenv(var, "")  # ghi lại trạng thái gốc để teardown khôi phục
        monkeypatch.delenv(var)
    yield


def _output(text: str, prompt_ids: list[int], token_ids: list[int]) -> Any:
    """Tạo đối tượng đầu ra kiểu vLLM với văn bản và danh sách token id."""
    comp = types.SimpleNamespace(text=text, token_ids=token_ids)
    return types.SimpleNamespace(prompt_token_ids=prompt_ids, outputs=[comp])


class FakeTokenizer:
    """Tokenizer giả, ghi lại đối số của apply_chat_template."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def apply_chat_template(self, messages: list[dict[str, Any]], **kw: Any) -> str:
        """Ghép nội dung các message bằng dấu | và lưu đối số."""
        self.kwargs = kw
        return "|".join(str(m.get("content")) for m in messages)


class FakeLLMEngine:
    """Engine chạy theo bước: mỗi step() mất `step_s` giây; request xong sau `n_steps` bước."""

    def __init__(self, *, step_s: float = 0.0, n_steps: int = 2) -> None:
        self.step_s, self.n_steps = step_s, n_steps
        self.added: list[tuple[str, str, Any]] = []
        self.aborted: list[str] = []
        self._left = 0

    def add_request(self, rid: str, prompt: str, params: Any) -> None:
        self.added.append((rid, prompt, params))
        self._left = self.n_steps

    def has_unfinished_requests(self) -> bool:
        return self._left > 0

    def abort_request(self, rids: list[str]) -> None:
        self.aborted += rids
        self._left = 0

    def step(self) -> list[Any]:
        import time as _t

        _t.sleep(self.step_s)
        self._left -= 1
        out = _output('{"action":"done"}', [1, 2, 3], [9, 9])
        out.request_id, out.finished = self.added[-1][0], self._left == 0
        return [out]


class FakeEngine:
    """LLM giả của vLLM, ghi lại đối số khởi tạo và đối số mỗi lần chat."""

    def __init__(self, *, reject_template_kwargs: bool = False, **init_kw: Any) -> None:
        self.init_kw = init_kw
        self.reject = reject_template_kwargs
        self.calls: list[dict[str, Any]] = []
        self.tokenizer = FakeTokenizer()
        self.llm_engine = FakeLLMEngine()

    def get_tokenizer(self) -> FakeTokenizer:
        return self.tokenizer

    def chat(self, messages: list[dict[str, Any]], **kw: Any) -> list[Any]:
        """Trả đầu ra cố định; khi `reject` bật thì từ chối chat_template_kwargs như vLLM cũ."""
        self.calls.append(kw)
        if self.reject and "chat_template_kwargs" in kw:
            raise TypeError("unexpected keyword enable_thinking")
        return [_output('{"action":"done"}', [1, 2, 3, 4], [9, 9])]


def _backend(cfg: LLMConfig = CFG, **kw: Any) -> tuple[VLLMBackend, FakeEngine]:
    """Tạo VLLMBackend với FakeEngine và trả về cả backend lẫn engine."""
    holder: dict[str, FakeEngine] = {}
    reject = kw.pop("reject", False)

    def factory(**init_kw: Any) -> FakeEngine:
        holder["e"] = FakeEngine(reject_template_kwargs=reject, **init_kw)
        return holder["e"]

    b = VLLMBackend(cfg, dtype="float16", engine_factory=factory, **kw)
    return b, holder["e"]


def test_env_var_only_when_batch_invariant(fake_vllm: None) -> None:
    """Bảo vệ: VLLM_BATCH_INVARIANT chỉ đặt khi bật batch_invariant; HF token qua biến môi trường, không vào engine."""
    b, eng = _backend(hf_cache="/c", hf_token="secret")
    assert "VLLM_BATCH_INVARIANT" not in os.environ
    assert os.environ["HF_TOKEN"] == "secret"
    assert eng.init_kw["model"] == "org/model" and eng.init_kw["revision"] == "abc123"
    assert eng.init_kw["download_dir"] == "/c" and eng.init_kw["enable_prefix_caching"] is True
    assert "hf_token" not in eng.init_kw and isinstance(b.version, str)
    _backend(replace(CFG, batch_invariant=True), compute_capability=(8, 0))
    assert os.environ["VLLM_BATCH_INVARIANT"] == "1"


def test_batch_invariant_on_sm75_raises(fake_vllm: None) -> None:
    """Bảo vệ: batch_invariant trên GPU sm75 báo ConfigError và không đặt biến môi trường."""
    with pytest.raises(ConfigError):
        _backend(replace(CFG, batch_invariant=True), compute_capability=(7, 5))
    assert "VLLM_BATCH_INVARIANT" not in os.environ


def test_generate_counts_and_params(fake_vllm: None) -> None:
    """Bảo vệ: đếm token đúng, tham số sampling tất định và schema đi qua structured_outputs."""
    b, eng = _backend()
    g = b.generate([{"role": "user", "content": "hi"}], SCHEMA, sample_id="x")
    assert g.text == '{"action":"done"}' and g.prompt_tokens == 4 and g.completion_tokens == 2
    assert g.seconds >= 0
    kw = eng.calls[0]
    assert kw["use_tqdm"] is False and kw["chat_template_kwargs"] == {"enable_thinking": False}
    p = kw["sampling_params"].kw
    assert p["temperature"] == 0 and p["max_tokens"] == 512 and p["seed"] == 7
    assert isinstance(p["structured_outputs"], _Structured) and p["structured_outputs"].json == SCHEMA


def test_structured_kwargs_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bảo vệ: vLLM cũ không có StructuredOutputsParams thì dùng guided_decoding."""
    vllm, sp = _fake_vllm(new_api=False)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sp)
    kw = _structured_kwargs(SCHEMA)
    assert isinstance(kw["guided_decoding"], _Guided) and kw["guided_decoding"].json == SCHEMA


def test_template_kwargs_typeerror_fallback_sticks(fake_vllm: None) -> None:
    """Bảo vệ: vLLM từ chối chat_template_kwargs thì bỏ đối số đó và giữ nguyên cho các lần sau."""
    b, eng = _backend(reject=True)
    b.generate([], SCHEMA, sample_id="a")
    b.generate([], SCHEMA, sample_id="b")
    assert ["chat_template_kwargs" in c for c in eng.calls] == [True, False, False]


def test_warmup_runs(fake_vllm: None) -> None:
    """Bảo vệ: warmup chạy đúng số lần cấu hình và ghi lại thời gian."""
    b, eng = _backend(replace(CFG, warmup_runs=3))
    b.warmup()
    # warmup đi cùng đường engine hủy được như mẫu thật, nên biên dịch đúng phần mẫu thật sẽ chạy
    added = eng.llm_engine.added
    assert len(added) == 3 and b.warmup_seconds is not None and eng.calls == []
    assert added[0][2].kw["structured_outputs"].json["required"] == ["action"]


def test_time_left_drives_engine_and_finishes(fake_vllm: None) -> None:
    """Có giới hạn thời gian thì request chạy từng bước trên engine (hủy được) và render giống LLM.chat."""
    b, eng = _backend()
    g = b.generate([{"role": "user", "content": "hi"}], SCHEMA, sample_id="x", time_left_s=10.0)
    assert not g.timed_out and g.text == '{"action":"done"}' and g.prompt_tokens == 3 and g.completion_tokens == 2
    assert eng.llm_engine.aborted == [] and eng.calls == []
    assert eng.tokenizer.kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    assert eng.llm_engine.added[0][1] == "hi"


def test_time_left_aborts_running_step(fake_vllm: None) -> None:
    """Bắt lỗi: bước chạy quá hạn của mẫu. Request bị hủy và không trả văn bản."""
    b, eng = _backend()
    eng.llm_engine = FakeLLMEngine(step_s=0.05, n_steps=1000)
    g = b.generate([], SCHEMA, sample_id="x", time_left_s=0.2)
    assert g.timed_out and g.text == "" and 0.2 <= g.seconds < 1.0
    assert eng.llm_engine.aborted == [eng.llm_engine.added[0][0]]


def test_engine_without_step_api_falls_back_to_chat(fake_vllm: None) -> None:
    """Engine thiếu add_request/step thì quay về LLM.chat (có ghi log) và giữ cách đó."""
    b, eng = _backend()
    eng.llm_engine = object()
    g = b.generate([], SCHEMA, sample_id="x", time_left_s=5.0)
    assert g.text == '{"action":"done"}' and len(eng.calls) == 1 and b._abortable is False
