import os
import sys
import types
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest

from lifemate.config import ConfigError, LLMConfig
from lifemate.llm import VLLMBackend, _structured_kwargs

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
    def __init__(self, **kw: Any) -> None:
        self.kw = kw


class _Structured:
    def __init__(self, *, json: Any) -> None:
        self.json = json


class _Guided(_Structured):
    pass


def _fake_vllm(*, new_api: bool) -> tuple[types.ModuleType, types.ModuleType]:
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
    vllm, sp = _fake_vllm(new_api=True)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sp)
    for var in ("VLLM_BATCH_INVARIANT", "HF_TOKEN"):
        monkeypatch.setenv(var, "")  # records original state so teardown restores it
        monkeypatch.delenv(var)
    yield


def _output(text: str, prompt_ids: list[int], token_ids: list[int]) -> Any:
    comp = types.SimpleNamespace(text=text, token_ids=token_ids)
    return types.SimpleNamespace(prompt_token_ids=prompt_ids, outputs=[comp])


class FakeEngine:
    def __init__(self, *, reject_template_kwargs: bool = False, **init_kw: Any) -> None:
        self.init_kw = init_kw
        self.reject = reject_template_kwargs
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, Any]], **kw: Any) -> list[Any]:
        self.calls.append(kw)
        if self.reject and "chat_template_kwargs" in kw:
            raise TypeError("unexpected keyword enable_thinking")
        return [_output('{"action":"done"}', [1, 2, 3, 4], [9, 9])]


def _backend(cfg: LLMConfig = CFG, **kw: Any) -> tuple[VLLMBackend, FakeEngine]:
    holder: dict[str, FakeEngine] = {}
    reject = kw.pop("reject", False)

    def factory(**init_kw: Any) -> FakeEngine:
        holder["e"] = FakeEngine(reject_template_kwargs=reject, **init_kw)
        return holder["e"]

    b = VLLMBackend(cfg, dtype="float16", engine_factory=factory, **kw)
    return b, holder["e"]


def test_env_var_only_when_batch_invariant(fake_vllm: None) -> None:
    b, eng = _backend(hf_cache="/c", hf_token="secret")
    assert "VLLM_BATCH_INVARIANT" not in os.environ
    assert os.environ["HF_TOKEN"] == "secret"
    assert eng.init_kw["model"] == "org/model" and eng.init_kw["revision"] == "abc123"
    assert eng.init_kw["download_dir"] == "/c" and eng.init_kw["enable_prefix_caching"] is True
    assert "hf_token" not in eng.init_kw and isinstance(b.version, str)
    _backend(replace(CFG, batch_invariant=True), compute_capability=(8, 0))
    assert os.environ["VLLM_BATCH_INVARIANT"] == "1"


def test_batch_invariant_on_sm75_raises(fake_vllm: None) -> None:
    with pytest.raises(ConfigError):
        _backend(replace(CFG, batch_invariant=True), compute_capability=(7, 5))
    assert "VLLM_BATCH_INVARIANT" not in os.environ


def test_generate_counts_and_params(fake_vllm: None) -> None:
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
    vllm, sp = _fake_vllm(new_api=False)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sp)
    kw = _structured_kwargs(SCHEMA)
    assert isinstance(kw["guided_decoding"], _Guided) and kw["guided_decoding"].json == SCHEMA


def test_template_kwargs_typeerror_fallback_sticks(fake_vllm: None) -> None:
    b, eng = _backend(reject=True)
    b.generate([], SCHEMA, sample_id="a")
    b.generate([], SCHEMA, sample_id="b")
    assert ["chat_template_kwargs" in c for c in eng.calls] == [True, False, False]


def test_warmup_runs(fake_vllm: None) -> None:
    b, eng = _backend(replace(CFG, warmup_runs=3))
    b.warmup()
    assert len(eng.calls) == 3 and b.warmup_seconds is not None
    assert eng.calls[0]["sampling_params"].kw["structured_outputs"].json["required"] == ["action"]
