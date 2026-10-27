"""LLM backends (SPEC 4.8): FakeLLM for tests, VLLMBackend for real runs."""

from __future__ import annotations

import importlib.metadata
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import structlog

from lifemate.core.config import ConfigError, LLMConfig
from lifemate.core.jsonutil import canonical_json
from lifemate.core.types import Generation, Sample

_DONE = canonical_json({"action": "done"})


class FakeLLM:
    """Replays each sample's `label.tool_calls`, then `{"action":"done"}` forever.

    `broken_outputs[sample_id]` raw strings are yielded first. With a `clock` (FakeClock-like, has `.advance`)
    each call advances it by `delay_s` instead of sleeping.
    """

    def __init__(
        self,
        samples: Iterable[Sample],
        *,
        delay_s: float = 0.0,
        broken_outputs: Mapping[str, Sequence[str]] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.delay_s = delay_s
        self.clock = clock
        self._queues: dict[str, list[str]] = {}
        broken = broken_outputs or {}
        for s in samples:
            calls = (s.label or {}).get("tool_calls") or []
            replies = [
                canonical_json({"action": "call", "name": c["name"], "arguments": c.get("arguments") or {}})
                for c in calls
            ]
            self._queues[s.id] = list(broken.get(s.id, ())) + replies
        for sid, raws in broken.items():
            self._queues.setdefault(sid, list(raws))
        self._cursor: dict[str, int] = {}

    def generate(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        *,
        sample_id: str,
        time_left_s: float | None = None,
    ) -> Generation:
        """Next scripted output for `sample_id`; a step longer than `time_left_s` is cut there (timed_out)."""
        queue = self._queues.get(sample_id, [])
        i = self._cursor.get(sample_id, 0)
        text = queue[i] if i < len(queue) else _DONE
        self._cursor[sample_id] = i + 1
        timed_out = time_left_s is not None and self.delay_s > time_left_s
        spent = time_left_s if timed_out and time_left_s is not None else self.delay_s
        if spent > 0:
            advance = getattr(self.clock, "advance", None)
            if advance is not None:
                advance(spent)
            elif self.clock is None:
                time.sleep(spent)
        prompt = " ".join(str(m.get("content") or "") for m in messages)
        return Generation(
            text="" if timed_out else text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=0 if timed_out else len(text.split()),
            seconds=spent,
            timed_out=timed_out,
        )

    def warmup(self) -> None:
        """Nothing to warm up."""


_log = structlog.get_logger(__name__)

_MIN_BATCH_INVARIANT_CC = (8, 0)
_WARMUP_SENTENCE = "Hôm nay trời đẹp, tôi muốn bật đèn phòng khách và đặt báo thức lúc bảy giờ sáng mai. "
_WARMUP_REPEATS = 200  # ~20 tokens each -> roughly 4K prompt tokens
_WARMUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"action": {"const": "done"}},
    "required": ["action"],
}


def _structured_kwargs(schema: dict[str, Any]) -> dict[str, Any]:
    """SamplingParams kwargs for JSON-schema constrained decoding, across vLLM versions."""
    try:
        from vllm.sampling_params import (
            StructuredOutputsParams,  # type: ignore[import-not-found,attr-defined,unused-ignore]
        )
    except ImportError:
        from vllm.sampling_params import (
            GuidedDecodingParams,  # type: ignore[import-not-found,attr-defined,unused-ignore]
        )

        return {"guided_decoding": GuidedDecodingParams(json=schema)}
    return {"structured_outputs": StructuredOutputsParams(json=schema)}


class VLLMBackend:
    """vLLM offline engine with JSON-schema structured output (SPEC 4.8)."""

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        dtype: str,
        hf_cache: str | None = None,
        hf_token: str | None = None,
        engine_factory: Callable[..., Any] | None = None,
        compute_capability: tuple[int, int] | None = None,
    ) -> None:
        self.cfg = cfg
        self.warmup_seconds: float | None = None
        self._use_template_kwargs = True
        self._abortable = True  # set False (and logged) if the engine lacks the step/abort API
        self._next_request = 0
        if cfg.batch_invariant:
            if compute_capability is not None and compute_capability < _MIN_BATCH_INVARIANT_CC:
                cc = compute_capability
                raise ConfigError(f"llm.batch_invariant needs compute capability >= 8.0, got {cc[0]}.{cc[1]}")
            os.environ["VLLM_BATCH_INVARIANT"] = "1"
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        if engine_factory is None:
            import vllm  # type: ignore[import-not-found,unused-ignore]  # extra `gpu`

            engine_factory = vllm.LLM
        try:
            self.version = importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError:
            self.version = "unknown"
        _log.info(
            "vllm_engine_init",
            model=cfg.model,
            revision=cfg.revision or None,
            dtype=dtype,
            vllm_version=self.version,
            batch_invariant=cfg.batch_invariant,
        )
        self.engine = engine_factory(
            model=cfg.model,
            revision=cfg.revision or None,
            dtype=dtype,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            seed=cfg.seed,
            enable_prefix_caching=True,
            download_dir=hf_cache,
        )

    def _sampling_params(self, schema: dict[str, Any]) -> Any:
        from vllm import SamplingParams  # type: ignore[import-not-found,unused-ignore]

        return SamplingParams(
            temperature=0,
            max_tokens=self.cfg.max_tokens,
            seed=self.cfg.seed,
            **_structured_kwargs(schema),
        )

    def generate(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        *,
        sample_id: str,
        time_left_s: float | None = None,
    ) -> Generation:
        """One constrained completion; `sample_id` is ignored.

        With `time_left_s`, the request is driven step by step on the engine and aborted when the time is up
        (`timed_out=True`), so a sample never runs past its deadline. Without it, `LLM.chat` is used.
        """
        params = self._sampling_params(schema)
        if time_left_s is not None and self._abortable:
            try:
                return self._generate_until(messages, params, time_left_s)
            except (AttributeError, TypeError) as exc:  # engine API differs from what this code expects
                _log.error("vllm_abortable_generate_unsupported", error=str(exc))
                self._abortable = False
        t0 = time.perf_counter()
        if self._use_template_kwargs:
            try:
                outputs = self.engine.chat(
                    messages,
                    sampling_params=params,
                    use_tqdm=False,
                    chat_template_kwargs={"enable_thinking": False},
                )
            except TypeError as exc:
                _log.warning("chat_template_kwargs_unsupported", error=str(exc))
                self._use_template_kwargs = False
                t0 = time.perf_counter()
                outputs = self.engine.chat(messages, sampling_params=params, use_tqdm=False)
        else:
            outputs = self.engine.chat(messages, sampling_params=params, use_tqdm=False)
        seconds = time.perf_counter() - t0
        out = outputs[0]
        completion = out.outputs[0]
        return Generation(
            text=completion.text,
            prompt_tokens=len(out.prompt_token_ids or ()),
            completion_tokens=len(completion.token_ids or ()),
            seconds=seconds,
        )

    def _render(self, messages: list[dict[str, Any]]) -> str:
        """Chat template exactly as `LLM.chat` applies it (generation prompt on, thinking off)."""
        tok = self.engine.get_tokenizer()
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if self._use_template_kwargs:
            kwargs["enable_thinking"] = False
        return str(tok.apply_chat_template(messages, **kwargs))

    def _generate_until(self, messages: list[dict[str, Any]], params: Any, time_left_s: float) -> Generation:
        """Add one request to the engine and step it until it finishes or `time_left_s` passes, then abort it."""
        engine = self.engine.llm_engine
        t0 = time.perf_counter()
        deadline = t0 + time_left_s
        self._next_request += 1
        rid = f"lm-{self._next_request}"
        engine.add_request(rid, self._render(messages), params)
        final: Any = None
        latest: Any = None
        timed_out = False
        while engine.has_unfinished_requests():
            if time.perf_counter() >= deadline:
                engine.abort_request([rid])
                timed_out = True
                break
            for out in engine.step():
                if out.request_id == rid:
                    latest = out
                    if out.finished:
                        final = out
        seconds = time.perf_counter() - t0
        out = final or latest
        completion = out.outputs[0] if out is not None and out.outputs else None
        if timed_out:
            _log.warning("llm_step_aborted_at_deadline", seconds=round(seconds, 3))
        return Generation(
            text="" if timed_out or completion is None else completion.text,
            prompt_tokens=len(out.prompt_token_ids or ()) if out is not None else 0,
            completion_tokens=len(completion.token_ids or ()) if completion is not None else 0,
            seconds=seconds,
            timed_out=timed_out,
        )

    def warmup(self) -> None:
        """Compile kernels / CUDA graphs on a ~4K-token synthetic conversation."""
        messages = [
            {"role": "system", "content": _WARMUP_SENTENCE * _WARMUP_REPEATS},
            {"role": "user", "content": "Bật đèn phòng khách."},
        ]
        t0 = time.perf_counter()
        for _ in range(self.cfg.warmup_runs):
            self.generate(messages, _WARMUP_SCHEMA, sample_id="__warmup__", time_left_s=300.0)
        self.warmup_seconds = time.perf_counter() - t0
        _log.info("vllm_warmup_done", runs=self.cfg.warmup_runs, seconds=round(self.warmup_seconds, 3))
