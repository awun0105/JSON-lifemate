"""Các backend LLM (SPEC 4.8): FakeLLM cho test, VLLMBackend cho lần chạy thật.

Cả hai cùng giao diện `generate(messages, schema, *, sample_id, time_left_s)` trả `Generation` và `warmup()`.

Cách xây dựng:
- `FakeLLM` phát lại `label.tool_calls` của từng mẫu dưới dạng JSON chuẩn hóa, rồi `{"action":"done"}`; có thể
  chèn trước các chuỗi hỏng và mô phỏng thời gian bằng đồng hồ giả, nên test được cả nhánh hết giờ mà không cần GPU.
- `VLLMBackend` import vllm bên trong hàm khởi tạo (extra `gpu`) và nhận `engine_factory` để test thay engine giả.
  Đầu ra bị ràng buộc theo JSON schema (structured output), với lớp tương thích giữa các phiên bản vLLM.
- Khi có `time_left_s`, request được đưa vào engine rồi chạy từng bước (`step`) và bị hủy (`abort_request`) ngay khi
  chạm hạn, nên một mẫu không bao giờ vượt hạn chót. Engine thiếu API này thì quay về `LLM.chat` và ghi log lỗi.
- Prefix caching luôn bật; `batch_invariant` đặt biến môi trường của vLLM để kết quả không phụ thuộc kích thước lô.
"""

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

# Đầu ra mặc định khi hàng đợi của mẫu đã hết
_DONE = canonical_json({"action": "done"})


class FakeLLM:
    """Phát lại `label.tool_calls` của từng mẫu, sau đó trả `{"action":"done"}` mãi mãi.

    Các chuỗi thô trong `broken_outputs[sample_id]` được trả trước. Khi có `clock` (kiểu FakeClock, có `.advance`),
    mỗi lần gọi đẩy đồng hồ tới `delay_s` thay vì ngủ thật.
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
        # Mẫu chỉ có đầu ra hỏng (không có trong `samples`) vẫn có hàng đợi riêng
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
        """Đầu ra kế tiếp đã soạn sẵn cho `sample_id`; bước dài hơn `time_left_s` bị cắt tại đó (timed_out).

        Khi hết giờ, văn bản trả về rỗng và thời gian tiêu tốn bằng đúng `time_left_s`, giống backend thật bị hủy.
        """
        queue = self._queues.get(sample_id, [])
        i = self._cursor.get(sample_id, 0)
        text = queue[i] if i < len(queue) else _DONE
        self._cursor[sample_id] = i + 1
        timed_out = time_left_s is not None and self.delay_s > time_left_s
        spent = time_left_s if timed_out and time_left_s is not None else self.delay_s
        # Có đồng hồ giả thì đẩy đồng hồ; không có đồng hồ thì ngủ thật; đồng hồ không có `.advance` thì bỏ qua
        if spent > 0:
            advance = getattr(self.clock, "advance", None)
            if advance is not None:
                advance(spent)
            elif self.clock is None:
                time.sleep(spent)
        # Số token chỉ ước lượng thô bằng số từ
        prompt = " ".join(str(m.get("content") or "") for m in messages)
        return Generation(
            text="" if timed_out else text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=0 if timed_out else len(text.split()),
            seconds=spent,
            timed_out=timed_out,
        )

    def warmup(self) -> None:
        """Không có gì cần khởi động."""


_log = structlog.get_logger(__name__)

# Chế độ batch-invariant của vLLM chỉ chạy trên GPU từ Ampere (sm80) trở lên
_MIN_BATCH_INVARIANT_CC = (8, 0)
_WARMUP_SENTENCE = "Hôm nay trời đẹp, tôi muốn bật đèn phòng khách và đặt báo thức lúc bảy giờ sáng mai. "
_WARMUP_REPEATS = 200  # mỗi câu ~20 token -> khoảng 4K token prompt
_WARMUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"action": {"const": "done"}},
    "required": ["action"],
}


def _structured_kwargs(schema: dict[str, Any]) -> dict[str, Any]:
    """Tham số SamplingParams cho giải mã ràng buộc theo JSON schema, tương thích nhiều phiên bản vLLM.

    vLLM mới dùng `StructuredOutputsParams` (khóa `structured_outputs`); bản cũ dùng `GuidedDecodingParams`
    (khóa `guided_decoding`).
    """
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
    """Engine vLLM offline với đầu ra có cấu trúc theo JSON schema (SPEC 4.8).

    Giải mã tham lam (temperature 0, seed cố định). Hai cờ tự hạ cấp khi engine không hỗ trợ:
    `_use_template_kwargs` (truyền `enable_thinking=False` vào chat template) và `_abortable` (API step/abort).
    """

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
        self._abortable = True  # đặt False (và ghi log) khi engine thiếu API step/abort
        self._next_request = 0
        if cfg.batch_invariant:
            if compute_capability is not None and compute_capability < _MIN_BATCH_INVARIANT_CC:
                cc = compute_capability
                raise ConfigError(f"llm.batch_invariant needs compute capability >= 8.0, got {cc[0]}.{cc[1]}")
            # Phải đặt trước khi engine được tạo thì vLLM mới đọc được
            os.environ["VLLM_BATCH_INVARIANT"] = "1"
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        if engine_factory is None:
            import vllm  # type: ignore[import-not-found,unused-ignore]  # extra `gpu`, import muộn

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
        """SamplingParams tham lam, giới hạn `max_tokens`, ràng buộc đầu ra theo `schema`."""
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
        """Một lần sinh có ràng buộc; `sample_id` bị bỏ qua.

        Khi có `time_left_s`, request được chạy từng bước trên engine và bị hủy khi hết giờ (`timed_out=True`), nên
        một mẫu không bao giờ chạy quá hạn chót. Khi không có, hàm dùng `LLM.chat`.
        """
        params = self._sampling_params(schema)
        if time_left_s is not None and self._abortable:
            try:
                return self._generate_until(messages, params, time_left_s)
            except (AttributeError, TypeError) as exc:  # API engine khác với điều mã này giả định
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
                # vLLM cũ không nhận `chat_template_kwargs`: tắt hẳn và gọi lại không có tham số này
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
        """Áp chat template đúng như `LLM.chat` làm (bật generation prompt, tắt thinking), trả chuỗi prompt."""
        tok = self.engine.get_tokenizer()
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if self._use_template_kwargs:
            kwargs["enable_thinking"] = False
        return str(tok.apply_chat_template(messages, **kwargs))

    def _generate_until(self, messages: list[dict[str, Any]], params: Any, time_left_s: float) -> Generation:
        """Thêm một request vào engine và chạy từng bước đến khi xong hoặc quá `time_left_s`, khi đó hủy request.

        Hạn chót được kiểm tra giữa các bước, nên thời gian vượt tối đa là một bước giải mã. Khi bị hủy, văn bản trả
        về rỗng (JSON dở dang không dùng được) nhưng số token vẫn lấy từ đầu ra mới nhất.
        """
        engine = self.engine.llm_engine
        t0 = time.perf_counter()
        deadline = t0 + time_left_s
        self._next_request += 1
        # id request duy nhất trong vòng đời engine
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
            # Một bước có thể trả đầu ra của nhiều request; chỉ giữ đầu ra của request `rid`
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
        """Biên dịch kernel / CUDA graph trên một hội thoại tổng hợp dài ~4K token.

        Chạy `warmup_runs` lần qua đúng đường `generate` có hạn giờ (300 giây) để làm nóng cả nhánh step/abort;
        tổng thời gian lưu vào `warmup_seconds`.
        """
        messages = [
            {"role": "system", "content": _WARMUP_SENTENCE * _WARMUP_REPEATS},
            {"role": "user", "content": "Bật đèn phòng khách."},
        ]
        t0 = time.perf_counter()
        for _ in range(self.cfg.warmup_runs):
            self.generate(messages, _WARMUP_SCHEMA, sample_id="__warmup__", time_left_s=300.0)
        self.warmup_seconds = time.perf_counter() - t0
        _log.info("vllm_warmup_done", runs=self.cfg.warmup_runs, seconds=round(self.warmup_seconds, 3))
