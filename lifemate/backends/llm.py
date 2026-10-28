"""Các backend LLM: `FakeLLM` cho test và `VLLMBackend` cho lần chạy thật (SPEC 4.8).

Nội dung chính:
- `FakeLLM`: phát lại đáp án có sẵn của mẫu; không cần GPU.
- `VLLMBackend`: chạy model thật bằng vLLM, đầu ra bị ràng buộc theo JSON schema.

Cách hoạt động:
- Cả hai có cùng giao diện: `generate(messages, schema, *, sample_id, time_left_s)` trả về `Generation`, và
  `warmup()`.
- `FakeLLM` phát lại `label.tool_calls` của từng mẫu dưới dạng JSON chuẩn tắc, rồi `{"action":"done"}`. Có thể
  chèn trước các chuỗi hỏng và giả lập thời gian bằng đồng hồ giả, nên test được cả nhánh hết giờ.
- `VLLMBackend` ràng buộc đầu ra theo JSON schema (structured output), có lớp tương thích giữa các phiên bản vLLM.
- Khi có `time_left_s`, request được đưa vào engine, chạy từng bước (`step`) và bị huỷ (`abort_request`) ngay khi
  chạm hạn. Nhờ vậy một mẫu không bao giờ vượt hạn giờ. Engine thiếu API này thì quay về `LLM.chat` và ghi log lỗi.

Lưu ý:
- `VLLMBackend` import vllm bên trong hàm khởi tạo (extra `gpu`) và nhận `engine_factory` để test thay engine giả.
- Prefix caching luôn bật. `batch_invariant` đặt biến môi trường của vLLM để kết quả không phụ thuộc kích thước
  batch.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import structlog

from lifemate.core.config import ConfigError, LLMConfig
from lifemate.core.jsonutil import canonical_json
from lifemate.core.types import Generation, Sample

# Đầu ra mặc định khi hàng đợi của mẫu đã hết.
_DONE = canonical_json({"action": "done"})


class FakeLLM:
    """LLM giả: phát lại `label.tool_calls` của từng mẫu, sau đó luôn trả `{"action":"done"}`.

    Các chuỗi thô trong `broken_outputs[sample_id]` được trả trước, để test nhánh parse lỗi. Khi có `clock` loại
    đồng hồ giả (có `.advance`), mỗi lần gọi đẩy đồng hồ thêm `delay_s` thay vì ngủ thật.

    Tham số:
        samples: các mẫu có `label` để phát lại.
        delay_s: thời gian giả lập của mỗi lần sinh, tính bằng giây.
        broken_outputs: id mẫu tới các chuỗi hỏng trả về trước đáp án.
        clock: đồng hồ dùng chung với agent; `None` thì ngủ thật `delay_s` giây.

    Thuộc tính:
        delay_s: như tham số cùng tên.
        clock: như tham số cùng tên.
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
        # Mẫu chỉ có đầu ra hỏng (không có trong `samples`) vẫn được tạo hàng đợi riêng.
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
        """Trả về đầu ra soạn sẵn kế tiếp của `sample_id`.

        Nếu `delay_s` lớn hơn `time_left_s`, bước bị coi là hết giờ: văn bản trả về rỗng và thời gian tiêu tốn bằng đúng
        `time_left_s`, giống backend thật bị huỷ.

        Tham số:
            messages: các tin nhắn của prompt; chỉ dùng để ước lượng số token.
            schema: bị bỏ qua.
            sample_id: mẫu cần lấy đầu ra.
            time_left_s: số giây còn lại; `None` là không giới hạn.

        Trả về:
            `Generation` với số token ước lượng bằng số từ.
        """
        queue = self._queues.get(sample_id, [])
        i = self._cursor.get(sample_id, 0)
        text = queue[i] if i < len(queue) else _DONE
        self._cursor[sample_id] = i + 1
        timed_out = time_left_s is not None and self.delay_s > time_left_s
        spent = time_left_s if timed_out and time_left_s is not None else self.delay_s
        # Có đồng hồ giả thì đẩy đồng hồ; không có đồng hồ thì ngủ thật; đồng hồ không có `.advance` thì bỏ qua.
        if spent > 0:
            advance = getattr(self.clock, "advance", None)
            if advance is not None:
                advance(spent)
            elif self.clock is None:
                time.sleep(spent)
        # Số token chỉ ước lượng thô bằng số từ.
        prompt = " ".join(str(m.get("content") or "") for m in messages)
        return Generation(
            text="" if timed_out else text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=0 if timed_out else len(text.split()),
            seconds=spent,
            timed_out=timed_out,
        )

    def warmup(self) -> None:
        """Không cần làm gì."""


_log = structlog.get_logger(__name__)

# Chế độ batch-invariant của vLLM chỉ chạy trên GPU từ Ampere (sm80) trở lên.
_MIN_BATCH_INVARIANT_CC = (8, 0)
_WARMUP_SENTENCE = "Hôm nay trời đẹp, tôi muốn bật đèn phòng khách và đặt báo thức lúc bảy giờ sáng mai. "
_WARMUP_REPEATS = 200  # mỗi câu khoảng 20 token, nhân 200 được khoảng 4K token prompt
_WARMUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"action": {"const": "done"}},
    "required": ["action"],
}


def _structured_kwargs(schema: dict[str, Any]) -> dict[str, Any]:
    """Tạo tham số `SamplingParams` để ràng buộc đầu ra theo JSON schema, chạy được trên nhiều phiên bản vLLM.

    vLLM mới dùng `StructuredOutputsParams` (khoá `structured_outputs`); bản cũ dùng `GuidedDecodingParams` (khoá
    `guided_decoding`).

    Tham số:
        schema: JSON schema của đầu ra.

    Trả về:
        Dict một khoá, truyền thẳng vào `SamplingParams(**...)`.
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
    """Engine vLLM chạy offline, đầu ra có cấu trúc theo JSON schema (SPEC 4.8).

    Giải mã tham lam (temperature 0, seed cố định). Hai cờ tự tắt khi engine không hỗ trợ:
    `_use_template_kwargs` (truyền `enable_thinking=False` vào chat template) và `_abortable` (API step/abort).

    Tham số:
        cfg: mục `[llm]` của config.
        dtype: dtype đã chọn (xem `resolve_dtype`).
        hf_cache: thư mục tải model; `None` là mặc định của Hugging Face.
        hf_token: token HF; có thì đặt vào biến môi trường `HF_TOKEN`.
        engine_factory: hàm tạo engine; `None` thì dùng `vllm.LLM`. Test truyền engine giả vào đây.
        compute_capability: compute capability của GPU, dùng để kiểm tra `batch_invariant`.

    Thuộc tính:
        cfg: như tham số cùng tên.
        engine: engine vLLM đã tạo.
        version: phiên bản vLLM, hoặc `"unknown"`.
        warmup_seconds: tổng thời gian warmup; `None` khi chưa warmup.

    Lỗi:
        ConfigError: khi bật `batch_invariant` trên GPU dưới sm80.
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
        # True sau lần đầu đường step/abort chạy trọn vẹn: từ đó lỗi trên đường này là bug thật, không phải thiếu API.
        self._abortable_verified = False
        self._next_request = 0
        if cfg.batch_invariant:
            if compute_capability is not None and compute_capability < _MIN_BATCH_INVARIANT_CC:
                cc = compute_capability
                raise ConfigError(f"llm.batch_invariant needs compute capability >= 8.0, got {cc[0]}.{cc[1]}")
            # Phải đặt trước khi tạo engine thì vLLM mới đọc được.
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
        """Tạo `SamplingParams` giải mã tham lam, giới hạn `max_tokens` và ràng buộc đầu ra theo `schema`."""
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
        """Sinh văn bản một lần, có ràng buộc theo `schema`; `sample_id` không dùng tới.

        Khi có `time_left_s`, request chạy từng bước trên engine và bị huỷ khi hết giờ (`timed_out=True`), nên một mẫu
        không bao giờ chạy quá hạn giờ. Khi không có `time_left_s` hoặc engine thiếu API step/abort, hàm dùng
        `LLM.chat`. Việc thiếu API chỉ được kết luận khi đường step/abort hỏng ngay lần đầu (thường là lúc warmup); đã
        chạy được rồi mà sau đó lỗi thì lỗi được ném ra, để không lặng lẽ mất hạn giờ cho cả phần còn lại của run.

        Tham số:
            messages: các tin nhắn của prompt.
            schema: JSON schema của đầu ra.
            sample_id: mã mẫu; không dùng.
            time_left_s: số giây còn lại của mẫu; `None` là không giới hạn.

        Trả về:
            `Generation` kèm số token prompt, số token sinh ra và thời gian.

        Lỗi:
            AttributeError, TypeError: khi đường step/abort đã từng chạy được nhưng lần này lỗi.
        """
        params = self._sampling_params(schema)
        # Thời gian của lần thử step/abort thất bại, được cộng vào `seconds` của lần gọi `LLM.chat` thay thế.
        failed_s = 0.0
        if time_left_s is not None and self._abortable:
            t_try = time.perf_counter()
            try:
                gen = self._generate_until(messages, params, time_left_s)
            except (AttributeError, TypeError) as exc:  # API engine khác với điều mã này giả định
                if self._abortable_verified:
                    raise
                _log.error("vllm_abortable_generate_unsupported", error=str(exc))
                self._abortable = False
                failed_s = time.perf_counter() - t_try
            else:
                self._abortable_verified = True
                return gen
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
                # vLLM cũ không nhận `chat_template_kwargs`: tắt hẳn cờ này và gọi lại không kèm tham số đó.
                _log.warning("chat_template_kwargs_unsupported", error=str(exc))
                self._use_template_kwargs = False
                # Đặt lại mốc thời gian để không tính lần gọi lỗi vừa rồi.
                t0 = time.perf_counter()
                outputs = self.engine.chat(messages, sampling_params=params, use_tqdm=False)
        else:
            outputs = self.engine.chat(messages, sampling_params=params, use_tqdm=False)
        seconds = time.perf_counter() - t0 + failed_s
        out = outputs[0]
        completion = out.outputs[0]
        return Generation(
            text=completion.text,
            prompt_tokens=len(out.prompt_token_ids or ()),
            completion_tokens=len(completion.token_ids or ()),
            seconds=seconds,
        )

    def _render(self, messages: list[dict[str, Any]]) -> str:
        """Áp chat template đúng như `LLM.chat` (bật generation prompt, tắt thinking) và trả về chuỗi prompt."""
        tok = self.engine.get_tokenizer()
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if self._use_template_kwargs:
            kwargs["enable_thinking"] = False
        return str(tok.apply_chat_template(messages, **kwargs))

    def _generate_until(self, messages: list[dict[str, Any]], params: Any, time_left_s: float) -> Generation:
        """Thêm một request vào engine, chạy từng bước tới khi xong hoặc quá `time_left_s` thì huỷ request.

        Hạn giờ được kiểm tra giữa các bước, nên thời gian vượt hạn tối đa là một bước giải mã. Khi bị huỷ, văn bản trả
        về rỗng (JSON dở dang không dùng được), nhưng số token vẫn lấy từ đầu ra mới nhất.

        Tham số:
            messages: các tin nhắn của prompt.
            params: `SamplingParams` đã tạo.
            time_left_s: số giây còn lại của mẫu.

        Trả về:
            `Generation`; `timed_out=True` khi request bị huỷ.

        Lỗi:
            AttributeError, TypeError: khi engine không có API step/abort như mong đợi; `generate` quyết định có quay về
            `LLM.chat` hay không. Mọi lỗi xảy ra sau khi đã thêm request đều huỷ request đó trước khi ném ra.
        """
        engine = self.engine.llm_engine
        t0 = time.perf_counter()
        deadline = t0 + time_left_s
        self._next_request += 1
        # id request không trùng trong suốt vòng đời engine
        rid = f"lm-{self._next_request}"
        engine.add_request(rid, self._render(messages), params)
        final: Any = None
        latest: Any = None
        timed_out = False
        try:
            while engine.has_unfinished_requests():
                if time.perf_counter() >= deadline:
                    engine.abort_request([rid])
                    timed_out = True
                    break
                # Một bước có thể trả đầu ra của nhiều request; chỉ giữ đầu ra của request `rid`.
                for out in engine.step():
                    if out.request_id == rid:
                        latest = out
                        if out.finished:
                            final = out
        except BaseException:
            # Thoát giữa chừng (lỗi engine, Ctrl-C): huỷ request, kẻo nó còn trong engine và chạy tiếp ở lần gọi sau.
            with contextlib.suppress(Exception):
                engine.abort_request([rid])
            raise
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
        """Biên dịch kernel và CUDA graph bằng một hội thoại tổng hợp dài khoảng 4K token.

        Chạy `warmup_runs` lần qua đúng đường `generate` có hạn giờ (300 giây), để làm nóng cả nhánh step/abort. Tổng
        thời gian được lưu vào `warmup_seconds`.
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
