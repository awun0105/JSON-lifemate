"""Token counting with the model's chat template (SPEC 4.8, HFTokenCounter)."""

from __future__ import annotations

from typing import Any

_PROBE = "xin chào"


class HFTokenCounter:
    """TokenCounter backed by `transformers.AutoTokenizer.apply_chat_template`; runs on CPU."""

    def __init__(self, model: str, revision: str | None = None, *, local_files_only: bool = False) -> None:
        # transformers comes from the `tokenizer` extra.
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=local_files_only)
        self._overhead: dict[str, int] = {}

    def count(self, messages: list[dict[str, Any]]) -> int:
        """Prompt tokens of `messages` rendered with the chat template plus the generation prompt."""
        ids: Any = self._tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        return len(ids)

    def count_text(self, text: str) -> int:
        """Tokens of raw text without special tokens."""
        return len(self._tok.encode(text, add_special_tokens=False))

    def template_overhead(self, role: str) -> int:
        """Tokens the template adds when one `role` message is appended (computed once by diffing)."""
        if role not in self._overhead:
            base = [{"role": "system", "content": _PROBE}, {"role": "user", "content": _PROBE}]
            if role == "user":
                base.append({"role": "assistant", "content": _PROBE})
            grown = [*base, {"role": role, "content": _PROBE}]
            self._overhead[role] = self.count(grown) - self.count(base) - self.count_text(_PROBE)
        return self._overhead[role]


class SimpleTokenCounter:
    """Whitespace tokens of each message content plus 4 per message; used with FakeLLM or without transformers."""

    def count(self, messages: list[dict[str, Any]]) -> int:
        """Approximate prompt tokens of `messages`."""
        return sum(len(str(m.get("content") or "").split()) + 4 for m in messages)
