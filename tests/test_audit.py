"""lifemate.eval.audit: per-call differences between a run and the gold labels (team lead's audit_labels)."""

from __future__ import annotations

from lifemate.eval.audit import audit_sample, summarize

KIND = {"a.list": "read", "a.set": "write", "sys.ask_missing_info": "meta"}


def _sample(gold: list[dict], history_results: bool = False) -> dict:
    history = [{"role": "assistant", "tool_results": [{"ok": True}]}] if history_results else []
    return {"id": "s", "history": history, "label": {"tool_calls": gold}}


def _types(issues: list[dict]) -> list[str]:
    return sorted(i["type"] for i in issues)


def test_order_does_not_matter_and_exact_match_is_clean() -> None:
    gold = [{"name": "a.list", "arguments": {}}, {"name": "a.set", "arguments": {"id": "x", "on": True}}]
    assert audit_sample(_sample(gold), list(reversed(gold)), KIND) == []


def test_argument_differences() -> None:
    gold = [{"name": "a.set", "arguments": {"id": "x", "name": "Lan", "n": 1, "gone": 2}}]
    pred = [{"name": "a.set", "arguments": {"id": "y", "name": "lan", "n": "1", "more": 3}}]
    assert _types(audit_sample(_sample(gold), pred, KIND)) == [
        "CASING_ONLY",
        "EXTRA_ARGUMENT",
        "MISSING_ARGUMENT",
        "TYPE_MISMATCH",
        "VALUE_MISMATCH",
    ]


def test_missing_extra_and_redundant_read() -> None:
    gold = [{"name": "a.set", "arguments": {}}]
    pred = [{"name": "a.list", "arguments": {}}]
    assert _types(audit_sample(_sample(gold), pred, KIND)) == ["EXTRA_CALL", "MISSING_CALL"]
    assert _types(audit_sample(_sample(gold, history_results=True), pred, KIND)) == ["MISSING_CALL", "REDUNDANT_READ"]


def test_calls_on_empty_label_and_summary() -> None:
    issues = audit_sample(_sample([]), [{"name": "a.set", "arguments": {}}], KIND)
    assert _types(issues) == ["CALLED_ON_EMPTY_LABEL"]
    summary = summarize([{"id": "s", "issues": issues}, {"id": "t", "issues": []}])
    assert summary["samples"] == 2 and summary["clean"] == 1 and summary["by_type"] == {"CALLED_ON_EMPTY_LABEL": 1}
