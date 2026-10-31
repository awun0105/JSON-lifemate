"""Test `lifemate/eval/agent_report.py`: chỉ số một run và so ghép cặp giữa hai run."""

from __future__ import annotations

import pytest

from lifemate.eval.agent_report import paired, run_metrics


def _run(scores: dict[str, float], stops: dict[str, str]) -> dict:
    details = {
        i: {"id": i, "score": s, "category": "execute", "predicted": "execute", "errors": []} for i, s in scores.items()
    }
    step = {"llm_s": 5.0, "error": None}
    trace = {i: {"stop_reason": stops[i], "timing": {"total_s": 10.0}, "steps": [step], "tokens": {}} for i in scores}
    return {"details": details, "trace": trace, "summary": {}}


def test_metrics_and_paired_diff() -> None:
    a = _run({"s1": 1.0, "s2": 0.0}, {"s1": "done", "s2": "timeout"})
    b = _run({"s1": 1.0, "s2": 1.0}, {"s1": "done", "s2": "done"})
    m = run_metrics(a)
    assert m["score"] == 50.0 and m["timeouts"] == 1 and m["perfect"] == 1 and m["decision_accuracy"] == 100.0
    p = paired(a["details"], b["details"])
    assert p["diff"] == 50.0 and (p["gained"], p["lost"]) == (1, 0)
    with pytest.raises(ValueError):
        paired(a["details"], {"s1": a["details"]["s1"]})
