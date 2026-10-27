"""check_submission on the real train split (SPEC 4.11 B3; T25)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lifemate.io.check_submission import check_submission
from lifemate.io.data import load_split

pytestmark = pytest.mark.data


def test_check_submission_flags_each_problem(data_root: Path, tmp_path: Path) -> None:
    """Missing id, duplicated id and a call missing a required param are each reported."""
    samples = [s for s in load_split(data_root, "train") if (s.label or {}).get("tool_calls")][:5]
    rows = [{"id": s.id, "tool_calls": (s.label or {})["tool_calls"], "state_hash": ""} for s in samples]
    missing = rows.pop(4)["id"]
    rows.append(dict(rows[0]))  # duplicate
    bad_call = dict(rows[1]["tool_calls"][0])
    bad_call["arguments"] = {}
    rows[1] = {**rows[1], "tool_calls": [bad_call]}
    sub = tmp_path / "submission.jsonl"
    sub.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows) + "{broken\n", encoding="utf-8")

    report = check_submission(sub, data_root, "train")
    text = report.to_text()
    assert not report.ok
    assert missing in report.missing_ids
    assert report.duplicate_ids == [rows[0]["id"]]
    assert report.invalid_lines == [6]
    assert report.validate_errors is not None and report.validate_errors >= 1, report.validate_output
    assert rows[1]["id"] in report.rejected
    for needle in ("missing ids", "duplicated ids", "invalid JSON", "schema errors", "replay"):
        assert needle in text
