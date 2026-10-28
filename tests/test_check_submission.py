"""Test `check_submission` trên tập train thật (SPEC 4.11 B3; T25).

Cách hoạt động:
- File submission được dựng trong `tmp_path`, cài sẵn từng loại lỗi.

Lưu ý:
- Cả file mang marker `data`: dùng fixture `data_root`, tự skip khi thiếu dữ liệu cuộc thi.
- Không cần tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lifemate.io.check_submission import check_submission
from lifemate.io.data import load_split

pytestmark = pytest.mark.data


def test_check_submission_flags_each_problem(data_root: Path, tmp_path: Path) -> None:
    """Thiếu id, id trùng, dòng JSON hỏng và lời gọi thiếu tham số bắt buộc đều được báo riêng."""
    samples = [s for s in load_split(data_root, "train") if (s.label or {}).get("tool_calls")][:5]
    rows = [{"id": s.id, "tool_calls": (s.label or {})["tool_calls"], "state_hash": ""} for s in samples]
    missing = rows.pop(4)["id"]
    rows.append(dict(rows[0]))  # id trùng
    bad_call = dict(rows[1]["tool_calls"][0])
    # Xoá hết tham số để lời gọi thiếu tham số bắt buộc.
    bad_call["arguments"] = {}
    rows[1] = {**rows[1], "tool_calls": [bad_call]}
    sub = tmp_path / "submission.jsonl"
    sub.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows) + "{broken\n", encoding="utf-8")

    report = check_submission(sub, data_root, "train")
    text = report.to_text()
    assert not report.ok
    assert missing in report.missing_ids
    assert report.duplicate_ids == [rows[0]["id"]]
    # 5 dòng hợp lệ, dòng thứ 6 là "{broken".
    assert report.invalid_lines == [6]
    assert report.validate_errors is not None and report.validate_errors >= 1, report.validate_output
    assert rows[1]["id"] in report.rejected
    for needle in ("missing ids", "duplicated ids", "invalid JSON", "schema errors", "replay"):
        assert needle in text


def test_malformed_tool_calls_reported_not_crash(data_root: Path, tmp_path: Path) -> None:
    """Lời gọi không phải object hoặc có `arguments` không phải object bị báo là dòng hỏng, không làm checker crash."""
    ids = [s.id for s in load_split(data_root, "train")[:3]]
    lines = [
        {"id": ids[0], "tool_calls": ["bad"]},
        {"id": ids[1], "tool_calls": [{"name": "x.y", "arguments": "oops"}]},
        {"id": ids[2], "tool_calls": []},
    ]
    sub = tmp_path / "submission.jsonl"
    sub.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    report = check_submission(sub, data_root, "train")
    assert report.invalid_lines == [1, 2]
    assert not report.ok and "invalid JSON" in report.to_text()
