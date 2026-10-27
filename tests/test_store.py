"""Kiểm tra RunStore: resume, dòng cuối hỏng, ghi file nguyên tử và bản tóm tắt (SPEC 4.11, 4.12; T12, T16).

Cách kiểm tra: dựng SampleResult giả bằng `result`, ghi trace và submission vào `tmp_path`; T16 dùng `monkeypatch`
thay `os.replace` bằng hàm lỗi. Không cần dữ liệu cuộc thi hay tokenizer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lifemate.core.types import ErrorCode, SampleResult, StopReason
from lifemate.io.store import RunStore, append_index, summarize


def result(sid: str, *, stop: StopReason = StopReason.DONE, total_s: float = 1.0) -> SampleResult:
    """Tạo SampleResult giả cho mẫu `sid`, có một tool call chứa tiếng Việt; mẫu EXCEPTION mang mã lỗi exception."""
    return SampleResult(
        id=sid,
        stop_reason=stop,
        errors=(ErrorCode.EXCEPTION,) if stop is StopReason.EXCEPTION else (),
        transcript=None,
        steps=(),
        tool_calls=({"name": "alarm.list", "arguments": {"q": "Phòng khách"}},),
        state_hash="h-" + sid,
        timing={"total_s": total_s},
        tokens={"max_prompt": 100, "completion_total": 5},
    )


def test_refuses_existing_trace_without_resume(tmp_path: Path) -> None:
    """Bảo vệ: không ghi đè trace đã có khi không bật resume."""
    RunStore(tmp_path, resume=False).append(result("a"))
    with pytest.raises(FileExistsError):
        RunStore(tmp_path, resume=False)


def test_resume_skips_done_and_corrupt_tail(tmp_path: Path) -> None:
    """T12: ba mẫu xong cộng một dòng bị cắt; resume giữ ba mẫu, bỏ dòng cắt, submission đủ mọi id."""
    ids = ["s1", "s2", "s3", "s4", "s5"]
    store = RunStore(tmp_path, resume=False)
    for sid in ids[:3]:
        store.append(result(sid))
    # Giả lập lượt chạy bị ngắt giữa chừng khi đang ghi dòng s4.
    with store.trace_path.open("a", encoding="utf-8") as f:
        f.write('{"id": "s4", "stop_reas')
    assert store.done_ids() == {"s1", "s2", "s3"}

    resumed = RunStore(tmp_path, resume=True)
    assert resumed.done_ids() == {"s1", "s2", "s3"}
    for sid in ids:
        if sid not in resumed.done_ids():
            resumed.append(result(sid))
    sub = resumed.finalize(ids)
    rows = [json.loads(ln) for ln in sub.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ids
    assert set(rows[0]) == {"id", "tool_calls", "state_hash"}
    assert "Phòng khách" in sub.read_text(encoding="utf-8")  # ensure_ascii=False: tiếng Việt ghi nguyên dạng
    assert len(resumed.records()) == 5


def test_corrupt_middle_line_raises(tmp_path: Path) -> None:
    """Bảo vệ: dòng hỏng ở giữa trace (không phải dòng cuối) là lỗi thật, báo kèm số dòng."""
    store = RunStore(tmp_path, resume=False)
    store.trace_path.write_text('{"id": "a"}\nnot json\n{"id": "b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt line 2"):
        store.records()


def test_atomic_submission_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T16: lệnh replace lỗi thì submission cũ còn nguyên và không sót file tmp."""
    store = RunStore(tmp_path, resume=False)
    store.append(result("a"))
    store.finalize(["a"])
    before = store.submission_path.read_text(encoding="utf-8")
    store.append(result("b"))

    def boom(src: object, dst: object) -> None:
        """Giả lập os.replace lỗi do đầy đĩa."""
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        store.finalize(["a", "b"])
    assert store.submission_path.read_text(encoding="utf-8") == before
    assert not list(tmp_path.glob("*.tmp"))


def test_summary_fields() -> None:
    """Bảo vệ: tóm tắt đếm đúng số mẫu, ngoại lệ, lý do dừng, lỗi và ngưỡng thời gian 25 s, 30 s."""
    recs = [result("a", total_s=1.0).to_dict(), result("b", total_s=26.0).to_dict()]
    recs.append(result("c", stop=StopReason.EXCEPTION, total_s=31.0).to_dict())
    s = summarize(recs)
    assert s["n"] == 3 and s["n_exception"] == 1
    assert set(s["stop_reason"]) == {r.value for r in StopReason}
    assert s["errors"] == {"exception": {"count": 1, "ids": ["c"]}}
    assert s["seconds"]["over_30"] == 1 and s["seconds"]["over_25"] == 2 and s["seconds"]["max"] == 31.0
    assert s["prompt_shrink_level"]["0"] == 3 and s["score"] is None


def test_append_index(tmp_path: Path) -> None:
    """Bảo vệ: append_index tạo thư mục, ghi tiêu đề một lần rồi nối thêm dòng."""
    path = tmp_path / "runs" / "index.csv"
    append_index(path, {"run": "r1", "n": 3})
    append_index(path, {"run": "r2", "n": 4})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("run,date,git_commit") and len(lines) == 3
