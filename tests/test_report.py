"""Kiểm tra lifemate.run.report: bảng văn bản dựng từ index.csv và metrics.jsonl (không phụ thuộc thư viện vẽ).

Cách kiểm tra: ghi index.csv bằng `append_index` và metrics.jsonl bằng `MetricsLogger` vào `tmp_path`, rồi đọc lại
qua `runs_table`, `train_report` và `main`. Không cần dữ liệu cuộc thi hay tokenizer.
"""

from pathlib import Path

import pytest

from lifemate.io.store import append_index
from lifemate.run.report import main, metric_columns, runs_table, train_report
from lifemate.train.common import MetricsLogger


def test_runs_table_newest_first(tmp_path: Path) -> None:
    """Bảo vệ: các dòng của index hiện theo thứ tự mới nhất trước, đúng các cột đã chọn."""
    idx = tmp_path / "index.csv"
    append_index(
        idx, {"run": "a", "date": "d1", "git_commit": "c", "model": "m", "split": "train", "n": 3, "score": 10.0}
    )
    append_index(
        idx, {"run": "b", "date": "d2", "git_commit": "c", "model": "m", "split": "train", "n": 3, "score": 20.0}
    )
    out = runs_table(idx)
    # Bỏ hai dòng đầu (tiêu đề và gạch ngang) để lấy các dòng dữ liệu.
    rows = out.splitlines()[2:]
    assert [r.split()[0] for r in rows] == ["b", "a"] and "p95_seconds" in out
    # limit=1: chỉ còn tiêu đề, gạch ngang và một dòng dữ liệu.
    assert runs_table(idx, limit=1).count("\n") == 2


def test_train_report_best_and_png_skip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Bảo vệ: tìm đúng các cột số, báo bước eval tốt nhất, thiếu matplotlib không bị coi là lỗi."""
    work = tmp_path / "w"
    log = MetricsLogger(work / "sft" / "metrics.jsonl")
    log.log({"step": 1, "epoch": 0.1, "loss": 2.0})
    log.log({"step": 2, "epoch": 0.2, "loss": 1.5, "eval_unseen_loss": 1.2})
    log.log({"step": 3, "epoch": 0.3, "loss": 1.0, "eval_unseen_loss": 1.4})
    assert metric_columns(log.read()) == ["loss", "eval_unseen_loss"]
    text = train_report(work, every=1, png=None)
    assert "best eval_unseen_loss: 1.2000 at step 2" in text and "3 rows" in text
    assert main(["train", str(work), "--every", "2"]) == 0
    assert "step" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError):
        train_report(tmp_path / "nothing", every=1, png=None)
