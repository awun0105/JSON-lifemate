"""T17: test phép so khớp tham số có xét kiểu và chế độ canon trong `lifemate/eval/score.py`, cùng điểm kiểm chứng.

Cách hoạt động:
- Gọi trực tiếp `values_equal`.
- Riêng `test_score_match_modes` chạy script chấm điểm bằng subprocess trên file gold và file toàn `[]`.

Lưu ý:
- `test_score_match_modes` mang marker `data`: cần gói dữ liệu cuộc thi, thiếu thì skip qua fixture `data_root`.
- Không cần tokenizer.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCORE = ["-m", "lifemate.eval.score"]


def _load_score():
    """Nạp module `lifemate.eval.score` để test gọi các hàm bên trong."""
    import lifemate.eval.score as mod

    return mod


score = _load_score()


@pytest.fixture
def data_root() -> Path:
    """Trả về thư mục dữ liệu mặc định của script chấm điểm; skip nếu thiếu gói dữ liệu cuộc thi."""
    root = score.DEFAULT_ROOT
    if not (root / "train" / "train.jsonl").exists():
        pytest.skip("competition data package is absent")
    return root


@pytest.mark.parametrize("canon", [False, True])
def test_bool_never_equals_number(canon):
    """bool không bao giờ bằng số (True khác 1, 0 khác False) ở cả hai chế độ."""
    assert not score.values_equal(True, 1, canon)
    assert not score.values_equal(0, False, canon)
    assert score.values_equal(True, True, canon)


@pytest.mark.parametrize("canon", [False, True])
def test_int_equals_float(canon):
    """Số nguyên bằng số thực cùng giá trị, nhưng chuỗi '1' khác số 1."""
    assert score.values_equal(1, 1.0, canon)
    assert not score.values_equal(1, 1.5, canon)
    assert not score.values_equal("1", 1, canon)


def test_string_case_strict_vs_canon():
    """Chế độ strict phân biệt hoa thường; canon bỏ qua hoa thường và khoảng trắng."""
    assert not score.values_equal("Phòng Khách", "phòng khách")
    assert score.values_equal("Phòng Khách", "phòng khách", canon=True)
    assert score.values_equal("700 cm", "700cm", canon=True)


def test_recursive_containers():
    """So khớp đệ quy trong list và dict; độ dài khác nhau thì không bằng."""
    assert score.values_equal([1, {"a": 2.0}], [1.0, {"a": 2}])
    assert not score.values_equal([True], [1])
    assert not score.values_equal({"a": "X"}, {"a": "x"})
    assert score.values_equal({"a": ["X  y"]}, {"a": ["x y"]}, canon=True)
    assert not score.values_equal([1], [1, 2])


def _run(*args: str) -> str:
    """Chạy script chấm điểm với các tham số đã cho và trả về stdout."""
    out = subprocess.run([sys.executable, *SCORE, *args], capture_output=True, text=True, check=True, cwd=REPO)
    return out.stdout


@pytest.mark.data
def test_score_match_modes(data_root, tmp_path):
    """File gold đạt 100 ở cả strict và canon; file toàn [] đạt đúng 7.18 (điểm của các mẫu ngoài phạm vi)."""
    gold = tmp_path / "gold.jsonl"
    _run("--root", str(data_root), "--make-gold", str(gold))
    out = _run(str(gold), "--root", str(data_root), "--match", "both")
    assert re.findall(r"Final score \[(\w+)\]: ([\d.]+)", out) == [("strict", "100.00"), ("canon", "100.00")]

    # Submission mọi mẫu đều rỗng: chỉ các mẫu out_of_scope được điểm.
    empty = tmp_path / "empty.jsonl"
    with open(gold, encoding="utf-8") as f, open(empty, "w", encoding="utf-8") as g:
        for line in f:
            g.write(json.dumps({"id": json.loads(line)["id"], "tool_calls": []}) + "\n")
    out = _run(str(empty), "--root", str(data_root), "--match", "both")
    assert re.findall(r"Final score \[(\w+)\]: ([\d.]+)", out) == [("strict", "7.18"), ("canon", "7.18")]


def test_report_empty_rows_no_crash(capsys):
    """Báo cáo trên tập rỗng không chia cho 0: độ chính xác quyết định in ra `nan`."""
    score.report([], 5)
    assert "decision accuracy: nan%" in capsys.readouterr().out


@pytest.mark.data
def test_empty_ids_file_exits_clearly(data_root, tmp_path):
    """File `--ids` rỗng làm script dừng với thông báo rõ ràng, không crash giữa chừng."""
    ids = tmp_path / "ids.txt"
    ids.write_text("", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, *SCORE, str(tmp_path / "none.jsonl"), "--root", str(data_root), "--ids", str(ids)],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
    )
    assert proc.returncode != 0 and "no samples selected for scoring" in proc.stderr
