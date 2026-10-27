"""T17: typed/canon argument matching in lifemate/eval/score.py and the gold / all-[] sanity scores."""

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
    import lifemate.eval.score as mod

    return mod


score = _load_score()


@pytest.fixture
def data_root() -> Path:
    root = score.DEFAULT_ROOT
    if not (root / "train" / "train.jsonl").exists():
        pytest.skip("competition data package is absent")
    return root


@pytest.mark.parametrize("canon", [False, True])
def test_bool_never_equals_number(canon):
    assert not score.values_equal(True, 1, canon)
    assert not score.values_equal(0, False, canon)
    assert score.values_equal(True, True, canon)


@pytest.mark.parametrize("canon", [False, True])
def test_int_equals_float(canon):
    assert score.values_equal(1, 1.0, canon)
    assert not score.values_equal(1, 1.5, canon)
    assert not score.values_equal("1", 1, canon)


def test_string_case_strict_vs_canon():
    assert not score.values_equal("Phòng Khách", "phòng khách")
    assert score.values_equal("Phòng Khách", "phòng khách", canon=True)
    assert score.values_equal("700 cm", "700cm", canon=True)


def test_recursive_containers():
    assert score.values_equal([1, {"a": 2.0}], [1.0, {"a": 2}])
    assert not score.values_equal([True], [1])
    assert not score.values_equal({"a": "X"}, {"a": "x"})
    assert score.values_equal({"a": ["X  y"]}, {"a": ["x y"]}, canon=True)
    assert not score.values_equal([1], [1, 2])


def _run(*args: str) -> str:
    out = subprocess.run([sys.executable, *SCORE, *args], capture_output=True, text=True, check=True, cwd=REPO)
    return out.stdout


@pytest.mark.data
def test_score_match_modes(data_root, tmp_path):
    gold = tmp_path / "gold.jsonl"
    _run("--root", str(data_root), "--make-gold", str(gold))
    out = _run(str(gold), "--root", str(data_root), "--match", "both")
    assert re.findall(r"Final score \[(\w+)\]: ([\d.]+)", out) == [("strict", "100.00"), ("canon", "100.00")]

    empty = tmp_path / "empty.jsonl"
    with open(gold, encoding="utf-8") as f, open(empty, "w", encoding="utf-8") as g:
        for line in f:
            g.write(json.dumps({"id": json.loads(line)["id"], "tool_calls": []}) + "\n")
    out = _run(str(empty), "--root", str(data_root), "--match", "both")
    assert re.findall(r"Final score \[(\w+)\]: ([\d.]+)", out) == [("strict", "7.18"), ("canon", "7.18")]
