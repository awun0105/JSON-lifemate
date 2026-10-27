"""T26: splits/smoke.txt is 30 val_seen ids covering every answer type, stable across runs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "train+public_test" / "train+public_test"
SPLITS = REPO / "splits"

pytestmark = pytest.mark.data


@pytest.fixture
def data_root() -> Path:
    if not (DATA / "train" / "train.jsonl").exists():
        pytest.skip("competition data package is absent")
    return DATA


def _ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _category(calls: list[dict]) -> str:
    names = [c["name"] for c in calls]
    if not names:
        return "out_of_scope"
    return "ask_missing_info" if "sys.ask_missing_info" in names else "execute"


def test_smoke_split(data_root):
    smoke = _ids(SPLITS / "smoke.txt")
    assert len(smoke) == 30
    assert len(set(smoke)) == 30
    assert set(smoke) <= set(_ids(SPLITS / "val_seen.txt"))
    with open(data_root / "train" / "train.jsonl", encoding="utf-8") as f:
        labels = {it["id"]: it["label"]["tool_calls"] for it in map(json.loads, f) if it}
    assert {_category(labels[i]) for i in smoke} == {"execute", "ask_missing_info", "out_of_scope"}


def test_smoke_stable(data_root, tmp_path):
    runs = []
    for name in ("a", "b"):
        out = tmp_path / name
        cmd = [sys.executable, str(REPO / "eval" / "make_splits.py"), "--root", str(data_root), "--out", str(out)]
        subprocess.run([*cmd, "--smoke", "30"], check=True, capture_output=True)
        runs.append((out / "smoke.txt").read_text())
    assert runs[0] == runs[1]
    assert runs[0] == (SPLITS / "smoke.txt").read_text()
