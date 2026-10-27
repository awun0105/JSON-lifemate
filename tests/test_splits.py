"""T26: configs/splits/smoke.txt is 30 val_seen ids covering every answer type, stable across runs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "train+public_test"
SPLITS = REPO / "configs" / "splits"


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


@pytest.mark.data
def test_smoke_split(data_root):
    smoke = _ids(SPLITS / "smoke.txt")
    assert len(smoke) == 30
    assert len(set(smoke)) == 30
    assert set(smoke) <= set(_ids(SPLITS / "val_seen.txt"))
    with open(data_root / "train" / "train.jsonl", encoding="utf-8") as f:
        labels = {it["id"]: it["label"]["tool_calls"] for it in map(json.loads, f) if it}
    assert {_category(labels[i]) for i in smoke} == {"execute", "ask_missing_info", "out_of_scope"}


@pytest.mark.data
def test_smoke_stable(data_root, tmp_path):
    runs = []
    for name in ("a", "b"):
        out = tmp_path / name
        script = REPO / "lifemate" / "eval" / "make_splits.py"
        cmd = [sys.executable, str(script), "--root", str(data_root), "--out", str(out)]
        subprocess.run([*cmd, "--smoke", "30"], check=True, capture_output=True)
        runs.append((out / "smoke.txt").read_text())
    assert runs[0] == runs[1]
    assert runs[0] == (SPLITS / "smoke.txt").read_text()


def _item(i: str, text: str, n_calls: int, ask: bool = False) -> dict:
    calls = [{"name": "alarm.create", "arguments": {}}] * n_calls
    if ask:
        calls = [*calls, {"name": "sys.ask_missing_info", "arguments": {}}]
    return {"id": i, "audio_text": text, "history": [], "label": {"tool_calls": calls}, "context": {}}


def test_near_duplicates_cluster_and_stay_in_one_fold() -> None:
    """Protects: near-duplicate requests (>= 0.85 similar) never span train and val (team lead's group split)."""
    from lifemate.eval.make_splits import cluster_near_duplicates, stratified_group_folds

    items = [
        _item("a1", "đặt báo thức sáu giờ sáng mai giúp tôi nhé", 1),
        _item("a2", "đặt báo thức sáu giờ sáng mai giúp tôi nha", 1),
        _item("b1", "chuyển cho mẹ năm trăm nghìn qua ví điện tử", 2),
        *[_item(f"c{i}", f"m{i}", i % 5, ask=i % 7 == 0) for i in range(40)],  # short texts: never fuzzy-merged
    ]
    clusters = cluster_near_duplicates(items, 0.85)
    assert ["a1", "a2"] in clusters
    folds = stratified_group_folds(clusters, {it["id"]: it for it in items}, 5, 42)
    assert sorted(x for f in folds for x in f) == sorted(it["id"] for it in items)
    assert sum(("a1" in f) and ("a2" in f) for f in folds) == 1
    assert max(map(len, folds)) - min(map(len, folds)) <= 2
    assert folds == stratified_group_folds(clusters, {it["id"]: it for it in items}, 5, 42)


def test_generated_splits_do_not_leak() -> None:
    """Catches the team lead repo's bug: val_unseen ids inside the training set, or val_seen overlapping train."""
    if not (SPLITS / "meta.json").is_file():
        pytest.skip("splits not generated (git-ignored): run `make splits`")
    train, seen, unseen = (set(_ids(SPLITS / f"{n}.txt")) for n in ("train", "val_seen", "val_unseen"))
    assert not train & unseen and not train & seen and not seen & unseen
    folds = [set(_ids(p)) for p in sorted((SPLITS / "folds").glob("fold_*.txt"))]
    assert len(folds) == 5 and folds[0] == seen and set().union(*folds[1:]) == train
    meta = json.loads((SPLITS / "meta.json").read_text(encoding="utf-8"))
    assert meta["leak_check"] == {"train_and_val_unseen": 0, "train_and_val_seen": 0}
