"""ASR pair counts and split hygiene (SPEC 6.4.1, 6.4.5 item 1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lifemate.data import load_split, read_ids
from lifemate.train import asr_data
from lifemate.train.asr import choose_checkpoint
from lifemate.train.common import CheckpointInfo

EXPECTED = {"train": 3547, "val_seen": 624, "val_unseen": 865}


@pytest.mark.data
@pytest.mark.parametrize("split", sorted(EXPECTED))
def test_pair_counts_match_spec(split: str, repo_root: Path, data_root: Path) -> None:
    samples = load_split(data_root, "train", read_ids(repo_root / "splits" / f"{split}.txt"))
    rows = asr_data.build_pairs(samples, split)
    assert len(rows) == EXPECTED[split]
    assert {r["split"] for r in rows} == {split}
    assert all((data_root / "train" / r["audio"]).is_file() for r in rows[:50])


@pytest.mark.data
def test_train_build_has_no_val_audio(tmp_path: Path, repo_root: Path, data_root: Path) -> None:
    out = tmp_path / "asr.jsonl"
    argv = ["--ids", str(repo_root / "splits" / "train.txt"), "--out", str(out)]
    meta = asr_data.run(asr_data.parse_args(argv), repo_root, argv=argv)
    assert meta["n_rows"] == EXPECTED["train"]
    rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    val_ids = set(read_ids(repo_root / "splits" / "val_seen.txt")) | set(
        read_ids(repo_root / "splits" / "val_unseen.txt")
    )
    assert not {r["id"] for r in rows} & val_ids
    assert not any(vid in r["audio"] for r in rows for vid in val_ids)
    assert rows[-1]["turn"] == "final" and json.loads(Path(f"{out}.meta.json").read_text())["sha256"] == meta["sha256"]


@pytest.mark.data
def test_refuses_val_ids(tmp_path: Path, repo_root: Path, data_root: Path) -> None:
    argv = ["--ids", str(repo_root / "splits" / "val_seen.txt"), "--out", str(tmp_path / "x.jsonl")]
    with pytest.raises(SystemExit, match="refusing"):
        asr_data.run(asr_data.parse_args(argv), repo_root, argv=argv)


def test_choose_checkpoint_entity_then_wer() -> None:
    a, b, c = (CheckpointInfo(Path(n), s, None) for n, s in (("a", 1), ("b", 2), ("c", 3)))
    scored = [
        (a, {"entity_miss_rate": 0.10, "wer": 0.20}),
        (b, {"entity_miss_rate": 0.05, "wer": 0.30}),
        (c, {"entity_miss_rate": 0.05, "wer": 0.25}),
    ]
    assert choose_checkpoint(scored) is c
