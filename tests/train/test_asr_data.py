"""Test số cặp dữ liệu ASR, việc tách split sạch, và cách chọn checkpoint ASR (SPEC 6.4.1, 6.4.5 mục 1).

Lưu ý:
- Ba test đầu đọc dữ liệu cuộc thi thật (marker `data`) qua fixture `repo_root`, `data_root` và các file
  `configs/splits/*.txt`.
- Test chọn checkpoint chạy thuần trong bộ nhớ với `CheckpointInfo` giả, không cần dữ liệu.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lifemate.io.data import load_split, read_ids
from lifemate.train import asr_data
from lifemate.train.asr import choose_checkpoint, normalize_text
from lifemate.train.common import CheckpointInfo

# Số cặp audio-transcript mong đợi cho từng split, lấy từ SPEC 6.4.1.
EXPECTED = {"train": 3593, "val_seen": 916, "val_unseen": 527}


@pytest.mark.data
@pytest.mark.parametrize("split", sorted(EXPECTED))
def test_pair_counts_match_spec(split: str, repo_root: Path, data_root: Path) -> None:
    """Mỗi split sinh đúng số cặp ghi trong SPEC, gắn đúng tên split, và trỏ tới file audio có thật."""
    samples = load_split(data_root, "train", read_ids(repo_root / "configs" / "splits" / f"{split}.txt"))
    rows = asr_data.build_pairs(samples, split)
    assert len(rows) == EXPECTED[split]
    assert {r["split"] for r in rows} == {split}
    # Chỉ kiểm tra 50 dòng đầu để test nhanh; vẫn đủ bắt lỗi sai đường dẫn audio.
    assert all((data_root / "train" / r["audio"]).is_file() for r in rows[:50])


@pytest.mark.data
def test_train_build_has_no_val_audio(tmp_path: Path, repo_root: Path, data_root: Path) -> None:
    """Dữ liệu ASR dựng từ split train không lẫn id hay audio của `val_seen`/`val_unseen` (chống lộ dữ liệu eval)."""
    out = tmp_path / "asr.jsonl"
    argv = ["--ids", str(repo_root / "configs" / "splits" / "train.txt"), "--out", str(out)]
    meta = asr_data.run(asr_data.parse_args(argv), repo_root, argv=argv)
    assert meta["n_rows"] == EXPECTED["train"]
    rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    val_ids = set(read_ids(repo_root / "configs" / "splits" / "val_seen.txt")) | set(
        read_ids(repo_root / "configs" / "splits" / "val_unseen.txt")
    )
    assert not {r["id"] for r in rows} & val_ids
    # Cả id lẫn đường dẫn audio đều không được chứa mẫu validation.
    assert not any(vid in r["audio"] for r in rows for vid in val_ids)
    # Dòng cuối là lượt `final`; file meta ghi đúng sha256 của file dữ liệu.
    assert rows[-1]["turn"] == "final" and json.loads(Path(f"{out}.meta.json").read_text())["sha256"] == meta["sha256"]


@pytest.mark.data
def test_refuses_val_ids(tmp_path: Path, repo_root: Path, data_root: Path) -> None:
    """Lệnh dựng dữ liệu từ chối danh sách id thuộc split validation."""
    argv = ["--ids", str(repo_root / "configs" / "splits" / "val_seen.txt"), "--out", str(tmp_path / "x.jsonl")]
    with pytest.raises(SystemExit, match="refusing"):
        asr_data.run(asr_data.parse_args(argv), repo_root, argv=argv)


def test_choose_checkpoint_entity_then_wer() -> None:
    """Chọn checkpoint có `entity_miss_rate` thấp nhất trước, hoà thì xét WER thấp hơn."""
    a, b, c = (CheckpointInfo(Path(n), s, None) for n, s in (("a", 1), ("b", 2), ("c", 3)))
    scored = [
        (a, {"entity_miss_rate": 0.10, "wer": 0.20}),
        (b, {"entity_miss_rate": 0.05, "wer": 0.30}),
        (c, {"entity_miss_rate": 0.05, "wer": 0.25}),
    ]
    # b và c cùng `entity_miss_rate` thấp nhất; c thắng nhờ WER thấp hơn.
    assert choose_checkpoint(scored) is c


def test_normalize_text_drops_unk() -> None:
    """Token `<unk>` và `⁇` bị xoá khi chuẩn hoá, nên không bị tính như một từ "unk" khi chọn checkpoint."""
    assert normalize_text("Bật <unk> đèn ⁇!") == "bật đèn"
    assert normalize_text("<unk>") == ""
