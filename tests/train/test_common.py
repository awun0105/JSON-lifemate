"""Kiểm tra tiện ích train dùng chung: seed_everything, CheckpointManager, MetricsLogger, write_train_config (SPEC 6.5).

Các test chạy trên thư mục tạm `tmp_path` với hàm ghi trọng số giả (`_writer`), không cần GPU, tokenizer hay dữ liệu
cuộc thi. Hai test cuối đọc file cấu hình thật trong `configs/train/` qua fixture `repo_root`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from lifemate.core.config import ConfigError
from lifemate.core.envinfo import EnvInfo
from lifemate.train.common import (
    CheckpointManager,
    MetricsLogger,
    seed_everything,
    sha256_file,
    write_train_config,
)
from lifemate.train.sft import load_llm_train_config


def _draw() -> tuple[list[float], list[float]]:
    """Rút 5 số từ `random` và 5 số từ numpy để so sánh trạng thái RNG."""
    return [random.random() for _ in range(5)], np.random.rand(5).tolist()


def test_seed_everything_reproducible() -> None:
    """Cùng seed cho cùng dãy số ngẫu nhiên; seed khác cho dãy khác."""
    seed_everything(123)
    a = _draw()
    seed_everything(123)
    assert _draw() == a
    seed_everything(124)
    assert _draw() != a


def _writer(payload: str):  # type: ignore[no-untyped-def]
    """Tạo hàm lưu trọng số giả: ghi `payload` vào `weights.txt` trong thư mục checkpoint."""

    def save(d: Path) -> None:
        """Ghi `payload` vào thư mục checkpoint `d`."""
        (d / "weights.txt").write_text(payload, encoding="utf-8")

    return save


def test_checkpoint_round_trip_restores_step_and_rng(tmp_path: Path) -> None:
    """Lưu rồi nạp checkpoint khôi phục đúng step, metric, trọng số và trạng thái RNG."""
    mgr = CheckpointManager(tmp_path / "ck", keep_last=1, keep_best=1)
    seed_everything(7)
    mgr.save(10, 1.0, _writer("w10"))
    expected = _draw()  # dãy số ngẫu nhiên ngay sau lúc lưu
    _draw()  # đẩy RNG đi tiếp để load phải khôi phục trạng thái cũ
    loaded: list[str] = []
    info = mgr.load(mgr.latest().path, lambda d: loaded.append((d / "weights.txt").read_text()))  # type: ignore[union-attr]
    assert info.step == 10 and info.metric == 1.0
    assert loaded == ["w10"]
    assert _draw() == expected


def test_checkpoint_latest_best_and_prune(tmp_path: Path) -> None:
    """Giữ đúng keep_last bản mới nhất cộng keep_best bản tốt nhất, xóa phần còn lại và không để sót thư mục tạm."""
    mgr = CheckpointManager(tmp_path / "ck", keep_last=2, keep_best=1)
    for step, loss in [(100, 0.9), (200, 0.5), (300, 0.7), (400, 0.8)]:
        mgr.save(step, loss, _writer(str(step)))
    steps = [c.step for c in mgr.all()]
    assert steps == [200, 300, 400]  # 2 bản mới nhất (300, 400) + 1 bản tốt nhất (200)
    assert mgr.latest().step == 400  # type: ignore[union-attr]
    assert mgr.best().step == 200  # type: ignore[union-attr]
    assert [c.step for c in mgr.best_k(2)] == [200, 300]
    # Không còn thư mục tạm sót lại sau khi lưu (ghi nguyên tử).
    assert not list((tmp_path / "ck").glob(".tmp-*"))


def test_checkpoint_higher_is_better_and_no_metric(tmp_path: Path) -> None:
    """Với lower_is_better=False, metric lớn hơn là tốt hơn; checkpoint không có metric không được tính là tốt nhất."""
    mgr = CheckpointManager(tmp_path / "ck", keep_last=1, keep_best=1, lower_is_better=False)
    mgr.save(1, 0.2, _writer("a"))
    mgr.save(2, 0.9, _writer("b"))
    mgr.save(3, None, _writer("c"))
    assert mgr.best().step == 2  # type: ignore[union-attr]
    # Bản 3 không có metric vẫn được giữ làm bản mới nhất; bản 1 bị xóa.
    assert [c.step for c in mgr.all()] == [2, 3]


def test_metrics_logger_appends(tmp_path: Path) -> None:
    """MetricsLogger ghi nối từng dòng và tự thêm trường `time`."""
    m = MetricsLogger(tmp_path / "m" / "metrics.jsonl")
    m.log({"step": 1, "loss": 2.0})
    m.log({"step": 2, "eval_unseen_loss": 1.5})
    rows = m.read()
    assert [r["step"] for r in rows] == [1, 2] and all("time" in r for r in rows)


def test_write_train_config(tmp_path: Path) -> None:
    """train_config.json ghi lại phiên bản mã, sha256 dữ liệu và tên checkpoint."""
    data = tmp_path / "d.jsonl"
    data.write_text("x\n", encoding="utf-8")
    info = EnvInfo(None, None, None, "abc", False, {"lifemate": "0.1.0"})
    out = tmp_path / "train_config.json"
    write_train_config(out, {"train": {"seed": 0}}, info, {"train": sha256_file(data)}, checkpoint="ckpt-1")
    got = json.loads(out.read_text(encoding="utf-8"))
    assert got["env_info"]["code_version"] == "abc"
    assert got["data_sha256"]["train"] == sha256_file(data) and got["checkpoint"] == "ckpt-1"


def test_llm_train_config_strict(tmp_path: Path, repo_root: Path) -> None:
    """Đọc đúng llm_v1.toml và từ chối khóa không khai báo (lora.rank)."""
    cfg = load_llm_train_config(repo_root / "configs" / "train" / "llm_v1.toml")
    assert cfg.base.model == "Qwen/Qwen3-4B-Instruct-2507" and cfg.lora.r == 32 and cfg.train.max_len == 10240
    bad = tmp_path / "bad.toml"
    text = (repo_root / "configs" / "train" / "llm_v1.toml").read_text(encoding="utf-8")
    # Thêm khóa lạ `rank` vào bảng [lora]: bộ đọc cấu hình nghiêm ngặt phải báo lỗi.
    bad.write_text(text.replace("r = 32", "r = 32\nrank = 3"), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"lora\.rank"):
        load_llm_train_config(bad)


def test_asr_train_config(repo_root: Path) -> None:
    """Đọc đúng asr_v1.toml; hàm wer chuẩn hóa văn bản trước khi tính."""
    from lifemate.train.asr import load_asr_train_config, wer

    cfg = load_asr_train_config(repo_root / "configs" / "train" / "asr_v1.toml")
    assert cfg.method.kind == "full" and cfg.method.spec_augment and cfg.base.model == "vinai/PhoWhisper-medium"
    # WER bỏ qua hoa thường và dấu câu.
    assert wer(["Xin chào, Lan!"], ["xin chào lan"]) == 0.0
    assert wer(["a b c d"], ["a x c"]) == 0.5
