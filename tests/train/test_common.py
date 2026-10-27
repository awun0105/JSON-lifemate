"""seed_everything, CheckpointManager, MetricsLogger, write_train_config (SPEC 6.5)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from lifemate.config import ConfigError
from lifemate.envinfo import EnvInfo
from lifemate.train.common import (
    CheckpointManager,
    MetricsLogger,
    seed_everything,
    sha256_file,
    write_train_config,
)
from lifemate.train.sft import load_llm_train_config


def _draw() -> tuple[list[float], list[float]]:
    return [random.random() for _ in range(5)], np.random.rand(5).tolist()


def test_seed_everything_reproducible() -> None:
    seed_everything(123)
    a = _draw()
    seed_everything(123)
    assert _draw() == a
    seed_everything(124)
    assert _draw() != a


def _writer(payload: str):  # type: ignore[no-untyped-def]
    def save(d: Path) -> None:
        (d / "weights.txt").write_text(payload, encoding="utf-8")

    return save


def test_checkpoint_round_trip_restores_step_and_rng(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path / "ck", keep_last=1, keep_best=1)
    seed_everything(7)
    mgr.save(10, 1.0, _writer("w10"))
    expected = _draw()  # what comes right after the save
    _draw()  # move the RNG on
    loaded: list[str] = []
    info = mgr.load(mgr.latest().path, lambda d: loaded.append((d / "weights.txt").read_text()))  # type: ignore[union-attr]
    assert info.step == 10 and info.metric == 1.0
    assert loaded == ["w10"]
    assert _draw() == expected


def test_checkpoint_latest_best_and_prune(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path / "ck", keep_last=2, keep_best=1)
    for step, loss in [(100, 0.9), (200, 0.5), (300, 0.7), (400, 0.8)]:
        mgr.save(step, loss, _writer(str(step)))
    steps = [c.step for c in mgr.all()]
    assert steps == [200, 300, 400]  # last 2 (300, 400) + best 1 (200)
    assert mgr.latest().step == 400  # type: ignore[union-attr]
    assert mgr.best().step == 200  # type: ignore[union-attr]
    assert [c.step for c in mgr.best_k(2)] == [200, 300]
    assert not list((tmp_path / "ck").glob(".tmp-*"))


def test_checkpoint_higher_is_better_and_no_metric(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path / "ck", keep_last=1, keep_best=1, lower_is_better=False)
    mgr.save(1, 0.2, _writer("a"))
    mgr.save(2, 0.9, _writer("b"))
    mgr.save(3, None, _writer("c"))
    assert mgr.best().step == 2  # type: ignore[union-attr]
    assert [c.step for c in mgr.all()] == [2, 3]


def test_metrics_logger_appends(tmp_path: Path) -> None:
    m = MetricsLogger(tmp_path / "m" / "metrics.jsonl")
    m.log({"step": 1, "loss": 2.0})
    m.log({"step": 2, "eval_unseen_loss": 1.5})
    rows = m.read()
    assert [r["step"] for r in rows] == [1, 2] and all("time" in r for r in rows)


def test_write_train_config(tmp_path: Path) -> None:
    data = tmp_path / "d.jsonl"
    data.write_text("x\n", encoding="utf-8")
    info = EnvInfo(None, None, None, "abc", False, {"lifemate": "0.1.0"})
    out = tmp_path / "train_config.json"
    write_train_config(out, {"train": {"seed": 0}}, info, {"train": sha256_file(data)}, checkpoint="ckpt-1")
    got = json.loads(out.read_text(encoding="utf-8"))
    assert got["env_info"]["code_version"] == "abc"
    assert got["data_sha256"]["train"] == sha256_file(data) and got["checkpoint"] == "ckpt-1"


def test_llm_train_config_strict(tmp_path: Path, repo_root: Path) -> None:
    cfg = load_llm_train_config(repo_root / "configs" / "train" / "llm_v1.toml")
    assert cfg.base.model == "" and cfg.lora.r == 16 and cfg.train.max_len == 16384
    bad = tmp_path / "bad.toml"
    text = (repo_root / "configs" / "train" / "llm_v1.toml").read_text(encoding="utf-8")
    bad.write_text(text.replace("r = 16", "r = 16\nrank = 3"), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"lora\.rank"):
        load_llm_train_config(bad)


def test_asr_train_config(repo_root: Path) -> None:
    from lifemate.train.asr import load_asr_train_config, wer

    cfg = load_asr_train_config(repo_root / "configs" / "train" / "asr_v1.toml")
    assert cfg.method.kind == "lora" and cfg.method.spec_augment and cfg.base.model == ""
    assert wer(["Xin chào, Lan!"], ["xin chào lan"]) == 0.0
    assert wer(["a b c d"], ["a x c"]) == 0.5
