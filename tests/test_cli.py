"""End-to-end CLI with FakeLLM, token leak and source scans (SPEC 4.11, 4.16; T1, T9, T21, T23)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lifemate.run import cli

SENTINEL = "hf_SENTINEL_do_not_leak_123"


def fake_config(repo_root: Path, tmp_path: Path) -> Path:
    text = (
        (repo_root / "configs" / "v0.toml").read_text(encoding="utf-8").replace('backend = "vllm"', 'backend = "fake"')
    )
    path = tmp_path / "fake.toml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.data
@pytest.mark.parametrize("ids", ["configs/splits/val_seen.txt", "configs/splits/val_unseen.txt"])
def test_fake_llm_end_to_end(repo_root: Path, data_root: Path, tmp_path: Path, ids: str) -> None:
    """T1: replaying gold through the whole Runner scores 100.00 in both match modes."""
    argv = ["--config", str(fake_config(repo_root, tmp_path)), "--env", "local", "--split", "train"]
    argv += ["--ids", ids, "--root", str(data_root), "--out", str(tmp_path / "run")]
    assert cli.main(argv) == 0
    score = (tmp_path / "run" / "score.txt").read_text(encoding="utf-8")
    assert "Final score [strict]: 100.00" in score
    assert "Final score [canon]: 100.00" in score
    n = len((repo_root / ids).read_text(encoding="utf-8").split())
    assert len((tmp_path / "run" / "submission.jsonl").read_text(encoding="utf-8").splitlines()) == n


@pytest.mark.data
def test_no_token_leak(repo_root: Path, data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T23: the HF token never reaches config.json, trace.jsonl or summary.json."""
    monkeypatch.setenv("HF_TOKEN", SENTINEL)
    argv = ["--config", str(fake_config(repo_root, tmp_path)), "--env", "local", "--split", "train"]
    argv += ["--ids", "configs/splits/smoke.txt", "--limit", "2", "--root", str(data_root)]
    argv += ["--out", str(tmp_path / "run")]
    assert cli.main(argv) == 0
    for name in ("config.json", "trace.jsonl", "summary.json"):
        assert SENTINEL not in (tmp_path / "run" / name).read_text(encoding="utf-8")


def _sources(repo_root: Path) -> list[Path]:
    return sorted((repo_root / "lifemate").rglob("*.py"))


def test_no_environment_access(repo_root: Path) -> None:
    """T9: only zaic.py may mention `_environment` or `state.json`."""
    for path in _sources(repo_root):
        if path.name == "zaic.py":
            continue
        text = path.read_text(encoding="utf-8")
        for needle in ("_environment", "state.json"):
            assert needle not in text, f"{path}: {needle}"


def test_no_hardcoded_paths(repo_root: Path) -> None:
    """T21: no machine paths inside lifemate/."""
    for path in _sources(repo_root):
        text = path.read_text(encoding="utf-8")
        for needle in ("/kaggle", "/content", "/home", "train+public_test"):
            assert needle not in text, f"{path}: {needle}"
