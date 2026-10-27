"""Docker entrypoint: split detection, symlink, exact CLI argv (SPEC 6.3); .dockerignore contents."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ENTRYPOINT = REPO / "ops" / "docker" / "entrypoint.sh"

FAKE_PYTHON = """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$ARGV_LOG"
"""


def _sandbox(tmp_path: Path, splits: list[str]) -> tuple[dict[str, str], Path, Path, Path]:
    data = tmp_path / "data"
    data.mkdir()
    for s in splits:
        (data / f"{s}_tools_catalog.json").write_text("{}", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "python"
    fake.write_text(FAKE_PYTHON, encoding="utf-8")
    fake.chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()
    log = tmp_path / "argv.txt"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "DATA_DIR": str(data),
        "ZAIC_ROOT": str(tmp_path / "zaic_root"),
        "ARGV_LOG": str(log),
    }
    env.pop("LIFEMATE_CONFIG", None)
    return env, data, work, log


def _run(env: dict[str, str], work: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(ENTRYPOINT), *extra], check=False, cwd=work, env=env, capture_output=True, text=True
    )


def test_detects_split_links_and_execs_cli(tmp_path: Path) -> None:
    env, data, work, log = _sandbox(tmp_path, ["private_test"])
    proc = _run(env, work, "--limit", "3")
    assert proc.returncode == 0, proc.stderr
    link = tmp_path / "zaic_root" / "private_test"
    assert link.is_symlink()
    assert link.resolve() == data.resolve()
    assert (work / "result").is_dir()
    argv = log.read_text(encoding="utf-8").splitlines()
    assert argv == [
        "-m",
        "lifemate.run.cli",
        "--config",
        "configs/docker.toml",
        "--env",
        "docker",
        "--split",
        "private_test",
        "--out",
        "result",
        "--resume",
        "--limit",
        "3",
    ]


def test_config_override(tmp_path: Path) -> None:
    env, _, work, log = _sandbox(tmp_path, ["public_test"])
    env["LIFEMATE_CONFIG"] = "configs/other.toml"
    assert _run(env, work).returncode == 0
    argv = log.read_text(encoding="utf-8").splitlines()
    assert argv[2:4] == ["--config", "configs/other.toml"]


def test_rerun_replaces_symlink(tmp_path: Path) -> None:
    env, data, work, _ = _sandbox(tmp_path, ["public_test"])
    assert _run(env, work).returncode == 0
    assert _run(env, work).returncode == 0
    assert (tmp_path / "zaic_root" / "public_test").resolve() == data.resolve()


@pytest.mark.parametrize("splits", [[], ["a", "b"]])
def test_fails_unless_exactly_one_catalog(tmp_path: Path, splits: list[str]) -> None:
    env, _, work, log = _sandbox(tmp_path, splits)
    proc = _run(env, work)
    assert proc.returncode != 0
    assert f"found {len(splits)}" in proc.stderr
    assert not log.exists()


def test_dockerignore_excludes_heavy_and_local_dirs() -> None:
    ignore = REPO / "ops" / "docker" / "Dockerfile.dockerignore"
    lines = {ln.strip() for ln in ignore.read_text(encoding="utf-8").splitlines()}
    for entry in ["data/", "artifacts/runs/", ".venv", "artifacts/dist/", "graphify-out/"]:
        assert entry in lines, entry
