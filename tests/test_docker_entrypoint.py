"""Test entrypoint Docker: nhận diện split, tạo symlink, argv CLI đúng (SPEC 6.3), và nội dung `.dockerignore`.

Cách hoạt động:
- Chạy `ops/docker/entrypoint.sh` thật bằng bash trong sandbox `tmp_path`.
- Lệnh `python` được thay bằng script giả `FAKE_PYTHON`, chỉ ghi argv ra file để so sánh.

Lưu ý:
- Không cần dữ liệu cuộc thi hay tokenizer.
"""

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
    """Dựng sandbox: thư mục data có catalog cho từng split, `python` giả đứng đầu `PATH`.

    Trả về:
        Bộ bốn (biến môi trường, thư mục data, thư mục làm việc, file log argv).
    """
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
    # Bỏ biến môi trường có sẵn để entrypoint dùng config mặc định.
    env.pop("LIFEMATE_CONFIG", None)
    return env, data, work, log


def _run(env: dict[str, str], work: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    """Chạy entrypoint trong thư mục `work`; không ném lỗi khi mã thoát khác 0."""
    return subprocess.run(
        ["bash", str(ENTRYPOINT), *extra], check=False, cwd=work, env=env, capture_output=True, text=True
    )


def test_detects_split_links_and_execs_cli(tmp_path: Path) -> None:
    """Entrypoint tìm split từ tên catalog, symlink data vào `ZAIC_ROOT`, tạo `result/` và gọi CLI đúng argv."""
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
    """Biến `LIFEMATE_CONFIG` thay đường dẫn config mặc định trong argv."""
    env, _, work, log = _sandbox(tmp_path, ["public_test"])
    env["LIFEMATE_CONFIG"] = "configs/other.toml"
    assert _run(env, work).returncode == 0
    argv = log.read_text(encoding="utf-8").splitlines()
    assert argv[2:4] == ["--config", "configs/other.toml"]


def test_rerun_replaces_symlink(tmp_path: Path) -> None:
    """Chạy lại lần hai không lỗi dù symlink đã có, và symlink vẫn trỏ đúng thư mục data."""
    env, data, work, _ = _sandbox(tmp_path, ["public_test"])
    assert _run(env, work).returncode == 0
    assert _run(env, work).returncode == 0
    assert (tmp_path / "zaic_root" / "public_test").resolve() == data.resolve()


@pytest.mark.parametrize("splits", [[], ["a", "b"]])
def test_fails_unless_exactly_one_catalog(tmp_path: Path, splits: list[str]) -> None:
    """Có 0 hoặc nhiều hơn 1 catalog thì entrypoint dừng với lỗi và không gọi python."""
    env, _, work, log = _sandbox(tmp_path, splits)
    proc = _run(env, work)
    assert proc.returncode != 0
    assert f"found {len(splits)}" in proc.stderr
    assert not log.exists()


def test_dockerignore_excludes_heavy_and_local_dirs() -> None:
    """`.dockerignore` loại các thư mục nặng hoặc chỉ dùng trên máy khỏi build context."""
    ignore = REPO / "ops" / "docker" / "Dockerfile.dockerignore"
    lines = {ln.strip() for ln in ignore.read_text(encoding="utf-8").splitlines()}
    for entry in ["data/", "artifacts/runs/", ".venv", "artifacts/dist/", "graphify-out/"]:
        assert entry in lines, entry
