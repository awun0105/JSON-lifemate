"""Test `ops/docker/entrypoint.sh` ở chế độ in lệnh (`LIFEMATE_DRY_RUN=1`), không cần Docker hay GPU (SPEC 6.3).

Cách hoạt động:
- Dựng thư mục `/data` giả trong `tmp_path` với một file `*_tools_catalog.json`, chạy script bằng bash và đọc lệnh
  CLI mà script sẽ exec.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[1] / "ops" / "docker" / "entrypoint.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def run_entrypoint(tmp_path: Path, data: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Chạy entrypoint với `DATA_DIR=data`, kết quả và zaic_root trong `tmp_path`, ở chế độ in lệnh."""
    env = {
        **os.environ,
        "DATA_DIR": str(data),
        "RESULT_DIR": str(tmp_path / "result"),
        "ZAIC_ROOT": str(tmp_path / "zaic_root"),
        "LIFEMATE_DRY_RUN": "1",
    }
    return subprocess.run(["bash", str(ENTRYPOINT), *args], env=env, capture_output=True, text=True, check=False)


def test_entrypoint_runs_fresh_without_resume(tmp_path: Path) -> None:
    """Lệnh CLI không có `--resume`, split lấy từ tên catalog, tham số thêm được chuyển tiếp, symlink tới /data."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "private_test_tools_catalog.json").write_text("{}", encoding="utf-8")
    proc = run_entrypoint(tmp_path, data, "--limit", "5")
    assert proc.returncode == 0, proc.stderr
    cmd = proc.stdout.splitlines()
    assert "--resume" not in cmd
    assert cmd[cmd.index("--split") + 1] == "private_test"
    assert cmd[cmd.index("--out") + 1] == str(tmp_path / "result")
    assert cmd[-2:] == ["--limit", "5"]
    assert (tmp_path / "zaic_root" / "private_test").resolve() == data.resolve()


def test_entrypoint_moves_old_trace_aside(tmp_path: Path) -> None:
    """Thư mục kết quả còn `trace.jsonl` cũ thì được đổi tên `.bak`, để lần chạy này là lần chạy mới."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "private_test_tools_catalog.json").write_text("{}", encoding="utf-8")
    result = tmp_path / "result"
    result.mkdir()
    (result / "trace.jsonl").write_text('{"id": "old"}\n', encoding="utf-8")
    proc = run_entrypoint(tmp_path, data)
    assert proc.returncode == 0, proc.stderr
    assert not (result / "trace.jsonl").exists()
    baks = list(result.glob("trace.*.bak"))
    assert len(baks) == 1 and baks[0].read_text(encoding="utf-8") == '{"id": "old"}\n'
    assert "moved to" in proc.stderr


def test_entrypoint_rejects_zero_or_many_catalogs(tmp_path: Path) -> None:
    """Không có hoặc có hơn một `*_tools_catalog.json` trong /data thì thoát mã 2."""
    data = tmp_path / "data"
    data.mkdir()
    assert run_entrypoint(tmp_path, data).returncode == 2
    (data / "a_tools_catalog.json").write_text("{}", encoding="utf-8")
    (data / "b_tools_catalog.json").write_text("{}", encoding="utf-8")
    assert run_entrypoint(tmp_path, data).returncode == 2
