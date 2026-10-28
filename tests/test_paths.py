"""Test `repo_path` và `local_data_root` trong `lifemate.core.paths`.

Lưu ý:
- Không cần dữ liệu cuộc thi: chỉ kiểm tra cách tính đường dẫn.
"""

from pathlib import Path

import pytest

from lifemate.core.paths import REPO_ROOT, local_data_root, repo_path


def test_repo_path_relative_absolute_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Đường dẫn tương đối tính từ gốc repo chứ không từ thư mục hiện tại; `~` được mở rộng; tuyệt đối giữ nguyên."""
    monkeypatch.chdir(tmp_path)
    assert repo_path("data/x", Path("/repo")) == Path("/repo/data/x")
    assert repo_path("/abs/x", Path("/repo")) == Path("/abs/x")
    assert repo_path("~/x", Path("/repo")) == Path.home() / "x"
    assert repo_path("data/x") == REPO_ROOT / "data" / "x"


def test_local_data_root_is_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Thư mục dữ liệu mặc định luôn là đường dẫn tuyệt đối, kể cả khi chạy từ thư mục khác."""
    monkeypatch.chdir(tmp_path)
    root = local_data_root()
    assert root.is_absolute()
    assert root == local_data_root(REPO_ROOT)
