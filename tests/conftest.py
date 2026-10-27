"""Fixture dùng chung cho toàn bộ test.

Cung cấp gốc repo và gốc dữ liệu cuộc thi. Fixture `data_root` tự bỏ qua (skip) test khi thiếu gói dữ liệu.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root() -> Path:
    """Trả về thư mục gốc của repo."""
    return REPO_ROOT


@pytest.fixture
def data_root(repo_root: Path) -> Path:
    """Trả về thư mục dữ liệu cuộc thi; bỏ qua test khi thiếu gói dữ liệu."""
    root = repo_root / "data" / "train+public_test"
    if not root.is_dir():
        pytest.skip(f"data package missing: {root}")
    return root
