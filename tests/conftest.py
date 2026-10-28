"""Fixture dùng chung cho toàn bộ test.

Nội dung chính:
- `repo_root`: thư mục gốc repo.
- `data_root`: thư mục dữ liệu cuộc thi; tự skip test khi thiếu gói dữ liệu.
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
    """Trả về thư mục dữ liệu cuộc thi; skip test khi thiếu gói dữ liệu."""
    root = repo_root / "data" / "train+public_test"
    if not root.is_dir():
        pytest.skip(f"data package missing: {root}")
    return root
