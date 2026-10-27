"""Shared fixtures."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root() -> Path:
    """Repository root."""
    return REPO_ROOT


@pytest.fixture
def data_root(repo_root: Path) -> Path:
    """Competition data root; skips when the data package is absent."""
    root = repo_root / "data" / "train+public_test"
    if not root.is_dir():
        pytest.skip(f"data package missing: {root}")
    return root
