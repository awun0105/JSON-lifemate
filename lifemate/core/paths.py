"""Repo root: one definition instead of `Path(__file__).parent.parent...` counted per module."""

from __future__ import annotations

from pathlib import Path

# lifemate/core/paths.py -> lifemate/core -> lifemate -> repo
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
