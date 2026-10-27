"""Load the per-split `zaic_tools` package and open one env per sample (SPEC 4.3).

Each split ships its own `zaic_tools` under the same module name, so one process runs one split.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from lifemate.core.types import ToolEnvLike


def catalog_path(root: Path, split: str) -> Path:
    """`<root>/<split>/<split>_tools_catalog.json`."""
    return Path(root) / split / f"{split}_tools_catalog.json"


def load_catalog(root: Path, split: str) -> dict[str, Any]:
    """Parse the split's tool catalog."""
    with catalog_path(root, split).open(encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))


def load_zaic(root: Path, split: str) -> ModuleType:
    """Import `<root>/<split>/zaic_tools`, purging any earlier copy; raise RuntimeError if another copy was imported."""
    split_dir = (Path(root) / split).resolve()
    for name in [m for m in sys.modules if m == "zaic_tools" or m.startswith("zaic_tools.")]:
        del sys.modules[name]
    sys.path.insert(0, str(split_dir))
    importlib.invalidate_caches()
    try:
        module = importlib.import_module("zaic_tools")
    except ImportError as e:
        raise RuntimeError(f"cannot import zaic_tools from {split_dir}: {e}") from e
    file = getattr(module, "__file__", None)
    if file is None or not Path(file).resolve().is_relative_to(split_dir):
        raise RuntimeError(f"zaic_tools imported from {file!r}, expected under {split_dir}")
    return module


class EnvRegistry:
    """Remembers which (split, sample_id) envs were opened in this process."""

    def __init__(self) -> None:
        self._opened: set[tuple[str, str]] = set()

    def claim(self, split: str, sample_id: str) -> None:
        """Mark as opened; raise RuntimeError if it already was."""
        key = (split, sample_id)
        if key in self._opened:
            raise RuntimeError(f"env for {split}/{sample_id} already opened in this process")
        self._opened.add(key)


class ZaicLoader:
    """Loads zaic_tools for one data root and opens each sample's env at most once."""

    def __init__(self, root: Path, split: str, registry: EnvRegistry | None = None) -> None:
        self.root = Path(root)
        self.split = split
        self.registry = registry or EnvRegistry()
        self._module: ModuleType | None = None

    def load(self) -> ModuleType:
        """Import (once) and return zaic_tools for this split."""
        if self._module is None:
            self._module = load_zaic(self.root, self.split)
        return self._module

    def open_env(self, sample_id: str) -> ToolEnvLike:
        """Fresh session for `sample_id`; RuntimeError on a second call for the same id."""
        return open_env(self.load(), self.root, self.split, sample_id, registry=self.registry)


_DEFAULT_REGISTRY = EnvRegistry()


def open_env(
    zaic: ModuleType, root: Path, split: str, sample_id: str, *, registry: EnvRegistry | None = None
) -> ToolEnvLike:
    """Call `zaic.ToolEnv(split, sample_id, root=root)` once per (split, sample_id) per registry."""
    (registry or _DEFAULT_REGISTRY).claim(split, sample_id)
    return cast(ToolEnvLike, zaic.ToolEnv(split, sample_id, root=Path(root)))
