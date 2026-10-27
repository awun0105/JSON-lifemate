"""Load the per-split `zaic_tools` package and open one env per sample (SPEC 4.3).

Each split ships its own `zaic_tools` under the same module name, so one process runs one split.

Nạp gói `zaic_tools` riêng của từng split và mở một môi trường cho mỗi mẫu (SPEC 4.3). Mỗi split có bản
`zaic_tools` riêng nhưng cùng tên module, nên mỗi tiến trình chỉ chạy một split.

Cách xây dựng:
- `load_zaic` xóa mọi bản `zaic_tools` cũ trong `sys.modules`, chèn thư mục split lên đầu `sys.path` rồi import
  lại; sau đó kiểm tra file của module thật sự nằm trong thư mục split, phòng trường hợp import nhầm bản khác.
- `EnvRegistry` ghi nhớ các cặp (split, sample_id) đã mở: mỗi mẫu chỉ được mở môi trường đúng một lần trong
  một registry, tránh dùng lại trạng thái công cụ của lần chạy trước.
- `ZaicLoader` gói hai việc trên cho một thư mục dữ liệu: import một lần, mở môi trường theo từng mẫu.
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
    """Đường dẫn catalog công cụ: `<root>/<split>/<split>_tools_catalog.json`."""
    return Path(root) / split / f"{split}_tools_catalog.json"


def load_catalog(root: Path, split: str) -> dict[str, Any]:
    """Đọc và parse catalog công cụ của split."""
    with catalog_path(root, split).open(encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))


def load_zaic(root: Path, split: str) -> ModuleType:
    """Import `<root>/<split>/zaic_tools` sau khi xóa mọi bản cũ; báo RuntimeError nếu import phải bản khác.

    RuntimeError cũng được báo khi không import được module từ thư mục split.
    """
    split_dir = (Path(root) / split).resolve()
    for name in [m for m in sys.modules if m == "zaic_tools" or m.startswith("zaic_tools.")]:
        del sys.modules[name]
    sys.path.insert(0, str(split_dir))
    # Xóa cache finder để thư mục vừa chèn vào sys.path được quét lại.
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
    """Ghi nhớ các môi trường (split, sample_id) đã mở trong tiến trình này."""

    def __init__(self) -> None:
        self._opened: set[tuple[str, str]] = set()

    def claim(self, split: str, sample_id: str) -> None:
        """Đánh dấu đã mở; báo RuntimeError nếu cặp này đã được mở trước đó."""
        key = (split, sample_id)
        if key in self._opened:
            raise RuntimeError(f"env for {split}/{sample_id} already opened in this process")
        self._opened.add(key)


class ZaicLoader:
    """Nạp zaic_tools cho một thư mục dữ liệu và mở môi trường của mỗi mẫu tối đa một lần."""

    def __init__(self, root: Path, split: str, registry: EnvRegistry | None = None) -> None:
        self.root = Path(root)
        self.split = split
        self.registry = registry or EnvRegistry()
        self._module: ModuleType | None = None

    def load(self) -> ModuleType:
        """Import zaic_tools của split này (chỉ một lần) và trả về module."""
        if self._module is None:
            self._module = load_zaic(self.root, self.split)
        return self._module

    def open_env(self, sample_id: str) -> ToolEnvLike:
        """Phiên mới cho `sample_id`; gọi lần thứ hai với cùng id thì báo RuntimeError."""
        return open_env(self.load(), self.root, self.split, sample_id, registry=self.registry)


# Registry dùng chung cho cả tiến trình khi nơi gọi không truyền registry riêng.
_DEFAULT_REGISTRY = EnvRegistry()


def open_env(
    zaic: ModuleType, root: Path, split: str, sample_id: str, *, registry: EnvRegistry | None = None
) -> ToolEnvLike:
    """Gọi `zaic.ToolEnv(split, sample_id, root=root)` một lần cho mỗi cặp (split, sample_id) trong mỗi registry.

    Báo RuntimeError khi cặp đó đã được mở trong `registry` (mặc định là registry chung của tiến trình).
    """
    (registry or _DEFAULT_REGISTRY).claim(split, sample_id)
    return cast(ToolEnvLike, zaic.ToolEnv(split, sample_id, root=Path(root)))
