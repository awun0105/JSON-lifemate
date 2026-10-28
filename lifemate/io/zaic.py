"""Nạp gói `zaic_tools` riêng của từng split và mở một môi trường tool (env) cho mỗi mẫu (SPEC 4.3).

Nội dung chính:
- `catalog_path`, `load_catalog`: tìm và đọc catalog tool của split.
- `load_zaic`: import `zaic_tools` của đúng split cần chạy.
- `EnvRegistry`: ghi nhớ các mẫu đã mở env, để mỗi mẫu chỉ mở một lần.
- `ZaicLoader`: gói hai việc trên cho một thư mục dữ liệu.
- `open_env`: mở env cho một mẫu.

Cách hoạt động:
- `load_zaic` xoá mọi bản `zaic_tools` cũ trong `sys.modules`, chèn thư mục split lên đầu `sys.path` rồi import
  lại. Sau đó kiểm tra file của module có thật sự nằm trong thư mục split, phòng khi import nhầm bản khác.
- `EnvRegistry` ghi nhớ các cặp (split, sample_id) đã mở. Mỗi mẫu chỉ được mở env đúng một lần trong một
  registry, để không dùng lại trạng thái tool của lần chạy trước.
- `ZaicLoader` import module một lần, rồi mở env theo từng mẫu.

Lưu ý:
- Mỗi split có bản `zaic_tools` riêng nhưng cùng tên module, nên mỗi tiến trình chỉ chạy một split.
- Đây là module duy nhất được phép đụng tới `*_environment/state.json` (qua `zaic_tools`); test kiểm tra điều này.
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
    """Trả về đường dẫn catalog tool: `<root>/<split>/<split>_tools_catalog.json`."""
    return Path(root) / split / f"{split}_tools_catalog.json"


def load_catalog(root: Path, split: str) -> dict[str, Any]:
    """Đọc và parse catalog tool của split."""
    with catalog_path(root, split).open(encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))


def load_zaic(root: Path, split: str) -> ModuleType:
    """Import `<root>/<split>/zaic_tools` sau khi xoá mọi bản cũ đã nạp.

    Tham số:
        root: thư mục dữ liệu.
        split: tên split.

    Trả về:
        Module `zaic_tools` của split.

    Lỗi:
        RuntimeError: khi không import được module từ thư mục split, hoặc module được import lại nằm ở chỗ khác.
    """
    split_dir = (Path(root) / split).resolve()
    for name in [m for m in sys.modules if m == "zaic_tools" or m.startswith("zaic_tools.")]:
        del sys.modules[name]
    sys.path.insert(0, str(split_dir))
    # Xoá cache của finder để thư mục vừa chèn vào `sys.path` được quét lại.
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
    """Ghi nhớ các env (split, sample_id) đã mở trong tiến trình này."""

    def __init__(self) -> None:
        self._opened: set[tuple[str, str]] = set()

    def claim(self, split: str, sample_id: str) -> None:
        """Đánh dấu cặp (split, sample_id) đã mở; báo `RuntimeError` nếu cặp này đã được mở trước đó."""
        key = (split, sample_id)
        if key in self._opened:
            raise RuntimeError(f"env for {split}/{sample_id} already opened in this process")
        self._opened.add(key)


class ZaicLoader:
    """Nạp `zaic_tools` cho một thư mục dữ liệu và mở env cho mỗi mẫu tối đa một lần.

    Tham số:
        root: thư mục dữ liệu.
        split: tên split.
        registry: registry ghi nhớ env đã mở; `None` thì tạo registry mới.

    Thuộc tính:
        root: như tham số cùng tên.
        split: như tham số cùng tên.
        registry: registry đang dùng.
    """

    def __init__(self, root: Path, split: str, registry: EnvRegistry | None = None) -> None:
        self.root = Path(root)
        self.split = split
        self.registry = registry or EnvRegistry()
        self._module: ModuleType | None = None

    def load(self) -> ModuleType:
        """Import `zaic_tools` của split này (chỉ lần đầu) và trả về module."""
        if self._module is None:
            self._module = load_zaic(self.root, self.split)
        return self._module

    def open_env(self, sample_id: str) -> ToolEnvLike:
        """Mở env mới cho `sample_id`; gọi lần thứ hai với cùng id thì báo `RuntimeError`."""
        return open_env(self.load(), self.root, self.split, sample_id, registry=self.registry)


# Registry dùng chung cho cả tiến trình, khi nơi gọi không truyền registry riêng.
_DEFAULT_REGISTRY = EnvRegistry()


def open_env(
    zaic: ModuleType, root: Path, split: str, sample_id: str, *, registry: EnvRegistry | None = None
) -> ToolEnvLike:
    """Gọi `zaic.ToolEnv(split, sample_id, root=root)`, mỗi cặp (split, sample_id) chỉ một lần trong một registry.

    Tham số:
        zaic: module `zaic_tools` đã nạp.
        root: thư mục dữ liệu.
        split: tên split.
        sample_id: id mẫu.
        registry: registry ghi nhớ env đã mở; `None` là registry chung của tiến trình.

    Trả về:
        Env của mẫu.

    Lỗi:
        RuntimeError: khi cặp (split, sample_id) đã được mở trong registry.
    """
    (registry or _DEFAULT_REGISTRY).claim(split, sample_id)
    return cast(ToolEnvLike, zaic.ToolEnv(split, sample_id, root=Path(root)))
