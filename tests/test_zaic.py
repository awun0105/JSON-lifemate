"""Kiểm tra lifemate.io.zaic: nạp module zaic của từng split, catalog và quy tắc mỗi mẫu một môi trường (T8).

Cách kiểm tra: cả file mang marker `data`, dùng gói dữ liệu cuộc thi thật qua fixture `data_root` và `EnvRegistry`.
Không cần tokenizer.
"""

from pathlib import Path

import pytest

from lifemate.io.data import load_split
from lifemate.io.zaic import EnvRegistry, ZaicLoader, load_catalog, load_zaic, open_env

pytestmark = pytest.mark.data


def test_load_zaic_file_under_split(data_root: Path) -> None:
    """Bảo vệ: module zaic được nạp từ đúng thư mục của split, catalog đọc ra dict."""
    z = load_zaic(data_root, "train")
    assert Path(z.__file__).resolve().is_relative_to((data_root / "train").resolve())
    assert isinstance(load_catalog(data_root, "train"), dict)


def test_one_env_per_sample(data_root: Path) -> None:
    """T8: mỗi mẫu chỉ được mở môi trường một lần; mở lần hai thì báo lỗi."""
    sid = load_split(data_root, "train")[0].id
    loader = ZaicLoader(data_root, "train")
    env = loader.open_env(sid)
    assert isinstance(env.get_state_hash(), str)
    with pytest.raises(RuntimeError):
        loader.open_env(sid)


def test_module_open_env_registry(data_root: Path) -> None:
    """Bảo vệ: hàm open_env cấp module cũng chặn mở trùng qua EnvRegistry."""
    z = load_zaic(data_root, "train")
    sid = load_split(data_root, "train")[1].id
    reg = EnvRegistry()
    open_env(z, data_root, "train", sid, registry=reg)
    with pytest.raises(RuntimeError):
        open_env(z, data_root, "train", sid, registry=reg)


def test_bogus_root_raises(tmp_path: Path, data_root: Path) -> None:
    """Bảo vệ: thư mục dữ liệu không tồn tại thì báo RuntimeError."""
    with pytest.raises(RuntimeError):
        load_zaic(tmp_path / "nowhere", "train")
