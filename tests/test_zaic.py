from pathlib import Path

import pytest

from lifemate.data import load_split
from lifemate.zaic import EnvRegistry, ZaicLoader, load_catalog, load_zaic, open_env

pytestmark = pytest.mark.data


def test_load_zaic_file_under_split(data_root: Path) -> None:
    z = load_zaic(data_root, "train")
    assert Path(z.__file__).resolve().is_relative_to((data_root / "train").resolve())
    assert isinstance(load_catalog(data_root, "train"), dict)


def test_one_env_per_sample(data_root: Path) -> None:  # T8
    sid = load_split(data_root, "train")[0].id
    loader = ZaicLoader(data_root, "train")
    env = loader.open_env(sid)
    assert isinstance(env.get_state_hash(), str)
    with pytest.raises(RuntimeError):
        loader.open_env(sid)


def test_module_open_env_registry(data_root: Path) -> None:
    z = load_zaic(data_root, "train")
    sid = load_split(data_root, "train")[1].id
    reg = EnvRegistry()
    open_env(z, data_root, "train", sid, registry=reg)
    with pytest.raises(RuntimeError):
        open_env(z, data_root, "train", sid, registry=reg)


def test_bogus_root_raises(tmp_path: Path, data_root: Path) -> None:
    with pytest.raises(RuntimeError):
        load_zaic(tmp_path / "nowhere", "train")
