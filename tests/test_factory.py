"""Kiểm tra cách `factory.build` lắp các thành phần và registry backend (SPEC 4.0).

Config lấy từ `configs/v0.toml`, đổi backend sang `fake` qua `fake_config`. Test `test_build_fake_gold` cần dữ liệu
cuộc thi (marker `data`, fixture `data_root`); test còn lại không cần. Không cần tokenizer.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from lifemate.backends.llm import FakeLLM
from lifemate.core.config import Config, ConfigError, load_config, load_profile
from lifemate.core.envinfo import EnvInfo
from lifemate.core.tokens import SimpleTokenCounter
from lifemate.run import factory
from lifemate.run.timing import TimedLLM

ENV_INFO = EnvInfo(None, None, None, "test", None, {})


def fake_config(repo_root: Path, tmp_path: Path, backend: str = "fake") -> Config:
    """Đọc config v0, thay backend vllm bằng `backend` để không cần GPU."""
    text = (
        (repo_root / "configs" / "v0.toml").read_text(encoding="utf-8").replace('backend = "vllm"', 'backend = "fake"')
    )
    (tmp_path / "fake.toml").write_text(text, encoding="utf-8")
    cfg = load_config(tmp_path / "fake.toml")
    return dataclasses.replace(cfg, llm=dataclasses.replace(cfg.llm, backend=backend))


@pytest.mark.data
def test_build_fake_gold(repo_root: Path, data_root: Path, tmp_path: Path) -> None:
    """Backend fake được bọc `TimedLLM`, dùng bộ đếm token đơn giản, và agent dùng chung catalog index."""
    cfg = fake_config(repo_root, tmp_path)
    profile = load_profile("local", repo_root).with_overrides(root=str(data_root))
    built = factory.build(cfg, profile, repo_root=repo_root, split="train", env_info=ENV_INFO)
    assert isinstance(built.llm, TimedLLM) and isinstance(built.llm.inner, FakeLLM)
    assert isinstance(built.counter, SimpleTokenCounter)
    assert built.dtype == "float16" and len(built.prompt_sha256) == 64
    assert built.loader.split == "train" and built.agent.index is built.index


def test_unknown_backend(repo_root: Path, tmp_path: Path) -> None:
    """Tên backend lạ gây `ConfigError` nêu khoá `llm.backend`."""
    cfg = fake_config(repo_root, tmp_path, backend="nope")
    profile = load_profile("local", repo_root)
    with pytest.raises(ConfigError, match=r"llm\.backend"):
        factory.build(cfg, profile, repo_root=repo_root, split="train", env_info=ENV_INFO)
