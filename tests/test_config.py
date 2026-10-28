"""Kiểm thử T19: profile môi trường và kiểm tra hợp lệ cấu hình chạy.

Đọc các file trong configs/ của repo và ghi cấu hình biến thể vào `tmp_path`.
Không cần dữ liệu cuộc thi hay tokenizer.
"""

from dataclasses import fields
from pathlib import Path

import pytest

from lifemate.core.config import ConfigError, EnvProfile, load_config, load_profile

PROFILES = ["local", "kaggle", "colab", "rented", "runpod", "docker"]


@pytest.mark.parametrize("name", PROFILES)
def test_env_profiles(name: str, repo_root: Path) -> None:
    """T19. Bảo vệ: mọi profile có đủ khóa; tham số CLI ghi đè profile. Không chứng minh: đường dẫn tồn tại."""
    prof = load_profile(name, repo_root)
    for f in fields(EnvProfile):
        if f.name != "hf_cache":
            assert getattr(prof, f.name), f.name
    assert prof.with_overrides() == prof
    over = prof.with_overrides(root="/r", out="/o")
    assert (over.data_root, over.runs_dir) == ("/r", "/o")
    assert over.hf_token_source == prof.hf_token_source
    assert prof.with_overrides(out="/o").data_root == prof.data_root


def test_profile_by_path_and_unknown_key(tmp_path: Path, repo_root: Path) -> None:
    """Bắt lỗi: khóa profile gõ sai mà vẫn lọt qua im lặng."""
    p = tmp_path / "x.toml"
    p.write_text('data_root="a"\nruns_dir="b"\nhf_token_source="none"\npersist="local"\nbogus=1\n')
    with pytest.raises(ConfigError, match="bogus"):
        load_profile(str(p), repo_root)


def test_hf_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bảo vệ: token được đọc từ biến môi trường; `none` hoặc biến vắng mặt cho None."""
    prof = EnvProfile("a", "b", "env:LM_TEST_TOKEN", "local")
    monkeypatch.setenv("LM_TEST_TOKEN", "secret")
    assert prof.resolve_hf_token() == "secret"
    monkeypatch.delenv("LM_TEST_TOKEN")
    assert prof.resolve_hf_token() is None
    assert EnvProfile("a", "b", "none", "local").resolve_hf_token() is None


def test_v0_config_loads_with_fake(tmp_path: Path, repo_root: Path) -> None:
    """Bảo vệ: v0.toml đọc được; với backend=fake thì model rỗng vẫn hợp lệ."""
    text = (
        (repo_root / "configs" / "v0.toml").read_text(encoding="utf-8").replace('backend = "vllm"', 'backend = "fake"')
    )
    p = tmp_path / "c.toml"
    p.write_text(text, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.agent.guards == ["allowed_tool", "schema", "duplicate_write"]


def test_v0_and_docker_configs_load(repo_root: Path) -> None:
    """Bảo vệ: v0.toml và docker.toml đọc được và đã điền mô hình đã chọn (SPEC 6.0)."""
    v0 = load_config(repo_root / "configs" / "v0.toml")
    assert v0.llm.model and v0.llm.revision and v0.asr.model and v0.asr.revision
    assert v0.asr.engine == "faster_whisper"
    dk = load_config(repo_root / "configs" / "docker.toml")
    assert dk.llm.model.startswith("/app/models/") and dk.asr.backend == "asr"
    assert dk.agent == v0.agent and dk.llm.max_model_len == v0.llm.max_model_len


def test_vllm_empty_model_raises(tmp_path: Path, repo_root: Path) -> None:
    """Bắt lỗi: khởi động vLLM khi chưa chọn mô hình."""
    text = (
        (repo_root / "configs" / "v0.toml")
        .read_text(encoding="utf-8")
        .replace('model = "Qwen/Qwen3-4B-Instruct-2507"', 'model = ""', 1)
    )
    p = tmp_path / "c.toml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=r"llm\.model"):
        load_config(p)


@pytest.mark.parametrize(
    ("old", "new", "key"),
    [
        ('engine = "faster_whisper"', 'engine = "nemo"', r"asr\.engine"),
        ('model = "Systran/faster-whisper-large-v3"', 'model = ""', r"asr\.model"),
        ('dtype = "auto"                         # auto: bfloat16', 'dtype = "int8"  # auto: bfloat16', r"asr\.dtype"),
    ],
)
def test_asr_engine_and_model_validated(tmp_path: Path, repo_root: Path, old: str, new: str, key: str) -> None:
    """Bắt lỗi: engine ASR không xác định, dtype ASR sai, hoặc backend=asr mà không có mô hình."""
    text = (repo_root / "configs" / "v0.toml").read_text(encoding="utf-8")
    text = (
        text.replace('backend = "vllm"', 'backend = "fake"')
        .replace('backend = "gold"', 'backend = "asr"')
        .replace(old, new, 1)
    )
    p = tmp_path / "c.toml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=key):
        load_config(p)


@pytest.mark.parametrize(
    ("old", "new", "key"), [("seed = 0", "seed = 0\nbogus = 1", "llm.bogus"), ("seed = 0\n", "", "llm.seed")]
)
def test_unknown_or_missing_key_raises(tmp_path: Path, repo_root: Path, old: str, new: str, key: str) -> None:
    """Bắt lỗi: khóa lạ hoặc thiếu khóa mà vẫn được chấp nhận im lặng."""
    text = (repo_root / "configs" / "v0.toml").read_text(encoding="utf-8").replace(old, new, 1)
    p = tmp_path / "c.toml"
    p.write_text(text.replace('backend = "vllm"', 'backend = "fake"'), encoding="utf-8")
    with pytest.raises(ConfigError, match=key):
        load_config(p)


def test_ct2_int8_dtype_only_for_faster_whisper(tmp_path: Path, repo_root: Path) -> None:
    """Bảo vệ: asr.dtype "int8_float16" hợp lệ với faster_whisper (Kaggle T4 cạnh vLLM), bị từ chối với engine khác."""
    text = (repo_root / "configs" / "v0.toml").read_text(encoding="utf-8")
    text = text.replace('backend = "vllm"', 'backend = "fake"').replace(
        'dtype = "auto"                         # auto: bfloat16', 'dtype = "int8_float16"  # auto: bfloat16', 1
    )
    p = tmp_path / "c.toml"
    p.write_text(text, encoding="utf-8")
    assert load_config(p).asr.dtype == "int8_float16"
    p.write_text(text.replace('engine = "faster_whisper"', 'engine = "hf_seq2seq"', 1), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"asr\.dtype"):
        load_config(p)
