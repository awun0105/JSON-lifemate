"""T19 env profiles and run config validation."""

from dataclasses import fields
from pathlib import Path

import pytest

from lifemate.core.config import ConfigError, EnvProfile, load_config, load_profile

PROFILES = ["local", "kaggle", "colab", "rented", "runpod", "docker"]


@pytest.mark.parametrize("name", PROFILES)
def test_env_profiles(name: str, repo_root: Path) -> None:
    """T19. Protects: every profile has all keys; CLI > profile. Doesn't prove: paths exist."""
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
    """Catches: typos in profile keys passing silently."""
    p = tmp_path / "x.toml"
    p.write_text('data_root="a"\nruns_dir="b"\nhf_token_source="none"\npersist="local"\nbogus=1\n')
    with pytest.raises(ConfigError, match="bogus"):
        load_profile(str(p), repo_root)


def test_hf_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protects: token read from env; none/absent give None."""
    prof = EnvProfile("a", "b", "env:LM_TEST_TOKEN", "local")
    monkeypatch.setenv("LM_TEST_TOKEN", "secret")
    assert prof.resolve_hf_token() == "secret"
    monkeypatch.delenv("LM_TEST_TOKEN")
    assert prof.resolve_hf_token() is None
    assert EnvProfile("a", "b", "none", "local").resolve_hf_token() is None


def test_v0_config_loads_with_fake(tmp_path: Path, repo_root: Path) -> None:
    """v0.toml parses; with backend=fake an empty model is fine."""
    text = (
        (repo_root / "configs" / "v0.toml").read_text(encoding="utf-8").replace('backend = "vllm"', 'backend = "fake"')
    )
    p = tmp_path / "c.toml"
    p.write_text(text, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.agent.guards == ["allowed_tool", "schema", "duplicate_write"]


def test_v0_and_docker_configs_load(repo_root: Path) -> None:
    """v0.toml and docker.toml parse with the chosen models filled in (SPEC 6.0)."""
    v0 = load_config(repo_root / "configs" / "v0.toml")
    assert v0.llm.model and v0.llm.revision and v0.asr.model and v0.asr.revision
    assert v0.asr.engine == "hf_seq2seq"
    dk = load_config(repo_root / "configs" / "docker.toml")
    assert dk.llm.model.startswith("/app/models/") and dk.asr.backend == "asr"
    assert dk.agent == v0.agent and dk.llm.max_model_len == v0.llm.max_model_len


def test_vllm_empty_model_raises(tmp_path: Path, repo_root: Path) -> None:
    """Catches: starting vLLM with no model chosen."""
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
        ('engine = "hf_seq2seq"', 'engine = "nemo"', r"asr\.engine"),
        ('model = "vinai/PhoWhisper-medium"', 'model = ""', r"asr\.model"),
        ('dtype = "auto"                         # auto: bfloat16', 'dtype = "int8"  # auto: bfloat16', r"asr\.dtype"),
    ],
)
def test_asr_engine_and_model_validated(tmp_path: Path, repo_root: Path, old: str, new: str, key: str) -> None:
    """Catches: unknown ASR engine, or backend=asr with no model."""
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
    """Catches: unknown or missing keys accepted silently."""
    text = (repo_root / "configs" / "v0.toml").read_text(encoding="utf-8").replace(old, new, 1)
    p = tmp_path / "c.toml"
    p.write_text(text.replace('backend = "vllm"', 'backend = "fake"'), encoding="utf-8")
    with pytest.raises(ConfigError, match=key):
        load_config(p)
