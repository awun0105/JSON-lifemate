"""Test T19: profile môi trường và kiểm tra config chạy.

Cách hoạt động:
- Đọc các file trong `configs/` của repo và ghi config biến thể vào `tmp_path`.

Lưu ý:
- Không cần dữ liệu cuộc thi hay tokenizer.
"""

from dataclasses import fields
from pathlib import Path

import pytest

from lifemate.core.config import ConfigError, EnvProfile, load_config, load_profile

PROFILES = ["local", "kaggle", "colab", "rented", "runpod", "docker"]


@pytest.mark.parametrize("name", PROFILES)
def test_env_profiles(name: str, repo_root: Path) -> None:
    """T19: mọi profile có đủ khoá và tham số CLI ghi đè được profile (test không kiểm tra đường dẫn có tồn tại)."""
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
    """Profile đọc được theo đường dẫn, và khoá gõ sai bị báo lỗi chứ không lọt qua im lặng."""
    p = tmp_path / "x.toml"
    p.write_text('data_root="a"\nruns_dir="b"\nhf_token_source="none"\npersist="local"\nbogus=1\n')
    with pytest.raises(ConfigError, match="bogus"):
        load_profile(str(p), repo_root)


def test_hf_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Token được đọc từ biến môi trường; nguồn `none` hoặc biến không có thì trả `None`."""
    prof = EnvProfile("a", "b", "env:LM_TEST_TOKEN", "local")
    monkeypatch.setenv("LM_TEST_TOKEN", "secret")
    assert prof.resolve_hf_token() == "secret"
    monkeypatch.delenv("LM_TEST_TOKEN")
    assert prof.resolve_hf_token() is None
    assert EnvProfile("a", "b", "none", "local").resolve_hf_token() is None


def test_v0_config_loads_with_fake(tmp_path: Path, repo_root: Path) -> None:
    """`v0.toml` đọc được; với `backend=fake` thì model rỗng vẫn hợp lệ."""
    text = (
        (repo_root / "configs" / "v0.toml").read_text(encoding="utf-8").replace('backend = "vllm"', 'backend = "fake"')
    )
    p = tmp_path / "c.toml"
    p.write_text(text, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.agent.guards == ["allowed_tool", "schema", "duplicate_write"]


def test_v0_and_docker_configs_load(repo_root: Path) -> None:
    """`v0.toml` và `docker.toml` đọc được và đã điền model đã chọn (SPEC 6.0)."""
    v0 = load_config(repo_root / "configs" / "v0.toml")
    assert v0.llm.model and v0.llm.revision and v0.asr.model and v0.asr.revision
    assert v0.asr.engine == "faster_whisper"
    dk = load_config(repo_root / "configs" / "docker.toml")
    assert dk.llm.model.startswith("/app/models/") and dk.asr.backend == "asr"
    assert dk.agent == v0.agent and dk.llm.max_model_len == v0.llm.max_model_len


def test_vllm_empty_model_raises(tmp_path: Path, repo_root: Path) -> None:
    """Báo lỗi khi khởi động vLLM mà chưa chọn model."""
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
    """Báo lỗi khi engine ASR không xác định, dtype ASR sai, hoặc `backend=asr` mà không có model."""
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
    """Khoá lạ hoặc thiếu khoá đều bị báo lỗi, không được chấp nhận im lặng."""
    text = (repo_root / "configs" / "v0.toml").read_text(encoding="utf-8").replace(old, new, 1)
    p = tmp_path / "c.toml"
    p.write_text(text.replace('backend = "vllm"', 'backend = "fake"'), encoding="utf-8")
    with pytest.raises(ConfigError, match=key):
        load_config(p)


def test_overrides_apply_before_validation(repo_root: Path) -> None:
    """`--set` ghi đè giá trị TOML (có hoặc không nháy), và vẫn qua bước kiểm tra; khoá gõ sai bị báo lỗi."""
    v0 = repo_root / "configs" / "v0.toml"
    cfg = load_config(v0, ["asr.dtype=float32", "asr.batch_size=1", 'asr.engine="hf_seq2seq"'])
    assert (cfg.asr.dtype, cfg.asr.batch_size, cfg.asr.engine) == ("float32", 1, "hf_seq2seq")
    with pytest.raises(ConfigError, match=r"unknown key asr\.bogus"):
        load_config(v0, ["asr.bogus=1"])
    with pytest.raises(ConfigError, match=r"section\.key=value"):
        load_config(v0, ["asr_dtype"])
    with pytest.raises(ConfigError, match=r"asr\.dtype"):
        load_config(v0, ["asr.dtype=float8"])


@pytest.mark.parametrize(
    ("extra", "key"),
    [
        ("prompt_guard_ngram = 1", "prompt_guard_ngram"),
        ("prompt_guard_ngram = 3", "needs asr.generate_kwargs.initial_prompt"),
        ("prompt_guard_max_compression = 0", "prompt_guard_max_compression"),
    ],
)
def test_prompt_guard_validated(tmp_path: Path, repo_root: Path, extra: str, key: str) -> None:
    """Lớp bảo vệ câu mồi: cụm phải từ 2 từ, cần có câu mồi, ngưỡng nén phải dương."""
    text = (
        (repo_root / "configs" / "v0.toml").read_text(encoding="utf-8").replace('backend = "vllm"', 'backend = "fake"')
    )
    p = tmp_path / "c.toml"
    p.write_text(text.replace("sample_rate = 16000", f"sample_rate = 16000\n{extra}", 1), encoding="utf-8")
    with pytest.raises(ConfigError, match=key):
        load_config(p)
    ok = text.replace(
        "condition_on_previous_text = false }", 'condition_on_previous_text = false, initial_prompt = "x" }'
    )
    p.write_text(ok.replace("sample_rate = 16000", "sample_rate = 16000\nprompt_guard_ngram = 3", 1), encoding="utf-8")
    assert load_config(p).asr.prompt_guard_ngram == 3


def test_initial_prompt_file_loaded(tmp_path: Path, repo_root: Path) -> None:
    """`asr.initial_prompt_file` được đọc vào `generate_kwargs.initial_prompt`; file thiếu hoặc khai cả hai thì lỗi."""
    text = (
        (repo_root / "configs" / "v0.toml").read_text(encoding="utf-8").replace('backend = "vllm"', 'backend = "fake"')
    )
    p = tmp_path / "c.toml"
    line = 'sample_rate = 16000\ninitial_prompt_file = "configs/asr_prompts/en_top10_v1.txt"'
    p.write_text(text.replace("sample_rate = 16000", line, 1), encoding="utf-8")
    gk = load_config(p).asr.generate_kwargs
    assert (
        gk["initial_prompt"] == (repo_root / "configs/asr_prompts/en_top10_v1.txt").read_text(encoding="utf-8").strip()
    )
    p.write_text(text.replace("sample_rate = 16000", 'sample_rate = 16000\ninitial_prompt_file = "nope.txt"', 1))
    with pytest.raises(ConfigError, match="not found"):
        load_config(p)
    both = text.replace(
        "condition_on_previous_text = false }", 'condition_on_previous_text = false, initial_prompt = "x" }'
    )
    p.write_text(both.replace("sample_rate = 16000", line, 1), encoding="utf-8")
    with pytest.raises(ConfigError, match="not both"):
        load_config(p)
