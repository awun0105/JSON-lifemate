"""Test CLI từ đầu đến cuối với `FakeLLM`, chống lộ token, và quét mã nguồn (SPEC 4.11, 4.16; T1, T9, T21, T23).

Cách hoạt động:
- Config `v0.toml` được đổi sang backend `fake`.
- Các test quét mã nguồn chỉ đọc thư mục `lifemate/`.

Lưu ý:
- Test có marker `data` cần dữ liệu cuộc thi (fixture `data_root`). Không cần tokenizer.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lifemate.run import cli

SENTINEL = "hf_SENTINEL_do_not_leak_123"


def fake_config(repo_root: Path, tmp_path: Path) -> Path:
    """Ghi bản sao `v0.toml` dùng backend LLM `fake` vào `tmp_path`."""
    text = (
        (repo_root / "configs" / "v0.toml").read_text(encoding="utf-8").replace('backend = "vllm"', 'backend = "fake"')
    )
    path = tmp_path / "fake.toml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.data
@pytest.mark.parametrize("ids", ["configs/splits/val_seen.txt", "configs/splits/val_unseen.txt"])
def test_fake_llm_end_to_end(repo_root: Path, data_root: Path, tmp_path: Path, ids: str) -> None:
    """T1: phát lại nhãn gold qua toàn bộ `Runner` đạt 100.00 ở cả hai chế độ so khớp."""
    argv = ["--config", str(fake_config(repo_root, tmp_path)), "--env", "local", "--split", "train"]
    argv += ["--ids", ids, "--root", str(data_root), "--out", str(tmp_path / "run")]
    assert cli.main(argv) == 0
    score = (tmp_path / "run" / "score.txt").read_text(encoding="utf-8")
    assert "Final score [strict]: 100.00" in score
    assert "Final score [canon]: 100.00" in score
    n = len((repo_root / ids).read_text(encoding="utf-8").split())
    assert len((tmp_path / "run" / "submission.jsonl").read_text(encoding="utf-8").splitlines()) == n


@pytest.mark.data
def test_no_token_leak(repo_root: Path, data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T23: HF token không bao giờ xuất hiện trong `config.json`, `trace.jsonl` hay `summary.json`."""
    monkeypatch.setenv("HF_TOKEN", SENTINEL)
    argv = ["--config", str(fake_config(repo_root, tmp_path)), "--env", "local", "--split", "train"]
    argv += ["--ids", "configs/splits/smoke.txt", "--limit", "2", "--root", str(data_root)]
    argv += ["--out", str(tmp_path / "run")]
    assert cli.main(argv) == 0
    for name in ("config.json", "trace.jsonl", "summary.json"):
        assert SENTINEL not in (tmp_path / "run" / name).read_text(encoding="utf-8")


@pytest.mark.data
def test_random_seeds_fixed_and_recorded(repo_root: Path, data_root: Path, tmp_path: Path) -> None:
    """Thể lệ mục 8: CLI cố định seed của Python (và NumPy, PyTorch nếu có) theo `[llm].seed`, ghi vào `config.json`."""
    import json

    argv = ["--config", str(fake_config(repo_root, tmp_path)), "--env", "local", "--split", "train"]
    argv += ["--ids", "configs/splits/smoke.txt", "--limit", "1", "--root", str(data_root)]
    argv += ["--out", str(tmp_path / "run")]
    assert cli.main(argv) == 0
    record = json.loads((tmp_path / "run" / "config.json").read_text(encoding="utf-8"))
    assert record["random_seeds"]["seed"] == 0 and "random" in record["random_seeds"]["libraries"]


def _sources(repo_root: Path) -> list[Path]:
    """Liệt kê mọi file `.py` trong `lifemate/`."""
    return sorted((repo_root / "lifemate").rglob("*.py"))


def test_no_environment_access(repo_root: Path) -> None:
    """T9: chỉ `zaic.py` được nhắc tới `_environment` hoặc `state.json`."""
    for path in _sources(repo_root):
        if path.name == "zaic.py":
            continue
        text = path.read_text(encoding="utf-8")
        # `trainer_state.json` (log train của HF/ms-swift) không phải `state.json` của simulator, nên được loại trừ.
        for needle in (r"_environment", r"(?<!trainer_)state\.json"):
            assert not re.search(needle, text), f"{path}: {needle}"


def test_no_hardcoded_paths(repo_root: Path) -> None:
    """T21: không có đường dẫn riêng của máy nào trong `lifemate/`."""
    for path in _sources(repo_root):
        text = path.read_text(encoding="utf-8")
        for needle in ("/kaggle", "/content", "/home", "train+public_test"):
            assert needle not in text, f"{path}: {needle}"


@pytest.mark.parametrize(("flag", "want"), [("config", "gold"), ("asr", "asr"), ("gold", "gold")])
def test_transcripts_flag_overrides_asr_backend(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str, want: str
) -> None:
    """`--transcripts` đổi `asr.backend` của config (v0 để gold); mặc định giữ nguyên giá trị trong config."""
    seen: list[str] = []

    def stop(config: object, *a: object, **k: object) -> None:
        """Ghi lại `asr.backend` mà factory nhận được rồi dừng, không dựng model."""
        seen.append(config.asr.backend)  # type: ignore[attr-defined]
        raise SystemExit(0)

    monkeypatch.setattr(cli, "load_split", lambda *a, **k: [])
    monkeypatch.setattr(cli.factory, "build", stop)
    argv = ["--config", str(fake_config(repo_root, tmp_path)), "--root", str(tmp_path), "--transcripts", flag]
    with pytest.raises(SystemExit):
        cli.run(cli.parse_args(argv), repo_root=repo_root)
    assert seen == [want]


@pytest.mark.parametrize(("failed", "code"), [(None, 0), ("check", 0), ("score", 1)])
def test_main_exit_code(
    failed: str | None, code: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Chấm điểm lỗi thì thoát 1; `check_submission` lỗi chỉ in cảnh báo, vẫn thoát 0 để container nộp không hỏng."""
    from lifemate.run.runner import RunSummary

    result = RunSummary(tmp_path, tmp_path / "submission.jsonl", {}, None, tmp_path / "check.txt", failed)
    monkeypatch.setattr(cli, "parse_args", lambda argv: None)
    monkeypatch.setattr(cli, "run", lambda args: result)
    assert cli.main([]) == code
    assert ("FAILED" in capsys.readouterr().err) == (failed is not None)


@pytest.mark.data
def test_existing_run_fails_before_model_load(
    repo_root: Path, data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thư mục run đã có trace mà quên `--resume`, hoặc resume bằng config khác, thì báo lỗi trước khi nạp model."""
    argv = ["--config", str(fake_config(repo_root, tmp_path)), "--env", "local", "--split", "train"]
    argv += ["--ids", "configs/splits/smoke.txt", "--limit", "1", "--root", str(data_root)]
    argv += ["--out", str(tmp_path / "run")]
    assert cli.main(argv) == 0

    def no_build(*args: object, **kwargs: object) -> None:
        raise AssertionError("factory.build must not run")

    monkeypatch.setattr(cli.factory, "build", no_build)
    with pytest.raises(FileExistsError):
        cli.main(argv)
    with pytest.raises(ValueError, match="config"):
        cli.main([*argv, "--resume", "--transcripts", "asr"])
