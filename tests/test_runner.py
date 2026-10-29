"""Test `Runner` cô lập lỗi theo từng mẫu (SPEC 4.11; T11) và hàm đọc điểm `parse_score`.

Nội dung chính:
- `FakeLoader`: thay `ZaicLoader`, mở `FakeEnv`.
- `FlakyTranscriber`: transcriber cố ý lỗi ở một mẫu.

Cách hoạt động:
- Dùng `FakeLLM`, `FakeEnv` (qua `FakeLoader`), `FlakyTranscriber` và `RunStore` trong `tmp_path`. Bước chấm điểm
  bên ngoài được thay bằng lambda.

Lưu ý:
- Không cần dữ liệu cuộc thi hay tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lifemate.backends.asr import GoldTranscriber
from lifemate.backends.llm import FakeLLM
from lifemate.core.types import Sample, StopReason, Transcript
from lifemate.io.store import RunStore
from lifemate.run.runner import Runner, parse_score
from tests.fakes import FakeEnv
from tests.test_agent import GOLD, make_agent, sample
from tests.test_guards import make_env

ZAIC_SENTINEL = object()


class FakeLoader:
    """Bản thay `ZaicLoader`: mở `FakeEnv` và kiểm tra mỗi mẫu chỉ mở env một lần."""

    def __init__(self) -> None:
        self.root = Path("unused")
        self.split = "train"
        self.envs: dict[str, FakeEnv] = {}

    def load(self) -> object:
        """Trả về một vật đánh dấu thay cho module `zaic_tools`, để test kiểm tra `Runner` truyền đúng module."""
        return ZAIC_SENTINEL

    def open_env(self, sample_id: str) -> FakeEnv:
        """Tạo `FakeEnv` mới cho mẫu; báo lỗi nếu mẫu đã được mở trước đó."""
        assert sample_id not in self.envs
        self.envs[sample_id] = make_env()
        return self.envs[sample_id]


class FlakyTranscriber(GoldTranscriber):
    """Transcriber trả transcript chuẩn, riêng mẫu `bad` thì ném `RuntimeError`."""

    def __init__(self, bad: str) -> None:
        self.bad = bad

    def transcribe(self, sample: Sample) -> Transcript:
        """Ném lỗi với mẫu `bad`, còn lại trả transcript chuẩn."""
        if sample.id == self.bad:
            raise RuntimeError("asr exploded")
        return super().transcribe(sample)


def test_exception_isolated(tmp_path: Path) -> None:
    """T11: transcriber lỗi ở s2 không dừng lần chạy; s2 vẫn có dòng ngoại lệ trong trace và dòng trong submission."""
    samples = [sample(f"s{i}", GOLD) for i in (1, 2, 3)]
    agent = make_agent(FakeLLM(samples))
    agent.transcriber = FlakyTranscriber("s2")
    store = RunStore(tmp_path / "run", resume=False)
    loader = FakeLoader()
    runner = Runner(agent, loader, store, repo_root=tmp_path, index_csv=tmp_path / "index.csv")
    checked: list[Path] = []
    # Thay bước kiểm tra submission bằng lambda ghi lại đường dẫn, để khỏi chạy script bên ngoài.
    runner._check = lambda sub: (checked.append(sub), (sub, True))[1]  # type: ignore[method-assign]
    out = runner.run(samples, ids_file=None, split_has_labels=False, config_record={"x": 1})
    assert checked == [out.submission] and out.score is None
    assert json.loads(store.config_path.read_text(encoding="utf-8"))["warmup_s"].keys() == {"llm_s", "transcriber_s"}
    assert set(loader.envs) == {"s1", "s2", "s3"}
    sub = out.submission
    rows = {r["id"]: r for r in map(json.loads, sub.read_text(encoding="utf-8").splitlines())}
    assert set(rows) == {"s1", "s2", "s3"}
    # Mẫu lỗi vẫn có dòng submission: không có lời gọi tool, `state_hash` là của môi trường chưa đổi.
    assert rows["s2"] == {"id": "s2", "tool_calls": [], "state_hash": FakeEnv(set()).get_state_hash()}
    assert rows["s1"]["tool_calls"] == GOLD
    trace = {r["id"]: r for r in store.records()}
    assert trace["s2"]["stop_reason"] == StopReason.EXCEPTION.value
    assert trace["s2"]["errors"] == ["exception"] and "asr exploded" in trace["s2"]["traceback"]
    assert store.read_summary()["n_exception"] == 1


def test_parse_score() -> None:
    """Đọc đúng điểm tổng, điểm theo chế độ so khớp và theo loại câu trả lời; văn bản lạ trả `None`."""
    # Mẫu đầu ra của script chấm điểm.
    text = "Final score [strict]: 100.00  (N = 5)\nFinal score [canon]: 99.50  (N = 5)\n\n== Score by answer type\n"
    text += "execute                3   100.00    100.0  1.000  1.000\n"
    score = parse_score(text)
    assert score == {"total": 100.0, "by_mode": {"strict": 100.0, "canon": 99.5}, "by_category": {"execute": 100.0}}
    assert parse_score("nothing") is None


def test_parse_score_ignores_confusion_matrix() -> None:
    """Điểm nhóm `execute` lấy từ bảng theo nhóm, không bị dòng `execute` của ma trận nhầm lẫn in sau ghi đè."""
    import contextlib
    import io

    from lifemate.eval.score import report

    rows = [
        {"category": "execute", "predicted": p, "score": 0.4, "hash_ok": True, "P": 0.4, "R": 0.4, "apps": []}
        for p in ("execute", "ask_missing_info")
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(rows, 5)
    score = parse_score(buf.getvalue())
    assert score is not None
    assert score["by_category"] == {"execute": 40.0}


def test_check_opens_no_second_tool_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Split không nhãn: `check_submission` chạy với `open_tool_envs=False`; báo cáo lỗi thì `failed` là "check"."""
    import lifemate.run.runner as runner_mod
    from lifemate.io.check_submission import CheckReport

    seen: dict[str, object] = {}

    def fake_check(*args: object, open_tool_envs: bool = True) -> CheckReport:
        seen["open_tool_envs"] = open_tool_envs
        report = CheckReport(split="train", n_expected=1)
        report.errors.append("1 missing ids: x")
        return report

    monkeypatch.setattr(runner_mod, "check_submission", fake_check)
    samples = [sample("s1", GOLD)]
    store = RunStore(tmp_path / "run", resume=False)
    r = Runner(make_agent(FakeLLM(samples)), FakeLoader(), store, repo_root=tmp_path)
    result = r.run(samples, ids_file=None, split_has_labels=False, config_record={})
    assert seen["open_tool_envs"] is False
    assert result.failed == "check" and (tmp_path / "run" / "check.txt").is_file()


class BrokenEnvLoader(FakeLoader):
    """Bản thay loader mà env hỏng luôn: đọc lời gọi đã gửi cũng ném lỗi."""

    def open_env(self, sample_id: str) -> FakeEnv:
        """Mở env như `FakeLoader`, rồi làm hỏng `get_tool_calls`."""
        env = super().open_env(sample_id)

        def broken() -> list[dict[str, object]]:
            raise RuntimeError("env also broken")

        env.get_tool_calls = broken  # type: ignore[method-assign]
        return env


def test_env_read_error_inside_handler_does_not_stop_run(tmp_path: Path) -> None:
    """Mẫu lỗi rồi env cũng lỗi khi đọc lại: mẫu vẫn có kết quả `exception`, các mẫu sau vẫn chạy."""
    samples = [sample(f"s{i}", GOLD) for i in (1, 2)]
    agent = make_agent(FakeLLM(samples))
    agent.transcriber = FlakyTranscriber("s1")
    store = RunStore(tmp_path / "run", resume=False)
    runner = Runner(agent, BrokenEnvLoader(), store, repo_root=tmp_path)
    runner._check = lambda sub: (sub, True)  # type: ignore[method-assign]
    runner.run(samples, ids_file=None, split_has_labels=False, config_record={})
    trace = {r["id"]: r for r in store.records()}
    assert set(trace) == {"s1", "s2"}
    assert trace["s1"]["stop_reason"] == StopReason.EXCEPTION.value
    assert "asr exploded" in trace["s1"]["traceback"] and "env also broken" in trace["s1"]["traceback"]
    assert trace["s1"]["tool_calls"] == [] and trace["s1"]["state_hash"] == ""


class SubmissionWatcher(GoldTranscriber):
    """Transcriber đọc `submission.jsonl` lúc đang xử lý mẫu `at`, để kiểm tra file nộp được ghi sau mỗi mẫu."""

    def __init__(self, path: Path, at: str) -> None:
        self.path = path
        self.at = at
        self.seen: list[dict[str, object]] = []

    def transcribe(self, sample: Sample) -> Transcript:
        """Ở mẫu `at`, chép các dòng của file nộp hiện có rồi mới trả transcript."""
        if sample.id == self.at:
            self.seen = [json.loads(ln) for ln in self.path.read_text(encoding="utf-8").splitlines()]
        return super().transcribe(sample)


def test_submission_written_after_every_sample(tmp_path: Path) -> None:
    """Lúc chạy s2, file nộp đã có đủ 3 dòng: s1 thật, s2 và s3 là dòng giữ chỗ (`[]`, hash rỗng)."""
    samples = [sample(f"s{i}", GOLD) for i in (1, 2, 3)]
    agent = make_agent(FakeLLM(samples))
    store = RunStore(tmp_path / "run", resume=False)
    watcher = SubmissionWatcher(store.submission_path, "s2")
    agent.transcriber = watcher
    runner = Runner(agent, FakeLoader(), store, repo_root=tmp_path)
    runner._check = lambda sub: (sub, True)  # type: ignore[method-assign]
    runner.run(samples, ids_file=None, split_has_labels=False, config_record={})
    assert [r["id"] for r in watcher.seen] == ["s1", "s2", "s3"]
    assert watcher.seen[0]["tool_calls"] and watcher.seen[0]["state_hash"]
    assert watcher.seen[1:] == [{"id": i, "tool_calls": [], "state_hash": ""} for i in ("s2", "s3")]


class KilledTranscriber(GoldTranscriber):
    """Transcriber giả lập tiến trình bị kill ở mẫu `at` (ném `KeyboardInterrupt`, không phải `Exception`)."""

    def __init__(self, at: str) -> None:
        self.at = at

    def transcribe(self, sample: Sample) -> Transcript:
        """Ném `KeyboardInterrupt` với mẫu `at`."""
        if sample.id == self.at:
            raise KeyboardInterrupt
        return super().transcribe(sample)


def test_killed_mid_run_leaves_complete_submission(tmp_path: Path) -> None:
    """Chết giữa chừng ở s3: file nộp vẫn đủ 3 dòng, s1 và s2 là kết quả thật, s3 là dòng giữ chỗ."""
    samples = [sample(f"s{i}", GOLD) for i in (1, 2, 3)]
    agent = make_agent(FakeLLM(samples))
    agent.transcriber = KilledTranscriber("s3")
    store = RunStore(tmp_path / "run", resume=False)
    runner = Runner(agent, FakeLoader(), store, repo_root=tmp_path)
    with pytest.raises(KeyboardInterrupt):
        runner.run(samples, ids_file=None, split_has_labels=False, config_record={})
    rows = [json.loads(ln) for ln in store.submission_path.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["s1", "s2", "s3"]
    assert all(r["tool_calls"] for r in rows[:2]) and rows[2] == {"id": "s3", "tool_calls": [], "state_hash": ""}


def test_post_run_errors_do_not_raise(tmp_path: Path) -> None:
    """`check_submission` hay `append_index` ném lỗi sau khi file nộp đã ghi: run vẫn trả về, chỉ báo `failed`."""
    samples = [sample("s1", GOLD)]
    agent = make_agent(FakeLLM(samples))
    store = RunStore(tmp_path / "run", resume=False)
    # `index.csv` trỏ vào một file thường làm thư mục cha, nên `append_index` sẽ lỗi.
    (tmp_path / "blocker").write_text("x", encoding="utf-8")
    runner = Runner(agent, FakeLoader(), store, repo_root=tmp_path, index_csv=tmp_path / "blocker" / "index.csv")

    def boom(sub: Path) -> tuple[Path, bool]:
        raise RuntimeError("zaic_tools missing")

    runner._check = boom  # type: ignore[method-assign]
    out = runner.run(samples, ids_file=None, split_has_labels=False, config_record={})
    assert out.failed == "check" and out.report is None
    assert store.submission_path.exists()
