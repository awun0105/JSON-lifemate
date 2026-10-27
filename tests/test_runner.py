"""Runner isolates per-sample exceptions (SPEC 4.11; T11)."""

from __future__ import annotations

import json
from pathlib import Path

from lifemate.backends.asr import GoldTranscriber
from lifemate.backends.llm import FakeLLM
from lifemate.core.types import Sample, StopReason, Transcript
from lifemate.io.store import RunStore
from lifemate.run.runner import Runner, parse_score
from tests.fakes import FakeEnv
from tests.test_agent import GOLD, make_agent, sample
from tests.test_guards import make_env


class FakeLoader:
    """ZaicLoader stand-in opening FakeEnvs."""

    def __init__(self) -> None:
        self.root = Path("unused")
        self.split = "train"
        self.envs: dict[str, FakeEnv] = {}

    def open_env(self, sample_id: str) -> FakeEnv:
        assert sample_id not in self.envs
        self.envs[sample_id] = make_env()
        return self.envs[sample_id]


class FlakyTranscriber(GoldTranscriber):
    def __init__(self, bad: str) -> None:
        self.bad = bad

    def transcribe(self, sample: Sample) -> Transcript:
        if sample.id == self.bad:
            raise RuntimeError("asr exploded")
        return super().transcribe(sample)


def test_exception_isolated(tmp_path: Path) -> None:
    """T11: transcriber raising on s2 does not stop the run; s2 gets an exception line in trace and submission."""
    samples = [sample(f"s{i}", GOLD) for i in (1, 2, 3)]
    agent = make_agent(FakeLLM(samples))
    agent.transcriber = FlakyTranscriber("s2")
    store = RunStore(tmp_path / "run", resume=False)
    loader = FakeLoader()
    runner = Runner(agent, loader, store, repo_root=tmp_path, index_csv=tmp_path / "index.csv")
    checked: list[Path] = []
    runner._check = lambda sub: checked.append(sub) or sub  # type: ignore[method-assign]
    out = runner.run(samples, ids_file=None, split_has_labels=False, config_record={"x": 1})
    assert checked == [out.submission] and out.score is None
    assert json.loads(store.config_path.read_text(encoding="utf-8"))["warmup_s"].keys() == {"llm_s", "transcriber_s"}
    assert set(loader.envs) == {"s1", "s2", "s3"}
    sub = out.submission
    rows = {r["id"]: r for r in map(json.loads, sub.read_text(encoding="utf-8").splitlines())}
    assert set(rows) == {"s1", "s2", "s3"}
    assert rows["s2"] == {"id": "s2", "tool_calls": [], "state_hash": FakeEnv(set()).get_state_hash()}
    assert rows["s1"]["tool_calls"] == GOLD
    trace = {r["id"]: r for r in store.records()}
    assert trace["s2"]["stop_reason"] == StopReason.EXCEPTION.value
    assert trace["s2"]["errors"] == ["exception"] and "asr exploded" in trace["s2"]["traceback"]
    assert store.read_summary()["n_exception"] == 1


def test_parse_score() -> None:
    text = "Final score [strict]: 100.00  (N = 5)\nFinal score [canon]: 99.50  (N = 5)\n\n== Score by answer type\n"
    text += "execute                3   100.00    100.0  1.000  1.000\n"
    score = parse_score(text)
    assert score == {"total": 100.0, "by_mode": {"strict": 100.0, "canon": 99.5}, "by_category": {"execute": 100.0}}
    assert parse_score("nothing") is None
