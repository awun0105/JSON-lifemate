import json
from pathlib import Path

import pytest

from lifemate.data import load_split, read_ids

ROWS = [
    {
        "id": "s1",
        "audio": "audio/s1/s1.wav",
        "audio_text": "xoá báo thức",
        "context": {"current_time": "t"},
        "history": [
            {"role": "user", "audio": "audio/s1/u1.wav", "audio_text": "có báo thức nào?"},
            {
                "role": "assistant",
                "text": "Có 1.",
                "tool_calls": [{"name": "alarm.list", "arguments": {}}],
                "tool_results": [{"ok": True}],
            },
        ],
        "label": {"tool_calls": [{"name": "alarm.delete", "arguments": {"id": "a"}}]},
    },
    {"id": "s2", "audio": "audio/s2/s2.wav", "context": {}, "history": []},
    {"id": "s3", "audio": "audio/s3/s3.wav", "context": {}, "history": [{"role": "assistant", "text": "hi"}]},
]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    d = tmp_path / "train"
    d.mkdir()
    (d / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in ROWS) + "\n\n", "utf-8")
    return tmp_path


def test_parse(root: Path) -> None:
    samples = load_split(root, "train")
    assert [s.id for s in samples] == ["s1", "s2", "s3"]
    s1 = samples[0]
    assert s1.split == "train" and s1.audio_text == "xoá báo thức"
    assert s1.history[0].role == "user" and s1.history[0].tool_calls == ()
    assert s1.history[1].tool_calls == ({"name": "alarm.list", "arguments": {}},)
    assert s1.history[1].tool_results == ({"ok": True},)
    assert s1.label is not None
    assert samples[1].audio_text is None and samples[1].label is None
    assert samples[2].history[0].tool_results == ()


def test_ids_filter_keeps_file_order(root: Path) -> None:
    assert [s.id for s in load_split(root, "train", ["s3", "s1"])] == ["s1", "s3"]


def test_unknown_ids(root: Path) -> None:
    with pytest.raises(ValueError, match="nope"):
        load_split(root, "train", ["s1", "nope"])


def test_read_ids(tmp_path: Path) -> None:
    p = tmp_path / "ids.txt"
    p.write_text("a\n\n b \n", "utf-8")
    assert read_ids(p) == ["a", "b"]
