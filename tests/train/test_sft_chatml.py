"""Test dữ liệu SFT ở định dạng ChatML (`sft_data.build_sample_chatml`, `swift_sft.swift_row`, `sft.build_labels`).

Nội dung chính:
- Phần đầu mỗi dòng giống hệt hội thoại agent ChatML dựng lúc chạy; lời gọi là tin `assistant` có `tool_calls`, kết quả
  là tin `tool`; câu kết là câu của lead.
- Dòng phục hồi lỗi: lượt gọi sai có `"loss": false` và tin `tool` mang lỗi thật của `env.validate`.
- `swift_row` đưa danh sách tool ra trường `tools` cấp dòng (ms-swift xoá khoá lạ trong tin nhắn).
- `build_labels` (chat template thật của Qwen, marker `tokenizer`) chỉ train lượt trợ lý và bỏ lượt gọi sai.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from lifemate.agent.chatml import FINAL_ASK, FINAL_DONE, FINAL_NO_CALL, ChatMLPrompt, parse_reply
from lifemate.agent.tooldocs import ASK_TOOL, CatalogIndex
from lifemate.backends.asr import GoldTranscriber
from lifemate.core.tokens import SimpleTokenCounter
from lifemate.io.data import load_split, read_ids
from lifemate.io.zaic import EnvRegistry, ZaicLoader, load_catalog
from lifemate.train import sft_data
from lifemate.train.sft import IGNORE_INDEX, build_labels
from lifemate.train.swift_sft import swift_row

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "train+public_test"


@pytest.fixture(scope="module")
def built() -> dict[str, Any]:
    """Dòng SFT ChatML (train mode, mọi mẫu chèn lỗi) từ 20 mẫu đầu của split train, theo `configs/v0.toml`."""
    if not DATA_ROOT.is_dir():
        pytest.skip(f"data package missing: {DATA_ROOT}")
    run_cfg = sft_data.load_run_config(REPO_ROOT / "configs" / "v0.toml")
    assert run_cfg.agent.format == "chatml"
    samples = load_split(DATA_ROOT, "train", read_ids(REPO_ROOT / "configs" / "splits" / "train.txt")[:20])
    index = CatalogIndex(load_catalog(DATA_ROOT, "train"))
    counter = SimpleTokenCounter()
    builder = sft_data.make_builder(run_cfg, REPO_ROOT, index, counter)
    assert isinstance(builder, ChatMLPrompt)
    result = sft_data.build_rows(
        samples,
        builder,
        ZaicLoader(DATA_ROOT, "train", EnvRegistry()),
        counter,
        max_model_len=run_cfg.llm.max_model_len,
        max_result_chars=run_cfg.agent.max_result_chars,
        train_mode=True,
        augment=sft_data.Augment(error_recovery_ratio=1.0),
    )
    return {"samples": {s.id: s for s in samples}, "result": result, "builder": builder}


@pytest.mark.data
def test_rows_start_like_runtime_and_end_with_lead_text(built: dict[str, Any]) -> None:
    """Phần đầu dòng trùng hội thoại lúc chạy; mỗi lời gọi đáp án có tin `tool` sau nó; câu kết đúng loại mẫu."""
    rows = built["result"].rows
    assert len(rows) == 20
    for row in rows:
        sample = built["samples"][row["id"]]
        start, _ = built["builder"].start(sample, GoldTranscriber().transcribe(sample), train_mode=True)
        assert row["messages"][: len(start)] == start
        calls = (sample.label or {}).get("tool_calls") or []
        tail = row["messages"][len(start) :]
        trained = [
            m for m in tail if m["role"] == "assistant" and m.get("loss", True) is not False and "tool_calls" in m
        ]
        assert [m["tool_calls"][0]["function"]["name"] for m in trained] == [c["name"] for c in calls]
        assert all(m["role"] == "tool" for m in (tail[i + 1] for i, x in enumerate(tail) if "tool_calls" in x))
        want = FINAL_NO_CALL if not calls else FINAL_ASK if calls[-1]["name"] == ASK_TOOL else FINAL_DONE
        assert tail[-1] == {"role": "assistant", "content": want}


@pytest.mark.data
def test_error_recovery_uses_real_validate_error(built: dict[str, Any]) -> None:
    """Lượt gọi sai có `loss: false`, tin `tool` sau nó là `{"ok": false, "error": ...}` với lỗi của `env.validate`."""
    rows = [r for r in built["result"].rows if r["error_recovery"]]
    assert rows
    for row in rows:
        k = next(i for i, m in enumerate(row["messages"]) if m.get("loss") is False)
        bad, err = row["messages"][k], row["messages"][k + 1]
        assert "tool_calls" in bad and err["role"] == "tool" and err["tool_call_id"] == bad["tool_calls"][0]["id"]
        content = json.loads(err["content"])
        assert content["ok"] is False and content["error"]


@pytest.mark.data
def test_swift_row_moves_tools_to_row_level(built: dict[str, Any]) -> None:
    """`swift_row`: `tools` ở cấp dòng, tin `system` không còn `tools`, lời gọi giữ `tool_calls` và cờ `loss`."""
    row = next(r for r in built["result"].rows if r["error_recovery"])
    out = swift_row(row)
    assert out["tools"] == row["messages"][0]["tools"]
    assert "tools" not in out["messages"][0]
    assert any(m.get("loss") is False and "tool_calls" in m for m in out["messages"])
    assert set().union(*(m.keys() for m in out["messages"])) <= {
        "role",
        "content",
        "loss",
        "tool_calls",
        "tool_call_id",
        "name",
    }


@pytest.mark.data
@pytest.mark.tokenizer
def test_labels_train_assistant_turns_only(built: dict[str, Any]) -> None:
    """Chat template thật của Qwen: nhãn phủ các lượt trợ lý trừ lượt gọi sai; phần nhãn của lời gọi đọc lại đúng."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B-Instruct-2507", revision="cdbee75f17c01a7cc42f958dc650907174af0554", local_files_only=True
        )
    except OSError:
        pytest.skip("Qwen tokenizer not in the local cache")
    row = next(r for r in built["result"].rows if r["error_recovery"])
    ids, labels = build_labels(tok, row["messages"], {"enable_thinking": False})
    trained = tok.decode([t for t, lab in zip(ids, labels, strict=True) if lab != IGNORE_INDEX])
    good = [m["tool_calls"][0]["function"] for m in row["messages"] if "tool_calls" in m and m.get("loss") is not False]
    assert trained.count("<tool_call>") == len(good)
    parsed = parse_reply(trained)
    assert isinstance(parsed, list) and [a.name for a in parsed] == [g["name"] for g in good]
    assert dataclasses.is_dataclass(parsed[0])
