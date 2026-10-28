"""Test phần tăng cường dữ liệu SFT chỉ dùng khi train (`sft_data.Augment`), chuyển thể từ bộ dựng của trưởng nhóm.

Lưu ý:
- Phần lớn test dựng `Sample` giả bằng `_sample` và dùng `FakeChatTokenizer` (mượn từ `test_sft_data`), nên không
  cần tokenizer thật.
- Riêng `test_augment_on_real_data` cần dữ liệu cuộc thi (marker `data`) và tự skip nếu thiếu.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lifemate.agent.agent import BLOCK_PREFIX
from lifemate.agent.tooldocs import CatalogIndex
from lifemate.core.tokens import SimpleTokenCounter
from lifemate.core.types import Sample
from lifemate.io.data import load_split, read_ids
from lifemate.io.zaic import EnvRegistry, ZaicLoader, load_catalog
from lifemate.train import sft_data
from lifemate.train.common import read_held_out
from lifemate.train.sft import IGNORE_INDEX, build_labels, load_llm_train_config
from tests.train.test_sft_data import FakeChatTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "train+public_test"
CFG = REPO_ROOT / "configs" / "train" / "llm_v1.toml"


def _sample(sid: str, calls: list[dict[str, Any]], text: str = "", apps: tuple[str, ...] = ("a",)) -> Sample:
    """Dựng `Sample` giả với danh sách app cài sẵn và nhãn `tool_calls` cho trước."""
    ctx = {"installed_apps": {app: {"tools": [f"{app}.x"]} for app in apps}}
    return Sample(sid, "train", f"{sid}.wav", text, ctx, (), {"tool_calls": calls, "tool_results": []})


# Lời gọi hỏi thêm thông tin và một lời gọi tool thường, dùng để dựng nhãn giả.
ASK = {"name": "sys.ask_missing_info", "arguments": {}}
CALL = {"name": "a.x", "arguments": {"k": 1}}


def test_category() -> None:
    """Không có lời gọi là `out_of_scope`, kết thúc bằng hỏi lại là `ask_missing_info`, còn lại là `execute`."""
    assert sft_data.category(_sample("1", [])) == "out_of_scope"
    assert sft_data.category(_sample("2", [CALL, ASK])) == "ask_missing_info"
    assert sft_data.category(_sample("3", [CALL])) == "execute"


def test_hard_copies_only_for_hard_rows_and_average_to_factor() -> None:
    """Chỉ dòng khó mới được nhân bản; số bản sao trung bình khớp hệ số và tất định."""
    assert sft_data.hard_copies(_sample("e", [CALL]), 2.0, 0) == 0
    assert sft_data.hard_copies(_sample("o", []), 1.0, 0) == 0
    assert sft_data.hard_copies(_sample("o", []), 3.0, 0) == 2
    extra = [sft_data.hard_copies(_sample(f"s{i}", []), 1.5, 0) for i in range(2000)]
    # Hệ số 1.5 nghĩa là trung bình 0.5 bản sao thêm; kết quả phải tất định theo id.
    assert set(extra) == {0, 1} and 0.45 < sum(extra) / len(extra) < 0.55
    assert extra == [sft_data.hard_copies(_sample(f"s{i}", []), 1.5, 0) for i in range(2000)]


def test_dedup_keeps_at_most_k_per_cluster_in_order() -> None:
    """Lọc trùng giữ tối đa k mẫu mỗi cụm theo thứ tự gốc; k=0 thì tắt lọc trùng."""
    same = "bật đèn phòng khách lên giúp anh nhé"
    samples = [_sample(f"t{i}", [CALL], same) for i in range(4)] + [_sample("u", [CALL], "xem thời tiết ngày mai")]
    kept = sft_data.dedup_clusters(samples, 2)
    assert [s.id for s in kept] == ["t0", "t1", "u"]
    assert sft_data.dedup_clusters(samples, 0) == samples


def test_distractors_skip_installed_held_out_and_out_of_scope() -> None:
    """App gây nhiễu không trùng app đã cài hay app held-out, không sửa mẫu gốc, và bỏ qua mẫu `out_of_scope`."""
    catalog = {"apps": [{"id": a, "tools": [{"name": f"{a}.x", "kind": "read"}]} for a in "abcde"]}
    index = CatalogIndex(catalog)
    sample = _sample("s", [CALL])
    out, picked = sft_data.with_distractors(sample, index, frozenset({"b"}), 2, 0)
    assert len(picked) == 2 and not {"a", "b"} & set(picked)
    assert set(out.context["installed_apps"]) == {"a", *picked}
    assert out.context["installed_apps"][picked[0]] == {"tools": [f"{picked[0]}.x"]}
    assert sample.context["installed_apps"].keys() == {"a"}  # mẫu gốc không bị sửa
    assert sft_data.with_distractors(sample, index, frozenset({"b"}), 2, 0)[1] == picked  # ổn định giữa các lần gọi
    assert sft_data.with_distractors(_sample("o", []), index, frozenset(), 2, 0) == (_sample("o", []), [])


def test_failed_call_is_not_trained() -> None:
    """Lượt gọi sai có `loss=False` bị che khỏi nhãn, còn lời gọi sửa lại vẫn được học."""
    tok = FakeChatTokenizer()
    bad, good = '{"name":"x.y","arguments":{}}', '{"name":"x.y","arguments":{"k":1}}'
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hỏi"},
        {"role": "assistant", "content": bad, "loss": False},
        {"role": "user", "content": BLOCK_PREFIX + "thiếu k"},
        {"role": "assistant", "content": good},
    ]
    ids, labels = build_labels(tok, messages, {"enable_thinking": False})
    trained = tok.decode([i for i, lab in zip(ids, labels, strict=True) if lab != IGNORE_INDEX])
    assert good in trained and bad not in trained
    plain = [{"role": m["role"], "content": m["content"]} for m in messages]
    assert ids == build_labels(tok, plain, {"enable_thinking": False})[0]  # cờ `loss` không làm đổi chuỗi token


@pytest.mark.data
def test_augment_on_real_data() -> None:
    """Lời gọi lỗi dùng đúng thông báo của guard thật, tắt với file eval; app gây nhiễu không là app held-out."""
    if not DATA_ROOT.is_dir():
        pytest.skip(f"data package missing: {DATA_ROOT}")
    cfg = load_llm_train_config(CFG)
    run_cfg = sft_data.load_run_config(REPO_ROOT / cfg.base.run_config)
    ids = read_ids(REPO_ROOT / "configs" / "splits" / "train.txt")[:30]
    samples = load_split(DATA_ROOT, "train", ids)
    index = CatalogIndex(load_catalog(DATA_ROOT, "train"))
    counter = SimpleTokenCounter()
    held_out = read_held_out(REPO_ROOT)
    aug = sft_data.Augment(error_recovery_ratio=1.0, oversample_hard=2.0, distractor_apps=2)

    def build(train_mode: bool) -> sft_data.BuildResult:
        """Dựng các dòng SFT từ mẫu thật, bật hoặc tắt chế độ train."""
        return sft_data.build_rows(
            samples,
            sft_data.make_builder(run_cfg, REPO_ROOT, index, counter),
            ZaicLoader(DATA_ROOT, "train", EnvRegistry()),
            counter,
            max_model_len=run_cfg.llm.max_model_len,
            max_result_chars=run_cfg.agent.max_result_chars,
            train_mode=train_mode,
            augment=aug,
            index=index,
            held_out=held_out,
        )

    rows = build(True).rows
    recovered = [r for r in rows if r["error_recovery"]]
    assert recovered
    for row in recovered:
        msgs = row["messages"]
        # Sau lượt gọi sai (loss=False) phải là thông báo chặn, rồi một lời gọi đúng cùng tên tool.
        k = next(i for i, m in enumerate(msgs) if m.get("loss") is False)
        assert msgs[k + 1]["role"] == "user" and msgs[k + 1]["content"].startswith(BLOCK_PREFIX)
        assert json.loads(msgs[k]["content"])["name"] == json.loads(msgs[k + 2]["content"])["name"]
    originals = [r for r in rows if "copy" not in r]
    copies = [r for r in rows if "copy" in r]
    # Với oversample_hard=2.0, mỗi dòng khó (không phải execute) có đúng một bản sao.
    assert len(copies) == sum(r["category"] != "execute" for r in originals)
    for row in rows:
        for app in held_out:
            assert f"{app}." not in row["messages"][0]["content"]
    plain = build(False).rows
    assert not any(r["error_recovery"] or "copy" in r for r in plain)
