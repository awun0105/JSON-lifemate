"""Kiểm tra quy tắc dữ liệu SFT (SPEC 6.2.1, 6.2.7 mục 1-2), che loss (6.2.2) và tiện ích pipeline (6.2.3, 6.2.6).

Nhóm đầu dùng fixture `built` dựng dòng SFT từ 20 mẫu train thật cộng mẫu rò rỉ LEAK_ID, nên cần dữ liệu cuộc thi
(marker `data`, tự bỏ qua nếu thiếu). Nhóm che loss dùng `FakeChatTokenizer` thay tokenizer thật. Nhóm pipeline chạy
trên `tmp_path` với runner giả, không cần GPU.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

from lifemate.agent.protocol import DONE_TEXT, parse
from lifemate.agent.tooldocs import ASK_TOOL, CatalogIndex
from lifemate.backends.asr import GoldTranscriber
from lifemate.core.jsonutil import canonical_json
from lifemate.core.tokens import SimpleTokenCounter
from lifemate.core.types import Action, Sample
from lifemate.io.data import load_split, read_ids
from lifemate.io.zaic import EnvRegistry, ZaicLoader, load_catalog
from lifemate.train import llm_pipeline, sft_data
from lifemate.train.common import read_held_out
from lifemate.train.sft import IGNORE_INDEX, build_labels, load_llm_train_config

# from lifemate.train.sft import pad_batch  # tiện ích của trình train Unsloth, đã tắt cùng trình train đó

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "train+public_test"
LEAK_ID = "train_0655"  # lịch sử hội thoại gọi một app held-out (SPEC 4.5)
CFG = REPO_ROOT / "configs" / "train" / "llm_v1.toml"


# ---------------------------------------------------------------- dữ liệu thật


@pytest.fixture(scope="module")
def built() -> dict[str, Any]:
    """Dựng dòng SFT (chế độ train) từ 20 mẫu đầu của split train cộng LEAK_ID; dùng chung cho cả module."""
    if not DATA_ROOT.is_dir():
        pytest.skip(f"data package missing: {DATA_ROOT}")
    cfg = load_llm_train_config(CFG)
    run_cfg = sft_data.load_run_config(REPO_ROOT / cfg.base.run_config)
    ids = [*read_ids(REPO_ROOT / "configs" / "splits" / "train.txt")[:20], LEAK_ID]
    samples = load_split(DATA_ROOT, "train", ids)
    index = CatalogIndex(load_catalog(DATA_ROOT, "train"))
    counter = SimpleTokenCounter()
    builder = sft_data.make_builder(run_cfg, REPO_ROOT, index, counter)
    result = sft_data.build_rows(
        samples,
        builder,
        ZaicLoader(DATA_ROOT, "train", EnvRegistry()),
        counter,
        max_model_len=run_cfg.llm.max_model_len,
        max_result_chars=run_cfg.agent.max_result_chars,
        train_mode=True,
    )
    return {"samples": {s.id: s for s in samples}, "result": result, "builder": builder, "index": index}


@pytest.mark.data
def test_prompt_matches_runtime_bytes(built: dict[str, Any]) -> None:
    """Hai tin nhắn đầu của dòng SFT trùng từng byte với prompt mà agent dựng lúc chạy thật."""
    rows = built["result"].rows
    assert len(rows) == 20
    for row in rows:
        sample = built["samples"][row["id"]]
        conv, _ = built["builder"].start(sample, GoldTranscriber().transcribe(sample), train_mode=True)
        assert row["messages"][:2] == list(conv.messages)
        assert row["n_assistant_turns"] == sum(m["role"] == "assistant" for m in row["messages"])


@pytest.mark.data
def test_turn_structure(built: dict[str, Any]) -> None:
    """Lượt assistant là JSON chuẩn tắc, lượt user mở đầu bằng RESULT_PREFIX; lượt cuối khớp loại mẫu."""
    for row in built["result"].rows:
        sample: Sample = built["samples"][row["id"]]
        calls = (sample.label or {}).get("tool_calls") or []
        tail = row["messages"][2:]
        for m in tail:
            if m["role"] == "assistant":
                assert canonical_json(json.loads(m["content"])) == m["content"]
            else:
                assert m["content"].startswith(sft_data.RESULT_PREFIX)
        if not calls:
            assert tail == [{"role": "assistant", "content": DONE_TEXT}]
        elif calls[-1]["name"] == ASK_TOOL:
            assert tail[-1]["role"] == "assistant" and json.loads(tail[-1]["content"])["name"] == ASK_TOOL
            assert DONE_TEXT not in [m["content"] for m in tail]
        else:
            assert tail[-1]["content"] == DONE_TEXT


@pytest.mark.data
def test_no_held_out_app_and_leaks_dropped(built: dict[str, Any]) -> None:
    """Không dòng nào nhắc app held-out; mẫu có lịch sử gọi app held-out bị loại với lý do held_out_leak."""
    held_out = read_held_out(REPO_ROOT)
    for row in built["result"].rows:
        for m in row["messages"]:
            for app in held_out:
                assert f"{app}." not in m["content"], (row["id"], app)
    index: CatalogIndex = built["index"]
    expected = {
        s.id
        for s in built["samples"].values()
        if any(index.app_of(c["name"]) in held_out for t in s.history if t.role == "assistant" for c in t.tool_calls)
    }
    assert LEAK_ID in expected
    assert {i for i, r in built["result"].dropped if r.startswith("held_out_leak")} == expected


@pytest.mark.data
def test_replay_of_rows_matches_gold_state(built: dict[str, Any]) -> None:
    """SPEC 6.2.7 mục 2: phát lại các JSON lời gọi trong dòng cho ra đúng trạng thái cuối của nhãn gold."""
    rows = built["result"].rows
    ids = [r["id"] for r in rows]
    from_rows = ZaicLoader(DATA_ROOT, "train", EnvRegistry())
    from_gold = ZaicLoader(DATA_ROOT, "train", EnvRegistry())
    for row in rows:
        env = from_rows.open_env(row["id"])
        for m in row["messages"][2:]:
            if m["role"] != "assistant":
                continue
            action = parse(m["content"])
            assert isinstance(action, Action)
            if action.kind == "call":
                env.call(str(action.name), dict(action.arguments or {}))
        gold = from_gold.open_env(row["id"])
        for c in (built["samples"][row["id"]].label or {}).get("tool_calls") or []:
            gold.call(c["name"], dict(c.get("arguments") or {}))
        assert env.get_state_hash() == gold.get_state_hash(), row["id"]
    assert len(ids) == 20
    assert built["result"].replay_warnings == []


@pytest.mark.data
def test_cli_writes_files_and_refuses_eval_ids(tmp_path: Path) -> None:
    """CLI ghi file dữ liệu, danh sách loại và meta; từ chối id eval trừ khi có --allow-eval."""
    if not DATA_ROOT.is_dir():
        pytest.skip("data package missing")
    ids = tmp_path / "ids.txt"
    ids.write_text(f"train_0002\n{LEAK_ID}\n", encoding="utf-8")
    out = tmp_path / "sft.jsonl"
    argv = ["--config", str(CFG), "--ids", str(ids), "--out", str(out)]
    meta = sft_data.run(sft_data.parse_args(argv), REPO_ROOT, argv=argv)
    assert meta["n_rows"] == 1 and meta["n_dropped"] == 1 and meta["train_mode"] is True
    assert (tmp_path / "sft.jsonl.dropped.txt").read_text(encoding="utf-8").startswith(f"{LEAK_ID}\theld_out_leak:")
    assert json.loads((tmp_path / "sft.jsonl.meta.json").read_text(encoding="utf-8"))["sha256"] == meta["sha256"]
    val = read_ids(REPO_ROOT / "configs" / "splits" / "val_unseen.txt")[:1]
    ids.write_text(val[0] + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing"):
        sft_data.run(sft_data.parse_args(argv), REPO_ROOT, argv=argv)
    # Với --allow-eval, file eval được dựng nhưng tắt chế độ train (không tăng cường).
    meta = sft_data.run(sft_data.parse_args([*argv, "--allow-eval"]), REPO_ROOT, argv=argv)
    assert meta["train_mode"] is False and meta["n_rows"] == 1


def test_asr_transcript_source_needs_cache(tmp_path: Path) -> None:
    """Nguồn transcript `asr` báo lỗi khi thiếu file cache."""
    with pytest.raises(FileNotFoundError):
        sft_data._transcriber("asr", tmp_path / "missing.json", 0.3, 0)


# ---------------------------------------------------------------- che loss với tokenizer giả


class FakeChatTokenizer:
    """Tokenizer giả theo kiểu ChatML: mỗi ký tự một token, `<|im_start|>`=1 và `<|im_end|>`=2."""

    START, END = 1, 2
    chat_template = "chatml {% if enable_thinking %}{% endif %}"

    def _encode(self, text: str) -> list[int]:
        """Mã hóa mỗi ký tự thành một token (lệch 10 để tránh id đặc biệt)."""
        return [ord(ch) + 10 for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Giải mã id về chuỗi, hiện token đặc biệt dưới dạng tên."""
        names = {self.START: "<|im_start|>", self.END: "<|im_end|>"}
        return "".join(names[i] if i in names else chr(i - 10) for i in ids)

    def apply_chat_template(
        self, conversation: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool, **kw: Any
    ) -> list[int]:
        """Dựng chuỗi token theo khuôn ChatML; chỉ chấp nhận tokenize=True và enable_thinking=False."""
        assert tokenize and kw == {"enable_thinking": False}
        ids: list[int] = []
        for m in conversation:
            ids += [self.START, *self._encode(m["role"] + "\n" + m["content"]), self.END, *self._encode("\n")]
        if add_generation_prompt:
            ids += [self.START, *self._encode("assistant\n")]
        return ids


def test_build_labels_masks_everything_but_assistant_turns() -> None:
    """build_labels giữ nguyên token của template nhưng che mọi phần trừ lượt assistant."""
    tok = FakeChatTokenizer()
    call = '{"action":"call","arguments":{},"name":"x.y"}'
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hỏi"},
        {"role": "assistant", "content": call},
        {"role": "user", "content": "KẾT QUẢ: {}"},
        {"role": "assistant", "content": DONE_TEXT},
    ]
    from lifemate.train.sft import template_kwargs

    ids, labels = build_labels(tok, messages, template_kwargs(tok))
    assert len(ids) == len(labels)
    assert ids == tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, enable_thinking=False)
    trained = tok.decode([t for t, lab in zip(ids, labels, strict=True) if lab != IGNORE_INDEX])
    # Chỉ nội dung assistant cùng token kết thúc lượt được học.
    assert trained == f"{call}<|im_end|>\n{DONE_TEXT}<|im_end|>\n"
    for t, lab in zip(ids, labels, strict=True):
        assert lab in (IGNORE_INDEX, t)
    assert "sys" not in trained and "KẾT QUẢ" not in trained


# Tiện ích của trình train Unsloth, tắt cùng `sft.pad_batch` (việc train nay chạy trên ms-swift).
# def test_pad_batch() -> None:
#     b = pad_batch([{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]}, {"input_ids": [5], "labels": [5]}], 0)
#     assert b["input_ids"] == [[5, 6, 7], [5, 0, 0]]
#     assert b["labels"][1] == [5, IGNORE_INDEX, IGNORE_INDEX]
#     assert b["attention_mask"] == [[1, 1, 1], [1, 0, 0]]


# ---------------------------------------------------------------- tiện ích pipeline (không cần GPU)


def test_choose_prefers_unseen_then_seen() -> None:
    """Chọn checkpoint theo điểm val_unseen trước, hòa thì xét val_seen."""
    cands = [
        {"path": "a", "val_unseen": 80.0, "val_seen": 90.0},
        {"path": "b", "val_unseen": 82.0, "val_seen": 70.0},
        {"path": "c", "val_unseen": 82.0, "val_seen": 75.0},
    ]
    assert llm_pipeline.choose(cands)["path"] == "c"


def test_dump_toml_round_trip(repo_root: Path) -> None:
    """dump_toml ghi ra TOML đọc lại được đúng dữ liệu gốc."""
    with open(repo_root / "configs" / "v0.toml", "rb") as f:
        raw = tomllib.load(f)
    assert tomllib.loads(llm_pipeline.dump_toml(raw)) == raw


def _ctx(tmp_path: Path) -> llm_pipeline.PipelineContext:
    """Dựng PipelineContext thử nghiệm trong `tmp_path` với cấu hình llm_v1."""
    cfg = load_llm_train_config(CFG)
    (tmp_path / "artifacts" / "models").mkdir(parents=True)
    return llm_pipeline.PipelineContext(REPO_ROOT, CFG, cfg, "local", "ft-x-sft1-r01", tmp_path / "work", 30)


def test_default_name_and_registry(tmp_path: Path) -> None:
    """Tên mô hình mặc định tăng số lượt chạy theo registry; registry có dòng tiêu đề đúng."""
    cfg = load_llm_train_config(CFG)
    reg = tmp_path / "registry.csv"
    name1 = "ft-qwen34binstruct2507-sft1-r01"
    assert llm_pipeline.default_name(cfg, reg) == name1
    llm_pipeline.append_registry(reg, {"name": name1, "base": "b"})
    assert llm_pipeline.default_name(cfg, reg) == "ft-qwen34binstruct2507-sft1-r02"
    assert reg.read_text(encoding="utf-8").splitlines()[0] == ",".join(llm_pipeline.REGISTRY_FIELDS)


def test_eval_choose_register_stages_with_fake_runner(tmp_path: Path) -> None:
    """Stage eval, choose, register với runner giả: chọn checkpoint tốt hơn trên val_unseen, ghi đủ file và registry."""
    ctx = _ctx(tmp_path)
    ctx = llm_pipeline.PipelineContext(tmp_path, CFG, ctx.cfg, "local", ctx.name, tmp_path / "work", 30)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "v0.toml").write_bytes((REPO_ROOT / "configs" / "v0.toml").read_bytes())
    scores = {"m1": {"val_seen": 70.0, "val_unseen": 60.0}, "m2": {"val_seen": 65.0, "val_unseen": 61.0}}
    calls: list[list[str]] = []

    def fake_runner(cmd: Any, cwd: Path) -> int:
        """Runner giả: ghi summary.json với điểm định sẵn theo tên mô hình và split."""
        cmd = list(cmd)
        calls.append(cmd)
        run_dir = Path(cmd[cmd.index("--out") + 1])
        run_cfg = tomllib.loads(Path(cmd[cmd.index("--config") + 1]).read_text(encoding="utf-8"))
        tag = Path(run_cfg["llm"]["model"]).name
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(json.dumps({"score": {"total": scores[tag][run_dir.name]}}))
        return 0

    state: dict[str, Any] = {
        "merge": [
            {"path": "ck/1", "step": 100, "loss": 0.5, "merged": str(tmp_path / "m1")},
            {"path": "ck/2", "step": 200, "loss": 0.6, "merged": str(tmp_path / "m2")},
        ]
    }
    # 2 checkpoint x 2 split = 4 lần chạy eval, mỗi lần giới hạn 30 mẫu.
    llm_pipeline.stage_eval(ctx, state, runner=fake_runner)
    assert len(calls) == 4 and all("--limit" in c and "30" in c for c in calls)
    llm_pipeline.stage_choose(ctx, state)
    assert state["choose"]["step"] == 200
    llm_pipeline.stage_register(ctx, state)
    assert json.loads((ctx.model_dir / "train_config.json").read_text())["checkpoint"] == "ck/2"
    assert json.loads((ctx.model_dir / "eval.json").read_text())["chosen"]["val_unseen"] == 61.0
    rows = llm_pipeline.read_registry(tmp_path / "artifacts" / "models" / "registry.csv")
    assert rows[0]["name"] == ctx.name and rows[0]["val_unseen"] == "61.0" and rows[0]["data_version"] == "v1"


def test_run_pipeline_resumes_from_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run_pipeline dừng được ở một stage và chạy tiếp từ stage sau mà không lặp lại."""
    ctx = _ctx(tmp_path)
    seen: list[str] = []
    for name in llm_pipeline.STAGES:
        monkeypatch.setitem(llm_pipeline.STAGE_FUNCS, name, lambda c, s, n=name: seen.append(n))
    # Chạy tới train, sau đó tiếp từ select: gộp lại phải đủ mọi stage, mỗi stage một lần.
    llm_pipeline.run_pipeline(ctx, to_stage="train")
    state = llm_pipeline.run_pipeline(ctx, from_stage="select")
    assert seen == list(llm_pipeline.STAGES)
    assert state["done"] == list(llm_pipeline.STAGES)


def test_parse_args_defaults() -> None:
    """Giá trị mặc định của CLI pipeline: limit=30 và bắt đầu từ stage data."""
    ns = llm_pipeline.parse_args(["--config", "x.toml"])
    assert isinstance(ns, argparse.Namespace) and ns.limit == 30 and ns.from_stage == "data"


def test_eval_command_uses_eval_transcripts(tmp_path: Path) -> None:
    """Bảo vệ: chấm checkpoint chạy agent với nguồn chữ `base.eval_transcripts` (llm_v1 đặt "asr", như lúc thi)."""
    ctx = _ctx(tmp_path)
    cmd = llm_pipeline.eval_command(ctx, tmp_path / "run.toml", "configs/splits/val_seen.txt", tmp_path / "r")
    assert cmd[cmd.index("--transcripts") + 1] == ctx.cfg.base.eval_transcripts == "asr"
