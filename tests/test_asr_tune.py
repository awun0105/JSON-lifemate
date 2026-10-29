"""Test `lifemate.eval.asr_tune`: dò từng bước, hàm mục tiêu, lựa chọn theo engine, không đo lại config trùng.

Lưu ý:
- Runner giả trả báo cáo dựng từ `generate_kwargs`, nên không cần GPU, model hay dữ liệu cuộc thi.
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any

import pytest

from lifemate.eval import asr_tune


def _report(miss: float, empty: int = 0, p95: float = 1.0, wer: float = 0.3) -> dict[str, Any]:
    """Tạo báo cáo `eval_asr` tối thiểu (100 câu) đủ cho hàm mục tiêu."""
    return {"entity_miss_rate": miss, "n_empty": empty, "n_utterances": 100, "p95_s": p95, "wer_nonum": wer, "wer": wer}


def test_objective_and_supported() -> None:
    """Hàm mục tiêu cộng bỏ sót thực thể với tỉ lệ câu rỗng; p95 quá ngưỡng xếp sau; lần đo lỗi đứng cuối."""
    assert asr_tune.objective(_report(0.3, empty=10), 6.0) == (0.0, 0.4, 0.3)
    # Vượt ngưỡng thời gian: xếp sau config đạt ngưỡng dù chính xác hơn, nhưng vẫn so được với nhau.
    slow_good, slow_bad = asr_tune.objective(_report(0.1, p95=7.0), 6.0), asr_tune.objective(_report(0.2, p95=7.0), 6.0)
    assert asr_tune.objective(_report(0.3), 6.0) < slow_good < slow_bad
    assert asr_tune.objective(None, 6.0) == (math.inf, math.inf, math.inf)
    assert asr_tune.supported("hf_seq2seq", {"beam_size": 5, "initial_prompt": "x"})
    assert not asr_tune.supported("hf_seq2seq", {"hotwords": "x"})
    assert not asr_tune.supported("hf_seq2seq", {"temperature": [0.0, 0.2]})
    assert asr_tune.supported("hf_seq2seq", {"prompt_guard_ngram": 3})
    assert not asr_tune.supported("chunkformer", {"prompt_guard_ngram": 3})
    assert asr_tune.supported("faster_whisper", {"vad_filter": True})
    assert asr_tune.supported("nemo_ctc", {"beam_size": 1})
    assert not asr_tune.supported("nemo_ctc", {"beam_size": 5})
    assert not asr_tune.supported("nemo_ctc", {"initial_prompt": "x"})
    assert asr_tune.supported("chunkformer", {}) and not asr_tune.supported("chunkformer", {"vad_filter": True})


def _gk_prompt(asr: dict[str, Any]) -> str:
    """Tên file câu mồi của một bảng `[asr]` đã ghi (rỗng khi không có câu mồi)."""
    return str(asr.get("initial_prompt_file", ""))


def test_prompt_options_are_full_grid() -> None:
    """Bước câu mồi có đủ 13 tổ hợp: không câu mồi, và 4 câu mồi x 3 mức bảo vệ; file câu mồi đều tồn tại."""
    opts = asr_tune.prompt_options()
    assert len(opts) == 13 and opts["none"] == {}
    assert opts["en_full+g4"] == {"initial_prompt_file": asr_tune.PROMPT_FILES["en_full"], "prompt_guard_ngram": 4}
    assert opts["short"] == {"initial_prompt_file": asr_tune.PROMPT_FILES["short"]}
    for path in asr_tune.PROMPT_FILES.values():
        assert (asr_tune.REPO_ROOT / path).read_text(encoding="utf-8").strip()


def test_tune_picks_per_stage_and_reuses_runs(tmp_path: Path, repo_root: Path) -> None:
    """Mỗi bước giữ lựa chọn tốt nhất, config trùng không đo lại, và PhoWhisper bỏ lựa chọn chỉ faster-whisper có.

    Câu mồi dài nhất chỉ tốt khi có lớp bảo vệ: dò cùng lúc câu mồi và lớp bảo vệ thì vẫn chọn được nó.
    Config sinh ra có bảng `[asr]` giống nhau giữa hai model, trừ engine, model và revision.
    """
    with open(repo_root / "configs" / "v0.toml", "rb") as f:
        base = tomllib.load(f)
    calls: list[Path] = []
    full = asr_tune.PROMPT_FILES["en_full"]

    def runner(cfg: Path, ids: str, out: Path, hyps: Path) -> dict[str, Any]:
        """Runner giả: beam 5 tốt hơn; `en_full` không bảo vệ sinh nhiều câu rỗng, có bảo vệ cụm 4 từ thì tốt nhất."""
        calls.append(cfg)
        asr = tomllib.loads(cfg.read_text(encoding="utf-8"))["asr"]
        miss = 0.4 - 0.1 * (asr["generate_kwargs"]["beam_size"] == 5)
        empty = 2
        if _gk_prompt(asr) == full:
            empty = 30 if not asr.get("prompt_guard_ngram") else 0
            miss -= 0.05 * (asr.get("prompt_guard_ngram") == 4)
        return _report(miss, empty=empty)

    res = asr_tune.tune("phowhisper", base, "ids.txt", tmp_path, runner, 6.0)
    assert [p["chosen"] for p in res["path"]] == ["beam5", "en_full+g4", "off", "none"]
    assert res["best_generate_kwargs"]["initial_prompt_file"] == full
    assert res["best_generate_kwargs"]["prompt_guard_ngram"] == 4
    assert "vad" not in {r["option"] for r in res["runs"]}
    # greedy, beam5, 12 tổ hợp có câu mồi ("none" dùng lại beam5), antirepeat on: 15 lần đo.
    assert len(calls) == 15
    # Khoá câu mồi và bảo vệ nằm ở bảng `[asr]`, không nằm trong `generate_kwargs`; config đọc lại được và có câu mồi.
    from lifemate.core.config import load_config

    cfg = load_config(tmp_path / "phowhisper__prompt_en_full+g4.toml")
    assert cfg.asr.prompt_guard_ngram == 4 and "prompt_guard_ngram" not in cfg.asr.generate_kwargs
    assert cfg.asr.generate_kwargs["initial_prompt"].endswith("coi miếng SJC luôn.")
    v3 = asr_tune.tune("large-v3", base, "ids.txt", tmp_path / "v3", runner, 6.0)
    assert {r["option"] for r in v3["runs"]} >= {"vad", "en_full+g3"}
    pk = asr_tune.tune("parakeet", base, "ids.txt", tmp_path / "pk", runner, 6.0)
    assert {(r["stage"], r["option"]) for r in pk["runs"]} == {
        ("decode", "greedy"),
        ("prompt", "none"),
        ("antirepeat", "off"),
        ("engine", "none"),
    }
    a = tomllib.loads(calls[0].read_text(encoding="utf-8"))["asr"]
    b = tomllib.loads((tmp_path / "v3" / "large-v3__decode_greedy.toml").read_text(encoding="utf-8"))["asr"]
    for k in ("engine", "model", "revision"):
        a.pop(k), b.pop(k)
    assert a == b


def test_tune_second_run_stages_start_and_objective(tmp_path: Path, repo_root: Path) -> None:
    """Lần 2: chỉ dò bước câu mồi, bật sẵn VAD cho large-v3, và hàm mục tiêu `en_mixed` tính cả từ tiếng Anh."""
    with open(repo_root / "configs" / "v0.toml", "rb") as f:
        base = tomllib.load(f)
    calls: list[dict[str, Any]] = []

    def runner(cfg: Path, ids: str, out: Path, hyps: Path) -> dict[str, Any]:
        """Runner giả: câu mồi chép đúng nhiều từ tiếng Anh hơn nhưng `en_top10` chèn nhiều từ của câu mồi."""
        asr = tomllib.loads(cfg.read_text(encoding="utf-8"))["asr"]
        calls.append(asr)
        prompt = _gk_prompt(asr)
        recall = 0.7 if prompt else 0.5
        ins = 40 if prompt == asr_tune.PROMPT_FILES["en_top10"] else 0
        return {**_report(0.3), "en_words": 100, "en_word_recall": recall, "prompt_insertions": ins}

    start = asr_tune.start_patch("faster_whisper", ["engine=vad"])
    res = asr_tune.tune(
        "large-v3", base, "ids.txt", tmp_path, runner, 6.0, stages=["prompt"], start=start, objective_kind="en_mixed"
    )
    assert [p["stage"] for p in res["path"]] == ["prompt"]
    # Hoà điểm giữa các câu mồi không chèn từ: giữ lựa chọn đứng trước theo thứ tự trong `prompt_options`.
    assert res["path"][0]["chosen"] == "short"
    assert all(c["generate_kwargs"]["vad_filter"] for c in calls) and len(calls) == 13
    assert asr_tune.start_patch("chunkformer", ["engine=vad"]) == {}
    rep = {**_report(0.3, empty=10), "en_words": 50, "en_word_recall": 0.8, "prompt_insertions": 5}
    assert asr_tune.objective(rep, 6.0, "en_mixed") == pytest.approx((0.0, 0.4 + 0.2 + 0.1, 0.3))
    assert asr_tune.objective(rep, 6.0) == pytest.approx((0.0, 0.4, 0.3))
    with pytest.raises(ValueError, match="stage=option"):
        asr_tune.start_patch("faster_whisper", ["engine=temp_fallback"])


def test_preflight_runs_each_supported_option_once(tmp_path: Path, repo_root: Path) -> None:
    """Chạy thử đo mỗi lựa chọn engine hỗ trợ một lần trên n mẫu; lựa chọn lỗi được đánh dấu FAIL."""
    with open(repo_root / "configs" / "v0.toml", "rb") as f:
        base = tomllib.load(f)
    ids = tmp_path / "ids.txt"
    ids.write_text("a\nb\nc\nd\n", encoding="utf-8")
    seen_ids: list[str] = []

    def runner(cfg: Path, ids_file: str, out: Path, hyps: Path) -> dict[str, Any] | None:
        """Runner giả: beam 5 lỗi, các lựa chọn khác thành công."""
        seen_ids.append(Path(ids_file).read_text(encoding="utf-8"))
        asr = tomllib.loads(cfg.read_text(encoding="utf-8"))["asr"]
        return None if asr["generate_kwargs"]["beam_size"] == 5 else _report(0.3)

    rows = asr_tune.preflight("phowhisper", base, str(ids), 2, tmp_path / "pho", runner)
    assert [r for r in rows if not r["ok"]] == [
        {"model": "phowhisper", "stage": "decode", "option": "beam5", "ok": False}
    ]
    # greedy (dùng chung cho mọi lựa chọn "giữ nguyên"), beam5, 12 tổ hợp có câu mồi, antirepeat on: 15 lần đo.
    assert len(seen_ids) == 15 and seen_ids[0] == "a\nb\n"


def test_emit_config_writes_untuned_config_per_model(tmp_path: Path, repo_root: Path) -> None:
    """`--emit-config` ghi config gốc (chưa dò) hợp lệ cho từng model, dùng được với `eval_asr --config`."""
    from lifemate.core.config import load_config

    args = ["--env", "local", "--models", "parakeet", "large-v3", "--emit-config", "--out", str(tmp_path)]
    assert asr_tune.main(args) == 0
    cfg = load_config(tmp_path / "parakeet.toml")
    assert cfg.asr.engine == "nemo_ctc" and cfg.asr.generate_kwargs == asr_tune.BASE_GENERATE_KWARGS
    assert load_config(tmp_path / "large-v3.toml").asr.engine == "faster_whisper"
