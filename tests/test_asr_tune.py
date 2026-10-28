"""Kiểm thử lifemate.eval.asr_tune: dò từng bước, hàm mục tiêu, lựa chọn theo engine, không đo lại cấu hình trùng.

Runner giả trả báo cáo dựng từ generate_kwargs, không cần GPU, mô hình hay dữ liệu cuộc thi.
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any

from lifemate.eval import asr_tune


def _report(miss: float, empty: int = 0, p95: float = 1.0, wer: float = 0.3) -> dict[str, Any]:
    """Báo cáo eval_asr tối thiểu cho hàm mục tiêu (100 câu)."""
    return {"entity_miss_rate": miss, "n_empty": empty, "n_utterances": 100, "p95_s": p95, "wer_nonum": wer, "wer": wer}


def test_objective_and_supported() -> None:
    """Hàm mục tiêu cộng bỏ sót thực thể với tỉ lệ câu rỗng; p95 quá ngưỡng xếp sau; lần đo lỗi đứng cuối."""
    assert asr_tune.objective(_report(0.3, empty=10), 6.0) == (0.0, 0.4, 0.3)
    # Vượt ngưỡng thời gian: xếp sau cấu hình đạt ngưỡng dù chính xác hơn, nhưng vẫn so được với nhau.
    slow_good, slow_bad = asr_tune.objective(_report(0.1, p95=7.0), 6.0), asr_tune.objective(_report(0.2, p95=7.0), 6.0)
    assert asr_tune.objective(_report(0.3), 6.0) < slow_good < slow_bad
    assert asr_tune.objective(None, 6.0) == (math.inf, math.inf, math.inf)
    assert asr_tune.supported("hf_seq2seq", {"beam_size": 5, "initial_prompt": "x"})
    assert not asr_tune.supported("hf_seq2seq", {"hotwords": "x"})
    assert not asr_tune.supported("hf_seq2seq", {"temperature": [0.0, 0.2]})
    assert asr_tune.supported("faster_whisper", {"vad_filter": True})
    assert asr_tune.supported("nemo_ctc", {"beam_size": 1})
    assert not asr_tune.supported("nemo_ctc", {"beam_size": 5})
    assert not asr_tune.supported("nemo_ctc", {"initial_prompt": "x"})
    assert asr_tune.supported("chunkformer", {}) and not asr_tune.supported("chunkformer", {"vad_filter": True})


def test_tune_picks_per_stage_and_reuses_runs(tmp_path: Path, repo_root: Path) -> None:
    """Mỗi bước giữ lựa chọn tốt nhất; cấu hình trùng (lựa chọn 'giữ nguyên') không đo lại; PhoWhisper bỏ lựa chọn
    chỉ faster-whisper có. Config sinh ra có [asr] giống nhau giữa hai mô hình trừ engine/model/revision."""
    with open(repo_root / "configs" / "v0.toml", "rb") as f:
        base = tomllib.load(f)
    calls: list[Path] = []

    def runner(cfg: Path, ids: str, out: Path, hyps: Path) -> dict[str, Any]:
        """Beam 5 và câu mồi ngắn tốt hơn; câu mồi của lead sinh nhiều câu rỗng."""
        calls.append(cfg)
        gk = tomllib.loads(cfg.read_text(encoding="utf-8"))["asr"]["generate_kwargs"]
        miss = 0.4 - 0.1 * (gk["beam_size"] == 5) - 0.05 * (gk.get("initial_prompt") == asr_tune.SHORT_PROMPT)
        empty = 30 if gk.get("initial_prompt") == asr_tune.LEAD_PROMPT else 2
        return _report(miss, empty=empty)

    res = asr_tune.tune("phowhisper", base, "ids.txt", tmp_path, runner, 6.0)
    assert [p["chosen"] for p in res["path"]] == ["beam5", "short", "off", "none"]
    assert res["best_generate_kwargs"]["initial_prompt"] == asr_tune.SHORT_PROMPT
    assert not {r["option"] for r in res["runs"]} & {"hotwords", "vad", "temp_fallback"}
    # greedy, beam5, lead, short, antirepeat on: 5 lần đo; các lựa chọn 'giữ nguyên' dùng lại kết quả.
    assert len(calls) == 5
    v3 = asr_tune.tune("large-v3", base, "ids.txt", tmp_path / "v3", runner, 6.0)
    assert {r["option"] for r in v3["runs"]} >= {"hotwords", "vad", "temp_fallback"}
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


def test_preflight_runs_each_supported_option_once(tmp_path: Path, repo_root: Path) -> None:
    """Chạy thử: mỗi lựa chọn engine hỗ trợ được đo một lần trên n mẫu; lựa chọn lỗi được đánh dấu FAIL."""
    with open(repo_root / "configs" / "v0.toml", "rb") as f:
        base = tomllib.load(f)
    ids = tmp_path / "ids.txt"
    ids.write_text("a\nb\nc\nd\n", encoding="utf-8")
    seen_ids: list[str] = []

    def runner(cfg: Path, ids_file: str, out: Path, hyps: Path) -> dict[str, Any] | None:
        """Beam 5 lỗi (giả lập), còn lại thành công."""
        seen_ids.append(Path(ids_file).read_text(encoding="utf-8"))
        gk = tomllib.loads(cfg.read_text(encoding="utf-8"))["asr"]["generate_kwargs"]
        return None if gk["beam_size"] == 5 else _report(0.3)

    rows = asr_tune.preflight("phowhisper", base, str(ids), 2, tmp_path / "pho", runner)
    assert [r for r in rows if not r["ok"]] == [
        {"model": "phowhisper", "stage": "decode", "option": "beam5", "ok": False}
    ]
    # greedy (dùng chung cho mọi lựa chọn "giữ nguyên"), beam5, lead, short, antirepeat on: 5 lần đo.
    assert len(seen_ids) == 5 and seen_ids[0] == "a\nb\n"
