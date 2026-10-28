"""Chọn model và config giải mã ASR một cách công bằng: cùng tập dò, cùng cách dò, cùng hàm mục tiêu.

Nội dung chính:
- `MODELS`: các model ASR được so sánh.
- `STAGES`: các bước dò và lựa chọn của từng bước.
- `objective`: hàm mục tiêu để so hai config (nhỏ hơn là tốt hơn).
- `tune`: dò lần lượt từng bước cho một model.
- `report`: đo config thắng trên các tập báo cáo.
- `preflight`: chạy thử mọi lựa chọn trên vài mẫu để bắt lỗi cài đặt.
- `main`: CLI.

Cách hoạt động:
1. Xuất phát từ config chung `BASE_GK`: giải mã tham lam, temperature 0, không câu mồi, không chống lặp, không
   VAD. Bộ lọc ảo giác bật như nhau cho mọi model; bộ lọc chạy sau ASR nên không phụ thuộc engine.
2. Dò lần lượt từng bước trong `STAGES`. Ở mỗi bước, thử mọi lựa chọn trên config thắng của các bước trước và giữ
   lựa chọn có `objective` nhỏ nhất. Lựa chọn mà engine không hỗ trợ thì bỏ qua (xem `supported`). Config nào đã
   đo thì dùng lại kết quả, không đo lại.
3. Hàm mục tiêu: tỉ lệ bỏ sót thực thể + tỉ lệ câu rỗng, vì đây là thứ ảnh hưởng điểm thi; hoà thì so
   `wer_nonum`. Config có p95 thời gian vượt `max_p95` xếp sau mọi config đạt ngưỡng, nhưng vẫn được so với nhau.
4. Config thắng của mỗi model được đo lại đúng một lần trên các tập báo cáo (`val_seen`, `val_unseen`). Các tập
   này không dùng để chọn.

Mỗi lần đo là một lệnh `python -m lifemate.eval.eval_asr` trên một config sinh từ `configs/v0.toml`, với bảng
`[asr]` được thay hoàn toàn.

Lưu ý:
- Các model: whisper-large-v3 (faster-whisper), PhoWhisper-medium (transformers), ChunkFormer CTC large và
  ChunkFormer RNNT large (gói `chunkformer`), Parakeet CTC 0,6B (NeMo).
- Tập dò là `configs/splits/asr_tune.txt` (300 mẫu lấy từ train, tạo bằng `make_splits --asr-tune 300`). ASR không
  học trên train nên không bị lộ đáp án, và các split chính không đổi.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from lifemate.core.paths import REPO_ROOT
from lifemate.core.tomlio import dump_toml

LEAD_PROMPT = (
    "LifeMate: trợ lý ảo tiếng Việt điều khiển smart home, tivi, đặt xe, ngân hàng, HTV7, VTV1, VTV3, deadline, "
    "voucher, Zalo."
)  # câu mồi trong repo của trưởng nhóm (src/preprocessing/asr/experiments/eval_fixes.py)
SHORT_PROMPT = "Nhắn Zalo cho Lan. Chuyển sang kênh VTV3."  # câu mồi ngắn, giống lời nói thật
HOTWORDS = "Zalo VTV1 VTV3 HTV7 voucher deadline"

MODELS: dict[str, dict[str, str]] = {
    "large-v3": {
        "engine": "faster_whisper",
        "model": "Systran/faster-whisper-large-v3",
        "revision": "edaa852ec7e145841d8ffdb056a99866b5f0a478",
    },
    "phowhisper": {
        "engine": "hf_seq2seq",
        "model": "vinai/PhoWhisper-medium",
        "revision": "55a7e3eb6c906de891f8f06a107754427dd3be79",
    },
    "chunkformer": {
        "engine": "chunkformer",
        "model": "khanhld/chunkformer-ctc-large-vie",
        "revision": "311fc03558a895dc2b32957f2fb4236c7fb1455b",
    },
    "chunkformer-rnnt": {
        "engine": "chunkformer",
        "model": "khanhld/chunkformer-rnnt-large-vie",
        "revision": "9863592e00d6c02634c4ce2db8a4f3414f1ba7d6",
    },
    "parakeet": {
        "engine": "nemo_ctc",
        "model": "nvidia/parakeet-ctc-0.6b-Vietnamese",
        "revision": "b0493142b49458810324e3db8be9e8e07b4ebc17",
    },
}

# Config chung để xuất phát. Tên khoá theo faster-whisper; engine transformers tự dịch sang tên khoá của transformers
# (`backends.asr.hf_generate_kwargs`).
BASE_GK: dict[str, Any] = {
    "language": "vi",
    "task": "transcribe",
    "temperature": 0.0,
    "beam_size": 1,
    "condition_on_previous_text": False,
}

# Các bước dò, theo thứ tự: (tên bước, {tên lựa chọn: các khoá ghi đè lên config hiện tại}).
STAGES: list[tuple[str, dict[str, dict[str, Any]]]] = [
    ("decode", {"greedy": {"beam_size": 1}, "beam5": {"beam_size": 5}}),
    (
        "prompt",
        {
            "none": {},
            "lead": {"initial_prompt": LEAD_PROMPT},
            "short": {"initial_prompt": SHORT_PROMPT},
            "hotwords": {"hotwords": HOTWORDS},
        },
    ),
    ("antirepeat", {"off": {}, "on": {"repetition_penalty": 1.2, "no_repeat_ngram_size": 3}}),
    (
        "engine",
        {
            "none": {},
            "vad": {"vad_filter": True, "vad_parameters": {"min_silence_duration_ms": 500}},
            "temp_fallback": {"temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},
        },
    ),
]

# Các khoá chỉ faster-whisper có. Temperature thử lại (list nhiều mức) cũng chỉ bật ở faster-whisper, vì
# transformers cần thêm ngưỡng riêng.
_FW_ONLY = frozenset({"hotwords", "vad_filter", "vad_parameters"})


def supported(engine: str, patch: Mapping[str, Any]) -> bool:
    """Kiểm tra engine có chạy được lựa chọn `patch` không.

    - `faster_whisper`: mọi lựa chọn.
    - `hf_seq2seq` (transformers): không có hotwords, VAD, temperature thử lại.
    - `nemo_ctc`, `chunkformer` (CTC, RNNT): chỉ giải mã tham lam; không có câu mồi, beam, chống lặp.

    Tham số:
        engine: tên engine, như trong `MODELS`.
        patch: các khoá mà lựa chọn ghi đè.

    Trả về:
        `True` khi engine hỗ trợ.
    """
    if engine in ("nemo_ctc", "chunkformer"):
        return all(k == "beam_size" and v == 1 for k, v in patch.items())
    if engine == "faster_whisper":
        return True
    return not (_FW_ONLY & patch.keys()) and not isinstance(patch.get("temperature"), list)


def objective(rep: Mapping[str, Any] | None, max_p95: float) -> tuple[float, float, float]:
    """Tính hàm mục tiêu của một lần đo; bộ ba nhỏ hơn là tốt hơn.

    Config có p95 vượt `max_p95` xếp sau mọi config đạt ngưỡng nhưng vẫn được so với nhau. Nhờ vậy model chậm trên T4
    (large-v3 có p95 khoảng 8 giây) vẫn chọn được config tốt nhất của model đó.

    Tham số:
        rep: báo cáo của `eval_asr`; `None` khi lần đo lỗi.
        max_p95: ngưỡng p95 thời gian, tính bằng giây.

    Trả về:
        Bộ ba (vượt ngưỡng thời gian 0/1, bỏ sót thực thể + tỉ lệ câu rỗng, `wer_nonum`). Lần đo lỗi cho
        `(inf, inf, inf)`.
    """
    if not rep:
        return (math.inf, math.inf, math.inf)
    empty_rate = rep["n_empty"] / max(1, rep["n_utterances"])
    over = float(rep.get("p95_s", math.inf) > max_p95)
    return (over, rep["entity_miss_rate"] + empty_rate, rep.get("wer_nonum", rep["wer"]))


def asr_section(model: str, gk: Mapping[str, Any]) -> dict[str, Any]:
    """Tạo bảng `[asr]` đầy đủ cho một model và một bộ `generate_kwargs`; mọi khoá khác giống nhau giữa các model."""
    return {
        "backend": "asr",
        **MODELS[model],
        "device": "auto",
        "dtype": "auto",
        "batch_size": 8,
        "generate_kwargs": dict(gk),
        "sample_rate": 16000,
        "sanitize": True,
    }


# Runner(config, ids, out_json, hyps_jsonl) -> báo cáo `eval_asr` dạng dict, hoặc None nếu lần đo lỗi.
Runner = Callable[[Path, str, Path, Path], dict[str, Any] | None]


def make_runner(env: str, split: str) -> Runner:
    """Tạo Runner thật: gọi `python -m lifemate.eval.eval_asr` trong tiến trình con rồi đọc file JSON kết quả.

    Tham số:
        env: tên profile môi trường.
        split: tên split chứa audio.
    """

    def run(cfg: Path, ids: str, out: Path, hyps: Path) -> dict[str, Any] | None:
        """Đo một config trên một file id; lỗi thì trả về `None` để vòng dò ghi lại rồi đi tiếp."""
        cmd = [sys.executable, "-m", "lifemate.eval.eval_asr", "--config", str(cfg), "--env", env]
        cmd += ["--split", split, "--ids", ids, "--out", str(out), "--hyps", str(hyps)]
        code = subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode
        if code != 0 or not out.is_file():
            return None
        data: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
        return data

    return run


def _measure(
    model: str, gk: Mapping[str, Any], base: Mapping[str, Any], ids: str, name: str, out_dir: Path, runner: Runner
) -> dict[str, Any] | None:
    """Ghi config `<name>.toml` (v0 với bảng `[asr]` thay hoàn toàn) rồi đo trên `ids`.

    Tham số:
        model: tên model trong `MODELS`.
        gk: `generate_kwargs` cần đo.
        base: config gốc đã đọc từ TOML.
        ids: đường dẫn file id.
        name: tên lần đo, dùng đặt tên file.
        out_dir: thư mục ghi config và kết quả.
        runner: hàm thực hiện lần đo.

    Trả về:
        Báo cáo `eval_asr`, hoặc `None` khi lần đo lỗi.
    """
    raw = {**base, "asr": asr_section(model, gk)}
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = out_dir / f"{name}.toml"
    cfg.write_text(dump_toml(raw), encoding="utf-8")
    return runner(cfg, ids, out_dir / f"{name}.json", out_dir / f"{name}.hyps.jsonl")


def tune(
    model: str, base: Mapping[str, Any], ids: str, out_dir: Path, runner: Runner, max_p95: float
) -> dict[str, Any]:
    """Dò lần lượt từng bước trong `STAGES` cho một model.

    Tham số:
        model: tên model trong `MODELS`.
        base: config gốc.
        ids: file id của tập dò.
        out_dir: thư mục ghi kết quả.
        runner: hàm thực hiện lần đo.
        max_p95: ngưỡng p95 thời gian, tính bằng giây.

    Trả về:
        Dict gồm config thắng (`best_generate_kwargs`), lựa chọn ở từng bước (`path`) và mọi lần đo (`runs`).
    """
    engine = MODELS[model]["engine"]
    gk: dict[str, Any] = dict(BASE_GK)
    cache: dict[str, tuple[str, dict[str, Any] | None]] = {}
    runs: list[dict[str, Any]] = []
    path: list[dict[str, str]] = []
    for stage, options in STAGES:
        cands: list[tuple[tuple[float, float, float], str, dict[str, Any]]] = []
        for opt, patch in options.items():
            if not supported(engine, patch):
                continue
            cand = {**gk, **patch}
            # Khoá cache là JSON của cả config: hai lựa chọn ra cùng config thì chỉ đo một lần.
            key = json.dumps(cand, sort_keys=True, ensure_ascii=False)
            if key not in cache:
                name = f"{model}__{stage}_{opt}"
                cache[key] = (name, _measure(model, cand, base, ids, name, out_dir, runner))
            name, rep = cache[key]
            score = objective(rep, max_p95)
            runs.append({"stage": stage, "option": opt, "run": name, "objective": list(score), "report": rep})
            cands.append((score, opt, cand))
        best_score, best_opt, gk = min(cands, key=lambda c: c[0])
        path.append({"stage": stage, "chosen": best_opt, "objective": json.dumps(list(best_score))})
    return {"model": model, "best_generate_kwargs": gk, "path": path, "runs": runs}


def report(
    model: str, gk: Mapping[str, Any], base: Mapping[str, Any], ids_files: Sequence[str], out_dir: Path, runner: Runner
) -> dict[str, Any]:
    """Đo config thắng đúng một lần trên mỗi tập báo cáo; các tập này không dùng để chọn."""
    return {f: _measure(model, gk, base, f, f"{model}__report_{Path(f).stem}", out_dir, runner) for f in ids_files}


def preflight(
    model: str, base: Mapping[str, Any], ids: str, n: int, out_dir: Path, runner: Runner
) -> list[dict[str, Any]]:
    """Chạy thử từng lựa chọn engine hỗ trợ (mỗi lần một lựa chọn, trên `BASE_GK`) với `n` mẫu đầu của `ids`.

    Dùng để bắt lỗi cài đặt, nạp model và tham số trước lần dò thật, ví dụ trên kernel CPU để khỏi tốn GPU.

    Tham số:
        model: tên model trong `MODELS`.
        base: config gốc.
        ids: file id; chỉ lấy `n` id đầu.
        n: số mẫu chạy thử.
        out_dir: thư mục ghi kết quả.
        runner: hàm thực hiện lần đo.

    Trả về:
        Mỗi lựa chọn một dòng `{model, stage, option, ok}`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(ids) if Path(ids).is_absolute() else REPO_ROOT / ids
    head = [x for x in src.read_text(encoding="utf-8").split() if x][:n]
    sub = out_dir / "preflight_ids.txt"
    sub.write_text("".join(f"{x}\n" for x in head), encoding="utf-8")
    engine = MODELS[model]["engine"]
    seen: dict[str, bool] = {}
    rows: list[dict[str, Any]] = []
    for stage, options in STAGES:
        for opt, patch in options.items():
            if not supported(engine, patch):
                continue
            cand = {**BASE_GK, **patch}
            key = json.dumps(cand, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen[key] = (
                    _measure(model, cand, base, str(sub), f"{model}__pre_{stage}_{opt}", out_dir, runner) is not None
                )
            rows.append({"model": model, "stage": stage, "option": opt, "ok": seen[key]})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy CLI: `python -m lifemate.eval.asr_tune --env kaggle --models <tên trong MODELS> ... --out <thư mục>`."""
    ap = argparse.ArgumentParser(prog="lifemate.eval.asr_tune", description=__doc__)
    ap.add_argument("--base", type=Path, default=REPO_ROOT / "configs" / "v0.toml", help="config gốc (trừ [asr])")
    ap.add_argument("--env", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--tune-ids", default="configs/splits/asr_tune.txt")
    ap.add_argument("--report-ids", nargs="*", default=["configs/splits/val_seen.txt", "configs/splits/val_unseen.txt"])
    ap.add_argument("--max-p95", type=float, default=6.0, help="giây; cấu hình có p95 lớn hơn xếp sau")
    ap.add_argument("--preflight", type=int, default=0, metavar="N", help="chỉ chạy thử mọi lựa chọn trên N mẫu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    with open(args.base, "rb") as f:
        base = tomllib.load(f)
    runner = make_runner(args.env, args.split)
    if args.preflight:
        rows = [r for m in args.models for r in preflight(m, base, args.tune_ids, args.preflight, args.out / m, runner)]
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "preflight.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        for r in rows:
            print(f"preflight {r['model']:<16} {r['stage']:<10} {r['option']:<14} {'OK' if r['ok'] else 'FAIL'}")
        return 0 if all(r["ok"] for r in rows) else 1
    summary: dict[str, Any] = {"tune_ids": args.tune_ids, "max_p95": args.max_p95, "models": {}}
    for model in args.models:
        res = tune(model, base, args.tune_ids, args.out / model, runner, args.max_p95)
        res["report"] = report(model, res["best_generate_kwargs"], base, args.report_ids, args.out / model, runner)
        summary["models"][model] = res
        # Ghi sau mỗi model, để giữ được kết quả nếu phiên bị ngắt giữa chừng.
        (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps({"model": model, "path": res["path"], "best": res["best_generate_kwargs"]}, ensure_ascii=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
