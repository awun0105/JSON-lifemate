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
1. Xuất phát từ config chung `BASE_GENERATE_KWARGS`: giải mã tham lam, temperature 0, không câu mồi, không chống
   lặp, không VAD. Bộ lọc ảo giác bật như nhau cho mọi model; bộ lọc chạy sau ASR nên không phụ thuộc engine.
2. Dò lần lượt từng bước trong `STAGES`. Ở mỗi bước, thử mọi lựa chọn trên config thắng của các bước trước và giữ
   lựa chọn có `objective` nhỏ nhất. Lựa chọn mà engine không hỗ trợ thì bỏ qua (xem `supported`). Config nào đã
   đo thì dùng lại kết quả, không đo lại.
3. Hàm mục tiêu: tỉ lệ bỏ sót thực thể + tỉ lệ câu rỗng, vì đây là thứ ảnh hưởng điểm thi; hoà thì so
   `wer_nonum`. Config có p95 thời gian vượt `max_p95` xếp sau mọi config đạt ngưỡng, nhưng vẫn được so với nhau.
   `--objective en_mixed` (benchmark lần 2) cộng thêm tỉ lệ từ tiếng Anh chép sai và tỉ lệ chèn từ của câu mồi.
4. Config thắng của mỗi model được đo lại đúng một lần trên các tập báo cáo (`val_seen`, `val_unseen`). Các tập
   này không dùng để chọn.

Mỗi lần đo là một lệnh `python -m lifemate.eval.eval_asr` trên một config sinh từ `configs/v0.toml`, với bảng
`[asr]` được thay hoàn toàn.

Benchmark lần 2 (câu chêm tiếng Anh, `docs/reports/asr_benchmark_en_mixed.md`): `--stages prompt` chỉ dò bước câu
mồi (13 tổ hợp câu mồi x lớp bảo vệ), `--start-option engine=vad` bật VAD từ đầu cho large-v3, `--objective en_mixed`.
Dò trên `asr_tune` đầy đủ và đo trên `val_seen`, `val_unseen` đầy đủ; số của các tập `*_en_mixed` lấy bằng cách chấm
lại tập con (`lifemate.eval.rescore_asr`), vì các tập đó là tập con.

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

# Câu mồi có phiên bản (`configs/asr_prompts/README.md`): tên lựa chọn -> file, đường dẫn tính từ gốc repo.
# `short` là câu mồi ngắn của lần 1 (thắng ở PhoWhisper); ba câu `en_*` của lần 2 chứa từ tiếng Anh mà large-v3 hay
# chép sai, lồng nhau theo độ dài.
PROMPT_FILES: dict[str, str] = {
    "short": "configs/asr_prompts/short_v1.txt",
    "en_top10": "configs/asr_prompts/en_top10_v1.txt",
    "en_top20": "configs/asr_prompts/en_top20_v1.txt",
    "en_full": "configs/asr_prompts/en_full_v1.txt",
}
# Mức bảo vệ câu mồi (`lifemate.backends.asr_guard`): hậu tố tên lựa chọn -> `prompt_guard_ngram`.
GUARDS: dict[str, int] = {"": 0, "+g3": 3, "+g4": 4}


def prompt_options() -> dict[str, dict[str, Any]]:
    """Các lựa chọn của bước `prompt`: không câu mồi, và mỗi câu mồi ghép với mỗi mức bảo vệ (1 + 4 x 3 = 13).

    Câu mồi và lớp bảo vệ được dò cùng lúc chứ không lần lượt: câu mồi dài có thể chỉ tốt khi có lớp bảo vệ (lớp bảo
    vệ cứu các câu bị chép lại câu mồi), nên dò câu mồi khi chưa có lớp bảo vệ sẽ loại oan nó.
    """
    out: dict[str, dict[str, Any]] = {"none": {}}
    for name, path in PROMPT_FILES.items():
        for suffix, ngram in GUARDS.items():
            out[name + suffix] = {"initial_prompt_file": path, **({"prompt_guard_ngram": ngram} if ngram else {})}
    return out


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
BASE_GENERATE_KWARGS: dict[str, Any] = {
    "language": "vi",
    "task": "transcribe",
    "temperature": 0.0,
    "beam_size": 1,
    "condition_on_previous_text": False,
}

# Các bước dò, theo thứ tự: (tên bước, {tên lựa chọn: các khoá ghi đè lên config hiện tại}).
# Đã bỏ sau lần 1: câu mồi của trưởng nhóm và `hotwords` (kém nhất ở large-v3), nhiệt độ thử lại (lấy mẫu ngẫu nhiên,
# không tất định như thể lệ yêu cầu); xem `docs/decisions.md`.
STAGES: list[tuple[str, dict[str, dict[str, Any]]]] = [
    ("decode", {"greedy": {"beam_size": 1}, "beam5": {"beam_size": 5}}),
    ("prompt", prompt_options()),
    ("antirepeat", {"off": {}, "on": {"repetition_penalty": 1.2, "no_repeat_ngram_size": 3}}),
    ("engine", {"none": {}, "vad": {"vad_filter": True, "vad_parameters": {"min_silence_duration_ms": 500}}}),
]
STAGE_NAMES = [name for name, _ in STAGES]

# Khoá của lựa chọn thuộc bảng `[asr]` chứ không thuộc `generate_kwargs`; `asr_section` chuyển chúng ra ngoài.
_ASR_SECTION_KEYS = frozenset({"prompt_guard_ngram", "initial_prompt_file"})

# Các khoá chỉ faster-whisper có. Temperature thử lại (list nhiều mức) cũng chỉ bật ở faster-whisper, vì
# transformers cần thêm ngưỡng riêng.
_FASTER_WHISPER_ONLY_KEYS = frozenset({"hotwords", "vad_filter", "vad_parameters"})


def supported(engine: str, patch: Mapping[str, Any]) -> bool:
    """Kiểm tra engine có chạy được lựa chọn `patch` không.

    - `faster_whisper`: mọi lựa chọn.
    - `hf_seq2seq` (transformers): không có hotwords, VAD, temperature thử lại.
    - `nemo_ctc`, `chunkformer` (CTC, RNNT): chỉ giải mã tham lam; không có câu mồi, beam, chống lặp, bảo vệ câu mồi.

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
    return not (_FASTER_WHISPER_ONLY_KEYS & patch.keys()) and not isinstance(patch.get("temperature"), list)


OBJECTIVES = ("entity", "en_mixed")


def objective(run_report: Mapping[str, Any] | None, max_p95: float, kind: str = "entity") -> tuple[float, float, float]:
    """Tính hàm mục tiêu của một lần đo; bộ ba nhỏ hơn là tốt hơn.

    Config có p95 vượt `max_p95` xếp sau mọi config đạt ngưỡng nhưng vẫn được so với nhau. Nhờ vậy model chậm trên T4
    (large-v3 có p95 khoảng 8 giây) vẫn chọn được config tốt nhất của model đó.

    Tham số:
        run_report: báo cáo của `eval_asr`; `None` khi lần đo lỗi.
        max_p95: ngưỡng p95 thời gian, tính bằng giây.
        kind: `"entity"` (lần 1) hoặc `"en_mixed"` (lần 2: cộng thêm tỉ lệ từ tiếng Anh chép sai và số lần chèn từ
            của câu mồi chia cho số từ tiếng Anh).

    Trả về:
        Bộ ba (vượt ngưỡng thời gian 0/1, tổng tỉ lệ lỗi, `wer_nonum`). Lần đo lỗi cho `(inf, inf, inf)`.
    """
    if not run_report:
        return (math.inf, math.inf, math.inf)
    empty_rate = run_report["n_empty"] / max(1, run_report["n_utterances"])
    over = float(run_report.get("p95_s", math.inf) > max_p95)
    err = run_report["entity_miss_rate"] + empty_rate
    if kind == "en_mixed":
        n_en = run_report.get("en_words", 0)
        if n_en:
            err += 1.0 - run_report["en_word_recall"] + run_report.get("prompt_insertions", 0) / n_en
    return (over, err, run_report.get("wer_nonum", run_report["wer"]))


def asr_section(model: str, generate_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Tạo bảng `[asr]` đầy đủ cho một model và một bộ `generate_kwargs`; mọi khoá khác giống nhau giữa các model.

    Khoá trong `_ASR_SECTION_KEYS` (ví dụ `prompt_guard_ngram`) được chuyển từ `generate_kwargs` ra bảng `[asr]`.
    """
    return {
        "backend": "asr",
        **MODELS[model],
        "device": "auto",
        "dtype": "auto",
        "batch_size": 8,
        "generate_kwargs": {k: v for k, v in generate_kwargs.items() if k not in _ASR_SECTION_KEYS},
        "sample_rate": 16000,
        "sanitize": True,
        **{k: v for k, v in generate_kwargs.items() if k in _ASR_SECTION_KEYS},
    }


def normalize_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Bỏ lớp bảo vệ câu mồi khi config không có câu mồi (lớp bảo vệ khi đó không làm gì), để không đo trùng."""
    if candidate.get("initial_prompt") or candidate.get("initial_prompt_file"):
        return dict(candidate)
    return {k: v for k, v in candidate.items() if k != "prompt_guard_ngram"}


def start_patch(engine: str, start_options: Sequence[str]) -> dict[str, Any]:
    """Gộp các lựa chọn khởi đầu dạng `bước=lựa_chọn` (ví dụ `engine=vad`) thành một bộ khoá ghi đè.

    Lựa chọn engine không hỗ trợ thì bỏ qua, để một lệnh dùng chung được cho nhiều model.

    Lỗi:
        ValueError: khi tên bước hoặc tên lựa chọn không có trong `STAGES`.
    """
    stages = dict(STAGES)
    out: dict[str, Any] = {}
    for item in start_options:
        stage, _, option = item.partition("=")
        if stage not in stages or option not in stages[stage]:
            raise ValueError(f"--start-option {item!r}: not a stage=option in STAGES")
        if supported(engine, stages[stage][option]):
            out.update(stages[stage][option])
    return out


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
    model: str,
    generate_kwargs: Mapping[str, Any],
    base: Mapping[str, Any],
    ids: str,
    name: str,
    out_dir: Path,
    runner: Runner,
) -> dict[str, Any] | None:
    """Ghi config `<name>.toml` (v0 với bảng `[asr]` thay hoàn toàn) rồi đo trên `ids`.

    Tham số:
        model: tên model trong `MODELS`.
        generate_kwargs: `generate_kwargs` cần đo.
        base: config gốc đã đọc từ TOML.
        ids: đường dẫn file id.
        name: tên lần đo, dùng đặt tên file.
        out_dir: thư mục ghi config và kết quả.
        runner: hàm thực hiện lần đo.

    Trả về:
        Báo cáo `eval_asr`, hoặc `None` khi lần đo lỗi.
    """
    raw = {**base, "asr": asr_section(model, generate_kwargs)}
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = out_dir / f"{name}.toml"
    cfg.write_text(dump_toml(raw), encoding="utf-8")
    return runner(cfg, ids, out_dir / f"{name}.json", out_dir / f"{name}.hyps.jsonl")


def tune(
    model: str,
    base: Mapping[str, Any],
    ids: str,
    out_dir: Path,
    runner: Runner,
    max_p95: float,
    *,
    stages: Sequence[str] | None = None,
    start: Mapping[str, Any] | None = None,
    objective_kind: str = "entity",
) -> dict[str, Any]:
    """Dò lần lượt từng bước trong `STAGES` cho một model.

    Tham số:
        model: tên model trong `MODELS`.
        base: config gốc.
        ids: file id của tập dò.
        out_dir: thư mục ghi kết quả.
        runner: hàm thực hiện lần đo.
        max_p95: ngưỡng p95 thời gian, tính bằng giây.
        stages: tên các bước cần dò, giữ thứ tự của `STAGES`; `None` là mọi bước.
        start: khoá ghi đè lên `BASE_GENERATE_KWARGS` trước khi dò (xem `start_patch`).
        objective_kind: một tên trong `OBJECTIVES`.

    Trả về:
        Dict gồm config thắng (`best_generate_kwargs`), lựa chọn ở từng bước (`path`) và mọi lần đo (`runs`).
    """
    engine = MODELS[model]["engine"]
    generate_kwargs: dict[str, Any] = {**BASE_GENERATE_KWARGS, **(start or {})}
    cache: dict[str, tuple[str, dict[str, Any] | None]] = {}
    runs: list[dict[str, Any]] = []
    path: list[dict[str, str]] = []
    for stage, options in STAGES:
        if stages is not None and stage not in stages:
            continue
        candidates: list[tuple[tuple[float, float, float], str, dict[str, Any]]] = []
        for option, patch in options.items():
            if not supported(engine, patch):
                continue
            candidate = normalize_candidate({**generate_kwargs, **patch})
            # Khoá cache là JSON của cả config: hai lựa chọn ra cùng config thì chỉ đo một lần.
            key = json.dumps(candidate, sort_keys=True, ensure_ascii=False)
            if key not in cache:
                name = f"{model}__{stage}_{option}"
                cache[key] = (name, _measure(model, candidate, base, ids, name, out_dir, runner))
            name, run_report = cache[key]
            score = objective(run_report, max_p95, objective_kind)
            runs.append({"stage": stage, "option": option, "run": name, "objective": list(score), "report": run_report})
            candidates.append((score, option, candidate))
        best_score, best_opt, generate_kwargs = min(candidates, key=lambda c: c[0])
        path.append({"stage": stage, "chosen": best_opt, "objective": json.dumps(list(best_score))})
    return {"model": model, "best_generate_kwargs": generate_kwargs, "path": path, "runs": runs}


def report(
    model: str,
    generate_kwargs: Mapping[str, Any],
    base: Mapping[str, Any],
    ids_files: Sequence[str],
    out_dir: Path,
    runner: Runner,
) -> dict[str, Any]:
    """Đo config thắng đúng một lần trên mỗi tập báo cáo; các tập này không dùng để chọn."""
    return {
        f: _measure(model, generate_kwargs, base, f, f"{model}__report_{Path(f).stem}", out_dir, runner)
        for f in ids_files
    }


def preflight(
    model: str,
    base: Mapping[str, Any],
    ids: str,
    n: int,
    out_dir: Path,
    runner: Runner,
    *,
    stages: Sequence[str] | None = None,
    start: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Chạy thử từng lựa chọn engine hỗ trợ trên `n` mẫu đầu của `ids`.

    Mỗi lần thử bật một lựa chọn trên nền `BASE_GENERATE_KWARGS` + `start`.

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
        if stages is not None and stage not in stages:
            continue
        for option, patch in options.items():
            if not supported(engine, patch):
                continue
            candidate = normalize_candidate({**BASE_GENERATE_KWARGS, **(start or {}), **patch})
            key = json.dumps(candidate, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen[key] = (
                    _measure(model, candidate, base, str(sub), f"{model}__pre_{stage}_{option}", out_dir, runner)
                    is not None
                )
            rows.append({"model": model, "stage": stage, "option": option, "ok": seen[key]})
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
    ap.add_argument("--stages", nargs="+", choices=STAGE_NAMES, default=None, help="chỉ dò các bước này")
    ap.add_argument(
        "--start-option",
        action="append",
        default=[],
        metavar="STAGE=OPTION",
        help="bật sẵn một lựa chọn trước khi dò, lặp được, ví dụ --start-option engine=vad",
    )
    ap.add_argument("--objective", choices=OBJECTIVES, default="entity", help="hàm mục tiêu (en_mixed: lần 2)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--emit-config",
        action="store_true",
        help="chỉ ghi config gốc (chưa dò) của từng model ra <out>/<model>.toml, để chạy eval_asr riêng",
    )
    args = ap.parse_args(argv)
    with open(args.base, "rb") as f:
        base = tomllib.load(f)
    if args.emit_config:
        args.out.mkdir(parents=True, exist_ok=True)
        for m in args.models:
            path = args.out / f"{m}.toml"
            path.write_text(dump_toml({**base, "asr": asr_section(m, BASE_GENERATE_KWARGS)}), encoding="utf-8")
            print(path)
        return 0
    runner = make_runner(args.env, args.split)
    try:
        starts = {m: start_patch(MODELS[m]["engine"], args.start_option) for m in args.models}
    except ValueError as e:
        ap.error(str(e))
    if args.preflight:
        rows = [
            r
            for m in args.models
            for r in preflight(
                m, base, args.tune_ids, args.preflight, args.out / m, runner, stages=args.stages, start=starts[m]
            )
        ]
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "preflight.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        for r in rows:
            print(f"preflight {r['model']:<16} {r['stage']:<10} {r['option']:<14} {'OK' if r['ok'] else 'FAIL'}")
        return 0 if all(r["ok"] for r in rows) else 1
    summary: dict[str, Any] = {
        "tune_ids": args.tune_ids,
        "max_p95": args.max_p95,
        "stages": args.stages or STAGE_NAMES,
        "start_option": args.start_option,
        "objective": args.objective,
        "models": {},
    }
    for model in args.models:
        res = tune(
            model,
            base,
            args.tune_ids,
            args.out / model,
            runner,
            args.max_p95,
            stages=args.stages,
            start=starts[model],
            objective_kind=args.objective,
        )
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
