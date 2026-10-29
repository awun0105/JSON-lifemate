"""Bộ chạy các phiên GPU của benchmark ASR lần 3 (`docs/reports/benchmark/asr_v3.md`, phiên K1–K3).

Cách dùng (trên máy có GPU, từ thư mục gốc repo):
  python -m lifemate.eval.asr_v3 transcribe --env kaggle --ids configs/splits/asr_tune.txt --variants E0 E1 E2 \\
      --rnnt --rerun 50 --out <thư mục>
  python -m lifemate.eval.asr_v3 score --env kaggle --ids configs/splits/asr_tune.txt \\
      --rnnt <thư mục>/rnnt.hyps.jsonl --v3 <thư mục>/v3_E1.hyps.jsonl --rerun 50 --out <thư mục>/scores_asr_tune

Nội dung chính:
- `RNNT_CFG`, `whisper_cfg`: cấu hình hai mô hình, đúng cấu hình thắng của lần 2 (RNNT tham lam; large-v3 tham lam,
  VAD, câu mồi `en_full_v1`, lớp bảo vệ g4, bộ lọc câu bịa).
- `utterance_keys`: khoá (lượt) của từng câu người dùng theo thứ tự file âm thanh.
- `transcribe_variants`: lệnh `transcribe` (phiên K1, K2): RNNT và large-v3 với từng biến thể câu mồi của E, ghi chữ
  từng câu, thời gian từng bước và bộ nhớ GPU.
- `score_candidates`: lệnh `score` (phiên K3): điểm âm thanh hai câu ứng viên của mọi vùng cần quyết định (G).
- `main`: CLI.

Cách hoạt động của `transcribe`:
1. Nạp RNNT và large-v3 một lần, khởi động mỗi mô hình trên 1 giây im lặng (không tính giờ).
2. Với mỗi mẫu: RNNT chép mọi câu (lượt người dùng trong lịch sử, rồi lượt cuối). Với mỗi biến thể câu mồi
   (`asr_prompt.build_sample_prompts`), large-v3 chép từng câu; câu hỏng theo lớp bảo vệ (`asr_guard.needs_retry`, so
   với **phần cố định** của câu mồi) thì chép lại không câu mồi; rồi lọc câu bịa (`asr_sanitizer.sanitize`).
3. Ghép chữ bằng luật của lần 2 (`context_other`) để đo thời gian của bước ghép.
4. Ghi `rnnt.hyps.jsonl`, `v3_<biến thể>.hyps.jsonl` (thêm trường `prompt`, `prompt_tokens`, `retried`, `scores`),
   `timing.jsonl` (mỗi mẫu một dòng), `gpu_memory.csv` và `summary.json` (bộ nhớ đỉnh, kiểm tra tất định).

Lưu ý:
- Kiểm tra tất định (`--rerun N`): chép lại N mẫu đầu với biến thể đầu tiên và so chữ từng ký tự với lần một.
- Chỉ chạy trên máy có GPU (Kaggle); máy dev không chạy mô hình.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from lifemate.backends.asr_guard import needs_retry
from lifemate.backends.asr_prompt import PROMPT_VARIANTS, build_sample_prompts
from lifemate.backends.asr_sanitizer import sanitize
from lifemate.core.config import ASRConfig
from lifemate.core.paths import REPO_ROOT
from lifemate.core.types import Sample
from lifemate.eval.asr_merge import english_filter, merge_transcripts, sample_context_words
from lifemate.eval.candidate_scoring import candidate_pairs
from lifemate.eval.foreign_words import find_foreign_words

FIXED_PROMPT_FILE = "configs/asr_prompts/en_full_v1.txt"
GUARD_NGRAM = 4
GUARD_MAX_COMPRESSION = 2.4
# Whisper giữ 223 token cuối của câu mồi (`max_length // 2 - 1`, `max_length = 448`).
PROMPT_TOKEN_LIMIT = 223

BASE_GENERATE_KWARGS: dict[str, Any] = {
    "language": "vi",
    "task": "transcribe",
    "temperature": 0.0,
    "beam_size": 1,
    "condition_on_previous_text": False,
}

RNNT_CFG = ASRConfig(
    backend="asr",
    engine="chunkformer",
    model="khanhld/chunkformer-rnnt-large-vie",
    revision="9863592e00d6c02634c4ce2db8a4f3414f1ba7d6",
    generate_kwargs=dict(BASE_GENERATE_KWARGS),
    sanitize=True,
)


def whisper_cfg(fixed_prompt: str) -> ASRConfig:
    """Cấu hình large-v3 thắng của lần 2: tham lam, VAD, câu mồi cố định `fixed_prompt` (lớp bảo vệ chạy ở đây)."""
    return ASRConfig(
        backend="asr",
        engine="faster_whisper",
        model="Systran/faster-whisper-large-v3",
        revision="edaa852ec7e145841d8ffdb056a99866b5f0a478",
        generate_kwargs={
            **BASE_GENERATE_KWARGS,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500},
            "initial_prompt": fixed_prompt,
        },
        sanitize=True,
    )


def utterance_keys(sample: Sample) -> list[tuple[str, str | None]]:
    """(lượt, đáp án) của từng câu người dùng theo thứ tự file âm thanh: chỉ số lượt trong lịch sử, rồi "final"."""
    keys: list[tuple[str, str | None]] = [
        (str(i), turn.audio_text) for i, turn in enumerate(sample.history) if turn.role == "user"
    ]
    keys.append(("final", sample.audio_text))
    return keys


def _load(env: str, split: str, ids_file: str, root: str | None) -> tuple[Path, list[Sample], dict[str, Any]]:
    """Đọc profile, các mẫu của `ids_file` và catalog (mã app -> mục catalog)."""
    from lifemate.core.config import load_profile
    from lifemate.core.paths import repo_path
    from lifemate.io.data import load_split, read_ids
    from lifemate.io.zaic import load_catalog

    profile = load_profile(env, REPO_ROOT).with_overrides(root=root)
    data_root = repo_path(profile.data_root, REPO_ROOT)
    samples = load_split(data_root, split, read_ids(repo_path(ids_file, REPO_ROOT)))
    catalog = {str(app["id"]): app for app in load_catalog(data_root, split)["apps"]}
    return data_root, samples, catalog


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], mode: str = "a") -> None:
    """Ghi thêm các dòng JSON vào `path`."""
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Đọc các dòng JSON của `path`."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _whisper_utterance(
    whisper: Any, audio: Any, prompt: str, fixed_prompt: str, is_history: bool
) -> tuple[str, str, bool, dict[str, float]]:
    """Chép một câu bằng large-v3 với câu mồi `prompt`, có lớp bảo vệ so với phần cố định; trả (hyp, raw, đã chép lại,
    điểm tin cậy)."""
    raw, scores = whisper.transcribe_one(audio, prompt)
    retried = needs_retry(raw, sanitize(raw, is_history=is_history), fixed_prompt, GUARD_NGRAM, GUARD_MAX_COMPRESSION)
    if retried:
        raw, scores = whisper.transcribe_one(audio, "")
    return sanitize(raw, is_history=is_history), raw, retried, scores


def transcribe_variants(args: argparse.Namespace) -> int:
    """Lệnh `transcribe`: xem docstring module."""
    from lifemate.backends.asr import ASRTranscriber, FasterWhisperASR, make_backend, read_wav
    from lifemate.eval.gpu_memory import GpuMemoryMonitor

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    fixed_prompt = (REPO_ROOT / FIXED_PROMPT_FILE).read_text(encoding="utf-8").strip()
    prompt_words = frozenset(w.folded for w in find_foreign_words(fixed_prompt))
    data_root, samples, catalog = _load(args.env, args.split, args.ids, args.root)
    for name in ["rnnt"] * bool(args.rnnt) + [f"v3_{v}" for v in args.variants]:
        (out / f"{name}.hyps.jsonl").write_text("", encoding="utf-8")
    (out / "timing.jsonl").write_text("", encoding="utf-8")
    summary: dict[str, Any] = {"ids": args.ids, "variants": list(args.variants), "n_samples": len(samples)}
    with GpuMemoryMonitor(out / "gpu_memory.csv") as monitor:
        t0 = time.perf_counter()
        wcfg = whisper_cfg(fixed_prompt)
        whisper = FasterWhisperASR(wcfg)
        # Chỉ dùng để lấy đường dẫn file âm thanh của mẫu theo đúng thứ tự.
        locator = ASRTranscriber(wcfg, data_root=data_root, split=args.split, backend=whisper)
        whisper.warmup()
        rnnt = None
        if args.rnnt:
            rnnt = ASRTranscriber(RNNT_CFG, data_root=data_root, split=args.split, backend=make_backend(RNNT_CFG))
            rnnt.warmup()
        summary["load_s"] = time.perf_counter() - t0
        tokenizer: Any = whisper._model.hf_tokenizer  # chỉ để đếm số token của câu mồi

        def run_sample(
            sample: Sample, variants: Sequence[str]
        ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
            """Chép một mẫu với RNNT và các biến thể; trả (tên hệ thống -> các dòng, dòng thời gian)."""
            keys = utterance_keys(sample)
            paths = locator.wav_paths(sample)
            audios = [read_wav(p, wcfg.sample_rate) for p in paths]
            timing: dict[str, Any] = {
                "id": sample.id,
                "n_utterances": len(keys),
                "audio_s": sum(len(a) for a in audios) / wcfg.sample_rate,
                "whisper_s": {},
                "merge_s": {},
                "retries": {},
            }
            rows: dict[str, list[dict[str, Any]]] = {}
            rnnt_hyps: list[str] = []
            if rnnt is not None:
                start = time.perf_counter()
                tr = rnnt.transcribe(sample)
                timing["rnnt_s"] = time.perf_counter() - start
                user_hist = [h for h, turn in zip(tr.history, sample.history, strict=True) if turn.role == "user"]
                rnnt_hyps = [str(h) for h in user_hist] + [tr.final]
                rows["rnnt"] = [
                    {"id": sample.id, "turn": k, "ref": ref, "hyp": hyp, "raw": raw}
                    for (k, ref), hyp, raw in zip(keys, rnnt_hyps, tr.raw, strict=True)
                ]
            accept = english_filter("context_other", sample_context_words(sample, catalog), prompt_words)
            for variant in variants:
                prompts = build_sample_prompts(sample, catalog, fixed_prompt, variant)
                start = time.perf_counter()
                done = [
                    _whisper_utterance(whisper, audio, prompt, fixed_prompt, i < len(audios) - 1)
                    for i, (audio, prompt) in enumerate(zip(audios, prompts, strict=True))
                ]
                timing["whisper_s"][variant] = time.perf_counter() - start
                timing["retries"][variant] = sum(d[2] for d in done)
                rows[f"v3_{variant}"] = [
                    {
                        "id": sample.id,
                        "turn": k,
                        "ref": ref,
                        "hyp": hyp,
                        "raw": raw,
                        "prompt": prompt,
                        "prompt_tokens": len(tokenizer.encode(" " + prompt).ids),
                        "retried": retried,
                        "scores": scores,
                    }
                    for (k, ref), (hyp, raw, retried, scores), prompt in zip(keys, done, prompts, strict=True)
                ]
                if rnnt_hyps:
                    start = time.perf_counter()
                    for r_hyp, (hyp, raw, _r, _s) in zip(rnnt_hyps, done, strict=True):
                        if hyp.strip() and " ".join(raw.split()) == " ".join(hyp.split()):
                            merge_transcripts(r_hyp, hyp, accept)
                    timing["merge_s"][variant] = time.perf_counter() - start
            return rows, timing

        for n, sample in enumerate(samples, 1):
            rows, timing = run_sample(sample, args.variants)
            for name, name_rows in rows.items():
                _write_jsonl(out / f"{name}.hyps.jsonl", name_rows)
            _write_jsonl(out / "timing.jsonl", [timing])
            if n % 25 == 0:
                print(f"[asr_v3] {n}/{len(samples)} samples", flush=True)
        if args.rerun:
            # Kiểm tra tất định: chép lại N mẫu đầu với biến thể đầu tiên, so chữ từng ký tự.
            first = {(r["id"], r["turn"]): r["hyp"] for r in _read_jsonl(out / f"v3_{args.variants[0]}.hyps.jsonl")}
            first_r = {(r["id"], r["turn"]): r["hyp"] for r in _read_jsonl(out / "rnnt.hyps.jsonl")} if rnnt else {}
            diff_v3 = diff_r = total = 0
            for sample in samples[: args.rerun]:
                rows, _timing = run_sample(sample, args.variants[:1])
                for row in rows[f"v3_{args.variants[0]}"]:
                    total += 1
                    diff_v3 += row["hyp"] != first[(row["id"], row["turn"])]
                for row in rows.get("rnnt", []):
                    diff_r += row["hyp"] != first_r[(row["id"], row["turn"])]
            summary["determinism"] = {"utterances": total, "diff_v3": diff_v3, "diff_rnnt": diff_r}
        summary["total_s"] = time.perf_counter() - t0
    summary["gpu_baseline_mb"] = monitor.baseline_mb
    summary["gpu_peak_mb"] = monitor.peak_mb
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


def score_candidates(args: argparse.Namespace) -> int:
    """Lệnh `score`: điểm âm thanh (large-v3 không câu mồi, ép chuỗi) hai câu ứng viên của mọi vùng cần quyết định.

    Ghi `<out>.jsonl` (mỗi vùng một dòng `{id, turn, part, a, b, score_a, n_a, score_b, n_b, long}`; `long` là câu dài
    hơn 30 giây, chỉ 30 giây đầu được chấm), `<out>.timing.jsonl` và `<out>.summary.json`.
    """
    from lifemate.backends.asr import ASRTranscriber, FasterWhisperASR, read_wav
    from lifemate.eval.gpu_memory import GpuMemoryMonitor

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    data_root, samples, _catalog = _load(args.env, args.split, args.ids, args.root)
    rnnt = {(r["id"], r["turn"]): r for r in _read_jsonl(args.rnnt)}
    v3 = {(r["id"], r["turn"]): r for r in _read_jsonl(args.v3)}
    rows_path, timing_path = Path(f"{out}.jsonl"), Path(f"{out}.timing.jsonl")
    rows_path.write_text("", encoding="utf-8")
    timing_path.write_text("", encoding="utf-8")
    summary: dict[str, Any] = {"ids": args.ids, "rnnt": str(args.rnnt), "v3": str(args.v3)}
    with GpuMemoryMonitor(Path(f"{out}.gpu_memory.csv")) as monitor:
        t0 = time.perf_counter()
        cfg = whisper_cfg("")
        whisper = FasterWhisperASR(cfg)
        locator = ASRTranscriber(cfg, data_root=data_root, split=args.split, backend=whisper)
        whisper.warmup()
        summary["load_s"] = time.perf_counter() - t0
        n_regions = 0
        first_rows: list[dict[str, Any]] = []
        for sample in samples:
            start = time.perf_counter()
            sample_rows: list[dict[str, Any]] = []
            for (key, _ref), path in zip(utterance_keys(sample), locator.wav_paths(sample), strict=True):
                r_row, v_row = rnnt.get((sample.id, key)), v3.get((sample.id, key))
                if r_row is None or v_row is None:
                    continue
                v_hyp = str(v_row["hyp"])
                if not v_hyp.strip() or " ".join(str(v_row.get("raw", v_hyp)).split()) != " ".join(v_hyp.split()):
                    continue
                pairs = candidate_pairs(str(r_row["hyp"]), v_hyp)
                if not pairs:
                    continue
                audio = read_wav(path, cfg.sample_rate)
                scored = whisper.score_texts(audio, [pairs[0].text_a] + [p.text_b for p in pairs])
                (score_a, n_a), rest = scored[0], scored[1:]
                for pair, (score_b, n_b) in zip(pairs, rest, strict=True):
                    sample_rows.append(
                        {
                            "id": sample.id,
                            "turn": key,
                            "part": pair.part_index,
                            "a": pair.text_a,
                            "b": pair.text_b,
                            "score_a": score_a,
                            "n_a": n_a,
                            "score_b": score_b,
                            "n_b": n_b,
                            "long": len(audio) > 30 * cfg.sample_rate,
                        }
                    )
            seconds = time.perf_counter() - start
            _write_jsonl(rows_path, sample_rows)
            _write_jsonl(timing_path, [{"id": sample.id, "regions": len(sample_rows), "score_s": seconds}])
            n_regions += len(sample_rows)
            if len(first_rows) < args.rerun:
                first_rows.extend(sample_rows)
        if args.rerun:
            # Kiểm tra tất định: chấm lại N vùng đầu, so điểm.
            diff = 0
            by_utt: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for row in first_rows[: args.rerun]:
                by_utt.setdefault((row["id"], row["turn"]), []).append(row)
            sample_by_id = {s.id: s for s in samples}
            for (sid, key), rows in by_utt.items():
                sample = sample_by_id[sid]
                path = dict(zip([k for k, _ in utterance_keys(sample)], locator.wav_paths(sample), strict=True))[key]
                again = whisper.score_texts(read_wav(path, cfg.sample_rate), [rows[0]["a"]] + [r["b"] for r in rows])
                diff += int(abs(again[0][0] - rows[0]["score_a"]) > 1e-6)
                diff += sum(abs(s - r["score_b"]) > 1e-6 for (s, _n), r in zip(again[1:], rows, strict=True))
            summary["determinism"] = {"regions": min(args.rerun, len(first_rows)), "diff": diff}
        summary["n_regions"] = n_regions
        summary["total_s"] = time.perf_counter() - t0
    summary["gpu_peak_mb"] = monitor.peak_mb
    Path(f"{out}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI với hai lệnh con `transcribe` và `score`."""
    ap = argparse.ArgumentParser(prog="lifemate.eval.asr_v3", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("transcribe", "score"):
        p = sub.add_parser(name)
        p.add_argument("--env", required=True)
        p.add_argument("--root", default=None)
        p.add_argument("--split", default="train")
        p.add_argument("--ids", required=True)
        p.add_argument("--rerun", type=int, default=0, help="kiểm tra tất định trên N mẫu (transcribe) hoặc N vùng")
        p.add_argument("--out", type=Path, required=True)
        if name == "transcribe":
            p.add_argument("--variants", nargs="+", choices=PROMPT_VARIANTS, default=["E0"])
            p.add_argument("--rnnt", action="store_true", help="chép cả bằng RNNT")
        else:
            p.add_argument("--rnnt", type=Path, required=True, help="hyps.jsonl của RNNT")
            p.add_argument("--v3", type=Path, required=True, help="hyps.jsonl của large-v3")
    args = ap.parse_args(argv)
    return transcribe_variants(args) if args.cmd == "transcribe" else score_candidates(args)


if __name__ == "__main__":
    raise SystemExit(main())
