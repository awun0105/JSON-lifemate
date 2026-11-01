"""So sánh nhiều hệ thống ASR trên cùng một tập mẫu, từ các file `hyps.jsonl` đã có, kèm khoảng tin cậy bootstrap.

Cách dùng (chạy từ thư mục gốc repo):
  python -m lifemate.eval.asr_compare --env local --ids configs/splits/val_seen_en_mixed.txt \\
      --system rnnt=<run>/chunkformer-rnnt__report_val_seen.hyps.jsonl \\
      --system v3=<run>/large-v3__report_val_seen.hyps.jsonl:<run>/large-v3__report_val_seen.toml --out cmp.json

Nội dung chính:
- `METRICS`: các chỉ số đưa vào bảng.
- `compare`: chấm mọi hệ thống, rồi bootstrap theo mẫu để ra khoảng tin cậy 95% của từng chỉ số và của chênh lệch so với
  hệ thống đầu tiên.
- `main`: CLI, in bảng Markdown và ghi JSON.

Cách hoạt động:
- Mỗi hệ thống là `tên=đường_dẫn_hyps[:đường_dẫn_config]`; config (tuỳ chọn) để lấy câu mồi cho các chỉ số câu mồi.
- `--prepare TÊN` (lặp được): áp `asr_merge_v2.prepare` (chữ thường, bỏ dấu câu hai đầu từ, đổi số đọc thành chữ số)
  lên chữ của hệ thống đó, để mô hình đứng riêng được so cùng cách viết số với pipeline ghép bản 2 (vốn đã `prepare`).
- `--timing TÊN=FILE:TRƯỜNG` (lặp được): thời gian mỗi mẫu của hệ thống, đọc từ file JSON Lines có `id` (trường lồng
  viết bằng dấu chấm, ví dụ `whisper_s.E2`). Nhiều mục cùng tên được cộng theo mẫu, ví dụ bản 2 = RNNT + Whisper + chấm
  điểm âm thanh. Kết quả ghi p50, p95, lớn nhất (`sample_seconds`).
- Bootstrap có ghép cặp: mỗi vòng lấy ngẫu nhiên có hoàn lại `len(ids)` mẫu (seed cố định), chấm mọi hệ thống trên
  cùng các mẫu đó. Khoảng tin cậy là phân vị 2,5 và 97,5 qua các vòng. Chênh lệch có khoảng tin cậy không chứa 0 mới
  coi là khác biệt thật.
- Lấy mẫu theo mẫu (sample) chứ không theo câu, vì các câu của cùng một mẫu (cùng người nói, cùng hội thoại) không độc
  lập với nhau.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lifemate.core.paths import REPO_ROOT, repo_path
from lifemate.core.types import Sample
from lifemate.eval.eval_asr import evaluate, f1
from lifemate.eval.rescore_asr import ReplayTranscriber

# Chỉ số -> (tử số, mẫu số) trong `AsrReport.counts`.
METRICS: dict[str, tuple[str, str]] = {
    "wer": ("wer_errors", "wer_words"),
    "wer_nonum": ("wer_nonum_errors", "wer_nonum_words"),
    "en_mixed_wer": ("en_mixed_errors", "en_mixed_words"),
    "vi_only_wer": ("vi_only_errors", "vi_only_words"),
    "en_word_recall": ("en_hits", "en_words"),
    "en_word_recall_lenient": ("en_hits_lenient", "en_words"),
    "en_word_precision": ("en_hyp_hits", "en_hyp_words"),
    "en_word_f1": ("", ""),  # tính từ `en_word_recall` và `en_word_precision`, không phải một tỉ lệ
    "entity_miss_rate": ("entity_missed", "entity_total"),
    "wer_digits": ("wer_digits_errors", "wer_digits_words"),
    "vi_only_wer_digits": ("vi_only_digits_errors", "vi_only_digits_words"),
    "final_wer_digits": ("final_digits_errors", "final_digits_words"),
    "number_recall": ("num_hit", "num_ref"),
    "number_precision": ("num_hit", "num_hyp"),
    "number_f1": ("", ""),
    "entity_miss_rate_all": ("entity_all_missed", "entity_all_total"),
}
# F1 -> (recall, precision): trung bình điều hoà, không phải một tỉ lệ tử số / mẫu số.
_F1 = {"en_word_f1": ("en_word_recall", "en_word_precision"), "number_f1": ("number_recall", "number_precision")}


def _metric_value(counts: Mapping[str, int], metric: str) -> float:
    """Tính một chỉ số từ tử số và mẫu số đã cộng dồn; mẫu số 0 cho 0.0."""
    if metric in _F1:
        recall, precision = _F1[metric]
        return f1(_metric_value(counts, recall), _metric_value(counts, precision))
    numerator, denominator = METRICS[metric]
    return counts[numerator] / counts[denominator] if counts[denominator] else 0.0


def _percentile_interval(values: list[float]) -> list[float]:
    """Khoảng tin cậy 95% theo phân vị (2,5 và 97,5) của các giá trị bootstrap."""
    ordered = sorted(values)
    return [ordered[int(0.025 * (len(ordered) - 1))], ordered[int(0.975 * (len(ordered) - 1))]]


def compare(
    samples: Sequence[Sample],
    systems: Mapping[str, tuple[Sequence[Mapping[str, Any]], str]],
    rounds: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Chấm mọi hệ thống và bootstrap có ghép cặp.

    Mỗi mẫu được chấm riêng một lần để lấy tử số và mẫu số (`AsrReport.counts`); mỗi vòng bootstrap chỉ cộng các số
    đó, nên kết quả giống hệt chấm lại cả tập mà nhanh hơn nhiều.

    Tham số:
        samples: các mẫu của tập so sánh.
        systems: tên -> (các dòng `hyps.jsonl`, câu mồi); hệ thống đầu tiên là mốc để tính chênh lệch.
        rounds: số vòng bootstrap.
        seed: seed của bộ sinh số ngẫu nhiên.

    Trả về:
        Dict `{tên: {"report": báo cáo đầy đủ, "ci": {chỉ số: [thấp, cao]}, "diff_ci": {chỉ số: [thấp, cao]}}}`;
        `diff_ci` là khoảng tin cậy của (hệ thống này - hệ thống mốc).
    """
    names = list(systems)
    full_reports = {
        name: evaluate(samples, ReplayTranscriber(rows), prompt=prompt).to_dict()
        for name, (rows, prompt) in systems.items()
    }
    # Hệ thống -> mỗi mẫu một dict tử số / mẫu số.
    per_sample_counts = {
        name: [evaluate([sample], ReplayTranscriber(rows), prompt=prompt).counts for sample in samples]
        for name, (rows, prompt) in systems.items()
    }
    count_keys = sorted({key for pair in METRICS.values() for key in pair if key})
    rng = random.Random(seed)
    # Hệ thống -> chỉ số -> giá trị ở từng vòng bootstrap.
    bootstrap_values: dict[str, dict[str, list[float]]] = {name: {m: [] for m in METRICS} for name in names}
    for _ in range(rounds):
        resampled = [rng.randrange(len(samples)) for _ in samples]
        for name in names:
            totals = {key: sum(per_sample_counts[name][i][key] for i in resampled) for key in count_keys}
            for metric in METRICS:
                bootstrap_values[name][metric].append(_metric_value(totals, metric))
    baseline = bootstrap_values[names[0]]
    result: dict[str, Any] = {}
    for name in names:
        values = bootstrap_values[name]
        result[name] = {
            "report": full_reports[name],
            "ci": {m: _percentile_interval(values[m]) for m in METRICS},
            "diff_ci": {
                m: _percentile_interval([a - b for a, b in zip(values[m], baseline[m], strict=True)]) for m in METRICS
            },
        }
    return result


def to_markdown(result: Mapping[str, Any]) -> str:
    """Bảng Markdown: mỗi hệ thống một dòng, mỗi chỉ số dạng `giá trị [thấp; cao]` (phần trăm)."""
    head = "| Hệ thống | " + " | ".join(METRICS) + " |"
    lines = [head, "|" + "---|" * (len(METRICS) + 1)]
    for name, r in result.items():
        cells = [f"{100 * r['report'][m]:.1f} [{100 * r['ci'][m][0]:.1f}; {100 * r['ci'][m][1]:.1f}]" for m in METRICS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def sample_seconds(specs: Sequence[str]) -> dict[str, dict[str, float]]:
    """Thời gian mỗi mẫu của từng hệ thống từ các mục `TÊN=FILE:TRƯỜNG`, cộng theo mẫu.

    Trả về `{tên: {"p50", "p95", "max", "n"}}`.

    Chỉ giữ mẫu có mặt ở mọi file của cùng hệ thống, để tổng không thiếu phần nào.
    """
    from lifemate.eval.eval_asr import percentile

    per: dict[str, list[dict[str, float]]] = {}
    for spec in specs:
        name, _, rest = spec.partition("=")
        path, _, field = rest.rpartition(":")
        times: dict[str, float] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            value: Any = row
            for key in field.split("."):
                value = value[key]
            times[str(row["id"])] = float(value)
        per.setdefault(name, []).append(times)
    out: dict[str, dict[str, float]] = {}
    for name, parts in per.items():
        ids = set.intersection(*(set(p) for p in parts))
        totals = [sum(p[i] for p in parts) for i in sorted(ids)]
        out[name] = {"n": len(totals), "p50": percentile(totals, 50), "p95": percentile(totals, 95),
                     "max": max(totals, default=0.0)}  # fmt: skip
    return out


def prepared(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Các dòng `hyps.jsonl` sau `prepare` (cả `hyp` lẫn `raw`), như đầu vào của pipeline ghép bản 2."""
    from lifemate.eval.asr_merge_v2 import prepare

    return [{**r, "hyp": prepare(str(r["hyp"])), "raw": prepare(str(r.get("raw", r["hyp"])))} for r in rows]


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy CLI so sánh và in bảng; trả về mã thoát 0."""
    from lifemate.core.config import load_config, load_profile
    from lifemate.io.data import load_split

    ap = argparse.ArgumentParser(prog="lifemate.eval.asr_compare", description=__doc__)
    ap.add_argument("--env", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--system", action="append", required=True, metavar="NAME=HYPS[:CONFIG]")
    ap.add_argument("--prepare", action="append", default=[], metavar="NAME", help="áp `prepare` lên hệ thống này")
    ap.add_argument("--timing", action="append", default=[], metavar="NAME=FILE:FIELD", help="thời gian mỗi mẫu")
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if "val_unseen" in str(args.ids):
        raise SystemExit("val_unseen bị khoá (D53)")
    profile = load_profile(args.env, REPO_ROOT).with_overrides(root=args.root)
    root = repo_path(profile.data_root, REPO_ROOT)
    ids = [x.strip() for x in args.ids.read_text(encoding="utf-8").splitlines() if x.strip()]
    samples = load_split(root, args.split, ids)
    systems: dict[str, tuple[list[dict[str, Any]], str]] = {}
    for item in args.system:
        name, _, rest = item.partition("=")
        hyps, _, cfg = rest.partition(":")
        rows = [json.loads(x) for x in Path(hyps).read_text(encoding="utf-8").splitlines() if x.strip()]
        if name in args.prepare:
            rows = prepared(rows)
        prompt = str(load_config(Path(cfg)).asr.generate_kwargs.get("initial_prompt") or "") if cfg else ""
        systems[name] = (rows, prompt)
    result = compare(samples, systems, rounds=args.rounds)
    print(to_markdown(result))
    if args.timing:
        seconds = sample_seconds(args.timing)
        for name, t in seconds.items():
            line = f"p50 {t['p50']:.2f} s, p95 {t['p95']:.2f} s, lớn nhất {t['max']:.2f} s (n={t['n']})"
            print(f"{name}: thời gian mỗi mẫu {line}")
        result = {**result, "sample_seconds": seconds}
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
