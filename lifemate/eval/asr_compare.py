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
from lifemate.eval.eval_asr import AsrReport, evaluate
from lifemate.eval.rescore_asr import ReplayTranscriber

METRICS = (
    "wer",
    "wer_nonum",
    "en_mixed_wer",
    "vi_only_wer",
    "en_word_recall",
    "en_word_recall_lenient",
    "entity_miss_rate",
)


def _score(samples: Sequence[Sample], rows: Sequence[Mapping[str, Any]], prompt: str) -> AsrReport:
    """Chấm một hệ thống trên `samples` từ các dòng `hyps.jsonl` đã có."""
    return evaluate(samples, ReplayTranscriber(rows), prompt=prompt)


def _ci(values: list[float]) -> list[float]:
    """Khoảng tin cậy 95% theo phân vị (2,5 và 97,5) của các giá trị bootstrap."""
    xs = sorted(values)
    return [xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))]]


def compare(
    samples: Sequence[Sample],
    systems: Mapping[str, tuple[Sequence[Mapping[str, Any]], str]],
    rounds: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Chấm mọi hệ thống và bootstrap có ghép cặp.

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
    full = {n: _score(samples, *systems[n]).to_dict() for n in names}
    rng = random.Random(seed)
    boot: dict[str, dict[str, list[float]]] = {n: {m: [] for m in METRICS} for n in names}
    for _ in range(rounds):
        pick = [samples[rng.randrange(len(samples))] for _ in samples]
        for n in names:
            rep = _score(pick, *systems[n])
            for m in METRICS:
                boot[n][m].append(float(getattr(rep, m)))
    out: dict[str, Any] = {}
    for n in names:
        base = boot[names[0]]
        out[n] = {
            "report": full[n],
            "ci": {m: _ci(boot[n][m]) for m in METRICS},
            "diff_ci": {m: _ci([a - b for a, b in zip(boot[n][m], base[m], strict=True)]) for m in METRICS},
        }
    return out


def to_markdown(result: Mapping[str, Any]) -> str:
    """Bảng Markdown: mỗi hệ thống một dòng, mỗi chỉ số dạng `giá trị [thấp; cao]` (phần trăm)."""
    head = "| Hệ thống | " + " | ".join(METRICS) + " |"
    lines = [head, "|" + "---|" * (len(METRICS) + 1)]
    for name, r in result.items():
        cells = [f"{100 * r['report'][m]:.1f} [{100 * r['ci'][m][0]:.1f}; {100 * r['ci'][m][1]:.1f}]" for m in METRICS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


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
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    profile = load_profile(args.env, REPO_ROOT).with_overrides(root=args.root)
    root = repo_path(profile.data_root, REPO_ROOT)
    ids = [x.strip() for x in args.ids.read_text(encoding="utf-8").splitlines() if x.strip()]
    samples = load_split(root, args.split, ids)
    systems: dict[str, tuple[list[dict[str, Any]], str]] = {}
    for item in args.system:
        name, _, rest = item.partition("=")
        hyps, _, cfg = rest.partition(":")
        rows = [json.loads(x) for x in Path(hyps).read_text(encoding="utf-8").splitlines() if x.strip()]
        prompt = str(load_config(Path(cfg)).asr.generate_kwargs.get("initial_prompt") or "") if cfg else ""
        systems[name] = (rows, prompt)
    result = compare(samples, systems, rounds=args.rounds)
    print(to_markdown(result))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
