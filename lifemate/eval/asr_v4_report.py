"""Tính các bảng số của báo cáo benchmark ASR lần 4 từ chữ đã lưu của các phiên GPU lần 3 (máy dev, không cần GPU).

Cách dùng (từ thư mục gốc repo):
  python -m lifemate.eval.asr_v4_report l1 --k1 <K1> --k2 <K2> --k3 <K3> --winner E2 --split train --out <thư mục>
  python -m lifemate.eval.asr_v4_report l1 --k1 <K1> --k2 <K2> --k3 <K3> --winner E2 --split val_seen --out <thư mục>

Nội dung chính:
- `h_full_train`: học lại bộ quyết định H trên cả 1 093 mẫu train như D54 (`asr_v3_report.fit_h`), trả về cách
  quyết định dùng được cho mọi tập.
- `merged_rows`: chữ ghép H của một tập (RNNT làm nền, large-v3 câu mồi thắng ở các đoạn H chọn).
- `stage_l1`: thử nghiệm L1: áp `asr_numbers.digitize` lên chữ RNNT và chữ ghép H, so bốn hệ thống bằng `compare`
  (bootstrap ghép cặp theo mẫu) với mốc là ghép H; thêm số câu có chữ số và số chữ số đổi sai.

Thư mục vào: như `asr_v3_report` (K1: tập dò; K2: `k2_<tập>/`; K3: `scores_<tập>.jsonl`). `--split train` gộp
`asr_tune` và `asr_learn` (tập phát triển); `val_seen` đo một lần; `val_unseen` khoá theo D53.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lifemate.backends.asr_numbers import digitize
from lifemate.eval import asr_v3_analysis as A  # noqa: N812
from lifemate.eval.asr_compare import compare
from lifemate.eval.asr_v3_report import _data, _fixed_prompt, _rows, fit_h
from lifemate.eval.eval_asr import _edit_distance, normalize_text

SPLIT_PARTS = {
    "train": ("asr_tune", "asr_learn"),
    "val_seen": ("val_seen",),
    "val_unseen": ("val_unseen",),
}
METRIC_KEYS = ("entity_miss_rate", "wer", "wer_nonum", "vi_only_wer", "en_word_recall")
_DIGIT = re.compile(r"\d+")


def _part_dirs(args: argparse.Namespace, part: str) -> tuple[Path, Path]:
    """(thư mục chữ K2, tiền tố file điểm K3) của một phần tập."""
    hdir = args.k1 if part == "asr_tune" else args.k2 / f"k2_{part}"
    return hdir, args.k3 / f"scores_{part}"


def load_parts(args: argparse.Namespace, parts: Sequence[str]) -> dict[str, Any]:
    """Mẫu, từ điển ngữ cảnh, chữ RNNT và các vùng cần quyết định của các phần tập gộp lại."""
    samples: list[Any] = []
    context: dict[str, frozenset[str]] = {}
    rnnt: list[dict[str, Any]] = []
    utts: list[A.UtteranceRegions] = []
    for part in parts:
        hdir, sprefix = _part_dirs(args, part)
        part_samples, catalog = _data(f"configs/splits/{part}.txt")
        part_rnnt = _rows(hdir / "rnnt.hyps.jsonl")
        v3 = _rows(hdir / f"v3_{args.winner}.hyps.jsonl")
        utts += A.utterance_regions(part_rnnt, v3, _rows(Path(f"{sprefix}.jsonl")))
        context |= A.context_words_by_sample(part_samples, catalog)
        samples += part_samples
        rnnt += part_rnnt
    return {"samples": samples, "context": context, "rnnt": rnnt, "utts": utts}


def h_full_train(args: argparse.Namespace) -> tuple[A.RegionDecider, dict[str, Any]]:
    """Bộ quyết định H học trên cả train (D54): mô hình của `fit_h`, bảng độ tin cậy đếm trên cả train.

    Trả về:
        (hàm quyết định dùng với từ điển ngữ cảnh của tập đích, thông tin đã học: l2, số đoạn).
    """
    _, prompt_words = _fixed_prompt()
    train = load_parts(args, SPLIT_PARTS["train"])
    if len(train["samples"]) != 1093:
        raise ValueError(f"cần 1 093 mẫu train, có {len(train['samples'])}")
    model, l2, _ = fit_h(train["utts"], train["context"], prompt_words)
    rel = A.reliability(A.count_prompt_words(train["utts"], prompt_words))
    info = {"l2": l2, "train_regions": sum(len(u.candidates) for u in train["utts"])}

    def make(context: Mapping[str, frozenset[str]]) -> A.RegionDecider:
        return A.h_decider(model, context, prompt_words, rel)

    return make, info  # type: ignore[return-value]


def digitized(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Các dòng `hyps.jsonl` sau khi đổi số đọc thành chữ số."""
    return [{**r, "hyp": digitize(str(r["hyp"])), "raw": digitize(str(r.get("raw", r["hyp"])))} for r in rows]


def digit_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Các số đếm về chữ số, so với đáp án **đã đổi số cùng luật** (lời chuẩn viết số lúc bằng chữ "tám giờ", lúc bằng
    chữ số "5 giờ", nên so với đáp án gốc không tách được lỗi đổi số khỏi khác biệt cách viết).

    Trả về:
        `ref_with_digits`: số câu đáp án gốc có chữ số; `hyp_with_digits`: trong đó số câu chữ ASR có chữ số;
        `ref_digits`: số chữ số của đáp án đã đổi; `ref_digits_found`: trong đó số có trong chữ ASR;
        `digits_not_in_ref`: chữ số của ASR không có trong đáp án đã đổi (đổi sai hoặc nghe sai); `wer_digitized_ref`:
        WER của chữ ASR so với đáp án đã đổi số.
    """
    out: dict[str, Any] = {
        "ref_with_digits": 0,
        "hyp_with_digits": 0,
        "digits_not_in_ref": 0,
        "ref_digits_found": 0,
        "ref_digits": 0,
    }
    errors = words = 0
    for r in rows:
        ref_norm = normalize_text(str(r["ref"]))
        ref_dig = digitize(ref_norm)
        hyp_norm = normalize_text(str(r["hyp"]))
        if _DIGIT.search(ref_norm):
            out["ref_with_digits"] += 1
            out["hyp_with_digits"] += bool(_DIGIT.search(hyp_norm))
        ref_digits = _DIGIT.findall(ref_dig)
        out["ref_digits"] += len(ref_digits)
        remaining = list(ref_digits)
        for d in _DIGIT.findall(hyp_norm):
            if d in remaining:
                remaining.remove(d)
                out["ref_digits_found"] += 1
            else:
                out["digits_not_in_ref"] += 1
        errors += _edit_distance(ref_dig.split(), hyp_norm.split())
        words += len(ref_dig.split())
    out["wer_digitized_ref"] = errors / words if words else 0.0
    return out


def stage_l1(args: argparse.Namespace) -> dict[str, Any]:
    """Thử nghiệm L1 trên `args.split`: RNNT, RNNT + L1, ghép H (mốc), ghép H + L1."""
    prompt_text, _ = _fixed_prompt()
    make_h, h_info = h_full_train(args)
    data = load_parts(args, SPLIT_PARTS[args.split])
    h_rows = A.apply_decider(data["rnnt"], data["utts"], make_h(data["context"]))
    systems = {
        "H": h_rows,
        "H+L1": digitized(h_rows),
        "RNNT": data["rnnt"],
        "RNNT+L1": digitized(data["rnnt"]),
    }
    cmp = compare(data["samples"], {n: (rows, prompt_text) for n, rows in systems.items()})
    table = {
        n: {
            **{k: {"value": cmp[n]["report"][k], "diff_ci": cmp[n]["diff_ci"][k]} for k in METRIC_KEYS},
            "entity_missed": cmp[n]["report"]["entity_missed"],
            "entity_total": cmp[n]["report"]["entity_total"],
            **digit_stats(rows),
            "changed_utterances": sum(a["hyp"] != b["hyp"] for a, b in zip(systems["H"], rows, strict=True))
            if n == "H+L1"
            else sum(a["hyp"] != b["hyp"] for a, b in zip(systems["RNNT"], rows, strict=True))
            if n == "RNNT+L1"
            else 0,
        }
        for n, rows in systems.items()
    }
    return {"split": args.split, "n_samples": len(data["samples"]), "h": h_info, "table": table}


def l1_markdown(res: Mapping[str, Any]) -> str:
    """Bảng Markdown ngắn của `stage_l1`."""

    def ci(pair: Sequence[float], scale: float) -> str:
        return f"[{pair[0] * scale:+.2f}; {pair[1] * scale:+.2f}]"

    lines = [
        f"{res['split']} ({res['n_samples']} mẫu); H: {res['h']}",
        "| Hệ thống | Giá trị tham số thiếu (chênh CI) | WER (chênh CI) | WER bỏ số (chênh CI)"
        " | WER thuần Việt (chênh CI) | Recall EN (chênh CI) | WER so đáp án đã đổi số | Câu có chữ số: đáp án / ASR"
        " | Chữ số đúng / tổng đáp án đã đổi | Chữ số lạ | Câu đổi |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n, t in res["table"].items():
        lines.append(
            f"| {n} | {t['entity_miss_rate']['value'] * 100:.1f}% ({t['entity_missed']}/{t['entity_total']})"
            f" {ci(t['entity_miss_rate']['diff_ci'], 100)}"
            f" | {t['wer']['value'] * 100:.2f}% {ci(t['wer']['diff_ci'], 100)}"
            f" | {t['wer_nonum']['value'] * 100:.2f}% {ci(t['wer_nonum']['diff_ci'], 100)}"
            f" | {t['vi_only_wer']['value'] * 100:.2f}% {ci(t['vi_only_wer']['diff_ci'], 100)}"
            f" | {t['en_word_recall']['value'] * 100:.1f}% {ci(t['en_word_recall']['diff_ci'], 100)}"
            f" | {t['wer_digitized_ref'] * 100:.2f}%"
            f" | {t['ref_with_digits']} / {t['hyp_with_digits']} | {t['ref_digits_found']} / {t['ref_digits']}"
            f" | {t['digits_not_in_ref']} | {t['changed_utterances']} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: lệnh `l1`, ghi `l1_<tập>.json` vào `--out` và in bảng Markdown."""
    ap = argparse.ArgumentParser(prog="lifemate.eval.asr_v4_report", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("l1")
    for k in ("--k1", "--k2", "--k3"):
        p1.add_argument(k, type=Path, required=True)
    p1.add_argument("--winner", default="E2")
    p1.add_argument("--split", choices=sorted(SPLIT_PARTS), default="train")
    p1.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.split == "val_unseen":
        raise SystemExit("val_unseen bị khoá (D53): chỉ đo khi chốt cấu hình nộp bài")
    res = stage_l1(args)
    (args.out / f"l1_{args.split}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(l1_markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
