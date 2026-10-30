"""Tính các bảng số của báo cáo benchmark ASR lần 4 từ chữ đã lưu của các phiên GPU lần 3 (máy dev, không cần GPU).

Cách dùng (từ thư mục gốc repo):
  python -m lifemate.eval.asr_v4_report l1 --k1 <K1> --k2 <K2> --k3 <K3> --winner E2 --split train --out <thư mục>
  python -m lifemate.eval.asr_v4_report l1 --k1 <K1> --k2 <K2> --k3 <K3> --winner E2 --split val_seen --out <thư mục>
  python -m lifemate.eval.asr_v4_report export-h --k1 <K1> --k2 <K2> --k3 <K3> --out configs/asr_merge/h_v1.json

Nội dung chính:
- `h_full_train`: học lại bộ quyết định H trên cả 1 093 mẫu train như D54 (`asr_v3_report.fit_h`), trả về cách
  quyết định dùng được cho mọi tập.
- `merged_rows`: chữ ghép H của một tập (RNNT làm nền, large-v3 câu mồi thắng ở các đoạn H chọn).
- `stage_l1`: thử nghiệm L1: áp `asr_numbers.digitize` lên chữ RNNT và chữ ghép H, so bốn hệ thống bằng `compare`
  (bootstrap ghép cặp theo mẫu) với mốc là ghép H; thêm số câu có chữ số và số chữ số đổi sai.
- `stage_export_h`: ghi bộ quyết định H (mô hình đã học trên cả train, bảng độ tin cậy, câu mồi, lớp bảo vệ) thành
  file JSON có phiên bản trong `configs/asr_merge/`, để engine `MergedASR` của hệ thống thật nạp lên.
- `stage_l2`: thử nghiệm L2: áp `asr_entities.correct` (danh sách ứng viên của từng mẫu tại từng lượt) lên chữ ghép
  H + L1, so với mốc H + L1; đếm số lần thay và số lần thay đúng (tên thay có trong đáp án), và tỉ lệ từ tiếng Anh bị
  mất mà có trong danh sách ứng viên (trần của L2).

Thư mục vào: như `asr_v3_report` (K1: tập dò; K2: `k2_<tập>/`; K3: `scores_<tập>.jsonl`). `--split train` gộp
`asr_tune` và `asr_learn` (tập phát triển); `val_seen` đo một lần; `val_unseen` khoá theo D53.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from lifemate.backends.asr_entities import correct, sample_candidates
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
# Hàm nhận từ điển ngữ cảnh của một tập và trả về cách quyết định H cho tập đó.
DeciderMaker = Callable[[Mapping[str, frozenset[str]]], A.RegionDecider]
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


def fit_h_full_train(args: argparse.Namespace) -> tuple[A.LogisticModel, dict[str, float], dict[str, Any]]:
    """Mô hình H và bảng độ tin cậy học trên cả 1 093 mẫu train (D54), kèm thông tin đã học (l2, số đoạn)."""
    _, prompt_words = _fixed_prompt()
    train = load_parts(args, SPLIT_PARTS["train"])
    if len(train["samples"]) != 1093:
        raise ValueError(f"cần 1 093 mẫu train, có {len(train['samples'])}")
    model, l2, _ = fit_h(train["utts"], train["context"], prompt_words)
    rel = A.reliability(A.count_prompt_words(train["utts"], prompt_words))
    info = {"l2": l2, "train_regions": sum(len(u.candidates) for u in train["utts"])}
    return model, rel, info


def h_full_train(args: argparse.Namespace) -> tuple[DeciderMaker, dict[str, Any]]:
    """Bộ quyết định H học trên cả train (D54).

    Trả về:
        (hàm quyết định dùng với từ điển ngữ cảnh của tập đích, thông tin đã học: l2, số đoạn).
    """
    _, prompt_words = _fixed_prompt()
    model, rel, info = fit_h_full_train(args)

    def make(context: Mapping[str, frozenset[str]]) -> A.RegionDecider:
        return A.h_decider(model, context, prompt_words, rel)

    return make, info


def stage_export_h(args: argparse.Namespace) -> dict[str, Any]:
    """File trọng số của `MergedASR` (`lifemate/backends/asr_merged.py`): mô hình H, bảng độ tin cậy, câu mồi, lớp
    bảo vệ, thứ tự đặc trưng. Mọi số đều học trên train; không có số nào từ val."""
    from lifemate.eval.asr_v3 import FIXED_PROMPT_FILE, GUARD_MAX_COMPRESSION, GUARD_NGRAM

    model, rel, info = fit_h_full_train(args)
    return {
        "version": args.out.stem,
        "description": "Bộ quyết định ghép H (asr_v3.md mục 7.2, D54) học trên cả 1 093 mẫu train; đổi số L1 (D56)",
        "prompt_file": FIXED_PROMPT_FILE,
        "prompt_variant": "E2",
        "guard": {"ngram": GUARD_NGRAM, "max_compression": GUARD_MAX_COMPRESSION},
        "features": list(A.FEATURES),
        "threshold": 0.5,
        "model": dataclasses.asdict(model),
        "reliability": dict(sorted(rel.items())),
        "digitize": True,
        "trained_on": info,
    }


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


def entity_corrected(
    rows: Sequence[Mapping[str, Any]], samples: Sequence[Any], catalog: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Chữ sau L2 và các số đếm: số lần thay, số lần thay đúng (dạng chuẩn hoá của tên thay có trong đáp án)."""
    from lifemate.eval.foreign_words import fold_text

    by_id = {s.id: s for s in samples}
    out: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"replacements": 0, "correct": 0, "examples": []}
    for r in rows:
        cands = sample_candidates(by_id[str(r["id"])], catalog, str(r["turn"]))
        new, reps = correct(str(r["hyp"]), cands)
        out.append({**r, "hyp": new, "raw": new})
        for x in reps:
            ok = fold_text(x.target) in fold_text(str(r["ref"]))
            stats["replacements"] += 1
            stats["correct"] += ok
            if len(stats["examples"]) < 40:
                stats["examples"].append({"id": r["id"], "turn": r["turn"], "ok": ok, **dataclasses.asdict(x)})
    return out, stats


def stage_l2(args: argparse.Namespace) -> dict[str, Any]:
    """Thử nghiệm L2 trên `args.split`: ghép H + L1 (mốc) và ghép H + L1 + L2; trần: từ tiếng Anh của đáp án không có
    ở cả RNNT lẫn large-v3 mà có trong danh sách ứng viên của mẫu."""
    from lifemate.eval.foreign_words import find_foreign_words, fold_text, match_words_in_transcript

    prompt_text, _ = _fixed_prompt()
    make_h, h_info = h_full_train(args)
    data = load_parts(args, SPLIT_PARTS[args.split])
    parts = SPLIT_PARTS[args.split]
    catalog: dict[str, Any] = {}
    for part in parts:
        catalog |= _data(f"configs/splits/{part}.txt")[1]
    base = digitized(A.apply_decider(data["rnnt"], data["utts"], make_h(data["context"])))
    fixed, stats = entity_corrected(base, data["samples"], catalog)
    cmp = compare(data["samples"], {"H+L1": (base, prompt_text), "H+L1+L2": (fixed, prompt_text)})
    table = {
        n: {k: {"value": cmp[n]["report"][k], "diff_ci": cmp[n]["diff_ci"][k]} for k in METRIC_KEYS}
        for n in ("H+L1", "H+L1+L2")
    }
    # Trần: từ tiếng Anh của đáp án không có trong chữ mốc, có / không có trong danh sách ứng viên.
    by_id = {s.id: s for s in data["samples"]}
    v3 = {}
    for part in parts:
        hdir, _ = _part_dirs(args, part)
        v3 |= {(str(r["id"]), str(r["turn"])): str(r["hyp"]) for r in _rows(hdir / f"v3_{args.winner}.hyps.jsonl")}
    missing = in_pool = 0
    for r in base:
        words = find_foreign_words(str(r["ref"]))
        if not words:
            continue
        in_a = match_words_in_transcript(words, str(r["hyp"]))
        in_b = match_words_in_transcript(words, v3.get((str(r["id"]), str(r["turn"])), ""))
        cands = None
        for w, x, y in zip(words, in_a, in_b, strict=True):
            if x or y:
                continue
            missing += 1
            if cands is None:
                cands = sample_candidates(by_id[str(r["id"])], catalog, str(r["turn"]))
            in_pool += any(fold_text(w.text) in c.folded for c in cands)
    return {
        "split": args.split,
        "n_samples": len(data["samples"]),
        "h": h_info,
        "table": table,
        "replacements": stats,
        "ceiling": {"missing_english_words": missing, "in_candidate_pool": in_pool},
    }


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
    """CLI: lệnh `l1` hoặc `l2`, ghi `l<n>_<tập>.json` vào `--out` và in tóm tắt."""
    ap = argparse.ArgumentParser(prog="lifemate.eval.asr_v4_report", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("l1")
    for k in ("--k1", "--k2", "--k3"):
        p1.add_argument(k, type=Path, required=True)
    p1.add_argument("--winner", default="E2")
    p1.add_argument("--split", choices=sorted(SPLIT_PARTS), default="train")
    p1.add_argument("--out", type=Path, required=True)
    p2 = sub.add_parser("l2")
    for k in ("--k1", "--k2", "--k3"):
        p2.add_argument(k, type=Path, required=True)
    p2.add_argument("--winner", default="E2")
    p2.add_argument("--split", choices=sorted(SPLIT_PARTS), default="train")
    p2.add_argument("--out", type=Path, required=True)
    pe = sub.add_parser("export-h")
    for k in ("--k1", "--k2", "--k3"):
        pe.add_argument(k, type=Path, required=True)
    pe.add_argument("--winner", default="E2")
    pe.add_argument("--out", type=Path, required=True, help="file JSON, ví dụ configs/asr_merge/h_v1.json")
    args = ap.parse_args(argv)
    if args.cmd == "export-h":
        weights = stage_export_h(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(weights, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({k: weights[k] for k in ("version", "trained_on", "model")}, ensure_ascii=False))
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    if args.split == "val_unseen":
        raise SystemExit("val_unseen bị khoá (D53): chỉ đo khi chốt cấu hình nộp bài")
    if args.cmd == "l2":
        res2 = stage_l2(args)
        path2 = args.out / f"l2_{args.split}.json"
        path2.write_text(json.dumps(res2, ensure_ascii=False, indent=1), encoding="utf-8")
        brief = {k: v for k, v in res2.items() if k != "replacements"} | {
            "replacements": {k: v for k, v in res2["replacements"].items() if k != "examples"}
        }
        print(json.dumps(brief, ensure_ascii=False, indent=1))
        return 0
    res = stage_l1(args)
    (args.out / f"l1_{args.split}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(l1_markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
