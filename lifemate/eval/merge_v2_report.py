"""Học và chọn bộ ghép bản 2 (H-Anh học lại trên vùng mới, H-Việt) từ chữ đã lưu và điểm âm thanh của phiên K5.

Cách dùng (từ thư mục gốc repo, sau khi có điểm của phiên K5):
  python -m lifemate.eval.merge_v2_report cv --k1 <K1> --k2 <K2> --k5 <K5> --out <thư mục>
  python -m lifemate.eval.merge_v2_report export --k1 <K1> --k2 <K2> --k5 <K5> --out configs/asr_merge/h_v2.json
  python -m lifemate.eval.merge_v2_report measure --k1 <K1> --k2 <K2> --k3 <K3> --k5 <K5> --split val_seen \
      --out <thư mục>

Nội dung chính:
- `build`: các câu của một tập ở dạng bản 2 (`asr_merge_v2`): vùng, loại vùng, nhãn tự động so với lời chuẩn đã đổi số,
  điểm âm thanh của K5.
- `fit_vi`: H-Việt (hồi quy logistic trên `VI_FEATURES`), hệ số phạt và ngưỡng chọn bằng kiểm định chéo bên trong.
- `stage_cv`: kiểm định chéo 5 phần theo mẫu trên cả train (1 093 mẫu), so ba hệ thống: giữ RNNT ở mọi vùng, chỉ H-Anh
  (vùng chữ Việt giữ RNNT), H-Anh cộng H-Việt; áp tiêu chí đặt trước (`asr_v4.md` mục 6c, D67).
- `stage_export`: học trên cả train, ghi `configs/asr_merge/h_v2.json` cho `MergedASR` bản 2.
- `stage_measure`: đo một lần trên một tập kiểm tra: bản 1 đang dùng (`h_v1` + đổi số), bản 2 chỉ H-Anh, bản 2 đủ.

Thư mục vào: K1 (`rnnt.hyps.jsonl`, `v3_E2.hyps.jsonl` của `asr_tune`), K2 (`k2_<tập>/`), K5 (`scores_<tập>.jsonl` do
`asr_v3 score --v2` ghi, mỗi vùng một dòng có `kind`, `score_a`, `score_b`, `n_a`, `n_b`, `long`). `val_unseen` khoá
theo D53.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lifemate.eval import asr_v3_analysis as A  # noqa: N812
from lifemate.eval.asr_compare import compare
from lifemate.eval.asr_merge import Region, assemble
from lifemate.eval.asr_merge_v2 import VI_FEATURES, prepare, region_indices, split_regions_v2, vi_features
from lifemate.eval.asr_v3_report import _data, _fixed_prompt, _rows, fit_h
from lifemate.eval.eval_asr import _edit_distance

SPLIT_PARTS = {"train": ("asr_tune", "asr_learn"), "val_seen": ("val_seen",), "val_unseen": ("val_unseen",)}
THETAS_VI = (0.4, 0.5, 0.6, 0.7)
CV_METRICS = ("vi_only_wer", "wer", "wer_nonum", "en_word_recall", "entity_miss_rate")


@dataclass
class Utt2:
    """Một câu ở dạng bản 2.

    Thuộc tính:
        en: `UtteranceRegions` của các vùng `en` (giao diện của H-Anh: `candidates`, `labels`, `acoustic`, `v3_scores`).
        vi: chỉ số các vùng `vi`; `vi_labels`: nhãn của chúng (1 lấy large-v3, 0 giữ RNNT, `None` hoà).
        acoustic_full: chỉ số vùng -> (điểm A, điểm B, số token A, số token B, câu dài hơn 30 giây).
    """

    row: dict[str, Any]
    en: A.UtteranceRegions
    vi: list[int] = field(default_factory=list)
    vi_labels: list[int | None] = field(default_factory=list)
    acoustic_full: dict[int, tuple[float, float, int, int, bool]] = field(default_factory=dict)


def _errors(ref: str, hyp: str) -> int:
    return _edit_distance(ref.split(), hyp.split())


def build(
    rnnt_rows: Sequence[Mapping[str, Any]], v3_rows: Sequence[Mapping[str, Any]], scores: Sequence[Mapping[str, Any]]
) -> list[Utt2]:
    """Mọi câu của tập ở dạng bản 2. Câu large-v3 rỗng hoặc bị bộ lọc câu bịa cắt thì không có vùng (giữ RNNT)."""
    v3_by = {(str(r["id"]), str(r["turn"])): r for r in v3_rows}
    sc = {(str(r["id"]), str(r["turn"]), int(r["part"])): r for r in scores}
    out: list[Utt2] = []
    for row in rnnt_rows:
        key = (str(row["id"]), str(row["turn"]))
        other = v3_by.get(key)
        a_text = prepare(str(row["hyp"]))
        ref = prepare(str(row.get("ref") or ""))
        b_raw = str(other["hyp"]) if other else ""
        filtered = other is None or not b_raw.strip() or " ".join(str(other.get("raw", b_raw)).split()) != " ".join(
            b_raw.split()
        )
        parts = [] if filtered else split_regions_v2(a_text, prepare(b_raw))
        if filtered or not any(isinstance(p, Region) for p in parts):
            parts = a_text.split()
        idx = region_indices(parts)
        e0 = _errors(ref, assemble(parts, lambda _k, _r: False))

        def label(k: int, parts: list[Any] = parts, e0: int = e0, ref: str = ref) -> int | None:
            ek = _errors(ref, assemble(parts, lambda i, _r: i == k))
            return None if ek == e0 else int(ek < e0)

        acoustic_full: dict[int, tuple[float, float, int, int, bool]] = {}
        for k in idx["en"] + idx["vi"]:
            s = sc.get((key[0], key[1], k))
            if s is not None:
                acoustic_full[k] = (
                    float(s["score_a"]), float(s["score_b"]), int(s["n_a"]), int(s["n_b"]), bool(s.get("long", False))
                )  # fmt: skip
        en = A.UtteranceRegions(
            key[0], key[1], str(row.get("ref") or ""), a_text, prepare(b_raw) if not filtered else "", parts,
            idx["en"], [label(k) for k in idx["en"]],
            {k: float(v) for k, v in ((other or {}).get("scores") or {}).items()},
            {k: (v[0], v[1], v[4]) for k, v in acoustic_full.items() if k in idx["en"]},
        )  # fmt: skip
        out.append(Utt2({**row, "ref": row.get("ref")}, en, idx["vi"], [label(k) for k in idx["vi"]], acoustic_full))
    return out


def load(args: argparse.Namespace, split: str) -> dict[str, Any]:
    """Mẫu, ngữ cảnh và các câu bản 2 của `split` (gộp các phần của `SPLIT_PARTS`)."""
    samples: list[Any] = []
    context: dict[str, frozenset[str]] = {}
    utts: list[Utt2] = []
    for part in SPLIT_PARTS[split]:
        hdir = args.k1 if part == "asr_tune" else args.k2 / f"k2_{part}"
        part_samples, catalog = _data(f"configs/splits/{part}.txt")
        utts += build(
            _rows(hdir / "rnnt.hyps.jsonl"), _rows(hdir / "v3_E2.hyps.jsonl"), _rows(args.k5 / f"scores_{part}.jsonl")
        )
        context |= A.context_words_by_sample(part_samples, catalog)
        samples += part_samples
    return {"samples": samples, "context": context, "utts": utts}


def _vi_xy(
    utts: Sequence[Utt2], context: Mapping[str, frozenset[str]]
) -> tuple[list[list[float]], list[int], list[str]]:
    x, y, sids = [], [], []
    for u in utts:
        for k, lab in zip(u.vi, u.vi_labels, strict=True):
            if lab is None:
                continue
            x.append(_vi_vec(u, k, context))
            y.append(lab)
            sids.append(u.en.sample_id)
    return x, y, sids


def _vi_vec(u: Utt2, k: int, context: Mapping[str, frozenset[str]]) -> list[float]:
    return vi_features(
        u.en.parts, k, u.acoustic_full.get(k), context.get(u.en.sample_id, frozenset()), u.en.v3_scores,
        u.en.turn == "final",
    )  # fmt: skip


def fit_vi(utts: Sequence[Utt2], context: Mapping[str, frozenset[str]]) -> tuple[A.LogisticModel, float, float]:
    """H-Việt học trên `utts`: hệ số phạt và ngưỡng chọn bằng kiểm định chéo bên trong (độ chính xác vùng có nhãn).

    Trả về:
        (mô hình học trên mọi vùng `vi` có nhãn của `utts`, hệ số phạt, ngưỡng).
    """
    x, y, sids = _vi_xy(utts, context)
    folds = sorted({A.fold_of(s) for s in sids})
    best = (-1.0, 1.0, 0.5)
    for l2 in A.L2_GRID:
        probs: list[tuple[float, int]] = []
        for f in folds:
            tr = [i for i, s in enumerate(sids) if A.fold_of(s) != f]
            te = [i for i, s in enumerate(sids) if A.fold_of(s) == f]
            if not tr or not te:
                continue
            m = A.fit_logistic([x[i] for i in tr], [y[i] for i in tr], l2)
            probs += [(A.predict(m, x[i]), y[i]) for i in te]
        for theta in THETAS_VI:
            acc = sum((p >= theta) == bool(t) for p, t in probs) / max(1, len(probs))
            if acc > best[0]:
                best = (acc, l2, theta)
    _, l2, theta = best
    return A.fit_logistic(x, y, l2), l2, theta


def assemble_rows(
    utts: Sequence[Utt2], decide_en: A.RegionDecider | None, decide_vi: Any | None
) -> list[dict[str, Any]]:
    """Các dòng `hyps.jsonl` sau khi ghép: vùng `en` theo `decide_en`, vùng `vi` theo `decide_vi`; `None` giữ RNNT."""
    out = []
    for u in utts:
        en_set, vi_set = set(u.en.candidates), set(u.vi)

        def take(k: int, region: Region, u: Utt2 = u, en_set: set[int] = en_set, vi_set: set[int] = vi_set) -> bool:
            if k in en_set:
                return decide_en is not None and bool(decide_en(u.en, k, region))
            if k in vi_set:
                return decide_vi is not None and bool(decide_vi(u, k))
            return False

        hyp = assemble(u.en.parts, take)
        out.append({"id": u.en.sample_id, "turn": u.en.turn, "ref": u.row.get("ref"), "hyp": hyp, "raw": hyp})
    return out


def _vi_decider(model: A.LogisticModel, theta: float, context: Mapping[str, frozenset[str]]) -> Any:
    return lambda u, k: A.predict(model, _vi_vec(u, k, context)) >= theta


def stage_cv(args: argparse.Namespace) -> dict[str, Any]:
    """Kiểm định chéo 5 phần theo mẫu trên train; mỗi mẫu được quyết định bởi mô hình học không có phần của nó."""
    _, prompt_words = _fixed_prompt()
    prompt_text, _ = _fixed_prompt()
    data = load(args, "train")
    utts, context = data["utts"], data["context"]
    folds = sorted({A.fold_of(u.en.sample_id) for u in utts})
    rows: dict[str, list[dict[str, Any]]] = {"keepA": [], "v2_en": [], "v2": []}
    per_fold: dict[str, Any] = {}
    vi_hits = {"rnnt_right": [0, 0], "whisper_right": [0, 0]}
    for f in folds:
        train = [u for u in utts if A.fold_of(u.en.sample_id) != f]
        test = [u for u in utts if A.fold_of(u.en.sample_id) == f]
        model_en, l2_en, _ = fit_h([u.en for u in train], context, prompt_words)
        rel = A.reliability(A.count_prompt_words([u.en for u in train], prompt_words))
        dec_en = A.h_decider(model_en, context, prompt_words, rel)
        model_vi, l2_vi, theta = fit_vi(train, context)
        dec_vi = _vi_decider(model_vi, theta, context)
        per_fold[str(f)] = {"l2_en": l2_en, "l2_vi": l2_vi, "theta_vi": theta}
        rows["keepA"] += assemble_rows(test, None, None)
        rows["v2_en"] += assemble_rows(test, dec_en, None)
        rows["v2"] += assemble_rows(test, dec_en, dec_vi)
        for u in test:
            for k, lab in zip(u.vi, u.vi_labels, strict=True):
                if lab is None:
                    continue
                side = "whisper_right" if lab == 1 else "rnnt_right"
                vi_hits[side][0] += 1
                vi_hits[side][1] += int(dec_vi(u, k)) == lab
    order = ("v2_en", "v2", "keepA")  # `compare` lấy hệ thống đầu tiên làm mốc
    cmp = compare(data["samples"], {n: (rows[n], prompt_text) for n in order})
    table = {n: {m: {"value": cmp[n]["report"][m], "diff_ci": cmp[n]["diff_ci"][m]} for m in CV_METRICS} for n in cmp}
    v2 = table["v2"]
    criteria = {
        "vi_only_wer_down": v2["vi_only_wer"]["diff_ci"][1] < 0,
        "en_recall_not_down": v2["en_word_recall"]["diff_ci"][1] >= 0,
        "entity_miss_not_up": v2["entity_miss_rate"]["diff_ci"][0] <= 0,
    }
    return {
        "n_samples": len(data["samples"]),
        "regions": {
            "en": sum(len(u.en.candidates) for u in utts),
            "vi": sum(len(u.vi) for u in utts),
            "vi_labelled": sum(lab is not None for u in utts for lab in u.vi_labels),
        },
        "per_fold": per_fold,
        "vi_accuracy_by_side": {k: (v[1] / v[0] if v[0] else 0.0, v[0]) for k, v in vi_hits.items()},
        "table": table,
        "baseline": "v2_en",
        "criteria": criteria,
        "keep_v2": all(criteria.values()),
    }


def _fit_all(args: argparse.Namespace) -> dict[str, Any]:
    _, prompt_words = _fixed_prompt()
    data = load(args, "train")
    utts, context = data["utts"], data["context"]
    model_en, l2_en, _ = fit_h([u.en for u in utts], context, prompt_words)
    rel = A.reliability(A.count_prompt_words([u.en for u in utts], prompt_words))
    model_vi, l2_vi, theta = fit_vi(utts, context)
    return {"en": model_en, "l2_en": l2_en, "rel": rel, "vi": model_vi, "l2_vi": l2_vi, "theta_vi": theta}


def stage_export(args: argparse.Namespace) -> dict[str, Any]:
    """Trọng số bản 2 học trên cả train, cho `MergedASR` (`configs/asr_merge/h_v2.json`)."""
    import dataclasses

    from lifemate.eval.asr_v3 import FIXED_PROMPT_FILE, GUARD_MAX_COMPRESSION, GUARD_NGRAM

    fit = _fit_all(args)
    return {
        "version": args.out.stem,
        "pipeline": 2,
        "description": "Ghép bản 2 (asr_v4.md mục 6c, D67): đổi số trước khi gióng, vùng chỉ khác dấu thanh, H-Anh học "
        "lại trên vùng mới, H-Việt cho vùng chữ Việt; học trên cả 1 093 mẫu train",
        "prompt_file": FIXED_PROMPT_FILE,
        "prompt_variant": "E2",
        "guard": {"ngram": GUARD_NGRAM, "max_compression": GUARD_MAX_COMPRESSION},
        "features": list(A.FEATURES),
        "threshold": 0.5,
        "model": dataclasses.asdict(fit["en"]),
        "reliability": dict(sorted(fit["rel"].items())),
        "digitize": True,
        "vi_features": list(VI_FEATURES),
        "vi_threshold": fit["theta_vi"],
        "vi_model": dataclasses.asdict(fit["vi"]),
        "trained_on": {"l2_en": fit["l2_en"], "l2_vi": fit["l2_vi"]},
    }


def stage_measure(args: argparse.Namespace) -> dict[str, Any]:
    """Đo một lần trên `args.split`: bản 1 đang dùng (`h_v1` + đổi số, điểm K3), bản 2 chỉ H-Anh, bản 2 đủ."""
    from lifemate.eval.asr_v4_report import digitized, h_full_train
    from lifemate.eval.asr_v4_report import load_parts as load_v1

    if args.split == "val_unseen":
        raise SystemExit("val_unseen bị khoá (D53)")
    prompt_text, prompt_words = _fixed_prompt()
    fit = _fit_all(args)
    data = load(args, args.split)
    context = data["context"]
    dec_en = A.h_decider(fit["en"], context, prompt_words, fit["rel"])
    dec_vi = _vi_decider(fit["vi"], fit["theta_vi"], context)
    make_v1, _ = h_full_train(argparse.Namespace(k1=args.k1, k2=args.k2, k3=args.k3, winner="E2"))
    v1 = load_v1(argparse.Namespace(k1=args.k1, k2=args.k2, k3=args.k3, winner="E2"), SPLIT_PARTS[args.split])
    systems = {
        "v1": digitized(A.apply_decider(v1["rnnt"], v1["utts"], make_v1(v1["context"]))),
        "v2_en": assemble_rows(data["utts"], dec_en, None),
        "v2": assemble_rows(data["utts"], dec_en, dec_vi),
    }
    cmp = compare(data["samples"], {n: (r, prompt_text) for n, r in systems.items()})
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, r in systems.items():
        (out_dir / f"{n}_{args.split}.hyps.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in r), encoding="utf-8"
        )
    return {
        "split": args.split,
        "table": {
            n: {m: {"value": cmp[n]["report"][m], "diff_ci": cmp[n]["diff_ci"][m]} for m in CV_METRICS} for n in cmp
        },
        "baseline": "v1",
    }


def _markdown(res: Mapping[str, Any]) -> str:
    lines = [
        f"mốc: {res['baseline']}",
        "| Hệ thống | " + " | ".join(CV_METRICS) + " |",
        "|---|" + "---|" * len(CV_METRICS),
    ]
    for n, t in res["table"].items():
        cells = [f"{100 * t[m]['value']:.2f}% [{100 * t[m]['diff_ci'][0]:+.2f}; {100 * t[m]['diff_ci'][1]:+.2f}]"
                 for m in CV_METRICS]  # fmt: skip
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: lệnh `cv`, `export` hoặc `measure` (xem docstring module)."""
    ap = argparse.ArgumentParser(prog="lifemate.eval.merge_v2_report", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("cv", "export", "measure"):
        p = sub.add_parser(name)
        for k in ("--k1", "--k2", "--k5"):
            p.add_argument(k, type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if name == "measure":
            p.add_argument("--k3", type=Path, required=True)
            p.add_argument("--split", default="val_seen")
    args = ap.parse_args(argv)
    random.seed(0)
    if args.cmd == "export":
        w = stage_export(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(w, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({k: w[k] for k in ("version", "vi_threshold", "trained_on")}, ensure_ascii=False))
        return 0
    res = stage_cv(args) if args.cmd == "cv" else stage_measure(args)
    args.out.mkdir(parents=True, exist_ok=True)
    name = "merge_v2_cv.json" if args.cmd == "cv" else f"merge_v2_{args.split}.json"
    (args.out / name).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(_markdown(res))
    if args.cmd == "cv":
        keys = ("regions", "vi_accuracy_by_side", "criteria", "keep_v2")
        print(json.dumps({k: res[k] for k in keys}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
