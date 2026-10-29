"""Tính các bảng số của báo cáo benchmark ASR lần 3 từ kết quả các phiên GPU (chạy trên máy dev, không cần GPU).

Cách dùng (từ thư mục gốc repo):
  python -m lifemate.eval.asr_v3_report e --k1 <K1> --out <thư mục>
  python -m lifemate.eval.asr_v3_report select --k1 <K1> --k2 <K2> --k3 <K3> --winner E1 --out <thư mục>

Nội dung chính:
- `stage_e`: thử nghiệm E: so ba biến thể câu mồi (bản ghép C0 và large-v3 đứng riêng) trên `asr_tune`, áp tiêu chí.
- `stage_select`: thử nghiệm F, G, H và bảng chọn chung trên `asr_tune`; đo cách thắng một lần trên val; I1; ghi
  chữ của các nguồn J cho phiên K4.

Thư mục vào:
- K1: `rnnt.hyps.jsonl`, `v3_E0|E1|E2.hyps.jsonl`, `timing.jsonl`, `summary.json` (tập dò).
- K2: `k2_asr_learn/`, `k2_val_seen/`, `k2_val_unseen/`, mỗi thư mục có `rnnt.hyps.jsonl`, `v3_<thắng>.hyps.jsonl`.
- K3: `scores_<tập>.jsonl`, `scores_<tập>.timing.jsonl`, `scores_<tập>.summary.json`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lifemate.backends.asr_guard import echoes_prompt
from lifemate.core.paths import REPO_ROOT
from lifemate.core.types import Sample
from lifemate.eval import asr_v3_analysis as A  # noqa: N812
from lifemate.eval.asr_compare import compare
from lifemate.eval.asr_merge import Region
from lifemate.eval.eval_asr import percentile
from lifemate.eval.foreign_words import contains_word, find_foreign_words, folded_words

PROMPT_FILE = "configs/asr_prompts/en_full_v1.txt"


def _rows(path: Path) -> list[dict[str, Any]]:
    """Đọc một file JSON Lines."""
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def _data(ids_file: str) -> tuple[list[Sample], dict[str, Any]]:
    """Các mẫu của `ids_file` và catalog, đọc theo profile `local`."""
    from lifemate.core.config import load_profile
    from lifemate.core.paths import repo_path
    from lifemate.io.data import load_split, read_ids
    from lifemate.io.zaic import load_catalog

    root = repo_path(load_profile("local", REPO_ROOT).data_root, REPO_ROOT)
    samples = load_split(root, "train", read_ids(repo_path(ids_file, REPO_ROOT)))
    return samples, {str(a["id"]): a for a in load_catalog(root, "train")["apps"]}


def _fixed_prompt() -> tuple[str, frozenset[str]]:
    """Câu mồi cố định và tập từ ngoại (đã chuẩn hoá) của câu mồi."""
    text = (REPO_ROOT / PROMPT_FILE).read_text(encoding="utf-8").strip()
    return text, frozenset(w.folded for w in find_foreign_words(text))


def en_context_recall(
    samples: Sequence[Sample],
    rows: Sequence[Mapping[str, Any]],
    context: Mapping[str, frozenset[str]],
) -> dict[str, list[int]]:
    """Chép đúng từ tiếng Anh, tách theo từ của đáp án có / không có trong từ điển ngữ cảnh: `{in|out: [số, đúng]}`."""
    hyp = {(str(r["id"]), str(r["turn"])): str(r["hyp"]) for r in rows}
    out = {"in": [0, 0], "out": [0, 0]}
    for s in samples:
        refs = [("final", s.audio_text)] + [(str(i), t.audio_text) for i, t in enumerate(s.history) if t.role == "user"]
        for turn, ref in refs:
            if not ref:
                continue
            words = folded_words(hyp.get((s.id, turn), ""))
            for w in find_foreign_words(ref):
                side = "in" if w.folded in context.get(s.id, frozenset()) else "out"
                out[side][0] += 1
                out[side][1] += contains_word(w.folded, words)
    return out


def assistant_echo(samples: Sequence[Sample], rows: Sequence[Mapping[str, Any]]) -> int:
    """Số câu mà chữ ASR có 4 từ liên tiếp trùng lời trợ lý liền trước, trong khi đáp án không có 4 từ đó."""
    hyp = {(str(r["id"]), str(r["turn"])): str(r["hyp"]) for r in rows}
    n = 0
    for s in samples:
        previous = None
        turns = [*((str(i), t) for i, t in enumerate(s.history)), ("final", None)]
        for key, turn in turns:
            role = turn.role if turn is not None else "user"
            if role == "user" and previous is not None and previous.role == "assistant" and previous.text:
                ref = s.audio_text if turn is None else turn.audio_text
                if echoes_prompt(hyp.get((s.id, key), ""), previous.text, 4) and not echoes_prompt(
                    ref or "", previous.text, 4
                ):
                    n += 1
            if turn is not None:
                previous = turn
    return n


def merge_word_errors(
    utterances: Sequence[A.UtteranceRegions], decide: A.RegionDecider, prompt_words: frozenset[str]
) -> dict[str, int]:
    """Từ tiếng Anh mới lấy nhầm (không có trong đáp án) và từ đúng bị bỏ, tách theo từ của câu mồi / ngoài câu mồi."""
    out = {"wrong_prompt": 0, "wrong_other": 0, "missed_prompt": 0, "missed_other": 0}
    for u in utterances:
        ref_words = folded_words(u.ref)
        for k in u.candidates:
            region = u.parts[k]
            assert isinstance(region, Region)
            take = decide(u, k, region)
            for w in region.new_english:
                side = "prompt" if w in prompt_words else "other"
                in_ref = contains_word(w, ref_words)
                if take and not in_ref:
                    out[f"wrong_{side}"] += 1
                elif not take and in_ref:
                    out[f"missed_{side}"] += 1
    return out


def _brief(result: Mapping[str, Any]) -> dict[str, Any]:
    """Các chỉ số chính của một kết quả `compare` (giá trị và CI của chênh lệch so với mốc)."""
    keys = ("en_word_recall", "vi_only_wer", "en_mixed_wer", "wer", "entity_miss_rate")
    return {k: {"value": result["report"][k], "diff_ci": result["diff_ci"][k]} for k in keys}


def stage_e(args: argparse.Namespace) -> dict[str, Any]:
    """Thử nghiệm E trên `asr_tune`: xem docstring module."""
    samples, catalog = _data("configs/splits/asr_tune.txt")
    prompt_text, prompt_words = _fixed_prompt()
    context = A.context_words_by_sample(samples, catalog)
    k1: Path = args.k1
    rnnt = _rows(k1 / "rnnt.hyps.jsonl")
    timing = _rows(k1 / "timing.jsonl")
    merged: dict[str, tuple[list[dict[str, Any]], str]] = {}
    alone: dict[str, tuple[list[dict[str, Any]], str]] = {}
    extra: dict[str, Any] = {}
    for v in ("E0", "E1", "E2"):
        v3 = _rows(k1 / f"v3_{v}.hyps.jsonl")
        utts = A.utterance_regions(rnnt, v3)
        merged[v] = (A.apply_decider(rnnt, utts, A.c0_decider(context, prompt_words)), prompt_text)
        alone[v] = (v3, prompt_text)
        extra[v] = {
            "en_context_v3": en_context_recall(samples, v3, context),
            "en_context_merged": en_context_recall(samples, merged[v][0], context),
            "assistant_echo": assistant_echo(samples, v3),
            "retries": sum(bool(r.get("retried")) for r in v3),
            "truncated": sum(int(r.get("prompt_tokens", 0)) > 223 for r in v3),
            "whisper_p50_p95_max": [percentile([t["whisper_s"][v] for t in timing], q) for q in (50, 95, 100)],
            "merge_p95": percentile([t["merge_s"].get(v, 0.0) for t in timing], 95),
            "total_p50_p95_max": [
                percentile([t["rnnt_s"] + t["whisper_s"][v] + t["merge_s"].get(v, 0.0) for t in timing], q)
                for q in (50, 95, 100)
            ],
        }
    extra["rnnt_p50_p95_max"] = [percentile([t["rnnt_s"] for t in timing], q) for q in (50, 95, 100)]
    extra["summary"] = json.loads((k1 / "summary.json").read_text(encoding="utf-8"))
    cm = compare(samples, merged)
    ca = compare(samples, alone)
    result = {"merged": {v: _brief(cm[v]) for v in cm}, "alone": {v: _brief(ca[v]) for v in ca}, "extra": extra}
    # Tiêu chí E (đặt trước): chép đúng cao nhất trong các biến thể mà WER câu thuần Việt của bản ghép không tăng so
    # với E0 (CI của chênh lệch không nằm hẳn trên 0) và p95 thời gian của large-v3 tăng không quá 1 giây.
    base_p95 = extra["E0"]["whisper_p50_p95_max"][1]
    ok = [
        v
        for v in ("E0", "E1", "E2")
        if cm[v]["diff_ci"]["vi_only_wer"][0] <= 0 and extra[v]["whisper_p50_p95_max"][1] - base_p95 <= 1.0
    ]
    result["winner"] = max(ok, key=lambda v: cm[v]["report"]["en_word_recall"]) if ok else "E0"
    return result


def stage_select(args: argparse.Namespace) -> dict[str, Any]:
    """Thử nghiệm F, G, H, bảng chọn chung, val, I1 và chữ cho J: xem docstring module."""
    w = args.winner
    prompt_text, prompt_words = _fixed_prompt()
    sets = {
        "asr_learn": (args.k2 / "k2_asr_learn", args.k3 / "scores_asr_learn"),
        "asr_tune": (args.k1, args.k3 / "scores_asr_tune"),
        "val_seen": (args.k2 / "k2_val_seen", args.k3 / "scores_val_seen"),
        "val_unseen": (args.k2 / "k2_val_unseen", args.k3 / "scores_val_unseen"),
    }
    data: dict[str, Any] = {}
    for name, (hdir, sprefix) in sets.items():
        samples, catalog = _data(f"configs/splits/{name}.txt")
        rnnt = _rows(hdir / "rnnt.hyps.jsonl")
        v3 = _rows(hdir / f"v3_{w}.hyps.jsonl")
        scores = _rows(Path(f"{sprefix}.jsonl")) if Path(f"{sprefix}.jsonl").is_file() else []
        timing = _rows(Path(f"{sprefix}.timing.jsonl")) if Path(f"{sprefix}.timing.jsonl").is_file() else []
        data[name] = {
            "samples": samples,
            "context": A.context_words_by_sample(samples, catalog),
            "rnnt": rnnt,
            "v3": v3,
            "utts": A.utterance_regions(rnnt, v3, scores),
            "score_timing": timing,
        }
    learn = data["asr_learn"]
    result: dict[str, Any] = {"winner_prompt": w}

    # F: chọn θ bằng kiểm định chéo 5 phần theo mẫu trên tập học.
    folds = [[u for u in learn["utts"] if A.fold_of(u.sample_id) == f] for f in range(A.N_FOLDS)]
    cv: dict[str, dict[str, float]] = {}
    oof_rel: dict[int, dict[str, float]] = {}
    for f in range(A.N_FOLDS):
        train_utts = [u for g in range(A.N_FOLDS) if g != f for u in folds[g]]
        oof_rel[f] = A.reliability(A.count_prompt_words(train_utts, prompt_words))
    c0_learn = A.c0_decider(learn["context"], prompt_words)
    cv["C0"] = {"accuracy": A.decision_accuracy(A.decisions(learn["utts"], c0_learn))}
    for theta in A.THETAS:
        rows = []
        for f in range(A.N_FOLDS):
            rows += A.decisions(folds[f], A.f_decider(learn["context"], prompt_words, oof_rel[f], theta))
        cv[str(theta)] = {"accuracy": A.decision_accuracy(rows)}
    best_theta = max(A.THETAS, key=lambda t: (cv[str(t)]["accuracy"], -abs(t - 0.5)))
    counts = A.count_prompt_words(learn["utts"], prompt_words)
    rel = A.reliability(counts)
    result["F"] = {
        "cv": cv,
        "theta": best_theta,
        "table": sorted(([w2, *c, rel[w2]] for w2, c in counts.items() if c[0] >= 3), key=lambda r: -r[1]),
    }

    # H: hồi quy logistic trên tập học; đặc trưng độ tin cậy dùng bảng đếm ngoài phần (out-of-fold).
    def features(utts: Sequence[A.UtteranceRegions], ctx: Mapping[str, frozenset[str]], relmap: Any) -> Any:
        xs, ys, sids = [], [], []
        for u in utts:
            r = relmap(u) if callable(relmap) else relmap
            for k, label in zip(u.candidates, u.labels, strict=True):
                region = u.parts[k]
                assert isinstance(region, Region)
                if label is None:
                    continue
                xs.append(A.region_features(u, region, k, ctx.get(u.sample_id, frozenset()), prompt_words, r))
                ys.append(label)
                sids.append(u.sample_id)
        return xs, ys, sids

    x, y, sids = features(learn["utts"], learn["context"], lambda u: oof_rel[A.fold_of(u.sample_id)])
    cv_h: dict[str, float] = {}
    for l2 in A.L2_GRID:
        hit = tot = 0
        for f in range(A.N_FOLDS):
            tr = [i for i, s in enumerate(sids) if A.fold_of(s) != f]
            te = [i for i, s in enumerate(sids) if A.fold_of(s) == f]
            if not tr or not te:
                continue
            m = A.fit_logistic([x[i] for i in tr], [y[i] for i in tr], l2)
            hit += sum((A.predict(m, x[i]) >= 0.5) == bool(y[i]) for i in te)
            tot += len(te)
        cv_h[str(l2)] = hit / tot if tot else 0.0
    best_l2 = max(A.L2_GRID, key=lambda v: (cv_h[str(v)], v))
    model = A.fit_logistic(x, y, best_l2)
    result["H"] = {
        "cv": cv_h,
        "l2": best_l2,
        "weights": dict(zip([*A.FEATURES, "intercept"], model.weights, strict=True)),
        "n_train": len(y),
        "labels": {
            "1": sum(y),
            "0": len(y) - sum(y),
            "tie": sum(lab is None for u in learn["utts"] for lab in u.labels),
        },
    }

    def deciders(ctx: Mapping[str, frozenset[str]]) -> dict[str, A.RegionDecider]:
        c0 = A.c0_decider(ctx, prompt_words)
        f_dec = A.f_decider(ctx, prompt_words, rel, best_theta)
        return {"C0": c0, "F": f_dec, "G": A.g_decider(f_dec), "H": A.h_decider(model, ctx, prompt_words, rel)}

    def evaluate_set(name: str, names: Sequence[str]) -> dict[str, Any]:
        """Chấm các cách quyết định `names` trên một tập; chênh lệch tính so với cách đầu tiên (C0)."""
        d = data[name]
        decs = deciders(d["context"])
        systems = {n: (A.apply_decider(d["rnnt"], d["utts"], decs[n]), prompt_text) for n in names}
        cmp = compare(d["samples"], {**systems, "RNNT": (d["rnnt"], prompt_text)})
        dec_rows = {n: A.decisions(d["utts"], decs[n]) for n in names}
        acc = A.bootstrap_accuracy(dec_rows, [s.id for s in d["samples"]], names[0])
        score_p95 = percentile([t["score_s"] for t in d["score_timing"]], 95) if d["score_timing"] else 0.0
        out: dict[str, Any] = {
            n: {
                **_brief(cmp[n]),
                "decision_accuracy": acc[n],
                "word_errors": merge_word_errors(d["utts"], decs[n], prompt_words),
                "extra_p95_s": score_p95 if n in ("G", "H") else 0.0,
            }
            for n in names
        }
        out["RNNT"] = _brief(cmp["RNNT"])
        out["_regions"] = {
            "utterances": len(d["utts"]),
            "regions": sum(len(u.candidates) for u in d["utts"]),
            "labelled": sum(lab is not None for u in d["utts"] for lab in u.labels),
            "ties": sum(lab is None for u in d["utts"] for lab in u.labels),
        }
        out["_rows"] = {n: systems[n][0] for n in names}
        out["_decisions"] = dec_rows
        return out

    order = ["C0", "F", "G", "H"]
    tune = evaluate_set("asr_tune", order)
    # Bảng chọn chung (đặt trước, mục 7 của báo cáo).
    # 1. Loại ứng viên tệ hơn C0 có ý nghĩa thống kê ở chép đúng từ tiếng Anh (CI của chênh lệch nằm hẳn dưới 0) hoặc ở
    #    WER câu thuần Việt (CI nằm hẳn trên 0), hoặc tốn thêm quá 1 giây ở p95 mỗi mẫu.
    eligible = ["C0"] + [
        n
        for n in order[1:]
        if tune[n]["en_word_recall"]["diff_ci"][1] >= 0
        and tune[n]["vi_only_wer"]["diff_ci"][0] <= 0
        and tune[n]["extra_p95_s"] <= 1.0
    ]
    # 2. Lấy ứng viên có độ chính xác quyết định đoạn cao nhất.
    best = max(eligible, key=lambda n: tune[n]["decision_accuracy"]["accuracy"])
    # 3. Không hơn có ý nghĩa thống kê một ứng viên đơn giản hơn thì chọn ứng viên đơn giản hơn.
    pairwise: dict[str, Any] = {}
    sample_ids = [s.id for s in data["asr_tune"]["samples"]]
    for simpler in [n for n in order[: order.index(best)] if n in eligible]:
        b = A.bootstrap_accuracy(
            {best: tune["_decisions"][best], simpler: tune["_decisions"][simpler]}, sample_ids, simpler
        )
        pairwise[f"{best}-{simpler}"] = b[best]["diff_ci"]
        if b[best]["diff_ci"][0] <= 0:
            best = simpler
            break
    result["tune"] = {k: v for k, v in tune.items() if not k.startswith("_")} | {"_regions": tune["_regions"]}
    result["eligible"] = eligible
    result["pairwise"] = pairwise
    result["selected"] = best

    # Đo một lần trên val: ứng viên thắng và C0 (cùng RNNT làm mốc tham khảo).
    names_val = ["C0"] if best == "C0" else ["C0", best]
    val: dict[str, Any] = {}
    rows_out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for vs in ("val_seen", "val_unseen"):
        ev = evaluate_set(vs, names_val)
        val[vs] = {k: v for k, v in ev.items() if not k.startswith("_")} | {"_regions": ev["_regions"]}
        rows_out[vs] = ev["_rows"]
    result["val"] = val

    # I1: hai phương án ở đoạn không chắc của cách thắng; trần bằng cách chọn phía ít lỗi hơn ở mọi đoạn không chắc.
    def uncertain_fn(ctx: Mapping[str, frozenset[str]]) -> Any:
        decs = deciders(ctx)
        if best == "G":
            return lambda u, k, r: (s := u.acoustic.get(k)) is not None and not s[2] and abs(s[1] - s[0]) < 2.3
        if best == "H":
            return lambda u, k, r: (
                0.2
                <= A.predict(model, A.region_features(u, r, k, ctx.get(u.sample_id, frozenset()), prompt_words, rel))
                <= 0.8
            )
        # Luật: đoạn mà bộ lọc loại (giữ RNNT) dù large-v3 có từ tiếng Anh mới.
        return lambda u, k, r: not decs[best](u, k, r)

    alts: dict[str, Any] = {}
    for name in ("asr_tune", "val_seen", "val_unseen"):
        d = data[name]
        decide = deciders(d["context"])[best]
        unsure = uncertain_fn(d["context"])
        texts, counts = A.alternatives(d["utts"], decide, unsure)

        def oracle(u: A.UtteranceRegions, k: int, r: Region, decide: Any = decide, unsure: Any = unsure) -> bool:
            if unsure(u, k, r):
                label = u.labels[u.candidates.index(k)]
                return decide(u, k, r) if label is None else bool(label)
            return bool(decide(u, k, r))

        chosen = A.apply_decider(d["rnnt"], d["utts"], decide)
        ceiling = A.apply_decider(d["rnnt"], d["utts"], oracle)
        cmp = compare(d["samples"], {"chosen": (chosen, prompt_text), "ceiling": (ceiling, prompt_text)})
        alt_rows = [{**r, "hyp": texts.get((str(r["id"]), str(r["turn"])), r["hyp"])} for r in chosen]
        alt_rows = [{**r, "raw": r["hyp"]} for r in alt_rows]
        vi_with_unsure = sum(
            1 for u in d["utts"] if not find_foreign_words(u.ref) and any(
                unsure(u, k, u.parts[k]) for k in u.candidates
            )
        )  # fmt: skip
        alts[name] = {
            "counts": counts,
            "chosen_recall": cmp["chosen"]["report"]["en_word_recall"],
            "ceiling_recall": cmp["ceiling"]["report"]["en_word_recall"],
            "ceiling_diff_ci": cmp["ceiling"]["diff_ci"]["en_word_recall"],
            "vi_only_utterances_with_uncertain": vi_with_unsure,
        }
        if name != "asr_tune":
            rows_out[name]["alternatives"] = alt_rows
    result["I1"] = alts

    # Chữ cho các nguồn của J (val): RNNT, ghép lần 2 (C0 trên chữ large-v3 E0 của lần 2 nếu có), ghép lần 3, hai
    # phương án.
    out_dir: Path = args.out
    for vs, rows in rows_out.items():
        d = data[vs]
        (out_dir / f"j1_rnnt_{vs}.hyps.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in d["rnnt"]), encoding="utf-8"
        )
        (out_dir / f"j3_merged_{vs}.hyps.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[best]), encoding="utf-8"
        )
        (out_dir / f"j4_alternatives_{vs}.hyps.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows["alternatives"]), encoding="utf-8"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: lệnh `e` hoặc `select`, ghi kết quả JSON vào `--out`."""
    ap = argparse.ArgumentParser(prog="lifemate.eval.asr_v3_report", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("e")
    pe.add_argument("--k1", type=Path, required=True)
    pe.add_argument("--out", type=Path, required=True)
    ps = sub.add_parser("select")
    for k in ("--k1", "--k2", "--k3"):
        ps.add_argument(k, type=Path, required=True)
    ps.add_argument("--winner", required=True)
    ps.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "e":
        res = stage_e(args)
        (args.out / "e.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({"winner": res["winner"], "merged": res["merged"]}, ensure_ascii=False, indent=1))
    else:
        clean = stage_select(args)
        (args.out / "select.json").write_text(json.dumps(clean, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({"selected": clean["selected"], "eligible": clean["eligible"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
