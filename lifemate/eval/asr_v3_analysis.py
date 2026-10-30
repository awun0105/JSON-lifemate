"""Phân tích offline của benchmark ASR lần 3 (`docs/reports/benchmark/asr_v3.md`): F, H, bảng chọn chung, I.

Module này không chạy mô hình; mọi đầu vào là file `hyps.jsonl` và file điểm âm thanh của các phiên GPU.

Nội dung chính:
- `utterance_regions`: các vùng cần quyết định của mọi câu, kèm nhãn tự động.
- `count_prompt_words`, `reliability`: bảng độ tin cậy từng từ của câu mồi (F).
- `RegionDecider` và các cách quyết định C0, F, G, H; `apply_decider`: ghép lại chữ theo một cách quyết định.
- `region_features`, `fit_logistic`, `predict`: bộ phân loại H (hồi quy logistic có phạt L2, phương pháp Newton).
- `decision_accuracy`, `bootstrap_accuracy`: độ chính xác quyết định đoạn và khoảng tin cậy bootstrap.
- `alternatives`: định dạng hai phương án và đếm trần (I1).

Cách hoạt động:
- Vùng cần quyết định: vùng khác nhau mà chữ large-v3 có từ tiếng Anh mới (`asr_merge.candidate_regions`). Câu
  large-v3 rỗng hoặc bị bộ lọc câu bịa cắt thì bỏ, như `merge_hyp_rows`.
- Nhãn tự động của một vùng: so hai câu (giữ RNNT ở mọi vùng; chỉ lấy large-v3 ở vùng đó) với đáp án theo số lỗi từ
  (chuẩn hoá như `eval_asr`); 1 khi lấy large-v3 ít lỗi hơn, 0 khi nhiều lỗi hơn, `None` khi hoà.
"""

from __future__ import annotations

import math
import random
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from lifemate.core.types import Sample
from lifemate.eval.asr_merge import Region, assemble, candidate_regions, english_filter, split_regions
from lifemate.eval.eval_asr import _edit_distance, normalize_text
from lifemate.eval.foreign_words import contains_word, find_foreign_words, folded_words

THETAS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
L2_GRID = (0.01, 0.1, 1.0, 10.0)
N_FOLDS = 5


@dataclass
class UtteranceRegions:
    """Các vùng cần quyết định của một câu nói.

    Thuộc tính:
        sample_id, turn: khoá của câu.
        ref: đáp án.
        rnnt, v3: chữ RNNT và chữ large-v3 (đã lọc câu bịa).
        parts: kết quả `split_regions(rnnt, v3)`.
        candidates: chỉ số các vùng cần quyết định trong `parts`.
        labels: nhãn tự động của từng vùng trong `candidates`.
        v3_scores: điểm tin cậy của large-v3 cho câu (`avg_logprob`, `no_speech_prob`).
        acoustic: chỉ số vùng -> (điểm âm thanh A, điểm âm thanh B, câu dài hơn 30 giây).
    """

    sample_id: str
    turn: str
    ref: str
    rnnt: str
    v3: str
    parts: list[str | Region]
    candidates: list[int]
    labels: list[int | None]
    v3_scores: dict[str, float] = field(default_factory=dict)
    acoustic: dict[int, tuple[float, float, bool]] = field(default_factory=dict)


def _errors(ref: str, hyp: str) -> int:
    """Số lỗi từ (thêm, xoá, thay) giữa đáp án và chữ ASR, chuẩn hoá như `eval_asr`."""
    return _edit_distance(normalize_text(ref).split(), normalize_text(hyp).split())


def utterance_regions(
    rnnt_rows: Sequence[Mapping[str, Any]],
    v3_rows: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]] = (),
) -> list[UtteranceRegions]:
    """Các câu có ít nhất một vùng cần quyết định, kèm nhãn tự động và điểm âm thanh (nếu có)."""
    v3_by = {(str(r["id"]), str(r["turn"])): r for r in v3_rows}
    score_by = {(str(r["id"]), str(r["turn"]), int(r["part"])): r for r in scores}
    out: list[UtteranceRegions] = []
    for row in rnnt_rows:
        key = (str(row["id"]), str(row["turn"]))
        other = v3_by.get(key)
        if other is None:
            continue
        v3_hyp = str(other["hyp"])
        if not v3_hyp.strip() or " ".join(str(other.get("raw", v3_hyp)).split()) != " ".join(v3_hyp.split()):
            continue
        parts = split_regions(str(row["hyp"]), v3_hyp)
        cands = candidate_regions(parts)
        if not cands:
            continue
        ref = str(row.get("ref") or "")
        keep_all = assemble(parts, lambda _k, _r: False)
        err_a = _errors(ref, keep_all)
        labels: list[int | None] = []
        for k in cands:
            err_b = _errors(ref, assemble(parts, lambda i, _r, k=k: i == k))
            labels.append(None if err_a == err_b else int(err_b < err_a))
        acoustic = {}
        for k in cands:
            s = score_by.get((key[0], key[1], k))
            if s is not None:
                acoustic[k] = (float(s["score_a"]), float(s["score_b"]), bool(s.get("long", False)))
        out.append(
            UtteranceRegions(
                key[0], key[1], ref, str(row["hyp"]), v3_hyp, parts, cands, labels,
                {k: float(v) for k, v in (other.get("scores") or {}).items()}, acoustic,
            )
        )  # fmt: skip
    return out


# ---------- F: độ tin cậy từng từ của câu mồi ----------


def count_prompt_words(
    utterances: Sequence[UtteranceRegions], prompt_words: frozenset[str]
) -> dict[str, tuple[int, int]]:
    """Đếm (số lần viết ra, số lần đúng) của từng từ câu mồi: large-v3 có từ đó mà RNNT cùng câu không có."""
    counts: dict[str, list[int]] = {}
    for u in utterances:
        v3_words = {w.folded for w in find_foreign_words(u.v3)} & prompt_words
        rnnt_words = set(folded_words(u.rnnt))
        ref_words = folded_words(u.ref)
        for w in v3_words - rnnt_words:
            c = counts.setdefault(w, [0, 0])
            c[0] += 1
            c[1] += contains_word(w, ref_words)
    return {w: (c[0], c[1]) for w, c in counts.items()}


def reliability(counts: Mapping[str, tuple[int, int]]) -> dict[str, float]:
    """Độ tin cậy `(đúng + 1) / (viết ra + 2)` (quy tắc kế tiếp của Laplace); từ chưa gặp tra ngoài là 0,5."""
    return {w: (correct + 1) / (written + 2) for w, (written, correct) in counts.items()}


# ---------- Các cách quyết định ----------

# Hàm quyết định: (câu, chỉ số vùng, vùng, ngữ cảnh của mẫu) -> có lấy chữ large-v3 không.
RegionDecider = Callable[[UtteranceRegions, int, Region], bool]


def c0_decider(context: Mapping[str, frozenset[str]], prompt_words: frozenset[str]) -> RegionDecider:
    """C0 (lần 2): `context_other` và điều kiện độ dài."""

    def decide(u: UtteranceRegions, _k: int, region: Region) -> bool:
        accept = english_filter("context_other", context.get(u.sample_id, frozenset()), prompt_words)
        words = {w for w in region.new_english if accept is None or accept(w)}
        return bool(words) and region.not_too_long

    return decide


def f_decider(
    context: Mapping[str, frozenset[str]], prompt_words: frozenset[str], rel: Mapping[str, float], theta: float
) -> RegionDecider:
    """F (`context_reliable`): như C0, nhưng từ câu mồi có độ tin cậy dưới θ phải có trong ngữ cảnh."""

    def decide(u: UtteranceRegions, _k: int, region: Region) -> bool:
        ctx = context.get(u.sample_id, frozenset())

        def accept(w: str) -> bool:
            if w in prompt_words and rel.get(w, 0.5) >= theta:
                return True
            return w in ctx

        return any(accept(w) for w in region.new_english) and region.not_too_long

    return decide


def g_decider(fallback: RegionDecider) -> RegionDecider:
    """G: lấy large-v3 khi điểm âm thanh của B cao hơn A; vùng thiếu điểm hoặc câu dài hơn 30 giây dùng `fallback`."""

    def decide(u: UtteranceRegions, k: int, region: Region) -> bool:
        s = u.acoustic.get(k)
        if s is None or s[2]:
            return fallback(u, k, region)
        return s[1] > s[0]

    return decide


def apply_decider(
    rnnt_rows: Sequence[Mapping[str, Any]], utterances: Sequence[UtteranceRegions], decide: RegionDecider
) -> list[dict[str, Any]]:
    """Các dòng `hyps.jsonl` sau khi ghép theo `decide`; câu không có vùng cần quyết định giữ chữ RNNT."""
    by_key = {(u.sample_id, u.turn): u for u in utterances}
    out = []
    for row in rnnt_rows:
        u = by_key.get((str(row["id"]), str(row["turn"])))
        hyp = str(row["hyp"])
        if u is not None:
            cand = set(u.candidates)
            hyp = assemble(u.parts, lambda k, r, u=u, cand=cand: k in cand and decide(u, k, r))
        out.append({**row, "hyp": hyp, "raw": hyp})
    return out


def decisions(utterances: Sequence[UtteranceRegions], decide: RegionDecider) -> list[tuple[str, int | None, bool]]:
    """(mã mẫu, nhãn, quyết định) của mọi vùng cần quyết định."""
    out = []
    for u in utterances:
        for k, label in zip(u.candidates, u.labels, strict=True):
            region = u.parts[k]
            assert isinstance(region, Region)
            out.append((u.sample_id, label, decide(u, k, region)))
    return out


def decision_accuracy(rows: Sequence[tuple[str, int | None, bool]]) -> float:
    """Tỉ lệ vùng có nhãn mà quyết định trùng nhãn (lấy large-v3 khi nhãn 1, giữ RNNT khi nhãn 0)."""
    labelled = [(lab, d) for _s, lab, d in rows if lab is not None]
    return sum(int(d) == lab for lab, d in labelled) / len(labelled) if labelled else 0.0


def bootstrap_accuracy(
    systems: Mapping[str, Sequence[tuple[str, int | None, bool]]],
    sample_ids: Sequence[str],
    baseline: str,
    rounds: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Độ chính xác quyết định đoạn của từng hệ thống, CI 95% và CI của chênh lệch so với `baseline`, bootstrap theo
    mẫu có ghép cặp."""
    per: dict[str, dict[str, list[int]]] = {}
    for name, rows in systems.items():
        d: dict[str, list[int]] = {}
        for sid, lab, dec in rows:
            if lab is not None:
                c = d.setdefault(sid, [0, 0])
                c[0] += int(dec) == lab
                c[1] += 1
        per[name] = d
    rng = random.Random(seed)
    vals: dict[str, list[float]] = {n: [] for n in systems}
    for _ in range(rounds):
        pick = [sample_ids[rng.randrange(len(sample_ids))] for _ in sample_ids]
        for name in systems:
            hit = sum(per[name].get(s, [0, 0])[0] for s in pick)
            tot = sum(per[name].get(s, [0, 0])[1] for s in pick)
            vals[name].append(hit / tot if tot else 0.0)

    def ci(xs: Sequence[float]) -> list[float]:
        s = sorted(xs)
        return [s[int(0.025 * len(s))], s[min(len(s) - 1, int(0.975 * len(s)))]]

    return {
        name: {
            "accuracy": decision_accuracy(systems[name]),
            "ci": ci(vals[name]),
            "diff_ci": ci([a - b for a, b in zip(vals[name], vals[baseline], strict=True)]),
        }
        for name in systems
    }


# ---------- H: bộ phân loại ----------

FEATURES = (
    "in_context",
    "in_prompt",
    "min_reliability",
    "avg_logprob",
    "no_speech_prob",
    "n_primary",
    "n_secondary",
    "primary_vietnamese",
    "is_final",
    "delta_acoustic",
)


def region_features(
    u: UtteranceRegions,
    region: Region,
    k: int,
    context: frozenset[str],
    prompt_words: frozenset[str],
    rel: Mapping[str, float],
) -> list[float]:
    """Vector đặc trưng của một vùng, theo thứ tự `FEATURES`."""
    in_prompt_words = region.new_english & prompt_words
    primary_text = " ".join(region.primary)
    n_primary = len(region.primary)
    n_foreign = len(find_foreign_words(primary_text)) if primary_text else 0
    s = u.acoustic.get(k)
    delta = 0.0 if s is None or s[2] else s[1] - s[0]
    return [
        float(bool(region.new_english & context)),
        float(bool(in_prompt_words)),
        min((rel.get(w, 0.5) for w in in_prompt_words), default=1.0),
        float(u.v3_scores.get("avg_logprob", 0.0)),
        float(u.v3_scores.get("no_speech_prob", 0.0)),
        float(n_primary),
        float(len(region.secondary)),
        (n_primary - n_foreign) / n_primary if n_primary else 0.0,
        float(u.turn == "final"),
        delta,
    ]


@dataclass
class LogisticModel:
    """Hồi quy logistic đã học: trung bình và độ lệch để chuẩn hoá, trọng số (phần tử cuối là hệ số chặn)."""

    mean: list[float]
    std: list[float]
    weights: list[float]
    l2: float


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Giải hệ `a x = b` bằng khử Gauss có chọn trụ (ma trận nhỏ, chạy tất định)."""
    n = len(b)
    m = [[*row, b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        if abs(m[col][col]) < 1e-12:
            continue
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                if f:
                    for c in range(col, n + 1):
                        m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


def fit_logistic(x: Sequence[Sequence[float]], y: Sequence[int], l2: float, iters: int = 50) -> LogisticModel:
    """Học hồi quy logistic có phạt L2 (không phạt hệ số chặn) bằng phương pháp Newton trên đặc trưng đã chuẩn hoá."""
    d = len(x[0])
    mean = [sum(r[j] for r in x) / len(x) for j in range(d)]
    std = [math.sqrt(sum((r[j] - mean[j]) ** 2 for r in x) / len(x)) or 1.0 for j in range(d)]
    z = [[(r[j] - mean[j]) / std[j] for j in range(d)] + [1.0] for r in x]
    w = [0.0] * (d + 1)
    for _ in range(iters):
        grad = [0.0] * (d + 1)
        hess = [[0.0] * (d + 1) for _ in range(d + 1)]
        for zi, yi in zip(z, y, strict=True):
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, sum(a * b for a, b in zip(w, zi, strict=True))))))
            for j in range(d + 1):
                grad[j] += (p - yi) * zi[j]
                for k in range(d + 1):
                    hess[j][k] += p * (1 - p) * zi[j] * zi[k]
        for j in range(d):
            grad[j] += l2 * w[j]
            hess[j][j] += l2
        step = _solve(hess, grad)
        w = [a - b for a, b in zip(w, step, strict=True)]
        if max(abs(s) for s in step) < 1e-8:
            break
    return LogisticModel(mean, std, w, l2)


def predict(model: LogisticModel, features: Sequence[float]) -> float:
    """Xác suất "nên lấy chữ large-v3" của một vùng."""
    z = [(f - m) / s for f, m, s in zip(features, model.mean, model.std, strict=True)] + [1.0]
    t = sum(a * b for a, b in zip(model.weights, z, strict=True))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, t))))


def fold_of(sample_id: str, n_folds: int = N_FOLDS) -> int:
    """Phần (0 tới `n_folds - 1`) của một mẫu trong kiểm định chéo, tất định theo mã mẫu."""
    return zlib.crc32(sample_id.encode("utf-8")) % n_folds


def always_take_decider(_u: UtteranceRegions, _k: int, _region: Region) -> bool:
    """ALL: lấy large-v3 ở mọi vùng cần quyết định (bản ghép không lọc của lần 2)."""
    return True


def fold_dispatch_decider(per_fold: Mapping[int, RegionDecider], n_folds: int = N_FOLDS) -> RegionDecider:
    """Cách quyết định ngoài phần: vùng của mẫu thuộc phần f dùng `per_fold[f]` (học trên các phần còn lại)."""

    def decide(u: UtteranceRegions, k: int, region: Region) -> bool:
        return per_fold[fold_of(u.sample_id, n_folds)](u, k, region)

    return decide


def out_of_fold_deciders(
    utterances: Sequence[UtteranceRegions],
    fit: Callable[[Sequence[UtteranceRegions], int], Mapping[str, RegionDecider]],
    n_folds: int = N_FOLDS,
) -> dict[str, RegionDecider]:
    """Ghép các cách quyết định ngoài phần: với mỗi phần f, `fit(câu của các phần khác, f)` trả về
    tên -> cách quyết định; kết quả là tên -> cách quyết định gọi đúng bản của phần chứa mẫu."""
    per_fold: dict[str, dict[int, RegionDecider]] = {}
    for f in range(n_folds):
        train = [u for u in utterances if fold_of(u.sample_id, n_folds) != f]
        for name, decide in fit(train, f).items():
            per_fold.setdefault(name, {})[f] = decide
    return {name: fold_dispatch_decider(d, n_folds) for name, d in per_fold.items()}


def h_decider(
    model: LogisticModel,
    context: Mapping[str, frozenset[str]],
    prompt_words: frozenset[str],
    rel: Mapping[str, float],
) -> RegionDecider:
    """H: lấy large-v3 khi xác suất của bộ phân loại từ 0,5 trở lên."""

    def decide(u: UtteranceRegions, k: int, region: Region) -> bool:
        feats = region_features(u, region, k, context.get(u.sample_id, frozenset()), prompt_words, rel)
        return predict(model, feats) >= 0.5

    return decide


# ---------- I: hai phương án ----------


def alternatives(
    utterances: Sequence[UtteranceRegions],
    decide: RegionDecider,
    uncertain: Callable[[UtteranceRegions, int, Region], bool],
) -> tuple[dict[tuple[str, str], str], dict[str, int]]:
    """Chữ dạng hai phương án ở các vùng không chắc, và số vùng theo bốn loại.

    Trả về:
        Cặp (khoá câu -> chữ, đếm): đếm gồm `uncertain` (số vùng không chắc), `rescuable` (bản chốt sai, phương án còn
        lại ít lỗi hơn), `risky` (bản chốt ít lỗi hơn), `tie` (vùng hoà: hai phương án cùng số lỗi).
    """
    texts: dict[tuple[str, str], str] = {}
    counts = {"uncertain": 0, "rescuable": 0, "risky": 0, "tie": 0}
    for u in utterances:
        flags: dict[int, tuple[bool, bool]] = {}
        for k, label in zip(u.candidates, u.labels, strict=True):
            region = u.parts[k]
            assert isinstance(region, Region)
            take = decide(u, k, region)
            unsure = uncertain(u, k, region)
            flags[k] = (take, unsure)
            if not unsure:
                continue
            counts["uncertain"] += 1
            if label is None:
                counts["tie"] += 1
            elif int(take) == label:
                counts["risky"] += 1
            else:
                counts["rescuable"] += 1
        words: list[str] = []
        for k, part in enumerate(u.parts):
            if not isinstance(part, Region):
                words.append(part)
                continue
            take, unsure = flags.get(k, (False, False))
            chosen = part.secondary if take else part.primary
            other = part.primary if take else part.secondary
            if unsure:
                words.append("{" + " ".join(chosen) + " | " + " ".join(other) + "}")
            else:
                words.extend(chosen)
        texts[(u.sample_id, u.turn)] = " ".join(words)
    return texts, counts


def context_words_by_sample(
    samples: Sequence[Sample], catalog: Mapping[str, Mapping[str, Any]]
) -> dict[str, frozenset[str]]:
    """Từ điển ngữ cảnh của từng mẫu (`asr_merge.sample_context_words`)."""
    from lifemate.eval.asr_merge import sample_context_words

    return {s.id: sample_context_words(s, catalog) for s in samples}
