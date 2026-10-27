"""ASR measurement: WER, CER, entity miss rate, latency (SPEC 6.4.2)."""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from lifemate.core.paths import REPO_ROOT
from lifemate.core.types import Sample, Transcriber, Transcript

_PUNCT = re.compile(r"[^\w\s]|_")


def _fold(s: str) -> str:
    """Lower-case, Vietnamese diacritics removed, spaces collapsed.

    Copied from `_fold` in the competition package zaic_tools (train/zaic_tools/utils.py) so lifemate never imports
    zaic_tools; keep in sync with that file.
    """
    t = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())


def _folded_tokens(s: str) -> str:
    """Folded text as space-padded word tokens, so matches respect word boundaries ("huy" not in "chuyen")."""
    return " " + " ".join(re.findall(r"[^\W_]+", _fold(s))) + " "


def normalize_text(s: str) -> str:
    """Lower-case, NFC, punctuation removed, spaces collapsed."""
    t = unicodedata.normalize("NFC", s.lower())
    return " ".join(_PUNCT.sub(" ", t).split())


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _rate(errors: int, n: int) -> float:
    if n == 0:
        return 0.0 if errors == 0 else 1.0
    return errors / n


def wer(ref: str, hyp: str) -> float:
    """Word error rate on normalized text."""
    r, h = normalize_text(ref).split(), normalize_text(hyp).split()
    return _rate(_edit_distance(r, h), len(r))


def cer(ref: str, hyp: str) -> float:
    """Character error rate on normalized text (spaces count as characters)."""
    r, h = normalize_text(ref), normalize_text(hyp)
    return _rate(_edit_distance(list(r), list(h)), len(r))


def _values(obj: Any) -> Iterator[str]:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, str):
        if obj.strip():
            yield obj
    elif isinstance(obj, int | float):
        yield str(int(obj)) if isinstance(obj, float) and obj.is_integer() else str(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _values(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            yield from _values(v)


def entity_values(sample: Sample) -> list[str]:
    """String and numeric argument values of the gold calls that appear (folded) in the gold final text."""
    if sample.label is None or sample.audio_text is None:
        return []
    gold = _folded_tokens(sample.audio_text)
    out: list[str] = []
    for call in sample.label.get("tool_calls") or []:
        if call.get("name") == "sys.ask_missing_info":
            continue
        for v in _values(call.get("arguments") or {}):
            fv = _folded_tokens(v)
            if fv.strip() and fv in gold and fv.strip() not in out:
                out.append(fv.strip())
    return out


@dataclass(frozen=True, slots=True)
class EntityDetail:
    """Per-sample entity check."""

    id: str
    entities: tuple[str, ...]
    missed: tuple[str, ...]


def entity_miss_rate(
    samples: Sequence[Sample], transcripts: Sequence[Transcript]
) -> tuple[int, int, float, list[EntityDetail]]:
    """(missed, total, rate, details): gold entities heard in audio_text but absent from the folded ASR final text."""
    missed = total = 0
    details: list[EntityDetail] = []
    for s, t in zip(samples, transcripts, strict=True):
        ents = entity_values(s)
        hyp = _folded_tokens(t.final)
        miss = tuple(e for e in ents if f" {e} " not in hyp)
        missed += len(miss)
        total += len(ents)
        details.append(EntityDetail(s.id, tuple(ents), miss))
    return missed, total, _rate(missed, total), details


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100); 0.0 for no values."""
    if not values:
        return 0.0
    xs = sorted(values)
    k = max(1, math.ceil(q / 100 * len(xs)))
    return xs[k - 1]


@dataclass(slots=True)
class AsrReport:
    """Aggregated ASR metrics."""

    n_samples: int
    n_utterances: int
    wer: float
    cer: float
    entity_missed: int
    entity_total: int
    entity_miss_rate: float
    p50_s: float
    p95_s: float
    max_s: float
    entity_details: list[EntityDetail] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict; details only for samples with misses."""
        d: dict[str, Any] = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "entity_details"}
        d["entity_misses"] = [
            {"id": e.id, "entities": list(e.entities), "missed": list(e.missed)}
            for e in self.entity_details
            if e.missed
        ]
        return d

    def to_text(self) -> str:
        """Short human summary."""
        return (
            f"samples={self.n_samples} utterances={self.n_utterances}\n"
            f"WER={self.wer:.4f} CER={self.cer:.4f}\n"
            f"entity_miss_rate={self.entity_miss_rate:.4f} ({self.entity_missed}/{self.entity_total})\n"
            f"seconds p50={self.p50_s:.3f} p95={self.p95_s:.3f} max={self.max_s:.3f}"
        )


def evaluate(samples: Sequence[Sample], transcriber: Transcriber) -> AsrReport:
    """Transcribe every sample; corpus WER/CER over final and history user turns (gold must exist)."""
    transcriber.warmup()
    transcripts = [transcriber.transcribe(s) for s in samples]
    w_err = w_n = c_err = c_n = n_utt = 0
    for s, t in zip(samples, transcripts, strict=True):
        pairs: list[tuple[str, str]] = []
        for turn, hyp in zip(s.history, t.history, strict=True):
            if turn.role == "user" and turn.audio_text is not None and hyp is not None:
                pairs.append((turn.audio_text, hyp))
        if s.audio_text is not None:
            pairs.append((s.audio_text, t.final))
        for ref, hyp in pairs:
            r, h = normalize_text(ref), normalize_text(hyp)
            w_err += _edit_distance(r.split(), h.split())
            w_n += len(r.split())
            c_err += _edit_distance(list(r), list(h))
            c_n += len(r)
            n_utt += 1
    missed, total, rate, details = entity_miss_rate(samples, transcripts)
    secs = [t.seconds for t in transcripts]
    return AsrReport(
        n_samples=len(samples),
        n_utterances=n_utt,
        wer=_rate(w_err, w_n),
        cer=_rate(c_err, c_n),
        entity_missed=missed,
        entity_total=total,
        entity_miss_rate=rate,
        p50_s=percentile(secs, 50),
        p95_s=percentile(secs, 95),
        max_s=max(secs, default=0.0),
        entity_details=details,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: python -m lifemate.eval.eval_asr --config ... --env ... --split train --ids <ids.txt> --out x.json."""
    from lifemate.backends.asr import ASRTranscriber, GoldTranscriber
    from lifemate.core.config import load_config, load_profile
    from lifemate.io.data import load_split

    ap = argparse.ArgumentParser(prog="lifemate.eval.eval_asr")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--ids", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    repo_root = REPO_ROOT
    cfg = load_config(args.config)
    profile = load_profile(args.env, repo_root).with_overrides(root=args.root)
    root = Path(profile.data_root)
    ids = None
    if args.ids is not None:
        ids = [x.strip() for x in args.ids.read_text(encoding="utf-8").splitlines() if x.strip()]
    samples = load_split(root, args.split, ids)
    # factory.TRANSCRIBERS only receives Config (no data_root/split), so build directly here.
    transcriber: Transcriber
    if cfg.asr.backend == "gold":
        transcriber = GoldTranscriber()
    else:
        transcriber = ASRTranscriber(cfg.asr, data_root=root, split=args.split)
    report = evaluate(samples, transcriber)
    print(report.to_text())
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
