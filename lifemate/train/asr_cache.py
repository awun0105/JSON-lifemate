"""Transcribe a split once with the run-time ASR and save the texts for SFT data (SPEC 6.2.1).

The cache lets `sft_data` build rows from real ASR output without running the ASR model again, and fixes the
transcripts so every rebuild of the data is identical. It holds exactly what the agent would see at run time:
the same `[asr]` section of the run config (engine, model, decoding options, sanitizer).

Output: one JSON object `{sample_id: {"final": str, "history": [str | null, ...]}}` plus `<out>.meta.json`
(config, ids file, counts, seconds). Needs a GPU box with the ASR extra.

Usage:
  python -m lifemate.train.asr_cache --config configs/v0.toml --env kaggle --out data/asr/train_asr.json \\
      --ids configs/splits/train.txt configs/splits/val_seen.txt configs/splits/val_unseen.txt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import structlog

from lifemate.core.config import load_config, load_profile
from lifemate.core.paths import REPO_ROOT
from lifemate.core.types import Sample, Transcriber, Transcript
from lifemate.io.data import load_split, read_ids

log = structlog.get_logger(__name__)


def to_entry(t: Transcript) -> dict[str, Any]:
    """Cache entry for one sample."""
    return {"final": t.final, "history": list(t.history)}


class CachedTranscriber:
    """Transcriber that replays a saved ASR cache; raises KeyError for a sample that is not in it."""

    source = "asr"

    def __init__(self, cache: Mapping[str, Mapping[str, Any]]) -> None:
        self._cache = cache

    @classmethod
    def from_file(cls, path: Path) -> CachedTranscriber:
        """Load `path` written by this module."""
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def has(self, sample_id: str) -> bool:
        """Whether the cache holds `sample_id`."""
        return sample_id in self._cache

    def transcribe(self, sample: Sample) -> Transcript:
        """Saved transcript of `sample`."""
        entry = self._cache[sample.id]
        history = tuple(entry.get("history") or ())
        if len(history) != len(sample.history):
            raise ValueError(f"{sample.id}: cache has {len(history)} history turns, sample has {len(sample.history)}")
        return Transcript(final=str(entry["final"]), history=history, seconds=0.0)

    def warmup(self) -> None:
        """Nothing to warm up."""


def build_cache(samples: Sequence[Sample], transcriber: Transcriber) -> tuple[dict[str, Any], float]:
    """Transcribe every sample in order; returns (cache, total seconds)."""
    transcriber.warmup()
    cache: dict[str, Any] = {}
    t0 = time.perf_counter()
    for i, sample in enumerate(samples, 1):
        cache[sample.id] = to_entry(transcriber.transcribe(sample))
        if i % 100 == 0:
            log.info("asr_cache_progress", done=i, total=len(samples), seconds=round(time.perf_counter() - t0, 1))
    return cache, time.perf_counter() - t0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI arguments."""
    ap = argparse.ArgumentParser(prog="python -m lifemate.train.asr_cache", description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=Path, help="run config whose [asr] section is used")
    ap.add_argument("--env", default="local", help="env profile name or path")
    ap.add_argument("--ids", required=True, type=Path, nargs="+", help="one or more files with one sample id per line")
    ap.add_argument("--out", required=True, type=Path, help="output JSON (e.g. data/asr/train_asr.json)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--root", help="data root override")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None, repo_root: Path = REPO_ROOT) -> int:
    """Entry point."""
    from lifemate.backends.asr import ASRTranscriber

    args = parse_args(argv)
    cfg = load_config(args.config if args.config.is_absolute() else repo_root / args.config)
    profile = load_profile(args.env, repo_root).with_overrides(root=args.root)
    root = Path(profile.data_root)
    root = root if root.is_absolute() else repo_root / root
    ids: list[str] = []
    for path in args.ids:
        ids += [i for i in read_ids(path if path.is_absolute() else repo_root / path) if i not in ids]
    samples = load_split(root, args.split, ids)
    asr_cfg = dataclasses.replace(cfg.asr, backend="asr")
    cache, seconds = build_cache(samples, ASRTranscriber(asr_cfg, data_root=root, split=args.split))
    out = args.out if args.out.is_absolute() else repo_root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cache, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    meta = {
        "asr": dataclasses.asdict(asr_cfg),
        "ids": [str(p) for p in args.ids],
        "split": args.split,
        "n_samples": len(cache),
        "seconds": round(seconds, 2),
    }
    out.with_name(out.name + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"samples={len(cache)} seconds={seconds:.1f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
