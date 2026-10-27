"""ASR fine-tune pairs (wav, verbatim audio_text) for the final turn and every user turn in history (SPEC 6.4.1)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import structlog

from lifemate.core.config import load_profile
from lifemate.core.envinfo import code_version
from lifemate.core.paths import REPO_ROOT
from lifemate.core.types import Sample
from lifemate.io.data import load_split, read_ids
from lifemate.train.common import abs_path, sha256_file
from lifemate.train.sft_data import eval_ids, write_jsonl

log = structlog.get_logger(__name__)

FINAL_TURN = "final"


def sample_pairs(sample: Sample, split_tag: str) -> list[dict[str, Any]]:
    """Rows `{id, turn, audio, text, split}`: history user turns `u1, u2, …` then the final turn (ASR order, 6.1)."""
    rows: list[dict[str, Any]] = []
    k = 0
    for turn in sample.history:
        if turn.role != "user":
            continue
        k += 1
        if turn.audio is None or turn.audio_text is None:
            raise ValueError(f"{sample.id}: user turn u{k} lacks audio or audio_text")
        rows.append(
            {"id": sample.id, "turn": f"u{k}", "audio": turn.audio, "text": turn.audio_text, "split": split_tag}
        )
    if sample.audio_text is None:
        raise ValueError(f"{sample.id}: final turn has no audio_text")
    rows.append(
        {"id": sample.id, "turn": FINAL_TURN, "audio": sample.audio, "text": sample.audio_text, "split": split_tag}
    )
    return rows


def build_pairs(samples: Sequence[Sample], split_tag: str) -> list[dict[str, Any]]:
    """Pairs of every sample in order; audio paths stay relative to the split folder (not copied)."""
    return [row for s in samples for row in sample_pairs(s, split_tag)]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI arguments."""
    ap = argparse.ArgumentParser(prog="python -m lifemate.train.asr_data", description="Build ASR pairs (6.4.1).")
    ap.add_argument("--env", default="local", help="env profile name or path")
    ap.add_argument("--ids", required=True, type=Path, help="file with one sample id per line")
    ap.add_argument("--out", required=True, type=Path, help="output jsonl (e.g. data/asr/v1.jsonl)")
    ap.add_argument("--split", default="train", help="data split folder holding the ids")
    ap.add_argument("--tag", help="value of the `split` field (default: ids file stem, e.g. train)")
    ap.add_argument("--root", help="data root override")
    ap.add_argument("--allow-eval", action="store_true", help="allow val ids (measurement files only)")
    return ap.parse_args(argv)


def run(args: argparse.Namespace, repo_root: Path = REPO_ROOT, argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Build one ASR pair file plus `<out>.meta.json`; returns the meta dict."""
    profile = load_profile(args.env, repo_root).with_overrides(root=args.root)
    root = abs_path(profile.data_root, repo_root)
    ids_path = abs_path(args.ids, repo_root)
    ids = read_ids(ids_path)
    leaked = sorted(set(ids) & eval_ids(repo_root))
    if leaked and not args.allow_eval:
        raise SystemExit(f"refusing: {len(leaked)} ids are in val_seen/val_unseen (e.g. {leaked[:3]}); --allow-eval")
    order = {sid: i for i, sid in enumerate(ids)}
    samples = sorted(load_split(root, args.split, ids), key=lambda s: order[s.id])
    rows = build_pairs(samples, args.tag or ids_path.stem)
    out = abs_path(args.out, repo_root)
    write_jsonl(out, rows)
    version, dirty = code_version(repo_root)
    meta = {
        "file": out.name,
        "sha256": sha256_file(out),
        "argv": list(argv if argv is not None else sys.argv),
        "code_version": version,
        "code_dirty": dirty,
        "split": args.split,
        "ids": ids,
        "n_samples": len(samples),
        "n_rows": len(rows),
    }
    Path(f"{out}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("asr_data_done", rows=len(rows), samples=len(samples))
    return meta


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    meta = run(parse_args(argv), argv=argv)
    print(f"rows={meta['n_rows']} samples={meta['n_samples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
