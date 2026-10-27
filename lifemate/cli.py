"""Command line entry: parse args, load config and profile, build, run (SPEC 4.11)."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
from collections.abc import Sequence
from pathlib import Path

import structlog

from lifemate import factory
from lifemate.config import EnvProfile, load_config, load_profile
from lifemate.data import load_split, read_ids
from lifemate.envinfo import detect
from lifemate.runner import Runner, RunSummary
from lifemate.store import RunStore

log = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _abs(path: str, repo_root: Path) -> Path:
    """Expand `~`; relative paths are taken from the repo root (SPEC 4.16 local profile)."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else repo_root / p


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI arguments."""
    ap = argparse.ArgumentParser(prog="python -m lifemate.cli", description="Run the LifeMate agent on one split.")
    ap.add_argument("--config", required=True, type=Path, help="run config TOML (e.g. configs/v0.toml)")
    ap.add_argument("--env", default="local", help="env profile name (configs/env/<name>.toml) or path")
    ap.add_argument("--split", default="train", help="split folder under the data root")
    ap.add_argument("--ids", type=Path, help="file with one sample id per line; default: every sample of the split")
    ap.add_argument("--name", help="run folder name under the profile's runs_dir (default: timestamp)")
    ap.add_argument("--out", help="full run folder path; overrides runs_dir/name")
    ap.add_argument("--root", help="data root override (folder holding the split folders)")
    ap.add_argument("--resume", action="store_true", help="continue a run folder whose trace.jsonl exists")
    ap.add_argument("--limit", type=int, help="run only the first N ids (install check)")
    ap.add_argument("--train-mode", action="store_true", help="strip held-out apps from prompts (SFT prompt checks)")
    return ap.parse_args(argv)


def run(args: argparse.Namespace, repo_root: Path = REPO_ROOT) -> RunSummary:
    """Do one run from parsed arguments."""
    config = load_config(_abs(str(args.config), repo_root))
    raw_profile = load_profile(args.env, repo_root).with_overrides(root=args.root)
    profile: EnvProfile = dataclasses.replace(
        raw_profile,
        data_root=str(_abs(raw_profile.data_root, repo_root)),
        runs_dir=str(_abs(raw_profile.runs_dir, repo_root)),
    )
    env_info = detect(repo_root)
    root = Path(profile.data_root)
    ids_file = _abs(str(args.ids), repo_root) if args.ids is not None else None
    ids = read_ids(ids_file) if ids_file is not None else None
    samples = load_split(root, args.split, ids)
    if args.limit is not None:
        samples = sorted(samples, key=lambda s: s.id)[: args.limit]
    has_labels = all(s.label is not None for s in samples)

    built = factory.build(config, profile, repo_root=repo_root, split=args.split, env_info=env_info, samples=samples)
    start = dt.datetime.now(dt.UTC)
    if args.out is not None:
        run_dir = _abs(args.out, repo_root)
        index_csv = run_dir.parent / "index.csv"
    else:
        run_dir = Path(profile.runs_dir) / (args.name or start.strftime("%Y-%m-%d_%H%M%S"))
        index_csv = Path(profile.runs_dir) / "index.csv"
    store = RunStore(run_dir, resume=args.resume)
    record = {
        "config": dataclasses.asdict(config),
        "dtype": {"config": config.llm.dtype, "resolved": built.dtype},
        "env_info": dataclasses.asdict(env_info),
        "profile": {"data_root": profile.data_root, "runs_dir": profile.runs_dir, "persist": profile.persist},
        "prompt_sha256": built.prompt_sha256,
        "split": args.split,
        "ids_file": str(ids_file) if ids_file is not None else None,
        "limit": args.limit,
        "train_mode": args.train_mode,
        "resume": args.resume,
        "started_at": start.isoformat(timespec="seconds"),
    }
    runner = Runner(
        built.agent,
        built.loader,
        store,
        repo_root=repo_root,
        index_csv=index_csv,
        train_mode=args.train_mode,
    )
    return runner.run(samples, ids_file=ids_file, split_has_labels=has_labels, config_record=record)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; prints the summary path and the score if any."""
    result = run(parse_args(argv))
    print(f"summary: {result.run_dir / 'summary.json'}")
    if result.score is not None:
        for mode, value in result.score["by_mode"].items():
            print(f"score [{mode}]: {value:.2f}")
    if result.report is not None:
        print(f"report: {result.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
