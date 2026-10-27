"""One command: ASR pairs → fine-tune → choose checkpoint by entity-miss rate → merge → measure (SPEC 6.4.5 item 6).

Stage outputs live in `<work>/pipeline_progress.json`; `--from-stage` resumes. Train, select, merge and measure need
the GPU extras and are not run in tests.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from lifemate.core.config import load_profile
from lifemate.core.paths import REPO_ROOT
from lifemate.train import asr, asr_data
from lifemate.train.common import CheckpointInfo, CheckpointManager, abs_path, write_json_atomic
from lifemate.train.sft import read_jsonl

log = structlog.get_logger(__name__)

STAGES = ("data", "train", "select", "merge", "measure")


@dataclass(frozen=True, slots=True)
class ASRPipelineContext:
    """Paths and options shared by every stage."""

    repo_root: Path
    config_path: Path
    cfg: asr.ASRTrainConfig
    env: str
    name: str
    work_dir: Path
    data_root: Path
    data_split: str = "train"

    @property
    def state_path(self) -> Path:
        """Where stage outputs are remembered (same layout as llm_pipeline)."""
        return self.work_dir / "pipeline_progress.json"

    def val_file(self, split: str) -> Path:
        """`data/asr/<v>_<split>.jsonl` next to the train file."""
        train = abs_path(self.cfg.data.train, self.repo_root)
        return train.with_name(f"{train.stem}_{split}{train.suffix}")


def stage_data(ctx: ASRPipelineContext, state: dict[str, Any]) -> None:
    """Train pairs from configs/splits/train.txt; val_unseen and val_seen pairs for measurement only."""
    jobs = [(abs_path(ctx.cfg.data.train, ctx.repo_root), "configs/splits/train.txt", False)]
    jobs += [(ctx.val_file(s), f"configs/splits/{s}.txt", True) for s in ("val_seen", "val_unseen")]
    out: dict[str, Any] = {}
    for path, ids, allow_eval in jobs:
        argv = ["--env", ctx.env, "--ids", ids, "--out", str(path), "--split", ctx.data_split]
        argv += ["--allow-eval"] if allow_eval else []
        meta = asr_data.run(asr_data.parse_args(argv), ctx.repo_root, argv=["lifemate.train.asr_data", *argv])
        out[str(path)] = {"sha256": meta["sha256"], "n_rows": meta["n_rows"]}
    state["data"] = out


def stage_train(ctx: ASRPipelineContext, state: dict[str, Any]) -> None:
    """Fine-tune; resumes from the latest checkpoint of an interrupted run."""
    out = ctx.work_dir / "asr"
    latest = CheckpointManager(out / "checkpoints").latest()
    asr.train(
        ctx.cfg,
        repo_root=ctx.repo_root,
        data_root=ctx.data_root,
        split=ctx.data_split,
        out_dir=out,
        resume_from=latest.path if latest else None,
    )
    state["train"] = {"out_dir": str(out)}


def _merged_or_raw(ctx: ASRPipelineContext, ckpt: CheckpointInfo) -> Path:
    if ctx.cfg.method.kind != "lora":
        return ckpt.path
    out = ctx.work_dir / "merged" / ckpt.path.name
    if not out.is_dir():
        asr.merge_lora(ctx.cfg.base.model, ckpt.path, out, revision=ctx.cfg.base.revision or None)
    return out


def stage_select(ctx: ASRPipelineContext, state: dict[str, Any]) -> None:
    """Score every kept checkpoint on val_unseen pairs; choose by entity-miss rate then WER."""
    mgr = CheckpointManager(ctx.work_dir / "asr" / "checkpoints", metric_name=asr.EVAL_METRIC)
    rows = read_jsonl(ctx.val_file("val_unseen"))
    split_dir = ctx.data_root / ctx.data_split
    processor = ctx.work_dir / "asr" / "processor"

    def transcribe(c: CheckpointInfo) -> list[str]:
        return asr.transcribe_rows(_merged_or_raw(ctx, c), processor, rows, split_dir)

    chosen, table = asr.select_checkpoint(mgr.all(), transcribe, rows)
    state["select"] = {"chosen": str(chosen.path), "step": chosen.step, "table": table}


def stage_merge(ctx: ASRPipelineContext, state: dict[str, Any]) -> None:
    """Merge the chosen checkpoint (LoRA) or keep it as is (full)."""
    path = Path(state["select"]["chosen"])
    state["merge"] = {"model": str(_merged_or_raw(ctx, CheckpointInfo(path, state["select"]["step"], None)))}


def stage_measure(ctx: ASRPipelineContext, state: dict[str, Any]) -> None:
    """WER / entity-miss rate of the final model on val_seen and val_unseen; written to <work>/measure.json."""
    model = Path(state["merge"]["model"])
    processor = model if (model / "preprocessor_config.json").is_file() else ctx.work_dir / "asr" / "processor"
    result: dict[str, Any] = {}
    for split in ("val_seen", "val_unseen"):
        rows = read_jsonl(ctx.val_file(split))
        hyps = asr.transcribe_rows(model, processor, rows, ctx.data_root / ctx.data_split)
        result[split] = asr.score_checkpoint(rows, hyps)
    (ctx.work_dir / "measure.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    state["measure"] = result


STAGE_FUNCS: dict[str, Callable[[ASRPipelineContext, dict[str, Any]], None]] = {
    "data": stage_data,
    "train": stage_train,
    "select": stage_select,
    "merge": stage_merge,
    "measure": stage_measure,
}


def run_pipeline(ctx: ASRPipelineContext, *, from_stage: str = STAGES[0], to_stage: str = STAGES[-1]) -> dict[str, Any]:
    """Run stages `from_stage..to_stage`, saving state after each."""
    state: dict[str, Any] = {}
    if ctx.state_path.is_file():
        state = dict(json.loads(ctx.state_path.read_text(encoding="utf-8")))
    for stage in STAGES[STAGES.index(from_stage) : STAGES.index(to_stage) + 1]:
        log.info("asr_pipeline_stage_start", stage=stage, name=ctx.name)
        STAGE_FUNCS[stage](ctx, state)
        state["done"] = [*[s for s in state.get("done", []) if s != stage], stage]
        write_json_atomic(ctx.state_path, state)
    return state


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI arguments."""
    ap = argparse.ArgumentParser(prog="python -m lifemate.train.asr_pipeline", description="ASR pipeline (6.4).")
    ap.add_argument("--config", required=True, type=Path, help="configs/train/asr_vN.toml")
    ap.add_argument("--env", default="local")
    ap.add_argument("--name", help="work folder name (default: config stem)")
    ap.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    ap.add_argument("--to-stage", choices=STAGES, default=STAGES[-1])
    ap.add_argument("--split", default="train", help="data split folder holding the audio")
    return ap.parse_args(argv)


def make_context(args: argparse.Namespace, repo_root: Path = REPO_ROOT) -> ASRPipelineContext:
    """Context from CLI arguments."""
    config_path = abs_path(args.config, repo_root)
    cfg = asr.load_asr_train_config(config_path)
    profile = load_profile(args.env, repo_root)
    name = args.name or f"asr-{config_path.stem}"
    work = abs_path(profile.runs_dir, repo_root) / "train" / name
    return ASRPipelineContext(
        repo_root, config_path, cfg, args.env, name, work, abs_path(profile.data_root, repo_root), args.split
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    ctx = make_context(args)
    state = run_pipeline(ctx, from_stage=args.from_stage, to_stage=args.to_stage)
    print(f"asr pipeline {ctx.name}: done {state.get('done')} state={ctx.state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
