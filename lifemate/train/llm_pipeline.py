"""One command: SFT data → train → 2 best checkpoints → merge → agent eval → choose → register (SPEC 6.2.3, 6.2.6).

Each stage is a function over a `PipelineContext`; progress is kept in `<work>/pipeline_progress.json` so
`--from-stage` can resume. The train and merge stages need the GPU extras and are not run in tests.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from lifemate.core.config import load_profile
from lifemate.core.envinfo import detect
from lifemate.core.paths import REPO_ROOT
from lifemate.train import sft, sft_data
from lifemate.train.common import CheckpointManager, abs_path, sha256_file, write_json_atomic, write_train_config

log = structlog.get_logger(__name__)

STAGES = ("data", "train", "select", "merge", "eval", "choose", "register")
REGISTRY_FIELDS = ("name", "base", "data_version", "checkpoint", "val_seen", "val_unseen", "location")
EVAL_SPLITS = ("val_seen", "val_unseen")
N_CANDIDATES = 2


@dataclass(frozen=True, slots=True)
class PipelineContext:
    """Paths and options shared by every stage."""

    repo_root: Path
    config_path: Path
    cfg: sft.LLMTrainConfig
    env: str
    name: str
    work_dir: Path
    limit: int
    data_split: str = "train"

    @property
    def state_path(self) -> Path:
        """Where stage outputs are remembered."""
        return self.work_dir / "pipeline_progress.json"

    @property
    def model_dir(self) -> Path:
        """`artifacts/models/<name>/` (in git: configs and scores, never weights)."""
        return self.repo_root / "artifacts" / "models" / self.name


def load_state(ctx: PipelineContext) -> dict[str, Any]:
    """Stage outputs so far."""
    if not ctx.state_path.is_file():
        return {}
    return dict(json.loads(ctx.state_path.read_text(encoding="utf-8")))


def save_state(ctx: PipelineContext, state: Mapping[str, Any]) -> None:
    """Atomically write the stage outputs."""
    write_json_atomic(ctx.state_path, dict(state))


def data_version(cfg: sft.LLMTrainConfig) -> str:
    """`v1` from `data/sft/v1.jsonl`."""
    return Path(cfg.data.train).stem


def default_name(cfg: sft.LLMTrainConfig, registry: Path) -> str:
    """`ft-<base>-sft<v>-r<nn>` with the next free run number in the registry (SPEC 6.2.6)."""
    base = re.sub(r"[^a-z0-9]+", "", cfg.base.model.rsplit("/", 1)[-1].lower()) or "base"
    version = re.sub(r"\D", "", data_version(cfg)) or "0"
    prefix = f"ft-{base}-sft{version}-r"
    used = {r["name"] for r in read_registry(registry)}
    n = 1
    while f"{prefix}{n:02d}" in used:
        n += 1
    return f"{prefix}{n:02d}"


def read_registry(path: Path) -> list[dict[str, str]]:
    """Rows of artifacts/models/registry.csv."""
    if not path.is_file():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_registry(path: Path, row: Mapping[str, Any]) -> None:
    """Append one row (header written when the file is new or empty)."""
    new = not path.is_file() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in REGISTRY_FIELDS})


# ---------------------------------------------------------------- stages


def stage_data(ctx: PipelineContext, state: dict[str, Any]) -> None:
    """Build train, eval_seen and eval_unseen SFT files."""
    jobs = [
        (ctx.cfg.data.train, "configs/splits/train.txt", False),
        (ctx.cfg.data.eval_seen, "configs/splits/val_seen.txt", True),
        (ctx.cfg.data.eval_unseen, "configs/splits/val_unseen.txt", True),
    ]
    metas: dict[str, Any] = {}
    for out, ids, allow_eval in jobs:
        argv = ["--config", str(ctx.config_path), "--env", ctx.env, "--ids", ids, "--out", out]
        argv += ["--split", ctx.data_split] + (["--allow-eval"] if allow_eval else [])
        meta = sft_data.run(sft_data.parse_args(argv), ctx.repo_root, argv=["lifemate.train.sft_data", *argv])
        metas[out] = {k: meta[k] for k in ("sha256", "n_rows", "n_dropped", "n_replay_warnings")}
    state["data"] = metas


def stage_train(ctx: PipelineContext, state: dict[str, Any]) -> None:
    """QLoRA SFT; resumes from the latest checkpoint of an interrupted run."""
    out = ctx.work_dir / "sft"
    latest = CheckpointManager(out / "checkpoints").latest()
    sft.train(ctx.cfg, repo_root=ctx.repo_root, out_dir=out, resume_from=latest.path if latest else None)
    state["train"] = {"out_dir": str(out)}


def stage_select(ctx: PipelineContext, state: dict[str, Any]) -> None:
    """The 2 checkpoints with the lowest eval_unseen loss."""
    mgr = CheckpointManager(ctx.work_dir / "sft" / "checkpoints", metric_name=sft.EVAL_METRIC)
    best = mgr.best_k(N_CANDIDATES)
    if not best:
        raise RuntimeError(f"no checkpoint with {sft.EVAL_METRIC} under {mgr.directory}")
    state["select"] = [{"path": str(c.path), "step": c.step, "loss": c.metric} for c in best]


def stage_merge(ctx: PipelineContext, state: dict[str, Any]) -> None:
    """Merge each candidate adapter into the bf16 base model."""
    merged = []
    for cand in state["select"]:
        out = ctx.work_dir / "merged" / Path(cand["path"]).name
        sft.merge_adapter(ctx.cfg.base.model, Path(cand["path"]), out, revision=ctx.cfg.base.revision or None)
        merged.append({**cand, "merged": str(out)})
    state["merge"] = merged


def _dump_toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ", ".join(_dump_toml_value(x) for x in v) + "]"
    if isinstance(v, Mapping):
        return "{" + ", ".join(f"{k} = {_dump_toml_value(x)}" for k, x in v.items()) + "}"
    raise TypeError(f"unsupported TOML value: {v!r}")


def dump_toml(data: Mapping[str, Mapping[str, Any]]) -> str:
    """Serialise top-level tables of scalars, lists and inline tables (enough for configs/v0.toml)."""
    out: list[str] = []
    for section, table in data.items():
        out.append(f"[{section}]")
        out += [f"{k} = {_dump_toml_value(v)}" for k, v in table.items()]
        out.append("")
    return "\n".join(out)


def run_config_for(ctx: PipelineContext, model_dir: str, out: Path) -> Path:
    """Copy of the run config with `[llm].model` pointing at a merged model."""
    with open(abs_path(ctx.cfg.base.run_config, ctx.repo_root), "rb") as f:
        raw = tomllib.load(f)
    raw["llm"]["model"] = model_dir
    raw["llm"]["revision"] = ""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_toml(raw), encoding="utf-8")
    return out


Runner = Callable[[Sequence[str], Path], int]


def _subprocess_runner(cmd: Sequence[str], cwd: Path) -> int:
    return subprocess.run(list(cmd), cwd=cwd, check=False).returncode


def eval_command(ctx: PipelineContext, run_config: Path, split_ids: str, run_dir: Path) -> list[str]:
    """`python -m lifemate.run.cli` on one val id file with `--limit`."""
    return [
        sys.executable,
        "-m",
        "lifemate.run.cli",
        "--config",
        str(run_config),
        "--env",
        ctx.env,
        "--split",
        ctx.data_split,
        "--ids",
        split_ids,
        "--limit",
        str(ctx.limit),
        "--out",
        str(run_dir),
    ]


def read_run_score(run_dir: Path) -> float | None:
    """`score.total` from a run's summary.json."""
    summary = run_dir / "summary.json"
    if not summary.is_file():
        return None
    score = json.loads(summary.read_text(encoding="utf-8")).get("score")
    return None if not score else float(score["total"])


def stage_eval(ctx: PipelineContext, state: dict[str, Any], runner: Runner = _subprocess_runner) -> None:
    """Run the real agent with each merged candidate on val_seen and val_unseen (`--limit`)."""
    results = []
    for cand in state["merge"]:
        tag = Path(cand["merged"]).name
        run_cfg = run_config_for(ctx, cand["merged"], ctx.work_dir / "eval" / tag / "run.toml")
        scores: dict[str, Any] = {}
        for split in EVAL_SPLITS:
            run_dir = ctx.work_dir / "eval" / tag / split
            code = runner(eval_command(ctx, run_cfg, f"configs/splits/{split}.txt", run_dir), ctx.repo_root)
            if code != 0:
                raise RuntimeError(f"lifemate.run.cli failed ({code}) for {tag} on {split}")
            scores[split] = read_run_score(run_dir)
            scores[f"{split}_run"] = str(run_dir)
        results.append({**cand, **scores})
    state["eval"] = results


def choose(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Highest val_unseen score; ties broken by val_seen (SPEC 6.2.3 item 3)."""

    def key(c: Mapping[str, Any]) -> tuple[float, float]:
        return (float(c.get("val_unseen") or 0.0), float(c.get("val_seen") or 0.0))

    if not candidates:
        raise ValueError("no candidates")
    return max(candidates, key=key)


def stage_choose(ctx: PipelineContext, state: dict[str, Any]) -> None:
    """Pick the checkpoint to keep."""
    state["choose"] = dict(choose(state["eval"]))


def stage_register(ctx: PipelineContext, state: dict[str, Any]) -> None:
    """Write artifacts/models/<name>/{train_config.json, eval.json, metrics.jsonl}; append the registry."""
    chosen = state["choose"]
    ctx.model_dir.mkdir(parents=True, exist_ok=True)
    data_sha = {
        k: sha256_file(abs_path(v, ctx.repo_root))
        for k, v in dataclasses.asdict(ctx.cfg.data).items()
        if abs_path(v, ctx.repo_root).is_file()
    }
    write_train_config(
        ctx.model_dir / "train_config.json",
        dataclasses.asdict(ctx.cfg),
        detect(ctx.repo_root),
        data_sha,
        name=ctx.name,
        seed=ctx.cfg.train.seed,
        checkpoint=chosen["path"],
        step=chosen["step"],
        eval_unseen_loss=chosen["loss"],
        candidates=state.get("eval", []),
    )
    eval_payload = {"name": ctx.name, "limit": ctx.limit, "chosen": chosen, "candidates": state.get("eval", [])}
    (ctx.model_dir / "eval.json").write_text(json.dumps(eval_payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    metrics = ctx.work_dir / "sft" / "metrics.jsonl"
    if metrics.is_file():
        (ctx.model_dir / "metrics.jsonl").write_bytes(metrics.read_bytes())
    append_registry(
        ctx.repo_root / "artifacts" / "models" / "registry.csv",
        {
            "name": ctx.name,
            "base": ctx.cfg.base.model,
            "data_version": data_version(ctx.cfg),
            "checkpoint": Path(chosen["path"]).name,
            "val_seen": chosen.get("val_seen"),
            "val_unseen": chosen.get("val_unseen"),
            "location": chosen.get("merged", ""),
        },
    )


STAGE_FUNCS: dict[str, Callable[[PipelineContext, dict[str, Any]], None]] = {
    "data": stage_data,
    "train": stage_train,
    "select": stage_select,
    "merge": stage_merge,
    "eval": stage_eval,
    "choose": stage_choose,
    "register": stage_register,
}


def run_pipeline(ctx: PipelineContext, *, from_stage: str = STAGES[0], to_stage: str = STAGES[-1]) -> dict[str, Any]:
    """Run stages `from_stage..to_stage` in order, saving state after each."""
    first, last = STAGES.index(from_stage), STAGES.index(to_stage)
    state = load_state(ctx)
    state.setdefault("name", ctx.name)
    for stage in STAGES[first : last + 1]:
        log.info("pipeline_stage_start", stage=stage, name=ctx.name)
        STAGE_FUNCS[stage](ctx, state)
        state["done"] = [*[s for s in state.get("done", []) if s != stage], stage]
        save_state(ctx, state)
    return state


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI arguments."""
    ap = argparse.ArgumentParser(prog="python -m lifemate.train.llm_pipeline", description="SFT pipeline (6.2.6).")
    ap.add_argument("--config", required=True, type=Path, help="configs/train/llm_vN.toml")
    ap.add_argument("--env", default="local", help="env profile; work dir under its runs_dir")
    ap.add_argument("--name", help="model name (default ft-<base>-sft<v>-r<nn>)")
    ap.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    ap.add_argument("--to-stage", choices=STAGES, default=STAGES[-1])
    ap.add_argument("--limit", type=int, default=30, help="ids per val split for the agent check")
    ap.add_argument("--split", default="train", help="data split folder holding train and val ids")
    return ap.parse_args(argv)


def make_context(args: argparse.Namespace, repo_root: Path = REPO_ROOT) -> PipelineContext:
    """Context from CLI arguments."""
    config_path = abs_path(args.config, repo_root)
    cfg = sft.load_llm_train_config(config_path)
    name = args.name or default_name(cfg, repo_root / "artifacts" / "models" / "registry.csv")
    runs = abs_path(load_profile(args.env, repo_root).runs_dir, repo_root)
    return PipelineContext(repo_root, config_path, cfg, args.env, name, runs / "train" / name, args.limit, args.split)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    ctx = make_context(args)
    state = run_pipeline(ctx, from_stage=args.from_stage, to_stage=args.to_stage)
    print(f"pipeline {ctx.name}: done {state.get('done')} state={ctx.state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
