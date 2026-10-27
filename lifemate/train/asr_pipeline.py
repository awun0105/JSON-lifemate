"""Một lệnh: cặp ASR → fine-tune → chọn checkpoint theo tỉ lệ sót thực thể → gộp → đo (SPEC 6.4.5 mục 6).

Kết quả từng giai đoạn nằm trong `<work>/pipeline_progress.json`; `--from-stage` để chạy tiếp. Các giai đoạn train,
select, merge và measure cần gói phụ GPU và không chạy trong test.

Cách xây dựng: các giai đoạn data→train→select→merge→measure là hàm `stage_*(ctx, state)` ghi kết quả vào dict
`state`; `run_pipeline` lưu state nguyên tử sau mỗi giai đoạn nên có thể dừng và chạy lại từ bất kỳ giai đoạn nào.
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
    """Đường dẫn và tùy chọn dùng chung cho mọi giai đoạn."""

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
        """Nơi ghi nhớ kết quả các giai đoạn (cùng bố cục với llm_pipeline)."""
        return self.work_dir / "pipeline_progress.json"

    def val_file(self, split: str) -> Path:
        """`data/asr/<v>_<split>.jsonl` cạnh file train."""
        train = abs_path(self.cfg.data.train, self.repo_root)
        return train.with_name(f"{train.stem}_{split}{train.suffix}")


def stage_data(ctx: ASRPipelineContext, state: dict[str, Any]) -> None:
    """Cặp train từ configs/splits/train.txt; cặp val_unseen và val_seen chỉ dùng để đo."""
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
    """Fine-tune; chạy tiếp từ checkpoint mới nhất nếu lần chạy trước bị ngắt."""
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
    """Thư mục mô hình dùng được ngay: checkpoint gốc (full) hoặc bản gộp LoRA, có cache dưới `<work>/merged`."""
    if ctx.cfg.method.kind != "lora":
        return ckpt.path
    out = ctx.work_dir / "merged" / ckpt.path.name
    if not out.is_dir():
        asr.merge_lora(ctx.cfg.base.model, ckpt.path, out, revision=ctx.cfg.base.revision or None)
    return out


def stage_select(ctx: ASRPipelineContext, state: dict[str, Any]) -> None:
    """Chấm mọi checkpoint còn giữ trên cặp val_unseen; chọn theo tỉ lệ sót thực thể rồi WER."""
    mgr = CheckpointManager(ctx.work_dir / "asr" / "checkpoints", metric_name=asr.EVAL_METRIC)
    rows = read_jsonl(ctx.val_file("val_unseen"))
    split_dir = ctx.data_root / ctx.data_split
    processor = ctx.work_dir / "asr" / "processor"

    def transcribe(c: CheckpointInfo) -> list[str]:
        return asr.transcribe_rows(_merged_or_raw(ctx, c), processor, rows, split_dir)

    chosen, table = asr.select_checkpoint(mgr.all(), transcribe, rows)
    state["select"] = {"chosen": str(chosen.path), "step": chosen.step, "table": table}


def stage_merge(ctx: ASRPipelineContext, state: dict[str, Any]) -> None:
    """Gộp checkpoint đã chọn (LoRA) hoặc giữ nguyên (full)."""
    path = Path(state["select"]["chosen"])
    state["merge"] = {"model": str(_merged_or_raw(ctx, CheckpointInfo(path, state["select"]["step"], None)))}


def stage_measure(ctx: ASRPipelineContext, state: dict[str, Any]) -> None:
    """WER / tỉ lệ sót thực thể của mô hình cuối trên val_seen và val_unseen; ghi vào <work>/measure.json."""
    model = Path(state["merge"]["model"])
    # Bản gộp LoRA có sẵn processor; bản full thì lấy processor lưu lúc train.
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
    """Chạy các giai đoạn `from_stage..to_stage`, lưu state sau mỗi giai đoạn."""
    state: dict[str, Any] = {}
    if ctx.state_path.is_file():
        state = dict(json.loads(ctx.state_path.read_text(encoding="utf-8")))
    for stage in STAGES[STAGES.index(from_stage) : STAGES.index(to_stage) + 1]:
        log.info("asr_pipeline_stage_start", stage=stage, name=ctx.name)
        STAGE_FUNCS[stage](ctx, state)
        # Đưa giai đoạn vừa xong xuống cuối danh sách, không lặp tên.
        state["done"] = [*[s for s in state.get("done", []) if s != stage], stage]
        write_json_atomic(ctx.state_path, state)
    return state


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Tham số dòng lệnh."""
    ap = argparse.ArgumentParser(prog="python -m lifemate.train.asr_pipeline", description="ASR pipeline (6.4).")
    ap.add_argument("--config", required=True, type=Path, help="configs/train/asr_vN.toml")
    ap.add_argument("--env", default="local")
    ap.add_argument("--name", help="work folder name (default: config stem)")
    ap.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    ap.add_argument("--to-stage", choices=STAGES, default=STAGES[-1])
    ap.add_argument("--split", default="train", help="data split folder holding the audio")
    return ap.parse_args(argv)


def make_context(args: argparse.Namespace, repo_root: Path = REPO_ROOT) -> ASRPipelineContext:
    """Tạo context từ tham số dòng lệnh."""
    config_path = abs_path(args.config, repo_root)
    cfg = asr.load_asr_train_config(config_path)
    profile = load_profile(args.env, repo_root)
    name = args.name or f"asr-{config_path.stem}"
    work = abs_path(profile.runs_dir, repo_root) / "train" / name
    return ASRPipelineContext(
        repo_root, config_path, cfg, args.env, name, work, abs_path(profile.data_root, repo_root), args.split
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Điểm vào CLI."""
    args = parse_args(argv)
    ctx = make_context(args)
    state = run_pipeline(ctx, from_stage=args.from_stage, to_stage=args.to_stage)
    print(f"asr pipeline {ctx.name}: done {state.get('done')} state={ctx.state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
