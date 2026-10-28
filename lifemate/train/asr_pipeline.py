"""Chạy cả quy trình ASR bằng một lệnh: tạo cặp dữ liệu, fine-tune, chọn checkpoint, gộp LoRA, đo (SPEC 6.4.5 mục 6).

Nội dung chính:
- `ASRPipelineContext`: đường dẫn và tuỳ chọn dùng chung cho mọi giai đoạn.
- `stage_data`, `stage_train`, `stage_select`, `stage_merge`, `stage_measure`: năm giai đoạn của pipeline.
- `run_pipeline`: chạy một đoạn giai đoạn liên tiếp và lưu tiến độ.
- `parse_args`, `make_context`, `main`: CLI.

Cách hoạt động:
- Các giai đoạn chạy theo thứ tự data -> train -> select -> merge -> measure. Mỗi giai đoạn là một hàm
  `stage_*(ctx, state)` ghi kết quả vào dict `state`.
- `run_pipeline` lưu `state` nguyên tử vào `<work>/pipeline_progress.json` sau mỗi giai đoạn. Nhờ vậy có thể dừng
  rồi chạy lại từ bất kỳ giai đoạn nào bằng `--from-stage`.
- Checkpoint được chọn theo tỉ lệ bỏ sót thực thể rồi WER.

Lưu ý:
- Các giai đoạn train, select, merge và measure cần extra GPU; test không chạy các giai đoạn này.
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
    """Đường dẫn và tuỳ chọn dùng chung cho mọi giai đoạn.

    Thuộc tính:
        repo_root: thư mục gốc repo.
        config_path: đường dẫn file config train.
        cfg: config train đã đọc.
        env: tên profile môi trường.
        name: tên thư mục làm việc.
        work_dir: thư mục làm việc của pipeline.
        data_root: thư mục dữ liệu.
        data_split: split chứa file audio.
    """

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
        """File ghi tiến độ các giai đoạn (cùng bố cục với `llm_pipeline`)."""
        return self.work_dir / "pipeline_progress.json"

    def val_file(self, split: str) -> Path:
        """Trả về đường dẫn file cặp của `split`: `data/asr/<v>_<split>.jsonl`, nằm cạnh file train."""
        train = abs_path(self.cfg.data.train, self.repo_root)
        return train.with_name(f"{train.stem}_{split}{train.suffix}")


def stage_data(ctx: ASRPipelineContext, state: dict[str, Any]) -> None:
    """Tạo cặp train từ `configs/splits/train.txt`, và cặp `val_seen`, `val_unseen` chỉ để đo."""
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
    """Fine-tune; nếu lần chạy trước bị ngắt thì chạy tiếp từ checkpoint mới nhất."""
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
    """Trả về thư mục model dùng được ngay cho một checkpoint.

    Với train toàn phần, đó là chính checkpoint. Với LoRA, đó là bản đã gộp, lưu cache trong `<work>/merged` để
    không gộp lại lần hai.
    """
    if ctx.cfg.method.kind != "lora":
        return ckpt.path
    out = ctx.work_dir / "merged" / ckpt.path.name
    if not out.is_dir():
        asr.merge_lora(ctx.cfg.base.model, ckpt.path, out, revision=ctx.cfg.base.revision or None)
    return out


def stage_select(ctx: ASRPipelineContext, state: dict[str, Any]) -> None:
    """Chấm mọi checkpoint còn giữ trên cặp `val_unseen`, chọn theo tỉ lệ bỏ sót thực thể rồi WER."""
    mgr = CheckpointManager(ctx.work_dir / "asr" / "checkpoints", metric_name=asr.EVAL_METRIC)
    rows = read_jsonl(ctx.val_file("val_unseen"))
    split_dir = ctx.data_root / ctx.data_split
    processor = ctx.work_dir / "asr" / "processor"

    def transcribe(c: CheckpointInfo) -> list[str]:
        """Chép lời các cặp `val_unseen` bằng checkpoint `c`."""
        return asr.transcribe_rows(_merged_or_raw(ctx, c), processor, rows, split_dir)

    chosen, table = asr.select_checkpoint(mgr.all(), transcribe, rows)
    state["select"] = {"chosen": str(chosen.path), "step": chosen.step, "table": table}


def stage_merge(ctx: ASRPipelineContext, state: dict[str, Any]) -> None:
    """Gộp checkpoint đã chọn (LoRA) hoặc giữ nguyên (train toàn phần)."""
    path = Path(state["select"]["chosen"])
    state["merge"] = {"model": str(_merged_or_raw(ctx, CheckpointInfo(path, state["select"]["step"], None)))}


def stage_measure(ctx: ASRPipelineContext, state: dict[str, Any]) -> None:
    """Đo WER và tỉ lệ bỏ sót thực thể của model cuối trên `val_seen` và `val_unseen`; ghi vào `<work>/measure.json`."""
    model = Path(state["merge"]["model"])
    # Bản gộp LoRA có sẵn processor; bản train toàn phần thì lấy processor đã lưu lúc train.
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
    """Chạy các giai đoạn từ `from_stage` tới `to_stage` (tính cả hai đầu), lưu `state` sau mỗi giai đoạn.

    Tham số:
        ctx: context của pipeline.
        from_stage: giai đoạn bắt đầu.
        to_stage: giai đoạn kết thúc.

    Trả về:
        `state` sau giai đoạn cuối.
    """
    state: dict[str, Any] = {}
    if ctx.state_path.is_file():
        state = dict(json.loads(ctx.state_path.read_text(encoding="utf-8")))
    for stage in STAGES[STAGES.index(from_stage) : STAGES.index(to_stage) + 1]:
        log.info("asr_pipeline_stage_start", stage=stage, name=ctx.name)
        STAGE_FUNCS[stage](ctx, state)
        # Đưa giai đoạn vừa xong xuống cuối danh sách `done`, không để trùng tên.
        state["done"] = [*[s for s in state.get("done", []) if s != stage], stage]
        write_json_atomic(ctx.state_path, state)
    return state


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    ap = argparse.ArgumentParser(prog="python -m lifemate.train.asr_pipeline", description="ASR pipeline (6.4).")
    ap.add_argument("--config", required=True, type=Path, help="configs/train/asr_vN.toml")
    ap.add_argument("--env", default="local")
    ap.add_argument("--name", help="work folder name (default: config stem)")
    ap.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    ap.add_argument("--to-stage", choices=STAGES, default=STAGES[-1])
    ap.add_argument("--split", default="train", help="data split folder holding the audio")
    return ap.parse_args(argv)


def make_context(args: argparse.Namespace, repo_root: Path = REPO_ROOT) -> ASRPipelineContext:
    """Tạo context của pipeline từ tham số dòng lệnh."""
    config_path = abs_path(args.config, repo_root)
    cfg = asr.load_asr_train_config(config_path)
    profile = load_profile(args.env, repo_root)
    name = args.name or f"asr-{config_path.stem}"
    work = abs_path(profile.runs_dir, repo_root) / "train" / name
    return ASRPipelineContext(
        repo_root, config_path, cfg, args.env, name, work, abs_path(profile.data_root, repo_root), args.split
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy CLI pipeline ASR và in các giai đoạn đã xong; trả về mã thoát 0."""
    args = parse_args(argv)
    ctx = make_context(args)
    state = run_pipeline(ctx, from_stage=args.from_stage, to_stage=args.to_stage)
    print(f"asr pipeline {ctx.name}: done {state.get('done')} state={ctx.state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
