"""Chạy cả quy trình fine-tune LLM bằng một lệnh, từ dữ liệu SFT tới đăng ký model (SPEC 6.2.3, 6.2.6).

Nội dung chính:
- `PipelineContext`: đường dẫn và tuỳ chọn dùng chung cho mọi stage.
- `stage_data`, `stage_train`, `stage_select`, `stage_merge`, `stage_eval`, `stage_choose`, `stage_register`:
  bảy stage của pipeline.
- `run_pipeline`: chạy một đoạn stage liên tiếp và lưu tiến độ.
- `read_registry`, `append_registry`, `default_name`: đọc, ghi registry model và đặt tên model.
- `run_config_for`: xuất config chạy (TOML) trỏ tới model đã merge.
- `parse_args`, `make_context`, `main`: CLI.

Cách hoạt động:
Mỗi stage là một hàm nhận `PipelineContext` và dict trạng thái, ghi kết quả vào khoá trùng tên stage. Thứ tự cố
định là data -> train -> select -> merge -> eval -> choose -> register:
1. data: gọi `sft_data` cho train, eval_seen, eval_unseen (augment chỉ áp dụng cho train).
2. train: LoRA SFT bằng `swift_sft.train`, tự chạy tiếp từ checkpoint mới nhất.
3. select: lấy 2 checkpoint có eval_unseen loss thấp nhất.
4. merge: gộp adapter vào base model bf16.
5. eval: chạy agent thật trên `val_seen` và `val_unseen` với từng model đã merge.
6. choose: chọn model có điểm `val_unseen` cao nhất, hoà thì xét `val_seen`.
7. register: ghi `artifacts/models/<name>/` và thêm một dòng vào `registry.csv`.

Lưu ý:
- Tiến độ được lưu vào `<work>/pipeline_progress.json` sau mỗi stage, nên `--from-stage` chạy tiếp được.
- Stage train và merge cần extra GPU; test không chạy hai stage này.
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
from lifemate.core.tomlio import dump_toml
from lifemate.train import sft, sft_data, swift_sft
from lifemate.train.common import abs_path, sha256_file, write_json_atomic, write_train_config

log = structlog.get_logger(__name__)

STAGES = ("data", "train", "select", "merge", "eval", "choose", "register")
REGISTRY_FIELDS = ("name", "base", "data_version", "checkpoint", "val_seen", "val_unseen", "location")
EVAL_SPLITS = ("val_seen", "val_unseen")
N_CANDIDATES = 2


@dataclass(frozen=True, slots=True)
class PipelineContext:
    """Đường dẫn và tuỳ chọn dùng chung cho mọi stage.

    Thuộc tính:
        repo_root: thư mục gốc repo.
        config_path: đường dẫn file config train.
        cfg: config train đã đọc.
        env: tên profile môi trường.
        name: tên model.
        work_dir: thư mục làm việc của pipeline.
        limit: số id mỗi tập val khi chấm bằng agent.
        data_split: split chứa id train và val.
    """

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
        """File lưu kết quả các stage."""
        return self.work_dir / "pipeline_progress.json"

    @property
    def model_dir(self) -> Path:
        """Thư mục `artifacts/models/<name>/`, nằm trong git: chỉ chứa config và điểm số, không chứa trọng số."""
        return self.repo_root / "artifacts" / "models" / self.name


def load_state(ctx: PipelineContext) -> dict[str, Any]:
    """Đọc kết quả các stage đã chạy tới giờ; trả về dict rỗng nếu chưa có file tiến độ."""
    if not ctx.state_path.is_file():
        return {}
    return dict(json.loads(ctx.state_path.read_text(encoding="utf-8")))


def save_state(ctx: PipelineContext, state: Mapping[str, Any]) -> None:
    """Ghi kết quả các stage theo cách nguyên tử, không để lại file ghi dở."""
    write_json_atomic(ctx.state_path, dict(state))


def data_version(cfg: sft.LLMTrainConfig) -> str:
    """Lấy phiên bản dữ liệu từ tên file train, ví dụ `v1` từ `data/sft/v1.jsonl`."""
    return Path(cfg.data.train).stem


def default_name(cfg: sft.LLMTrainConfig, registry: Path) -> str:
    """Tạo tên model `ft-<base>-sft<v>-r<nn>`, với `<nn>` là số lần chạy còn trống tiếp theo trong registry (SPEC
    6.2.6).

    `<base>` là tên model đã bỏ ký tự ngoài a-z0-9; `<v>` là phần số của phiên bản dữ liệu.

    Tham số:
        cfg: config train LLM.
        registry: file `registry.csv`.

    Trả về:
        Tên model chưa có trong registry.
    """
    base = re.sub(r"[^a-z0-9]+", "", cfg.base.model.rsplit("/", 1)[-1].lower()) or "base"
    version = re.sub(r"\D", "", data_version(cfg)) or "0"
    prefix = f"ft-{base}-sft{version}-r"
    used = {r["name"] for r in read_registry(registry)}
    n = 1
    while f"{prefix}{n:02d}" in used:
        n += 1
    return f"{prefix}{n:02d}"


def read_registry(path: Path) -> list[dict[str, str]]:
    """Đọc các dòng của `artifacts/models/registry.csv`; trả về danh sách rỗng nếu chưa có file."""
    if not path.is_file():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_registry(path: Path, row: Mapping[str, Any]) -> None:
    """Thêm một dòng vào registry; ghi dòng tiêu đề khi file mới hoặc rỗng, cột thiếu để trống."""
    new = not path.is_file() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in REGISTRY_FIELDS})


# ---------------------------------------------------------------- Các stage


def stage_data(ctx: PipelineContext, state: dict[str, Any]) -> None:
    """Tạo các file SFT train, eval_seen và eval_unseen.

    Mỗi file gọi `sft_data.run` với argv tương ứng; hai file eval có thêm `--allow-eval`. Sha256 và số dòng của
    từng file được ghi vào `state["data"]`.
    """
    jobs = [
        (ctx.cfg.data.train, "configs/splits/train.txt", False),
        (ctx.cfg.data.eval_seen, "configs/splits/val_seen.txt", True),
        (ctx.cfg.data.eval_unseen, "configs/splits/val_unseen.txt", True),
    ]
    metas: dict[str, Any] = {}
    for out, ids, allow_eval in jobs:
        argv = ["--config", str(ctx.config_path), "--env", ctx.env, "--ids", ids, "--out", out]
        argv += ["--split", ctx.data_split] + (["--allow-eval"] if allow_eval else [])
        argv += ["--transcript-source", ctx.cfg.data.transcript_source, "--asr-ratio", str(ctx.cfg.data.asr_ratio)]
        if ctx.cfg.data.asr_cache:
            argv += ["--asr-cache", ctx.cfg.data.asr_cache]
        d = ctx.cfg.data  # cờ augment chỉ áp dụng cho file train (sft_data bỏ qua chúng khi có --allow-eval)
        argv += ["--error-recovery", str(d.error_recovery_ratio), "--oversample-hard", str(d.oversample_hard)]
        argv += ["--distractor-apps", str(d.distractor_apps), "--dedup-max-k", str(d.dedup_max_k)]
        meta = sft_data.run(sft_data.parse_args(argv), ctx.repo_root, argv=["lifemate.train.sft_data", *argv])
        metas[out] = {k: meta[k] for k in ("sha256", "n_rows", "n_dropped", "n_replay_warnings")}
    state["data"] = metas


def stage_train(ctx: PipelineContext, state: dict[str, Any]) -> None:
    """LoRA SFT bằng ms-swift; nếu lần trước bị ngắt thì chạy tiếp từ checkpoint mới nhất."""
    out = ctx.work_dir / "sft"
    # Trainer Unsloth (trước khi chuyển sang ms-swift):
    # latest = CheckpointManager(out / "checkpoints").latest()
    # sft.train(ctx.cfg, repo_root=ctx.repo_root, out_dir=out, resume_from=latest.path if latest else None)
    swift_dir = swift_sft.train(ctx.cfg, repo_root=ctx.repo_root, out_dir=out)
    state["train"] = {"out_dir": str(out), "swift_dir": str(swift_dir)}


def stage_select(ctx: PipelineContext, state: dict[str, Any]) -> None:
    """Chọn 2 checkpoint có eval_unseen loss thấp nhất.

    Loss lấy từ `eval_loss` của ms-swift; file val duy nhất truyền cho ms-swift là `val_unseen`.

    Lỗi:
        RuntimeError: khi không có checkpoint nào có eval loss.
    """
    # Trainer Unsloth (trước khi chuyển sang ms-swift):
    # mgr = CheckpointManager(ctx.work_dir / "sft" / "checkpoints", metric_name=sft.EVAL_METRIC)
    # best = mgr.best_k(N_CANDIDATES)
    swift_dir = ctx.work_dir / "sft" / "swift"
    best = swift_sft.best_k(swift_dir, N_CANDIDATES)
    if not best:
        raise RuntimeError(f"no checkpoint with an eval loss under {swift_dir}")
    state["select"] = [{"path": str(c.path), "step": c.step, "loss": c.eval_loss} for c in best]


def stage_merge(ctx: PipelineContext, state: dict[str, Any]) -> None:
    """Gộp adapter của từng ứng viên vào base model bf16."""
    merged = []
    for cand in state["select"]:
        out = ctx.work_dir / "merged" / Path(cand["path"]).name
        sft.merge_adapter(ctx.cfg.base.model, Path(cand["path"]), out, revision=ctx.cfg.base.revision or None)
        merged.append({**cand, "merged": str(out)})
    state["merge"] = merged


def run_config_for(ctx: PipelineContext, model_dir: str, out: Path) -> Path:
    """Tạo bản sao config chạy với `[llm].model` trỏ tới model đã merge.

    `revision` để trống vì model là thư mục trên máy.

    Tham số:
        ctx: context của pipeline.
        model_dir: thư mục model đã merge.
        out: file TOML đích.

    Trả về:
        `out`.
    """
    with open(abs_path(ctx.cfg.base.run_config, ctx.repo_root), "rb") as f:
        raw = tomllib.load(f)
    raw["llm"]["model"] = model_dir
    raw["llm"]["revision"] = ""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_toml(raw), encoding="utf-8")
    return out


# Runner(lệnh, thư mục làm việc) -> mã thoát; test thay bằng hàm giả để không chạy tiến trình con thật.
Runner = Callable[[Sequence[str], Path], int]


def _subprocess_runner(cmd: Sequence[str], cwd: Path) -> int:
    """Chạy lệnh trong `cwd` và trả về mã thoát; không ném lỗi khi mã khác 0."""
    return subprocess.run(list(cmd), cwd=cwd, check=False).returncode


def eval_command(ctx: PipelineContext, run_config: Path, split_ids: str, run_dir: Path) -> list[str]:
    """Dựng lệnh `python -m lifemate.run.cli` trên một file id val, kèm `--limit` và nguồn chữ của config."""
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
        "--transcripts",
        ctx.cfg.base.eval_transcripts,
        "--out",
        str(run_dir),
    ]


def read_run_score(run_dir: Path) -> float | None:
    """Đọc `score.total` từ `summary.json` của một run; trả về `None` nếu thiếu file hoặc thiếu điểm."""
    summary = run_dir / "summary.json"
    if not summary.is_file():
        return None
    score = json.loads(summary.read_text(encoding="utf-8")).get("score")
    return None if not score else float(score["total"])


def stage_eval(ctx: PipelineContext, state: dict[str, Any], runner: Runner = _subprocess_runner) -> None:
    """Chạy agent thật với từng ứng viên đã merge trên `val_seen` và `val_unseen` (giới hạn bởi `--limit`).

    Mỗi ứng viên có config chạy riêng.

    Tham số:
        ctx: context của pipeline.
        state: trạng thái pipeline; đọc `state["merge"]`, ghi `state["eval"]`.
        runner: hàm chạy tiến trình con; test thay bằng hàm giả.

    Lỗi:
        RuntimeError: khi một lệnh chấm thất bại.
    """
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
    """Chọn ứng viên có điểm `val_unseen` cao nhất; hoà thì xét `val_seen` (SPEC 6.2.3 mục 3).

    Lỗi:
        ValueError: khi không có ứng viên nào.
    """

    def key(c: Mapping[str, Any]) -> tuple[float, float]:
        """Khoá so sánh: (điểm `val_unseen`, điểm `val_seen`)."""
        # Điểm thiếu (None) được tính là 0.
        return (float(c.get("val_unseen") or 0.0), float(c.get("val_seen") or 0.0))

    if not candidates:
        raise ValueError("no candidates")
    return max(candidates, key=key)


def stage_choose(ctx: PipelineContext, state: dict[str, Any]) -> None:
    """Chọn checkpoint giữ lại trong các ứng viên đã chấm."""
    state["choose"] = dict(choose(state["eval"]))


def stage_register(ctx: PipelineContext, state: dict[str, Any]) -> None:
    """Ghi `artifacts/models/<name>/{train_config.json, eval.json, metrics.jsonl}` và thêm một dòng vào registry.

    `train_config.json` kèm sha256 của mọi file dữ liệu có thật trong config.
    """
    chosen = state["choose"]
    ctx.model_dir.mkdir(parents=True, exist_ok=True)
    data_sha = {
        k: sha256_file(abs_path(v, ctx.repo_root))
        for k, v in dataclasses.asdict(ctx.cfg.data).items()
        if isinstance(v, str) and v and abs_path(v, ctx.repo_root).is_file()
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
    """Chạy các stage từ `from_stage` tới `to_stage` theo thứ tự, lưu trạng thái sau mỗi stage.

    Tham số:
        ctx: context của pipeline.
        from_stage: stage bắt đầu.
        to_stage: stage kết thúc (tính cả stage này).

    Trả về:
        Trạng thái sau stage cuối.
    """
    first, last = STAGES.index(from_stage), STAGES.index(to_stage)
    state = load_state(ctx)
    state.setdefault("name", ctx.name)
    for stage in STAGES[first : last + 1]:
        log.info("pipeline_stage_start", stage=stage, name=ctx.name)
        STAGE_FUNCS[stage](ctx, state)
        # Đưa stage vừa xong xuống cuối danh sách `done`, để chạy lại không sinh tên trùng.
        state["done"] = [*[s for s in state.get("done", []) if s != stage], stage]
        save_state(ctx, state)
    return state


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
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
    """Tạo context từ tham số dòng lệnh; thư mục làm việc là `<runs_dir>/train/<name>` của profile môi trường."""
    config_path = abs_path(args.config, repo_root)
    cfg = sft.load_llm_train_config(config_path)
    name = args.name or default_name(cfg, repo_root / "artifacts" / "models" / "registry.csv")
    runs = abs_path(load_profile(args.env, repo_root).runs_dir, repo_root)
    return PipelineContext(repo_root, config_path, cfg, args.env, name, runs / "train" / name, args.limit, args.split)


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy CLI pipeline LLM và in các stage đã xong; trả về mã thoát 0."""
    args = parse_args(argv)
    ctx = make_context(args)
    state = run_pipeline(ctx, from_stage=args.from_stage, to_stage=args.to_stage)
    print(f"pipeline {ctx.name}: done {state.get('done')} state={ctx.state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
