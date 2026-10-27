"""Một lệnh: dữ liệu SFT → train → 2 checkpoint tốt nhất → merge → eval agent → chọn → đăng ký (SPEC 6.2.3, 6.2.6).

Nội dung: `PipelineContext`, bảy hàm stage, các hàm đọc/ghi registry, xuất TOML cho run config và CLI.

Cách dựng: mỗi stage là một hàm nhận `PipelineContext` và dict trạng thái, ghi kết quả vào khoá trùng tên stage.
Thứ tự cố định data→train→select→merge→eval→choose→register:
- data: gọi `sft_data` cho train, eval_seen, eval_unseen (augment chỉ áp dụng cho train).
- train: LoRA SFT bằng `swift_sft.train`, tự nối tiếp từ checkpoint mới nhất.
- select: 2 checkpoint có eval_unseen loss thấp nhất.
- merge: gộp adapter vào base bf16.
- eval: chạy agent thật trên val_seen và val_unseen với từng model đã merge.
- choose: điểm val_unseen cao nhất, hoà thì xét val_seen.
- register: ghi artifacts/models/<name>/ và thêm một dòng vào registry.csv.

Tiến độ lưu trong `<work>/pipeline_progress.json` sau mỗi stage nên `--from-stage` chạy tiếp được. Stage train và
merge cần gói GPU và không chạy trong test.
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
from lifemate.train import sft, sft_data, swift_sft
from lifemate.train.common import abs_path, sha256_file, write_json_atomic, write_train_config

log = structlog.get_logger(__name__)

STAGES = ("data", "train", "select", "merge", "eval", "choose", "register")
REGISTRY_FIELDS = ("name", "base", "data_version", "checkpoint", "val_seen", "val_unseen", "location")
EVAL_SPLITS = ("val_seen", "val_unseen")
N_CANDIDATES = 2


@dataclass(frozen=True, slots=True)
class PipelineContext:
    """Đường dẫn và tuỳ chọn dùng chung cho mọi stage."""

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
        """Nơi lưu kết quả các stage."""
        return self.work_dir / "pipeline_progress.json"

    @property
    def model_dir(self) -> Path:
        """`artifacts/models/<name>/` (nằm trong git: cấu hình và điểm số, không bao giờ chứa trọng số)."""
        return self.repo_root / "artifacts" / "models" / self.name


def load_state(ctx: PipelineContext) -> dict[str, Any]:
    """Kết quả các stage đã chạy tới hiện tại (dict rỗng nếu chưa có file tiến độ)."""
    if not ctx.state_path.is_file():
        return {}
    return dict(json.loads(ctx.state_path.read_text(encoding="utf-8")))


def save_state(ctx: PipelineContext, state: Mapping[str, Any]) -> None:
    """Ghi kết quả các stage theo cách nguyên tử (không để lại file ghi dở)."""
    write_json_atomic(ctx.state_path, dict(state))


def data_version(cfg: sft.LLMTrainConfig) -> str:
    """Lấy phiên bản dữ liệu từ tên file train, ví dụ `v1` từ `data/sft/v1.jsonl`."""
    return Path(cfg.data.train).stem


def default_name(cfg: sft.LLMTrainConfig, registry: Path) -> str:
    """Tạo tên `ft-<base>-sft<v>-r<nn>` với số lần chạy còn trống tiếp theo trong registry (SPEC 6.2.6).

    `<base>` là tên model bỏ ký tự ngoài a-z0-9, `<v>` là phần số của phiên bản dữ liệu.
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
    """Đọc các dòng của artifacts/models/registry.csv (danh sách rỗng nếu chưa có file)."""
    if not path.is_file():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_registry(path: Path, row: Mapping[str, Any]) -> None:
    """Thêm một dòng (ghi header khi file mới hoặc rỗng); cột thiếu được để trống."""
    new = not path.is_file() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in REGISTRY_FIELDS})


# ---------------------------------------------------------------- stages


def stage_data(ctx: PipelineContext, state: dict[str, Any]) -> None:
    """Tạo các file SFT train, eval_seen và eval_unseen.

    Mỗi file gọi `sft_data.run` với argv tương ứng; hai file eval có `--allow-eval`. Ghi sha256 và số dòng
    của từng file vào `state["data"]`.
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
    """LoRA SFT bằng ms-swift; chạy tiếp từ checkpoint mới nhất nếu lần trước bị ngắt."""
    out = ctx.work_dir / "sft"
    # Unsloth trainer (before ms-swift):
    # latest = CheckpointManager(out / "checkpoints").latest()
    # sft.train(ctx.cfg, repo_root=ctx.repo_root, out_dir=out, resume_from=latest.path if latest else None)
    swift_dir = swift_sft.train(ctx.cfg, repo_root=ctx.repo_root, out_dir=out)
    state["train"] = {"out_dir": str(out), "swift_dir": str(swift_dir)}


def stage_select(ctx: PipelineContext, state: dict[str, Any]) -> None:
    """Chọn 2 checkpoint có eval_unseen loss thấp nhất (eval_loss của ms-swift, file val duy nhất là val_unseen)."""
    # Unsloth trainer (before ms-swift):
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


def _dump_toml_value(v: Any) -> str:
    """Chuyển một giá trị Python (bool, số, chuỗi, list, mapping) thành chuỗi TOML; kiểu khác thì ném TypeError."""
    # Xét bool trước int vì bool là lớp con của int.
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
    """Xuất các bảng cấp cao nhất gồm giá trị đơn, list và bảng inline (đủ cho configs/v0.toml)."""
    out: list[str] = []
    for section, table in data.items():
        out.append(f"[{section}]")
        out += [f"{k} = {_dump_toml_value(v)}" for k, v in table.items()]
        out.append("")
    return "\n".join(out)


def run_config_for(ctx: PipelineContext, model_dir: str, out: Path) -> Path:
    """Tạo bản sao run config với `[llm].model` trỏ tới model đã merge (revision để trống vì là thư mục cục bộ)."""
    with open(abs_path(ctx.cfg.base.run_config, ctx.repo_root), "rb") as f:
        raw = tomllib.load(f)
    raw["llm"]["model"] = model_dir
    raw["llm"]["revision"] = ""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_toml(raw), encoding="utf-8")
    return out


Runner = Callable[[Sequence[str], Path], int]


def _subprocess_runner(cmd: Sequence[str], cwd: Path) -> int:
    """Chạy lệnh trong `cwd` và trả về mã thoát (không ném lỗi khi mã khác 0)."""
    return subprocess.run(list(cmd), cwd=cwd, check=False).returncode


def eval_command(ctx: PipelineContext, run_config: Path, split_ids: str, run_dir: Path) -> list[str]:
    """Lệnh `python -m lifemate.run.cli` chạy trên một file id val, kèm `--limit`."""
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
    """Đọc `score.total` từ summary.json của một lần chạy (None nếu thiếu file hoặc thiếu điểm)."""
    summary = run_dir / "summary.json"
    if not summary.is_file():
        return None
    score = json.loads(summary.read_text(encoding="utf-8")).get("score")
    return None if not score else float(score["total"])


def stage_eval(ctx: PipelineContext, state: dict[str, Any], runner: Runner = _subprocess_runner) -> None:
    """Chạy agent thật với từng ứng viên đã merge trên val_seen và val_unseen (`--limit`).

    Mỗi ứng viên có run config riêng; lệnh thất bại thì ném RuntimeError. `runner` cho phép test thay tiến trình con.
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
    """Chọn ứng viên có điểm val_unseen cao nhất; hoà thì xét val_seen (SPEC 6.2.3 mục 3)."""

    def key(c: Mapping[str, Any]) -> tuple[float, float]:
        # Điểm thiếu (None) được tính là 0.
        return (float(c.get("val_unseen") or 0.0), float(c.get("val_seen") or 0.0))

    if not candidates:
        raise ValueError("no candidates")
    return max(candidates, key=key)


def stage_choose(ctx: PipelineContext, state: dict[str, Any]) -> None:
    """Chọn checkpoint giữ lại."""
    state["choose"] = dict(choose(state["eval"]))


def stage_register(ctx: PipelineContext, state: dict[str, Any]) -> None:
    """Ghi artifacts/models/<name>/{train_config.json, eval.json, metrics.jsonl}; thêm một dòng vào registry.

    train_config.json kèm sha256 của mọi file dữ liệu có thật trong cấu hình.
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
    """Chạy các stage từ `from_stage` tới `to_stage` theo thứ tự, lưu trạng thái sau mỗi stage."""
    first, last = STAGES.index(from_stage), STAGES.index(to_stage)
    state = load_state(ctx)
    state.setdefault("name", ctx.name)
    for stage in STAGES[first : last + 1]:
        log.info("pipeline_stage_start", stage=stage, name=ctx.name)
        STAGE_FUNCS[stage](ctx, state)
        # Đưa stage vừa xong xuống cuối danh sách done, không lặp lại khi chạy lại.
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
    """Tạo context từ tham số dòng lệnh; thư mục làm việc là `<runs_dir>/train/<name>` của profile env."""
    config_path = abs_path(args.config, repo_root)
    cfg = sft.load_llm_train_config(config_path)
    name = args.name or default_name(cfg, repo_root / "artifacts" / "models" / "registry.csv")
    runs = abs_path(load_profile(args.env, repo_root).runs_dir, repo_root)
    return PipelineContext(repo_root, config_path, cfg, args.env, name, runs / "train" / name, args.limit, args.split)


def main(argv: Sequence[str] | None = None) -> int:
    """Điểm vào của CLI."""
    args = parse_args(argv)
    ctx = make_context(args)
    state = run_pipeline(ctx, from_stage=args.from_stage, to_stage=args.to_stage)
    print(f"pipeline {ctx.name}: done {state.get('done')} state={ctx.state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
