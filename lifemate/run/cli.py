"""Điểm vào dòng lệnh: đọc tham số, nạp cấu hình và profile môi trường, lắp ráp, chạy (SPEC 4.11).

Luồng: `parse_args` -> `run` (nạp config TOML và env profile, chuẩn hóa đường dẫn theo gốc repo, nạp mẫu của split,
gọi `factory.build`, chọn thư mục run, ghi lại toàn bộ thông tin cấu hình vào bản ghi run, giao cho `Runner`)
-> `main` in đường dẫn summary, điểm và báo cáo. Module không chứa logic agent; mọi việc lắp ráp đối tượng nằm ở
`factory`.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
from collections.abc import Sequence
from pathlib import Path

import structlog

from lifemate.core.config import EnvProfile, load_config, load_profile
from lifemate.core.envinfo import detect
from lifemate.core.paths import REPO_ROOT
from lifemate.io.data import load_split, read_ids
from lifemate.io.store import RunStore
from lifemate.run import factory
from lifemate.run.runner import Runner, RunSummary

log = structlog.get_logger(__name__)


def _abs(path: str, repo_root: Path) -> Path:
    """Mở rộng `~`; đường dẫn tương đối được tính từ gốc repo (SPEC 4.16, profile local)."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else repo_root / p


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    ap = argparse.ArgumentParser(prog="python -m lifemate.run.cli", description="Run the LifeMate agent on one split.")
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
    """Thực hiện một lần chạy từ tham số đã đọc.

    `--out` ghi đè thư mục run (index.csv đặt ở thư mục cha); nếu không, thư mục run là `runs_dir/<name>` với tên
    mặc định là dấu thời gian UTC. Điểm chỉ được tính khi mọi mẫu đều có nhãn.
    """
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
    # Sắp theo id trước khi cắt để `--limit` luôn chọn cùng một tập mẫu
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
    # Bản ghi cấu hình đầy đủ của run, để tái lập kết quả sau này
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
    """Điểm vào chương trình; in đường dẫn summary, điểm (nếu có) và báo cáo; trả mã thoát 0."""
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
