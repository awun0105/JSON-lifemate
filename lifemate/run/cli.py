"""Điểm vào dòng lệnh để chạy agent trên một split (SPEC 4.11).

Nội dung chính:
- `parse_args`: đọc tham số dòng lệnh.
- `run`: thực hiện một lần chạy từ tham số đã đọc.
- `main`: gọi `run` rồi in đường dẫn summary, điểm và báo cáo.

Cách hoạt động:
1. `parse_args` đọc tham số.
2. `run` nạp config TOML và profile môi trường, đổi đường dẫn tương đối theo thư mục gốc repo, rồi nạp mẫu
   của split.
3. `run` gọi `factory.build` để lắp ráp đối tượng, chọn thư mục run, ghi toàn bộ thông tin config vào bản ghi
   run, rồi giao cho `Runner`.
4. `main` in kết quả.

Lưu ý:
- Module không chứa logic agent; mọi việc lắp ráp đối tượng nằm ở `factory`.
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
    """Mở rộng `~` và tính đường dẫn tương đối từ thư mục gốc repo (SPEC 4.16, profile `local`)."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else repo_root / p


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh; `argv` là `None` thì đọc từ `sys.argv`."""
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
    ap.add_argument(
        "--transcripts",
        choices=("config", "gold", "asr"),
        default="config",
        help="override asr.backend: gold transcripts or real ASR (default: keep the config value)",
    )
    return ap.parse_args(argv)


def run(args: argparse.Namespace, repo_root: Path = REPO_ROOT) -> RunSummary:
    """Thực hiện một lần chạy từ tham số đã đọc.

    `--out` ghi đè thư mục run, khi đó `index.csv` đặt ở thư mục cha. Nếu không có `--out`, thư mục run là
    `runs_dir/<name>`, với tên mặc định là thời điểm bắt đầu theo giờ UTC.

    Tham số:
        args: tham số dòng lệnh đã đọc.
        repo_root: thư mục gốc repo.

    Trả về:
        `RunSummary` của lần chạy. Điểm chỉ được tính khi mọi mẫu đều có nhãn.
    """
    config = load_config(_abs(str(args.config), repo_root))
    # `--transcripts asr` chạy ASR thật trên cùng config, khỏi phải chép cả file config chỉ để đổi `asr.backend`.
    if args.transcripts != "config":
        config = dataclasses.replace(config, asr=dataclasses.replace(config.asr, backend=args.transcripts))
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
    # Sắp theo id trước khi cắt, để `--limit` luôn chọn cùng một tập mẫu.
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
    # Bản ghi đầy đủ config của run, để sau này chạy lại được đúng kết quả.
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
    """Chạy CLI, in đường dẫn summary, điểm (nếu có) và đường dẫn báo cáo.

    Tham số:
        argv: tham số dòng lệnh; `None` thì đọc từ `sys.argv`.

    Trả về:
        Mã thoát, luôn là 0 khi chạy xong.
    """
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
