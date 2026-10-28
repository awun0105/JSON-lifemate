"""Chép lời một split một lần bằng ASR lúc chạy và lưu văn bản làm dữ liệu SFT (SPEC 6.2.1).

Cách dùng:
  python -m lifemate.train.asr_cache --config configs/v0.toml --env kaggle --out data/asr/train_asr.json \\
      --ids configs/splits/train.txt configs/splits/val_seen.txt configs/splits/val_unseen.txt

Nội dung chính:
- `to_entry`: tạo một mục cache từ `Transcript`.
- `CachedTranscriber`: phát lại cache theo giao diện `Transcriber`.
- `build_cache`: chép lời toàn bộ mẫu.
- `parse_args`, `main`: CLI.

Cách hoạt động:
1. Đọc các file id và nạp mẫu của split.
2. `ASRTranscriber` chép lời từng mẫu theo thứ tự.
3. Ghi file JSON `{sample_id: {"final": str, "history": [str | null, ...]}}` và `<out>.meta.json` (config, file
   id, số mẫu, số giây).

Lưu ý:
- Cache giúp `sft_data` tạo dữ liệu từ đầu ra ASR thật mà không phải chạy lại model ASR. Cache cũng cố định bản
  chép lời, nên mọi lần dựng lại dữ liệu đều giống hệt nhau.
- Cache chứa đúng những gì agent thấy lúc chạy: cùng mục `[asr]` của config chạy (engine, model, tuỳ chọn giải mã,
  sanitizer).
- Cần máy có GPU và đã cài extra ASR.
- Dòng đầu của docstring này cũng là mô tả của CLI (`--help`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import structlog

from lifemate.core.config import load_config, load_profile
from lifemate.core.paths import REPO_ROOT
from lifemate.core.types import Sample, Transcriber, Transcript
from lifemate.io.data import load_split, read_ids

log = structlog.get_logger(__name__)


def to_entry(t: Transcript) -> dict[str, Any]:
    """Tạo mục cache của một mẫu: văn bản lượt cuối và lịch sử."""
    return {"final": t.final, "history": list(t.history)}


class CachedTranscriber:
    """Transcriber phát lại cache ASR đã lưu.

    Tham số:
        cache: id mẫu tới mục cache (xem `to_entry`).

    Thuộc tính:
        source: nguồn transcript, luôn là `"asr"`.
    """

    source = "asr"

    def __init__(self, cache: Mapping[str, Mapping[str, Any]]) -> None:
        self._cache = cache

    @classmethod
    def from_file(cls, path: Path) -> CachedTranscriber:
        """Nạp cache từ file `path` do module này ghi ra."""
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def has(self, sample_id: str) -> bool:
        """Trả về `True` khi cache có `sample_id`."""
        return sample_id in self._cache

    def transcribe(self, sample: Sample) -> Transcript:
        """Trả về bản chép lời đã lưu của `sample`.

        Lỗi:
            KeyError: khi cache không có mẫu này.
            ValueError: khi số lượt lịch sử trong cache khác với mẫu (cache cũ hoặc sai split).
        """
        entry = self._cache[sample.id]
        history = tuple(entry.get("history") or ())
        if len(history) != len(sample.history):
            raise ValueError(f"{sample.id}: cache has {len(history)} history turns, sample has {len(sample.history)}")
        return Transcript(final=str(entry["final"]), history=history, seconds=0.0)

    def warmup(self) -> None:
        """Không cần làm gì."""


def build_cache(samples: Sequence[Sample], transcriber: Transcriber) -> tuple[dict[str, Any], float]:
    """Chép lời mọi mẫu theo thứ tự.

    Tham số:
        samples: các mẫu cần chép lời.
        transcriber: transcriber ASR.

    Trả về:
        Cặp (cache, tổng số giây chép lời, không tính warmup).
    """
    transcriber.warmup()
    cache: dict[str, Any] = {}
    t0 = time.perf_counter()
    for i, sample in enumerate(samples, 1):
        cache[sample.id] = to_entry(transcriber.transcribe(sample))
        if i % 100 == 0:
            log.info("asr_cache_progress", done=i, total=len(samples), seconds=round(time.perf_counter() - t0, 1))
    return cache, time.perf_counter() - t0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    ap = argparse.ArgumentParser(prog="python -m lifemate.train.asr_cache", description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=Path, help="run config whose [asr] section is used")
    ap.add_argument("--env", default="local", help="env profile name or path")
    ap.add_argument("--ids", required=True, type=Path, nargs="+", help="one or more files with one sample id per line")
    ap.add_argument("--out", required=True, type=Path, help="output JSON (e.g. data/asr/train_asr.json)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--root", help="data root override")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None, repo_root: Path = REPO_ROOT) -> int:
    """Chạy CLI: nạp config và mẫu, chép lời, ghi cache và file meta; trả về mã thoát 0."""
    from lifemate.backends.asr import ASRTranscriber

    args = parse_args(argv)
    cfg = load_config(args.config if args.config.is_absolute() else repo_root / args.config)
    profile = load_profile(args.env, repo_root).with_overrides(root=args.root)
    root = Path(profile.data_root)
    root = root if root.is_absolute() else repo_root / root
    ids: list[str] = []
    # Gộp nhiều file id, giữ thứ tự xuất hiện và bỏ id trùng.
    for path in args.ids:
        ids += [i for i in read_ids(path if path.is_absolute() else repo_root / path) if i not in ids]
    samples = load_split(root, args.split, ids)
    # Ép backend "asr" để cache luôn là đầu ra ASR thật, kể cả khi config chạy dùng backend khác.
    asr_cfg = dataclasses.replace(cfg.asr, backend="asr")
    cache, seconds = build_cache(samples, ASRTranscriber(asr_cfg, data_root=root, split=args.split))
    out = args.out if args.out.is_absolute() else repo_root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cache, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    meta = {
        "asr": dataclasses.asdict(asr_cfg),
        "ids": [str(p) for p in args.ids],
        "split": args.split,
        "n_samples": len(cache),
        "seconds": round(seconds, 2),
    }
    out.with_name(out.name + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"samples={len(cache)} seconds={seconds:.1f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
