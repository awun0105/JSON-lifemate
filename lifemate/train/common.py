"""Shared training frame: seeding, checkpoints, metrics, train_config.json, strict TOML sections (SPEC 6.5)."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import os
import random
import shutil
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar

import structlog

from lifemate.config import ConfigError
from lifemate.envinfo import EnvInfo
from lifemate.jsonutil import canonical_json

log = structlog.get_logger(__name__)

STATE_FILE = "ckpt_meta.json"
TORCH_RNG_FILE = "rng_torch.pt"
CKPT_PREFIX = "ckpt-"
_HASH_CHUNK = 1 << 20

_T = TypeVar("_T")


def _optional_module(name: str) -> ModuleType | None:
    """Import an optional package (numpy, torch from the `gpu`/train extras) or return None."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA); turn on PyTorch deterministic algorithms when possible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np = _optional_module("numpy")
    if np is not None:
        np.random.seed(seed)
    torch = _optional_module("torch")
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError, TypeError) as exc:
            log.warning("deterministic_algorithms_unavailable", error=str(exc))
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _to_json(obj: Any) -> Any:
    """Tuples and numpy arrays to lists, recursively."""
    if isinstance(obj, tuple | list):
        return [_to_json(x) for x in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def _to_tuple(obj: Any) -> Any:
    if isinstance(obj, list):
        return tuple(_to_tuple(x) for x in obj)
    return obj


def capture_rng() -> dict[str, Any]:
    """JSON-serialisable Python and NumPy RNG states."""
    state: dict[str, Any] = {"python": _to_json(random.getstate())}
    np = _optional_module("numpy")
    if np is not None:
        state["numpy"] = _to_json(np.random.get_state())
    return state


def restore_rng(state: Mapping[str, Any]) -> None:
    """Inverse of `capture_rng`."""
    if "python" in state:
        random.setstate(_to_tuple(state["python"]))
    np = _optional_module("numpy")
    if np is not None and "numpy" in state:
        name, keys, pos, has_gauss, cached = state["numpy"]
        np.random.set_state((name, np.asarray(keys, dtype=np.uint32), int(pos), int(has_gauss), float(cached)))


def _save_torch_rng(path: Path) -> None:
    torch = _optional_module("torch")
    if torch is None:
        return
    rng: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng, path)


def _load_torch_rng(path: Path) -> None:
    torch = _optional_module("torch")
    if torch is None or not path.is_file():
        return
    rng = torch.load(path, weights_only=False)
    torch.set_rng_state(rng["cpu"])
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    """One saved checkpoint."""

    path: Path
    step: int
    metric: float | None


class CheckpointManager:
    """Directory of `ckpt-<step>` folders with `ckpt_meta.json`; keeps the last N plus the best N by one metric."""

    def __init__(
        self,
        directory: Path,
        *,
        keep_last: int = 2,
        keep_best: int = 2,
        metric_name: str = "eval_unseen_loss",
        lower_is_better: bool = True,
    ) -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1 (resume needs the latest checkpoint)")
        self.directory = Path(directory)
        self.keep_last = keep_last
        self.keep_best = keep_best
        self.metric_name = metric_name
        self.lower_is_better = lower_is_better
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, step: int) -> Path:
        return self.directory / f"{CKPT_PREFIX}{step:08d}"

    def save(self, step: int, metric: float | None, save_fn: Callable[[Path], None]) -> CheckpointInfo:
        """Write weights via `save_fn(tmp_dir)`, then RNG and step to ckpt_meta.json; rename atomically; prune."""
        final = self._path(step)
        tmp = self.directory / f".tmp-{final.name}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        save_fn(tmp)
        _save_torch_rng(tmp / TORCH_RNG_FILE)
        state = {
            "step": step,
            "metric": metric,
            "metric_name": self.metric_name,
            "rng": capture_rng(),
            "saved_at": time.time(),
        }
        with open(tmp / STATE_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(state))
            f.flush()
            os.fsync(f.fileno())
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
        log.info("checkpoint_saved", step=step, metric=metric, path=str(final))
        self.prune()
        return CheckpointInfo(final, step, metric)

    def all(self) -> list[CheckpointInfo]:
        """Complete checkpoints sorted by step."""
        out: list[CheckpointInfo] = []
        for d in self.directory.glob(f"{CKPT_PREFIX}*"):
            state_file = d / STATE_FILE
            if not state_file.is_file():
                continue
            state = json.loads(state_file.read_text(encoding="utf-8"))
            metric = state.get("metric")
            out.append(CheckpointInfo(d, int(state["step"]), None if metric is None else float(metric)))
        return sorted(out, key=lambda c: c.step)

    def latest(self) -> CheckpointInfo | None:
        """Checkpoint with the highest step."""
        ckpts = self.all()
        return ckpts[-1] if ckpts else None

    def best_k(self, k: int) -> list[CheckpointInfo]:
        """Up to `k` checkpoints with a metric, best first (ties: earlier step first)."""
        scored = [c for c in self.all() if c.metric is not None]
        sign = 1.0 if self.lower_is_better else -1.0
        return sorted(scored, key=lambda c: (sign * float(c.metric or 0.0), c.step))[:k]

    def best(self) -> CheckpointInfo | None:
        """Best checkpoint by the metric."""
        top = self.best_k(1)
        return top[0] if top else None

    def prune(self) -> list[Path]:
        """Delete checkpoints that are neither among the last `keep_last` nor the best `keep_best`."""
        ckpts = self.all()
        keep = {c.path for c in ckpts[-self.keep_last :]} | {c.path for c in self.best_k(self.keep_best)}
        removed = [c.path for c in ckpts if c.path not in keep]
        for p in removed:
            shutil.rmtree(p)
        for stale in self.directory.glob(f".tmp-{CKPT_PREFIX}*"):
            shutil.rmtree(stale)
        return removed

    def load(self, path: Path, load_fn: Callable[[Path], None] | None = None) -> CheckpointInfo:
        """Call `load_fn(path)`, restore RNG states, return the checkpoint's step and metric."""
        path = Path(path)
        state = json.loads((path / STATE_FILE).read_text(encoding="utf-8"))
        if load_fn is not None:
            load_fn(path)
        restore_rng(state.get("rng") or {})
        _load_torch_rng(path / TORCH_RNG_FILE)
        metric = state.get("metric")
        return CheckpointInfo(path, int(state["step"]), None if metric is None else float(metric))


class MetricsLogger:
    """Appends one JSON object per line to metrics.jsonl and flushes each line."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Mapping[str, Any]) -> None:
        """Append `record` plus a wall-clock `time` field."""
        row = {"time": time.time(), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(canonical_json(row) + "\n")
            f.flush()

    def read(self) -> list[dict[str, Any]]:
        """All rows logged so far."""
        if not self.path.is_file():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(ln) for ln in f if ln.strip()]


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def write_train_config(
    path: Path, config: Mapping[str, Any], env_info: EnvInfo, data_sha256: Mapping[str, str], **extra: Any
) -> None:
    """Write train_config.json: config, EnvInfo, data sha256 and any extra fields (checkpoint, loss, seed)."""
    payload = {
        "config": dict(config),
        "env_info": dataclasses.asdict(env_info),
        "data_sha256": dict(data_sha256),
        **extra,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write indented JSON via a temp file and rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def abs_path(path: str | Path, repo_root: Path) -> Path:
    """Expand `~`; relative paths are taken from the repo root (same rule as lifemate.cli)."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else repo_root / p


def build_section(cls: type[_T], section: str, raw: Any) -> _T:
    """Dataclass from one TOML table; unknown or missing keys raise ConfigError (like lifemate.config)."""
    if not isinstance(raw, dict):
        raise ConfigError(f"[{section}] must be a table")
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    for key in raw:
        if key not in names:
            raise ConfigError(f"unknown key {section}.{key}")
    for name in sorted(names - set(raw)):
        raise ConfigError(f"missing key {section}.{name}")
    return cls(**raw)


def load_toml(path: Path, sections: Sequence[str]) -> dict[str, Any]:
    """Parse TOML and require exactly `sections` at the top level."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    for key in raw:
        if key not in sections:
            raise ConfigError(f"unknown key {key}")
    for key in sections:
        if key not in raw:
            raise ConfigError(f"missing key {key}")
    return raw


def read_held_out(repo_root: Path) -> frozenset[str]:
    """`held_out_apps` from splits/meta.json (empty when the file is absent)."""
    meta = repo_root / "splits" / "meta.json"
    if not meta.is_file():
        return frozenset()
    return frozenset(json.loads(meta.read_text(encoding="utf-8")).get("held_out_apps", []))
