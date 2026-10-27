"""Khung huấn luyện dùng chung: seed, checkpoint, metrics, train_config.json, mục TOML chặt chẽ (SPEC 6.5).

Cách xây dựng: mọi trainer (LLM, ASR) dùng chung các khối này. `CheckpointManager` ghi checkpoint vào thư mục tạm
rồi đổi tên nguyên tử, kèm trạng thái RNG để chạy tiếp tái lập được; `MetricsLogger` ghi jsonl từng dòng;
`write_train_config` lưu config, EnvInfo và sha256 dữ liệu; `build_section`/`load_toml` từ chối khóa lạ hoặc thiếu.
numpy và torch là tùy chọn, thiếu thì bỏ qua phần tương ứng.
"""

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

from lifemate.core.config import ConfigError
from lifemate.core.envinfo import EnvInfo
from lifemate.core.jsonutil import canonical_json

log = structlog.get_logger(__name__)

STATE_FILE = "ckpt_meta.json"
TORCH_RNG_FILE = "rng_torch.pt"
CKPT_PREFIX = "ckpt-"
_HASH_CHUNK = 1 << 20

_T = TypeVar("_T")


def _optional_module(name: str) -> ModuleType | None:
    """Import gói tùy chọn (numpy, torch từ gói phụ `gpu`/train) hoặc trả về None."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Đặt seed cho Python, NumPy và PyTorch (CPU và CUDA); bật thuật toán tất định của PyTorch khi có thể."""
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
        # cuBLAS cần biến này để phép nhân ma trận tất định.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError, TypeError) as exc:
            log.warning("deterministic_algorithms_unavailable", error=str(exc))
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _to_json(obj: Any) -> Any:
    """Chuyển tuple và mảng numpy thành list, đệ quy."""
    if isinstance(obj, tuple | list):
        return [_to_json(x) for x in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def _to_tuple(obj: Any) -> Any:
    """Ngược của `_to_json` cho list: list lồng nhau thành tuple (random.setstate cần tuple)."""
    if isinstance(obj, list):
        return tuple(_to_tuple(x) for x in obj)
    return obj


def capture_rng() -> dict[str, Any]:
    """Trạng thái RNG của Python và NumPy ở dạng tuần tự hóa JSON được."""
    state: dict[str, Any] = {"python": _to_json(random.getstate())}
    np = _optional_module("numpy")
    if np is not None:
        state["numpy"] = _to_json(np.random.get_state())
    return state


def restore_rng(state: Mapping[str, Any]) -> None:
    """Ngược của `capture_rng`."""
    if "python" in state:
        random.setstate(_to_tuple(state["python"]))
    np = _optional_module("numpy")
    if np is not None and "numpy" in state:
        # Dựng lại tuple trạng thái của NumPy với đúng kiểu từng phần.
        name, keys, pos, has_gauss, cached = state["numpy"]
        np.random.set_state((name, np.asarray(keys, dtype=np.uint32), int(pos), int(has_gauss), float(cached)))


def _save_torch_rng(path: Path) -> None:
    """Lưu RNG của torch (CPU, CUDA nếu có) vào file nhị phân; không có torch thì bỏ qua."""
    torch = _optional_module("torch")
    if torch is None:
        return
    rng: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng, path)


def _load_torch_rng(path: Path) -> None:
    """Khôi phục RNG của torch từ file; thiếu torch hoặc file thì bỏ qua."""
    torch = _optional_module("torch")
    if torch is None or not path.is_file():
        return
    rng = torch.load(path, weights_only=False)
    torch.set_rng_state(rng["cpu"])
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    """Một checkpoint đã lưu: đường dẫn, step và metric (có thể None)."""

    path: Path
    step: int
    metric: float | None


class CheckpointManager:
    """Thư mục chứa các `ckpt-<step>` kèm `ckpt_meta.json`; giữ N bản mới nhất cộng N bản tốt nhất theo một metric."""

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
        """Thư mục của `step`, đệm 8 chữ số để sắp theo tên cũng đúng thứ tự."""
        return self.directory / f"{CKPT_PREFIX}{step:08d}"

    def save(self, step: int, metric: float | None, save_fn: Callable[[Path], None]) -> CheckpointInfo:
        """Ghi trọng số qua `save_fn(tmp_dir)`, rồi RNG và step vào ckpt_meta.json; đổi tên nguyên tử; dọn bớt.

        ckpt_meta.json được ghi cuối cùng và fsync, nên thư mục thiếu file này là checkpoint dở dang.
        """
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
        """Các checkpoint hoàn chỉnh (có ckpt_meta.json), sắp theo step."""
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
        """Checkpoint có step cao nhất."""
        ckpts = self.all()
        return ckpts[-1] if ckpts else None

    def best_k(self, k: int) -> list[CheckpointInfo]:
        """Tối đa `k` checkpoint có metric, tốt nhất trước (bằng nhau thì step sớm hơn trước)."""
        scored = [c for c in self.all() if c.metric is not None]
        sign = 1.0 if self.lower_is_better else -1.0
        return sorted(scored, key=lambda c: (sign * float(c.metric or 0.0), c.step))[:k]

    def best(self) -> CheckpointInfo | None:
        """Checkpoint tốt nhất theo metric."""
        top = self.best_k(1)
        return top[0] if top else None

    def prune(self) -> list[Path]:
        """Xóa checkpoint không thuộc `keep_last` bản mới nhất lẫn `keep_best` bản tốt nhất; dọn thư mục tạm sót lại."""
        ckpts = self.all()
        keep = {c.path for c in ckpts[-self.keep_last :]} | {c.path for c in self.best_k(self.keep_best)}
        removed = [c.path for c in ckpts if c.path not in keep]
        for p in removed:
            shutil.rmtree(p)
        for stale in self.directory.glob(f".tmp-{CKPT_PREFIX}*"):
            shutil.rmtree(stale)
        return removed

    def load(self, path: Path, load_fn: Callable[[Path], None] | None = None) -> CheckpointInfo:
        """Gọi `load_fn(path)`, khôi phục trạng thái RNG, trả về step và metric của checkpoint."""
        path = Path(path)
        state = json.loads((path / STATE_FILE).read_text(encoding="utf-8"))
        if load_fn is not None:
            load_fn(path)
        restore_rng(state.get("rng") or {})
        _load_torch_rng(path / TORCH_RNG_FILE)
        metric = state.get("metric")
        return CheckpointInfo(path, int(state["step"]), None if metric is None else float(metric))


class MetricsLogger:
    """Ghi nối một đối tượng JSON mỗi dòng vào metrics.jsonl và flush từng dòng."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Mapping[str, Any]) -> None:
        """Ghi nối `record` kèm trường `time` theo đồng hồ thực."""
        row = {"time": time.time(), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(canonical_json(row) + "\n")
            f.flush()

    def read(self) -> list[dict[str, Any]]:
        """Mọi dòng đã ghi đến thời điểm hiện tại."""
        if not self.path.is_file():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(ln) for ln in f if ln.strip()]


def sha256_file(path: Path) -> str:
    """sha256 dạng hex của một file, đọc theo từng khối."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def write_train_config(
    path: Path, config: Mapping[str, Any], env_info: EnvInfo, data_sha256: Mapping[str, str], **extra: Any
) -> None:
    """Ghi train_config.json: config, EnvInfo, sha256 dữ liệu và các trường thêm (checkpoint, loss, seed)."""
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
    """Ghi JSON thụt lề qua file tạm rồi đổi tên."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def abs_path(path: str | Path, repo_root: Path) -> Path:
    """Mở rộng `~`; đường dẫn tương đối tính từ gốc repo (cùng quy tắc với lifemate.run.cli)."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else repo_root / p


def build_section(cls: type[_T], section: str, raw: Any) -> _T:
    """Tạo dataclass từ một bảng TOML; khóa lạ hoặc thiếu gây ConfigError (giống lifemate.core.config)."""
    if not isinstance(raw, dict):
        raise ConfigError(f"[{section}] must be a table")
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    for key in raw:
        if key not in names:
            raise ConfigError(f"unknown key {section}.{key}")
    # Báo khóa thiếu đầu tiên theo thứ tự chữ cái để thông báo lỗi ổn định.
    for name in sorted(names - set(raw)):
        raise ConfigError(f"missing key {section}.{name}")
    return cls(**raw)


def load_toml(path: Path, sections: Sequence[str]) -> dict[str, Any]:
    """Đọc TOML và yêu cầu cấp cao nhất có đúng các mục `sections`."""
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
    """Đọc `held_out_apps` từ configs/splits/meta.json.

    File bị git bỏ qua, nên bản clone mới phải chạy `make splits` trước; thiếu file là lỗi, không bao giờ trả về
    tập rỗng (vì như vậy sẽ rò rỉ schema của các app giữ lại).
    """
    meta = repo_root / "configs" / "splits" / "meta.json"
    if not meta.is_file():
        raise FileNotFoundError(f"{meta} is missing: run `make splits`")
    return frozenset(json.loads(meta.read_text(encoding="utf-8")).get("held_out_apps", []))
