"""Khung train dùng chung cho mọi trainer (LLM, ASR): seed, checkpoint, metrics, `train_config.json`, đọc TOML chặt
(SPEC 6.5).

Nội dung chính:
- `seed_everything`: đặt seed cho Python, NumPy, PyTorch và bật chế độ tất định.
- `capture_rng`, `restore_rng`: lưu và khôi phục trạng thái bộ sinh số ngẫu nhiên (RNG).
- `CheckpointManager`, `CheckpointInfo`: lưu, liệt kê, dọn và nạp checkpoint.
- `MetricsLogger`: ghi metrics ra `metrics.jsonl`, mỗi dòng một bản ghi.
- `write_train_config`, `write_json_atomic`, `sha256_file`: ghi config của job train và băm file dữ liệu.
- `build_section`, `load_toml`: đọc config TOML, từ chối khoá lạ hoặc thiếu.
- `abs_path`, `read_held_out`: tiện ích đường dẫn và đọc danh sách app held-out.

Cách hoạt động:
- `CheckpointManager` ghi checkpoint vào thư mục tạm rồi đổi tên nguyên tử, kèm trạng thái RNG, để chạy tiếp
  (resume) ra đúng kết quả như chạy liền một mạch.
- `MetricsLogger` ghi jsonl từng dòng và flush ngay.
- `write_train_config` lưu config, `EnvInfo` và sha256 của dữ liệu để truy vết job.

Lưu ý:
- numpy và torch là tuỳ chọn; thiếu gói nào thì bỏ qua phần tương ứng.
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
from lifemate.core.paths import repo_path

log = structlog.get_logger(__name__)

STATE_FILE = "ckpt_meta.json"
TORCH_RNG_FILE = "rng_torch.pt"
CKPT_PREFIX = "ckpt-"
_HASH_CHUNK = 1 << 20

_T = TypeVar("_T")


def _optional_module(name: str) -> ModuleType | None:
    """Import gói tuỳ chọn (numpy, torch từ extra `gpu` hoặc gói train); không có thì trả về `None`."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Đặt seed cho Python, NumPy và PyTorch (CPU và CUDA).

    Tham số:
        seed: giá trị seed.
        deterministic: `True` để bật thuật toán tất định của PyTorch khi có thể (chậm hơn nhưng chạy lại ra cùng
            kết quả).
    """
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
        # cuBLAS cần biến này thì phép nhân ma trận mới tất định.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            # `warn_only=True`: phép tính không có bản tất định chỉ cảnh báo, không làm hỏng job.
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError, TypeError) as exc:
            log.warning("deterministic_algorithms_unavailable", error=str(exc))
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _to_json(obj: Any) -> Any:
    """Chuyển đệ quy tuple và mảng numpy thành list, để ghi được ra JSON."""
    if isinstance(obj, tuple | list):
        return [_to_json(x) for x in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def _to_tuple(obj: Any) -> Any:
    """Ngược của `_to_json` cho list: list lồng nhau thành tuple, vì `random.setstate` cần tuple."""
    if isinstance(obj, list):
        return tuple(_to_tuple(x) for x in obj)
    return obj


def capture_rng() -> dict[str, Any]:
    """Lấy trạng thái RNG của Python và NumPy ở dạng ghi được ra JSON."""
    state: dict[str, Any] = {"python": _to_json(random.getstate())}
    np = _optional_module("numpy")
    if np is not None:
        state["numpy"] = _to_json(np.random.get_state())
    return state


def restore_rng(state: Mapping[str, Any]) -> None:
    """Khôi phục trạng thái RNG đã lấy bằng `capture_rng`."""
    if "python" in state:
        random.setstate(_to_tuple(state["python"]))
    np = _optional_module("numpy")
    if np is not None and "numpy" in state:
        # Dựng lại tuple trạng thái của NumPy, ép đúng kiểu cho từng phần (JSON làm mất kiểu gốc).
        name, keys, pos, has_gauss, cached = state["numpy"]
        np.random.set_state((name, np.asarray(keys, dtype=np.uint32), int(pos), int(has_gauss), float(cached)))


def _save_torch_rng(path: Path) -> None:
    """Lưu RNG của torch (CPU, và CUDA nếu có) vào file nhị phân; không có torch thì bỏ qua."""
    torch = _optional_module("torch")
    if torch is None:
        return
    rng: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng, path)


def _load_torch_rng(path: Path) -> None:
    """Khôi phục RNG của torch từ file; thiếu torch hoặc thiếu file thì bỏ qua."""
    torch = _optional_module("torch")
    if torch is None or not path.is_file():
        return
    rng = torch.load(path, weights_only=False)
    torch.set_rng_state(rng["cpu"])
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    """Một checkpoint đã lưu.

    Thuộc tính:
        path: thư mục checkpoint.
        step: step lúc lưu.
        metric: giá trị metric lúc lưu; `None` khi không đo.
    """

    path: Path
    step: int
    metric: float | None


class CheckpointManager:
    """Quản lý thư mục chứa các checkpoint `ckpt-<step>`, mỗi checkpoint có file `ckpt_meta.json`.

    Giữ lại `keep_last` bản mới nhất cộng `keep_best` bản tốt nhất theo một metric; các bản khác bị xoá.

    Tham số:
        directory: thư mục chứa checkpoint; được tạo nếu chưa có.
        keep_last: số bản mới nhất cần giữ, tối thiểu 1.
        keep_best: số bản tốt nhất cần giữ.
        metric_name: tên metric dùng để xếp hạng.
        lower_is_better: `True` khi metric càng nhỏ càng tốt (ví dụ loss).

    Lỗi:
        ValueError: khi `keep_last < 1`, vì chạy tiếp cần checkpoint mới nhất.
    """

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
        """Trả về thư mục của `step`; số step đệm đủ 8 chữ số để sắp theo tên cũng đúng thứ tự."""
        return self.directory / f"{CKPT_PREFIX}{step:08d}"

    def save(self, step: int, metric: float | None, save_fn: Callable[[Path], None]) -> CheckpointInfo:
        """Lưu một checkpoint mới rồi dọn bớt các bản cũ.

        Các bước: ghi trọng số qua `save_fn(tmp_dir)`, ghi RNG và step vào `ckpt_meta.json`, đổi tên thư mục tạm thành
        thư mục chính, rồi gọi `prune`. `ckpt_meta.json` được ghi cuối cùng và fsync, nên thư mục nào thiếu file này là
        checkpoint dở dang.

        Tham số:
            step: step hiện tại.
            metric: giá trị metric; `None` khi không đo.
            save_fn: hàm ghi trọng số vào thư mục được truyền vào.

        Trả về:
            `CheckpointInfo` của checkpoint vừa lưu.
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
        # Lưu lại cùng step (ví dụ sau khi chạy tiếp) thì thay bản cũ.
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
        log.info("checkpoint_saved", step=step, metric=metric, path=str(final))
        self.prune()
        return CheckpointInfo(final, step, metric)

    def all(self) -> list[CheckpointInfo]:
        """Liệt kê các checkpoint hoàn chỉnh (có `ckpt_meta.json`), sắp theo step."""
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
        """Trả về checkpoint có step cao nhất, hoặc `None` khi chưa có."""
        ckpts = self.all()
        return ckpts[-1] if ckpts else None

    def best_k(self, k: int) -> list[CheckpointInfo]:
        """Trả về tối đa `k` checkpoint có metric, tốt nhất trước; bằng nhau thì step sớm hơn đứng trước."""
        scored = [c for c in self.all() if c.metric is not None]
        # Đổi dấu khi metric càng lớn càng tốt, để luôn sắp tăng dần.
        sign = 1.0 if self.lower_is_better else -1.0
        return sorted(scored, key=lambda c: (sign * float(c.metric or 0.0), c.step))[:k]

    def best(self) -> CheckpointInfo | None:
        """Trả về checkpoint tốt nhất theo metric, hoặc `None` khi chưa có bản nào có metric."""
        top = self.best_k(1)
        return top[0] if top else None

    def prune(self) -> list[Path]:
        """Xoá checkpoint không thuộc `keep_last` bản mới nhất và không thuộc `keep_best` bản tốt nhất.

        Đồng thời dọn các thư mục tạm còn sót từ lần lưu bị ngắt.

        Trả về:
            Đường dẫn các checkpoint đã xoá.
        """
        ckpts = self.all()
        keep = {c.path for c in ckpts[-self.keep_last :]} | {c.path for c in self.best_k(self.keep_best)}
        removed = [c.path for c in ckpts if c.path not in keep]
        for p in removed:
            shutil.rmtree(p)
        for stale in self.directory.glob(f".tmp-{CKPT_PREFIX}*"):
            shutil.rmtree(stale)
        return removed

    def load(self, path: Path, load_fn: Callable[[Path], None] | None = None) -> CheckpointInfo:
        """Nạp một checkpoint: gọi `load_fn(path)` rồi khôi phục trạng thái RNG.

        Tham số:
            path: thư mục checkpoint.
            load_fn: hàm nạp trọng số; `None` thì chỉ khôi phục RNG.

        Trả về:
            `CheckpointInfo` với step và metric của checkpoint.
        """
        path = Path(path)
        state = json.loads((path / STATE_FILE).read_text(encoding="utf-8"))
        if load_fn is not None:
            load_fn(path)
        restore_rng(state.get("rng") or {})
        _load_torch_rng(path / TORCH_RNG_FILE)
        metric = state.get("metric")
        return CheckpointInfo(path, int(state["step"]), None if metric is None else float(metric))


class MetricsLogger:
    """Ghi nối mỗi bản ghi thành một dòng JSON vào `metrics.jsonl`, flush ngay từng dòng.

    Tham số:
        path: file `metrics.jsonl`; thư mục cha được tạo nếu chưa có.

    Thuộc tính:
        path: như tham số cùng tên.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Mapping[str, Any]) -> None:
        """Ghi nối `record`, kèm trường `time` theo đồng hồ thực."""
        row = {"time": time.time(), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(canonical_json(row) + "\n")
            f.flush()

    def read(self) -> list[dict[str, Any]]:
        """Đọc mọi dòng đã ghi tới thời điểm hiện tại; trả về danh sách rỗng khi file chưa có."""
        if not self.path.is_file():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(ln) for ln in f if ln.strip()]


def sha256_file(path: Path) -> str:
    """Tính sha256 dạng hex của một file, đọc theo từng khối 1 MB."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def write_train_config(
    path: Path, config: Mapping[str, Any], env_info: EnvInfo, data_sha256: Mapping[str, str], **extra: Any
) -> None:
    """Ghi `train_config.json`: config, `EnvInfo`, sha256 dữ liệu và các trường thêm (checkpoint, loss, seed).

    File được ghi qua file `.tmp` rồi `os.replace`, nên không bao giờ thấy file ghi dở.

    Tham số:
        path: file đích.
        config: config của job train.
        env_info: thông tin máy.
        data_sha256: tên file dữ liệu tới sha256.
        extra: các trường thêm, ghi ở cấp cao nhất.
    """
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
    """Ghi JSON có thụt lề qua file tạm rồi đổi tên."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def abs_path(path: str | Path, repo_root: Path) -> Path:
    """Mở rộng `~` và tính đường dẫn tương đối từ thư mục gốc repo (dùng `lifemate.core.paths.repo_path`)."""
    return repo_path(path, repo_root)


def build_section(cls: type[_T], section: str, raw: Any) -> _T:
    """Tạo dataclass từ một bảng TOML, giống `lifemate.core.config`.

    Tham số:
        cls: lớp dataclass cần tạo.
        section: tên mục, dùng trong thông báo lỗi.
        raw: giá trị đọc từ TOML.

    Trả về:
        Đối tượng `cls`.

    Lỗi:
        ConfigError: khi `raw` không phải bảng, có khoá lạ hoặc thiếu khoá.
    """
    if not isinstance(raw, dict):
        raise ConfigError(f"[{section}] must be a table")
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    for key in raw:
        if key not in names:
            raise ConfigError(f"unknown key {section}.{key}")
    # Báo khoá thiếu đầu tiên theo thứ tự chữ cái, để thông báo lỗi luôn giống nhau.
    for name in sorted(names - set(raw)):
        raise ConfigError(f"missing key {section}.{name}")
    return cls(**raw)


def load_toml(path: Path, sections: Sequence[str]) -> dict[str, Any]:
    """Đọc file TOML và yêu cầu cấp cao nhất có đúng các mục trong `sections`.

    Lỗi:
        ConfigError: khi có mục lạ hoặc thiếu mục.
    """
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
    """Đọc `held_out_apps` từ `configs/splits/meta.json`.

    File này bị git bỏ qua, nên bản clone mới phải chạy `make splits` trước. Thiếu file là lỗi, không bao giờ trả về
    tập rỗng, vì như vậy sẽ để lộ schema của các app held-out vào dữ liệu train.

    Lỗi:
        FileNotFoundError: khi không có `configs/splits/meta.json`.
    """
    meta = repo_root / "configs" / "splits" / "meta.json"
    if not meta.is_file():
        raise FileNotFoundError(f"{meta} is missing: run `make splits`")
    return frozenset(json.loads(meta.read_text(encoding="utf-8")).get("held_out_apps", []))
