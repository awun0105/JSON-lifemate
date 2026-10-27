"""SFT cho LLM bằng ms-swift (`swift sft`), trainer của trưởng nhóm (ThanhDanh1510/ZaloAI2026 docs/llm_finetuning.md).

Nội dung: xuất dòng SFT sang định dạng ms-swift, kiểm tra template, dựng argv cho `swift sft`, chạy tiến trình con
và đọc các checkpoint đã ghi.

Module thay cho trainer Unsloth từng nằm trong `sft.train` (vẫn giữ ở đó dưới dạng chú thích). Phần vẫn do dự án
tự làm: các dòng SFT (`sft_data`), chọn checkpoint theo điểm agent, merge và registry (`llm_pipeline`).

Luồng dữ liệu của `train`:
1. `export_rows` chép dòng train và val_unseen, chỉ giữ `messages` (kèm cờ `loss` từng lượt).
2. `check_template` mã hoá một số dòng mẫu theo cả hai cách và dừng nếu khác nhau.
3. `swift_argv` dựng tham số; nếu đã có checkpoint thì nối tiếp từ checkpoint mới nhất.
4. `runner` gọi `swift sft`; sau đó `checkpoints`/`best_k`/`latest` đọc kết quả từ trainer_state.json.

An toàn train/serve: ms-swift dựng hội thoại bằng template riêng (`qwen3_nothinking` cho
Qwen3-4B-Instruct-2507), còn vLLM phục vụ bằng chat template của tokenizer. `check_template` mã hoá các dòng mẫu
theo cả hai cách trước khi train và dừng ở điểm khác biệt đầu tiên về input ids hoặc nhãn loss. Đo trên 60 dòng thật
(25 dòng có phục hồi lỗi) với ms-swift 4.6.0: giống hệt nhau.

Checkpoint: `swift sft` ghi `<out>/swift/checkpoint-<step>/` (adapter PEFT + trainer_state.json). Chỉ truyền một
file val là val_unseen, vì ms-swift gộp nhiều file val thành một loss duy nhất; do đó `eval_loss` của ms-swift
chính là `eval_unseen_loss` của dự án.

Các gói nặng (ms-swift, torch, transformers) chỉ được import bên trong hàm cần dùng.
"""

from __future__ import annotations

import importlib
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from lifemate.core.config import ConfigError
from lifemate.train.common import abs_path
from lifemate.train.sft import LLMTrainConfig, build_labels, read_jsonl, template_kwargs

log = structlog.get_logger(__name__)

SWIFT_VERSION = "4.6.0"  # ops/requirements/train.txt; tên tham số bên dưới đã đối chiếu với phiên bản này
CKPT_GLOB = "checkpoint-*"
N_TEMPLATE_CHECK = 20

Runner = Callable[[Sequence[str], Path], int]


class TemplateMismatchError(RuntimeError):
    """ms-swift và chat template lúc phục vụ mã hoá một dòng khác nhau: dữ liệu train sẽ lệch so với lúc chạy."""


@dataclass(frozen=True, slots=True)
class SwiftCheckpoint:
    """Một thư mục `checkpoint-<step>` do `swift sft` ghi ra.

    `eval_loss` là loss trên val_unseen được ghi đúng tại bước đó, hoặc None nếu bước đó không có lượt eval.
    """

    path: Path
    step: int
    eval_loss: float | None


def export_rows(src: Path, dst: Path) -> int:
    """Chép các dòng `sft_data`, chỉ giữ `messages` (kèm cờ `loss` của từng lượt); trả về số dòng.

    Các khoá khác (id, metadata) bị bỏ để ms-swift không hiểu nhầm thành cột dữ liệu.
    """
    rows = read_jsonl(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for row in rows:
            msgs = [{k: m[k] for k in ("role", "content", "loss") if k in m} for m in row["messages"]]
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    return len(rows)


def pick_check_rows(rows: Sequence[Mapping[str, Any]], n: int = N_TEMPLATE_CHECK) -> list[Mapping[str, Any]]:
    """Lấy tối đa `n` dòng: mọi dòng phục hồi lỗi trước (có `loss: false`), sau đó các dòng còn lại theo thứ tự.

    Dòng phục hồi lỗi được ưu tiên vì cờ `loss: false` là chỗ dễ lệch nhãn nhất giữa hai template.
    """
    special = [r for r in rows if any(m.get("loss") is False for m in r["messages"])]
    rest = [r for r in rows if r not in special]
    return [*special, *rest][:n]


def compare_encodings(
    row_id: str, ours: tuple[list[int], list[int]], theirs: tuple[list[int], list[int]], decode: Callable[..., str]
) -> None:
    """Ném TemplateMismatchError tại token đầu tiên mà input ids hoặc nhãn khác nhau.

    `ours` và `theirs` là cặp (input_ids, labels); thông báo lỗi kèm đoạn văn bản giải mã quanh vị trí lệch.
    """
    for what, a, b in (("input_ids", ours[0], theirs[0]), ("labels", ours[1], theirs[1])):
        if a == b:
            continue
        # Vị trí lệch đầu tiên; nếu một dãy là tiền tố của dãy kia thì lệch tại độ dài ngắn hơn.
        i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        context = decode(theirs[0][max(0, i - 12) : i + 12])
        raise TemplateMismatchError(
            f"{row_id}: {what} differ at token {i} (ours {len(a)}, swift {len(b)}): {context!r}"
        )


def check_template(cfg: LLMTrainConfig, rows: Sequence[Mapping[str, Any]]) -> int:
    """Mã hoá `rows` bằng template train của ms-swift và bằng chat template lúc phục vụ; ném lỗi nếu có khác biệt.

    Phía dự án dùng `build_labels` (chat template của tokenizer, giống vLLM); phía ms-swift dùng
    `get_template(...).encode` ở chế độ train. Trả về số dòng đã kiểm tra.
    """
    swift_model = importlib.import_module("swift.model")
    swift_template = importlib.import_module("swift.template")
    proc = swift_model.get_processor(cfg.base.model, use_hf=True, revision=cfg.base.revision or None)
    tpl = swift_template.get_template(proc, max_length=None)
    tpl.set_mode("train")
    kwargs = template_kwargs(proc)
    checked = pick_check_rows(rows)
    for row in checked:
        msgs = [{k: m[k] for k in ("role", "content", "loss") if k in m} for m in row["messages"]]
        enc = tpl.encode({"messages": msgs})
        theirs = ([int(x) for x in enc["input_ids"]], [int(x) for x in enc["labels"]])
        compare_encodings(str(row.get("id")), build_labels(proc, row["messages"], kwargs), theirs, proc.decode)
    log.info("swift_template_checked", rows=len(checked))
    return len(checked)


def swift_argv(
    cfg: LLMTrainConfig,
    *,
    train_file: Path,
    val_file: Path,
    out_dir: Path,
    dtype: str,
    resume_from: Path | None = None,
) -> list[str]:
    """Dựng danh sách tham số của `swift sft` cho một lần chạy (tên theo ms-swift 4.6.0).

    Thêm tham số revision, lượng tử hoá nf4 (bitsandbytes) và `--resume_from_checkpoint` khi cấu hình yêu cầu.
    """
    lora, tr = cfg.lora, cfg.train
    argv = [
        "--model", cfg.base.model,
        "--use_hf", "true",
        "--tuner_type", "lora",
        "--tuner_backend", tr.tuner_backend,
        "--lora_rank", str(lora.r),
        "--lora_alpha", str(lora.alpha),
        "--lora_dropout", str(lora.dropout),
        "--target_modules", *lora.target_modules,
        "--torch_dtype", dtype,
        "--dataset", str(train_file),
        "--val_dataset", str(val_file),
        "--max_length", str(tr.max_len),
        "--truncation_strategy", "delete",  # dòng dài hơn max_len bị bỏ, không bao giờ bị cắt (giữ như trước)
        "--num_train_epochs", str(tr.epochs),
        "--per_device_train_batch_size", str(tr.batch_size),
        "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", str(tr.grad_accum),
        "--learning_rate", str(tr.lr),
        "--lr_scheduler_type", tr.lr_scheduler,
        "--warmup_ratio", str(tr.warmup_ratio),
        "--eval_strategy", "steps",
        "--eval_steps", str(tr.eval_every_steps),
        "--save_strategy", "steps",
        "--save_steps", str(tr.save_every_steps),
        "--logging_steps", "10",
        "--seed", str(tr.seed),
        "--data_seed", str(tr.seed),
        "--output_dir", str(out_dir),
        "--add_version", "false",
        "--report_to", "none",
    ]  # fmt: skip
    if cfg.base.revision:
        argv += ["--model_revision", cfg.base.revision]
    if lora.quant == "nf4":
        argv += ["--quant_method", "bnb", "--quant_bits", "4", "--bnb_4bit_quant_type", "nf4"]
        argv += ["--bnb_4bit_compute_dtype", dtype]
    if resume_from is not None:
        argv += ["--resume_from_checkpoint", str(resume_from)]
    return argv


def checkpoints(swift_dir: Path) -> list[SwiftCheckpoint]:
    """Liệt kê checkpoint trong `swift_dir` theo thứ tự bước, kèm eval loss được ghi tại đúng bước đó.

    Thư mục chưa có trainer_state.json (đang ghi dở) bị bỏ qua.
    """
    out: list[SwiftCheckpoint] = []
    for d in swift_dir.glob(CKPT_GLOB):
        state = d / "trainer_state.json"
        if not state.is_file():
            continue
        step = int(d.name.rsplit("-", 1)[1])
        history = json.loads(state.read_text(encoding="utf-8")).get("log_history", [])
        # Chỉ lấy eval_loss ghi tại đúng bước của checkpoint, không lấy loss của bước khác.
        losses = [h["eval_loss"] for h in history if "eval_loss" in h and int(h.get("step", -1)) == step]
        out.append(SwiftCheckpoint(d, step, float(losses[-1]) if losses else None))
    return sorted(out, key=lambda c: c.step)


def best_k(swift_dir: Path, k: int) -> list[SwiftCheckpoint]:
    """Lấy tối đa `k` checkpoint có eval loss thấp nhất (bằng nhau thì bước sớm hơn đứng trước)."""
    scored = [c for c in checkpoints(swift_dir) if c.eval_loss is not None]
    return sorted(scored, key=lambda c: (float(c.eval_loss or 0.0), c.step))[:k]


def latest(swift_dir: Path) -> SwiftCheckpoint | None:
    """Checkpoint có bước cao nhất (để chạy tiếp một lần train bị ngắt)."""
    ckpts = checkpoints(swift_dir)
    return ckpts[-1] if ckpts else None


def write_metrics(swift_dir: Path, out: Path) -> int:
    """Ghi `metrics.jsonl` (định dạng của MetricsLogger) từ log_history của checkpoint mới nhất; trả số dòng.

    ms-swift không ghi `metrics.jsonl`, nhưng `lifemate.run.report train` và bước register của pipeline đọc file
    này. `eval_loss` được đổi tên thành `eval_unseen_loss`, vì tập val duy nhất truyền cho ms-swift là val_unseen.
    Không có checkpoint nào thì không ghi gì và trả 0.
    """
    last = latest(swift_dir)
    if last is None:
        return 0
    history = json.loads((last.path / "trainer_state.json").read_text(encoding="utf-8")).get("log_history", [])
    rows = [{("eval_unseen_loss" if k == "eval_loss" else k): v for k, v in h.items()} for h in history]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


def _subprocess_runner(cmd: Sequence[str], cwd: Path) -> int:
    """Chạy lệnh trong `cwd` và trả về mã thoát (không ném lỗi khi mã khác 0)."""
    return subprocess.run(list(cmd), cwd=cwd, check=False).returncode


def _gpu_dtype() -> str:
    """Chọn bfloat16 nếu GPU hỗ trợ, ngược lại float16."""
    torch = importlib.import_module("torch")
    return "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"


def train(
    cfg: LLMTrainConfig,
    *,
    repo_root: Path,
    out_dir: Path,
    resume: bool = True,
    runner: Runner = _subprocess_runner,
    dtype: str | None = None,
    check: Callable[[LLMTrainConfig, Sequence[Mapping[str, Any]]], int] = check_template,
) -> Path:
    """Xuất dữ liệu, kiểm tra template, chạy `swift sft`; trả về thư mục chứa các `checkpoint-<step>`.

    Khi `resume` bật và đã có checkpoint, lần chạy nối tiếp từ checkpoint mới nhất. `runner`, `dtype` và `check`
    cho phép test thay thế tiến trình con, GPU và bước kiểm tra template. Ném RuntimeError khi `swift sft` lỗi.
    """
    if not cfg.base.model:
        raise ConfigError("base.model: required to train")
    data_dir = out_dir / "swift_data"
    train_file, val_file = data_dir / "train.jsonl", data_dir / "val_unseen.jsonl"
    export_rows(abs_path(cfg.data.train, repo_root), train_file)
    export_rows(abs_path(cfg.data.eval_unseen, repo_root), val_file)
    check(cfg, read_jsonl(abs_path(cfg.data.train, repo_root)))
    swift_dir = out_dir / "swift"
    last = latest(swift_dir) if resume else None
    argv = swift_argv(
        cfg,
        train_file=train_file,
        val_file=val_file,
        out_dir=swift_dir,
        dtype=dtype or _gpu_dtype(),
        resume_from=last.path if last else None,
    )
    log.info("swift_sft_start", resume_from=str(last.path) if last else None)
    code = runner(["swift", "sft", *argv], repo_root)
    if code != 0:
        raise RuntimeError(f"swift sft failed with exit code {code}")
    # Đường loss cho `make report-train` và cho artifacts/models/<tên>/metrics.jsonl ở bước register.
    write_metrics(swift_dir, out_dir / "metrics.jsonl")
    return swift_dir
