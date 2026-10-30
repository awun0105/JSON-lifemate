"""SFT cho LLM bằng ms-swift (`swift sft`), trainer do trưởng nhóm chọn (ThanhDanh1510/ZaloAI2026
docs/llm_finetuning.md).

Nội dung chính:
- `export_rows`: xuất dòng SFT sang định dạng ms-swift.
- `check_template`, `compare_encodings`, `pick_check_rows`: kiểm tra template train và template lúc phục vụ khớp
  nhau.
- `swift_argv`: dựng tham số cho `swift sft`.
- `train`: chạy cả quy trình trong tiến trình con.
- `checkpoints`, `best_k`, `latest`, `write_metrics`: đọc các checkpoint và log đã ghi.

Cách hoạt động (hàm `train`):
1. `export_rows` chép dòng train và `val_unseen`, chỉ giữ `messages` (kèm cờ `loss` của từng lượt).
2. `check_template` mã hoá một số dòng mẫu theo cả hai cách và dừng nếu khác nhau.
3. `swift_argv` dựng tham số; nếu đã có checkpoint thì chạy tiếp từ checkpoint mới nhất.
4. `runner` gọi `swift sft`; sau đó `checkpoints`, `best_k`, `latest` đọc kết quả từ `trainer_state.json`.

Lưu ý:
- Module này thay cho trainer Unsloth từng nằm trong `sft.train` (vẫn giữ ở đó dưới dạng chú thích). Phần dự án
  vẫn tự làm: dựng dòng SFT (`sft_data`), chọn checkpoint theo điểm agent, merge và registry (`llm_pipeline`).
- Khớp giữa train và serve: ms-swift dựng hội thoại bằng template riêng (`qwen3_nothinking` cho
  Qwen3-4B-Instruct-2507), còn vLLM phục vụ bằng chat template của tokenizer. `check_template` mã hoá các dòng mẫu
  theo cả hai cách trước khi train và dừng ở chỗ khác đầu tiên về input ids hoặc nhãn loss. Đo trên 60 dòng thật
  (25 dòng có phục hồi lỗi) với ms-swift 4.6.0: giống hệt nhau.
- Checkpoint: `swift sft` ghi `<out>/swift/checkpoint-<step>/` (adapter PEFT + `trainer_state.json`). Chỉ truyền
  một file val là `val_unseen`, vì ms-swift gộp nhiều file val thành một loss duy nhất. Do đó `eval_loss` của
  ms-swift chính là `eval_unseen_loss` của dự án.
- Các gói nặng (ms-swift, torch, transformers) chỉ được import bên trong hàm cần dùng.
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

from lifemate.core.chat import split_tools
from lifemate.core.config import ConfigError
from lifemate.train.common import abs_path
from lifemate.train.sft import LLMTrainConfig, build_labels, read_jsonl, template_kwargs

log = structlog.get_logger(__name__)

SWIFT_VERSION = "4.6.0"  # khớp ops/requirements/train.txt; tên tham số bên dưới đã đối chiếu với phiên bản này
CKPT_GLOB = "checkpoint-*"
N_TEMPLATE_CHECK = 20

# Runner(lệnh, thư mục làm việc) -> mã thoát; test thay bằng hàm giả để không chạy `swift` thật.
Runner = Callable[[Sequence[str], Path], int]


class TemplateMismatchError(RuntimeError):
    """Lỗi khi ms-swift và chat template lúc phục vụ mã hoá một dòng khác nhau: dữ liệu train sẽ lệch với lúc chạy."""


@dataclass(frozen=True, slots=True)
class SwiftCheckpoint:
    """Một thư mục `checkpoint-<step>` do `swift sft` ghi ra.

    Thuộc tính:
        path: thư mục checkpoint.
        step: step lúc lưu.
        eval_loss: loss trên `val_unseen` ghi đúng tại step đó; `None` nếu step đó không có lần eval.
    """

    path: Path
    step: int
    eval_loss: float | None


def swift_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Một dòng `sft_data` ở dạng ms-swift đọc được.

    Định dạng JSON cũ: chỉ giữ `role`, `content`, `loss`. Định dạng ChatML: danh sách tool chuyển từ trường `tools` của
    tin `system` ra trường `tools` cấp dòng (ms-swift xoá mọi khoá lạ trong tin nhắn); tin `assistant` giữ `tool_calls`
    dạng OpenAI (ms-swift tự đổi sang vai `tool_call`), tin `tool` giữ `tool_call_id` và `name`; mọi tin giữ `loss`.
    """
    msgs, tools = split_tools(row["messages"])
    if tools is None:
        return {"messages": [{k: m[k] for k in ("role", "content", "loss") if k in m} for m in msgs]}
    keep = ("role", "content", "loss", "tool_calls", "tool_call_id", "name")
    return {"tools": tools, "messages": [{k: m[k] for k in keep if k in m} for m in msgs]}


def export_rows(src: Path, dst: Path) -> int:
    """Chép các dòng của `sft_data` sang dạng ms-swift (`swift_row`).

    Các khoá khác (id, metadata) bị bỏ để ms-swift không hiểu nhầm thành cột dữ liệu.

    Tham số:
        src: file jsonl nguồn.
        dst: file jsonl đích; thư mục cha được tạo nếu chưa có.

    Trả về:
        Số dòng đã chép.
    """
    rows = read_jsonl(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(swift_row(row), ensure_ascii=False) + "\n")
    return len(rows)


def pick_check_rows(rows: Sequence[Mapping[str, Any]], n: int = N_TEMPLATE_CHECK) -> list[Mapping[str, Any]]:
    """Chọn tối đa `n` dòng để kiểm tra template: mọi dòng phục hồi lỗi (có `loss: false`) trước, rồi các dòng khác.

    Dòng phục hồi lỗi được ưu tiên vì cờ `loss: false` là chỗ dễ lệch nhãn nhất giữa hai template.
    """
    special = [r for r in rows if any(m.get("loss") is False for m in r["messages"])]
    rest = [r for r in rows if r not in special]
    return [*special, *rest][:n]


def compare_encodings(
    row_id: str, ours: tuple[list[int], list[int]], theirs: tuple[list[int], list[int]], decode: Callable[..., str]
) -> None:
    """So hai cách mã hoá một dòng và báo lỗi tại token đầu tiên mà input ids hoặc nhãn khác nhau.

    Tham số:
        row_id: id dòng, để ghi vào thông báo lỗi.
        ours: cặp (input_ids, labels) của dự án.
        theirs: cặp (input_ids, labels) của ms-swift.
        decode: hàm giải mã id thành văn bản, để in đoạn quanh chỗ lệch.

    Lỗi:
        TemplateMismatchError: khi hai cách mã hoá khác nhau.
    """
    for what, a, b in (("input_ids", ours[0], theirs[0]), ("labels", ours[1], theirs[1])):
        if a == b:
            continue
        # Vị trí lệch đầu tiên; nếu một dãy là phần đầu của dãy kia thì lệch tại độ dài của dãy ngắn hơn.
        i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        context = decode(theirs[0][max(0, i - 12) : i + 12])
        raise TemplateMismatchError(
            f"{row_id}: {what} differ at token {i} (ours {len(a)}, swift {len(b)}): {context!r}"
        )


def check_template(cfg: LLMTrainConfig, rows: Sequence[Mapping[str, Any]]) -> int:
    """Mã hoá `rows` bằng template train của ms-swift và bằng chat template lúc phục vụ, báo lỗi nếu khác nhau.

    Phía dự án dùng `build_labels` (chat template của tokenizer, giống vLLM); phía ms-swift dùng
    `get_template(...).encode` ở chế độ train.

    Tham số:
        cfg: config train LLM.
        rows: các dòng SFT; chỉ một phần được kiểm tra (xem `pick_check_rows`).

    Trả về:
        Số dòng đã kiểm tra.

    Lỗi:
        TemplateMismatchError: khi có dòng mã hoá khác nhau.
    """
    swift_model = importlib.import_module("swift.model")
    swift_template = importlib.import_module("swift.template")
    proc = swift_model.get_processor(cfg.base.model, use_hf=True, revision=cfg.base.revision or None)
    tpl = swift_template.get_template(proc, max_length=None)
    tpl.set_mode("train")
    kwargs = template_kwargs(proc)
    checked = pick_check_rows(rows)
    for row in checked:
        enc = tpl.encode(swift_row(row))
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
    """Dựng danh sách tham số của `swift sft` cho một lần chạy (tên tham số theo ms-swift 4.6.0).

    Thêm tham số revision, lượng tử hoá nf4 (bitsandbytes) và `--resume_from_checkpoint` khi config yêu cầu.

    Tham số:
        cfg: config train LLM.
        train_file: file train đã xuất.
        val_file: file `val_unseen` đã xuất.
        out_dir: thư mục ghi checkpoint.
        dtype: dtype train (`bfloat16` hoặc `float16`).
        resume_from: checkpoint để chạy tiếp; `None` là train từ đầu.

    Trả về:
        Danh sách tham số, chưa gồm `swift sft`.
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
    """Liệt kê checkpoint trong `swift_dir` theo thứ tự step, kèm eval loss ghi tại đúng step đó.

    Thư mục chưa có `trainer_state.json` (đang ghi dở) bị bỏ qua.
    """
    out: list[SwiftCheckpoint] = []
    for d in swift_dir.glob(CKPT_GLOB):
        state = d / "trainer_state.json"
        if not state.is_file():
            continue
        step = int(d.name.rsplit("-", 1)[1])
        history = json.loads(state.read_text(encoding="utf-8")).get("log_history", [])
        # Chỉ lấy `eval_loss` ghi tại đúng step của checkpoint, không lấy loss của step khác.
        losses = [h["eval_loss"] for h in history if "eval_loss" in h and int(h.get("step", -1)) == step]
        out.append(SwiftCheckpoint(d, step, float(losses[-1]) if losses else None))
    return sorted(out, key=lambda c: c.step)


def best_k(swift_dir: Path, k: int) -> list[SwiftCheckpoint]:
    """Trả về tối đa `k` checkpoint có eval loss thấp nhất; bằng nhau thì step sớm hơn đứng trước."""
    scored = [c for c in checkpoints(swift_dir) if c.eval_loss is not None]
    return sorted(scored, key=lambda c: (float(c.eval_loss or 0.0), c.step))[:k]


def latest(swift_dir: Path) -> SwiftCheckpoint | None:
    """Trả về checkpoint có step cao nhất (để chạy tiếp một lần train bị ngắt), hoặc `None` khi chưa có."""
    ckpts = checkpoints(swift_dir)
    return ckpts[-1] if ckpts else None


def write_metrics(swift_dir: Path, out: Path) -> int:
    """Ghi `metrics.jsonl` (định dạng của `MetricsLogger`) từ `log_history` của checkpoint mới nhất.

    ms-swift không ghi `metrics.jsonl`, nhưng `lifemate.run.report train` và bước register của pipeline đọc file này.
    `eval_loss` được đổi tên thành `eval_unseen_loss`, vì tập val duy nhất truyền cho ms-swift là `val_unseen`.

    Tham số:
        swift_dir: thư mục checkpoint của ms-swift.
        out: file `metrics.jsonl` đích.

    Trả về:
        Số dòng đã ghi; 0 và không ghi gì khi chưa có checkpoint nào.
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
    """Chạy lệnh trong `cwd` và trả về mã thoát; không ném lỗi khi mã khác 0."""
    return subprocess.run(list(cmd), cwd=cwd, check=False).returncode


def _gpu_dtype() -> str:
    """Chọn bfloat16 nếu GPU hỗ trợ, không thì float16."""
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
    """Xuất dữ liệu, kiểm tra template, chạy `swift sft`.

    Khi `resume` bật và đã có checkpoint, lần chạy nối tiếp từ checkpoint mới nhất.

    Tham số:
        cfg: config train LLM.
        repo_root: thư mục gốc repo.
        out_dir: thư mục làm việc của job train.
        resume: `True` để chạy tiếp từ checkpoint mới nhất nếu có.
        runner: hàm chạy tiến trình con; test thay bằng hàm giả.
        dtype: dtype train; `None` thì chọn theo GPU.
        check: hàm kiểm tra template; test có thể thay.

    Trả về:
        Thư mục chứa các `checkpoint-<step>`.

    Lỗi:
        ConfigError: khi chưa chọn `base.model`.
        RuntimeError: khi `swift sft` kết thúc với mã khác 0.
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
    # Đường loss cho `make report-train` và cho `artifacts/models/<tên>/metrics.jsonl` ở bước register.
    write_metrics(swift_dir, out_dir / "metrics.jsonl")
    return swift_dir
