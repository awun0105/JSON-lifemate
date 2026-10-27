"""Cấu hình fine-tune LLM, nhãn loss chỉ trên lượt assistant và gộp adapter (SPEC 6.2.2, 6.2.3, 6.2.4).

Nội dung file:
- Các dataclass `BaseSection`, `DataSection`, `LoraSection`, `TrainSection`, `LLMTrainConfig` và
  `load_llm_train_config`: đọc và kiểm tra configs/train/llm_vN.toml với khóa chặt chẽ.
- `build_labels` và các hàm phụ: tính nhãn loss chỉ cho lượt assistant, thuần Python, kiểm thử bằng tokenizer giả.
- `merge_adapter`: gộp adapter LoRA vào model gốc bf16.
- `main`: giao diện dòng lệnh, gọi `swift_sft.train` rồi ghi train_config.json.

Cách dựng: bản thân việc train chạy trong ms-swift (`lifemate.train.swift_sft`). Trainer Unsloth trước đây được
giữ bên dưới ở dạng chú thích. `build_labels` vẫn còn: đây là bản tham chiếu phía phục vụ mà
`swift_sft.check_template` dùng để so với cách ms-swift mã hóa hội thoại.

Ví dụ:
    python -m lifemate.train.sft --config configs/train/llm_v1.toml --env local
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json

# import random  # Unsloth trainer, unused
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# from types import ModuleType, SimpleNamespace  # Unsloth trainer, unused
from typing import Any, Protocol

import structlog

from lifemate.core.config import ConfigError, load_profile
from lifemate.core.envinfo import detect
from lifemate.core.paths import REPO_ROOT
from lifemate.train.common import (
    # CheckpointManager,  # Unsloth trainer, unused
    # MetricsLogger,  # Unsloth trainer, unused
    abs_path,
    build_section,
    load_toml,
    # seed_everything,  # Unsloth trainer, unused
    sha256_file,
    write_train_config,
)

log = structlog.get_logger(__name__)

IGNORE_INDEX = -100
# TRAIN_PACKAGES = ("unsloth", "trl", "transformers", "peft", "torch", "datasets")  # Unsloth trainer, unused
EVAL_METRIC = "eval_unseen_loss"


@dataclass(frozen=True, slots=True)
class BaseSection:
    """[base]: model cần fine-tune (tên, revision) và cấu hình run có thiết lập prompt mà dữ liệu phải khớp."""

    model: str
    revision: str
    run_config: str


@dataclass(frozen=True, slots=True)
class DataSection:
    """[data]: các file SFT jsonl (train, eval_seen, eval_unseen) và tham số dựng dữ liệu cho sft_data."""

    train: str
    eval_seen: str
    eval_unseen: str
    transcript_source: str  # gold | asr | mix (lifemate.train.sft_data); mix = 30% gold / 70% ASR của trưởng nhóm
    asr_ratio: float  # chỉ cho mix: tỉ lệ mẫu dựng từ transcript ASR
    asr_cache: str  # cache ASR từ lifemate.train.asr_cache ("" khi transcript_source = gold)
    error_recovery_ratio: float  # tỉ lệ dòng train có một lời gọi lỗi rồi lời gọi sửa (sft_data.Augment)
    oversample_hard: float  # hệ số nhân bản dòng ask_missing_info / out_of_scope; 1.0 = không
    distractor_apps: int  # số app cài thêm cho mỗi mẫu train; 0 = không
    dedup_max_k: int  # tối đa số mẫu train mỗi cụm gần trùng; 0 = giữ tất cả


@dataclass(frozen=True, slots=True)
class LoraSection:
    """[lora]: lượng tử hóa (nf4 | none), hạng r, alpha, dropout và các module đích của LoRA."""

    quant: str
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]


@dataclass(frozen=True, slots=True)
class TrainSection:
    """[train]: siêu tham số train, seed và chu kỳ eval/lưu checkpoint (tính theo bước)."""

    lr: float
    lr_scheduler: str  # tên scheduler của transformers, ví dụ cosine
    tuner_backend: str  # giá trị --tuner_backend của ms-swift: peft | unsloth
    warmup_ratio: float
    epochs: int
    batch_size: int
    grad_accum: int
    max_len: int
    seed: int
    eval_every_steps: int
    save_every_steps: int


@dataclass(frozen=True, slots=True)
class LLMTrainConfig:
    """Toàn bộ nội dung configs/train/llm_vN.toml."""

    base: BaseSection
    data: DataSection
    lora: LoraSection
    train: TrainSection


def load_llm_train_config(path: Path) -> LLMTrainConfig:
    """Đọc và kiểm tra file TOML cấu hình train LLM (khóa chặt chẽ: thiếu hay thừa khóa đều báo lỗi).

    Ngoài kiểu dữ liệu, hàm còn kiểm tra miền giá trị: quant, transcript_source, các tỉ lệ trong [0, 1],
    tuner_backend, và save_every_steps phải là bội của eval_every_steps (checkpoint chỉ lưu sau một lượt eval).
    """
    raw = load_toml(path, ("base", "data", "lora", "train"))
    cfg = LLMTrainConfig(
        base=build_section(BaseSection, "base", raw["base"]),
        data=build_section(DataSection, "data", raw["data"]),
        lora=build_section(LoraSection, "lora", raw["lora"]),
        train=build_section(TrainSection, "train", raw["train"]),
    )
    if cfg.lora.quant not in {"nf4", "none"}:
        raise ConfigError(f"lora.quant: {cfg.lora.quant!r} not in nf4|none")
    if cfg.data.transcript_source not in {"gold", "asr", "mix"}:
        raise ConfigError(f"data.transcript_source: {cfg.data.transcript_source!r} not in gold|asr|mix")
    if cfg.data.transcript_source != "gold" and not cfg.data.asr_cache:
        raise ConfigError("data.asr_cache: required when data.transcript_source is asr or mix")
    if not 0.0 <= cfg.data.error_recovery_ratio <= 1.0:
        raise ConfigError("data.error_recovery_ratio must be in [0, 1]")
    if cfg.data.oversample_hard < 1.0 or cfg.data.distractor_apps < 0 or cfg.data.dedup_max_k < 0:
        raise ConfigError("data.oversample_hard must be >= 1; data.distractor_apps and data.dedup_max_k >= 0")
    if not 0.0 <= cfg.data.asr_ratio <= 1.0:
        raise ConfigError("data.asr_ratio must be in [0, 1]")
    if cfg.train.tuner_backend not in {"peft", "unsloth"}:
        raise ConfigError(f"train.tuner_backend: {cfg.train.tuner_backend!r} not in peft|unsloth")
    if cfg.train.save_every_steps % cfg.train.eval_every_steps != 0:
        raise ConfigError("train.save_every_steps must be a multiple of train.eval_every_steps")
    return cfg


# ---------------------------------------------------------------- che loss (thuần, kiểm thử bằng tokenizer giả)


class ChatTokenizer(Protocol):
    """Phần của tokenizer HF mà `build_labels` dùng."""

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any: ...


def _ids(out: Any) -> list[int]:
    """Chuẩn hóa đầu ra của apply_chat_template (list, dict/BatchEncoding, dạng batch) thành list id."""
    if isinstance(out, Mapping) or hasattr(out, "input_ids"):
        out = out["input_ids"]
    if out and isinstance(out[0], list):  # dạng batch: lấy phần tử đầu
        out = out[0]
    return [int(x) for x in out]


def _render(
    tokenizer: ChatTokenizer, messages: Sequence[Mapping[str, Any]], gen_prompt: bool, kwargs: Mapping[str, Any]
) -> list[int]:
    """Render hội thoại qua chat template thành list id, có hoặc không có generation prompt."""
    return _ids(
        tokenizer.apply_chat_template(list(messages), tokenize=True, add_generation_prompt=gen_prompt, **kwargs)
    )


def _common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    """Độ dài tiền tố chung của hai dãy id."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def template_kwargs(tokenizer: Any) -> dict[str, Any]:
    """Trả `enable_thinking=False` khi chat template biết cờ này (giống VLLMBackend lúc chạy, S6), ngược lại rỗng."""
    template = getattr(tokenizer, "chat_template", None)
    return {"enable_thinking": False} if isinstance(template, str) and "enable_thinking" in template else {}


def build_labels(
    tokenizer: ChatTokenizer,
    messages: Sequence[Mapping[str, Any]],
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[int], list[int]]:
    """Id token của cả hội thoại và nhãn; nhãn là IGNORE_INDEX ngoài lượt assistant và trên lượt `"loss": false`.

    Lượt assistant có `"loss": false` là lời gọi lỗi của một dòng error-recovery.

    Đoạn assistant thứ k = các token nằm sau bản render của messages[:k] kèm generation prompt, kéo tới hết bản
    render của messages[:k+1] (nhờ vậy token kết thúc lượt cũng được train). Điểm bắt đầu là tiền tố chung giữa bản
    render có generation prompt và toàn bộ id, cách này xử lý được template render generation prompt khác với một
    lượt assistant đã hoàn tất. Template không ổn định tiền tố gây ValueError.
    """
    kwargs = dict(chat_template_kwargs or {})
    trained = [m.get("role") == "assistant" and m.get("loss", True) is not False for m in messages]
    # Khóa `loss` không bao giờ được đưa vào template.
    plain = [{"role": m["role"], "content": m["content"]} for m in messages]
    full = _render(tokenizer, plain, False, kwargs)
    labels = [IGNORE_INDEX] * len(full)
    for k in range(len(plain)):
        if not trained[k]:
            continue
        upto = _render(tokenizer, plain[: k + 1], False, kwargs)
        # Bản render ngắn hơn phải là tiền tố của bản đầy đủ, nếu không vị trí nhãn sẽ lệch.
        if full[: len(upto)] != upto:
            raise ValueError(f"chat template is not prefix-stable at message {k}")
        start = min(_common_prefix(_render(tokenizer, plain[:k], True, kwargs), full), len(upto))
        labels[start : len(upto)] = full[start : len(upto)]
    return full, labels


@dataclass(frozen=True, slots=True)
class Example:
    """Một dòng SFT đã token hóa: id mẫu, input_ids và labels."""

    id: str
    input_ids: list[int]
    labels: list[int]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Đọc các dòng của file jsonl (bỏ dòng trống)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


# --- Không dùng từ khi chuyển sang ms-swift (lifemate/train/swift_sft.py); giữ dạng chú thích để tham khảo/quay lui.
# def tokenize_rows(
#     tokenizer: ChatTokenizer, rows: Sequence[Mapping[str, Any]], max_len: int, kwargs: Mapping[str, Any]
# ) -> list[Example]:
#     """`build_labels` per row; rows longer than `max_len` are skipped with a warning, never truncated."""
#     out: list[Example] = []
#     for row in rows:
#         ids, labels = build_labels(tokenizer, row["messages"], kwargs)
#         if len(ids) > max_len:
#             log.warning("sft_row_too_long", id=row["id"], tokens=len(ids), max_len=max_len)
#             continue
#         out.append(Example(str(row["id"]), ids, labels))
#     return out
#
#
# def pad_batch(examples: Sequence[Mapping[str, Sequence[int]]], pad_id: int) -> dict[str, list[list[int]]]:
#     """Right-pad input_ids (pad_id), labels (IGNORE_INDEX) and build attention_mask."""
#     width = max(len(e["input_ids"]) for e in examples)
#     batch: dict[str, list[list[int]]] = {"input_ids": [], "labels": [], "attention_mask": []}
#     for e in examples:
#         n = len(e["input_ids"])
#         batch["input_ids"].append([*e["input_ids"], *([pad_id] * (width - n))])
#         batch["labels"].append([*e["labels"], *([IGNORE_INDEX] * (width - n))])
#         batch["attention_mask"].append([1] * n + [0] * (width - n))
#     return batch


# ---------------------------------------------------------------- phần GPU (không chạy trong test)


# --- Không dùng từ khi chuyển sang ms-swift (lifemate/train/swift_sft.py); giữ dạng chú thích để tham khảo/quay lui.
# def _import_training_stack() -> SimpleNamespace:
#     """Import the training extras; RuntimeError naming the missing packages."""
#     mods: dict[str, ModuleType] = {}
#     missing: list[str] = []
#     for name in TRAIN_PACKAGES:
#         try:
#             mods[name] = importlib.import_module(name)
#         except ImportError:
#             missing.append(name)
#     if missing:
#         raise RuntimeError(
#             f"SFT needs packages {', '.join(missing)} (install: pip install unsloth trl transformers peft torch "
#             "datasets; GPU machine, SPEC 6.2.3)"
#         )
#     return SimpleNamespace(**mods)
#
#
# def _make_callback(stack: SimpleNamespace, mgr: CheckpointManager, metrics: MetricsLogger, save_every: int) -> Any:
#     """TrainerCallback: log every metric line; save a checkpoint after the eval_unseen pass at save steps."""
#     torch = stack.torch
#     np = importlib.import_module("numpy")
#     base_cls: Any = stack.transformers.TrainerCallback
#
#     class _Callback(base_cls):  # type: ignore[misc]
#         trainer: Any = None
#
#         def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kw: Any) -> None:
#             metrics.log({"step": state.global_step, "epoch": state.epoch, **(logs or {})})
#
#         def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Any = None, **kw: Any) -> None:
#             if not metrics or EVAL_METRIC not in metrics or state.global_step % save_every != 0:
#                 return
#             tr = self.trainer
#
#             def save(d: Path) -> None:
#                 tr.model.save_pretrained(str(d))
#                 torch.save(tr.optimizer.state_dict(), d / "optimizer.pt")
#                 torch.save(tr.lr_scheduler.state_dict(), d / "scheduler.pt")
#                 tr.state.save_to_json(str(d / "trainer_progress.json"))
#                 # Same layout as transformers' rng_state.pth so Trainer restores it on resume.
#                 rng = {"python": random.getstate(), "numpy": np.random.get_state(), "cpu": torch.get_rng_state()}
#                 if torch.cuda.is_available():
#                     rng["cuda"] = torch.cuda.get_rng_state_all()
#                 torch.save(rng, d / "rng_state.pth")
#
#             mgr.save(state.global_step, float(metrics[EVAL_METRIC]), save)
#
#     return _Callback()
#
#
# def train(cfg: LLMTrainConfig, *, repo_root: Path, out_dir: Path, resume_from: Path | None = None) -> Path:
#     """Run QLoRA SFT; checkpoints and metrics.jsonl go to `out_dir`. Returns the checkpoint directory."""
#     if not cfg.base.model:
#         raise ConfigError("base.model: required to train (model not chosen yet, SPEC 6.0)")
#     stack = _import_training_stack()
#     seed_everything(cfg.train.seed)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     mgr = CheckpointManager(out_dir / "checkpoints", metric_name=EVAL_METRIC, lower_is_better=True)
#     metrics = MetricsLogger(out_dir / "metrics.jsonl")
#     model, tokenizer = stack.unsloth.FastLanguageModel.from_pretrained(
#         model_name=cfg.base.model,
#         revision=cfg.base.revision or None,
#         max_seq_length=cfg.train.max_len,
#         load_in_4bit=cfg.lora.quant == "nf4",
#         dtype=None,
#     )
#     model = stack.unsloth.FastLanguageModel.get_peft_model(
#         model,
#         r=cfg.lora.r,
#         lora_alpha=cfg.lora.alpha,
#         lora_dropout=cfg.lora.dropout,
#         target_modules=list(cfg.lora.target_modules),
#         use_gradient_checkpointing="unsloth",
#         random_state=cfg.train.seed,
#     )
#     kwargs = template_kwargs(tokenizer)
#
#     def dataset(rel: str) -> Any:
#         rows = tokenize_rows(tokenizer, read_jsonl(abs_path(rel, repo_root)), cfg.train.max_len, kwargs)
#         return stack.datasets.Dataset.from_list([{"input_ids": e.input_ids, "labels": e.labels} for e in rows])
#
#     train_ds = dataset(cfg.data.train)
#     eval_ds = {"seen": dataset(cfg.data.eval_seen), "unseen": dataset(cfg.data.eval_unseen)}
#     pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
#     torch = stack.torch
#
#     def collate(examples: list[dict[str, Any]]) -> dict[str, Any]:
#         return {k: torch.tensor(v) for k, v in pad_batch(examples, int(pad_id)).items()}
#
#     bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
#     args = stack.transformers.TrainingArguments(
#         output_dir=str(out_dir / "hf"),
#         learning_rate=cfg.train.lr,
#         lr_scheduler_type=cfg.train.lr_scheduler,
#         warmup_ratio=cfg.train.warmup_ratio,
#         num_train_epochs=cfg.train.epochs,
#         per_device_train_batch_size=cfg.train.batch_size,
#         per_device_eval_batch_size=1,
#         gradient_accumulation_steps=cfg.train.grad_accum,
#         eval_strategy="steps",
#         eval_steps=cfg.train.eval_every_steps,
#         save_strategy="no",  # CheckpointManager saves (SPEC 6.5)
#         logging_steps=10,
#         seed=cfg.train.seed,
#         data_seed=cfg.train.seed,
#         fp16=not bf16,
#         bf16=bf16,
#         report_to=[],
#         remove_unused_columns=False,
#     )
#     callback = _make_callback(stack, mgr, metrics, cfg.train.save_every_steps)
#     trainer = stack.transformers.Trainer(
#         model=model,
#         args=args,
#         train_dataset=train_ds,
#         eval_dataset=eval_ds,
#         data_collator=collate,
#         callbacks=[callback],
#     )
#     callback.trainer = trainer
#     resume: str | None = None
#     if resume_from is not None:
#         mgr.load(resume_from)
#         resume = str(resume_from)
#     trainer.train(resume_from_checkpoint=resume)
#     final_metrics = trainer.evaluate()
#     mgr.save(trainer.state.global_step, float(final_metrics.get(EVAL_METRIC, float("nan"))), _final_saver(trainer))
#     return mgr.directory
#
#
# def _final_saver(trainer: Any) -> Any:
#     def save(d: Path) -> None:
#         trainer.model.save_pretrained(str(d))
#
#     return save


def merge_adapter(base: str, adapter_dir: Path, out_dir: Path, *, revision: str | None = None) -> Path:
    """Gộp adapter LoRA vào model gốc bf16 và lưu safetensors đầy đủ (chạy trên CPU cũng được, SPEC 6.2.4).

    Tokenizer của model gốc được lưu cùng thư mục, để thư mục đầu ra phục vụ được ngay. Trả về `out_dir`.
    """
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        peft = importlib.import_module("peft")
    except ImportError as exc:
        raise RuntimeError(f"merge needs torch, transformers, peft: {exc}") from exc
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base, revision=revision, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    merged = peft.PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    transformers.AutoTokenizer.from_pretrained(base, revision=revision).save_pretrained(str(out_dir))
    log.info("adapter_merged", base=base, adapter=str(adapter_dir), out=str(out_dir))
    return out_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    ap = argparse.ArgumentParser(prog="python -m lifemate.train.sft", description="LoRA SFT via ms-swift (6.2.3).")
    ap.add_argument("--config", required=True, type=Path, help="train config TOML (configs/train/llm_v1.toml)")
    ap.add_argument("--env", default="local", help="env profile; checkpoints go under its runs_dir")
    ap.add_argument("--out-dir", type=Path, help="override output dir (default <runs_dir>/train/<config stem>)")
    # ap.add_argument("--resume-from", type=Path, help="checkpoint folder to continue from")  # Unsloth trainer
    ap.add_argument("--no-resume", action="store_true", help="start over instead of resuming the latest checkpoint")
    return ap.parse_args(argv)


def default_out_dir(env: str, config_path: Path, repo_root: Path) -> Path:
    """Thư mục đầu ra mặc định: `<profile.runs_dir>/train/<config stem>`."""
    profile = load_profile(env, repo_root)
    return abs_path(profile.runs_dir, repo_root) / "train" / config_path.stem


def main(argv: Sequence[str] | None = None, repo_root: Path = REPO_ROOT) -> int:
    """Điểm vào dòng lệnh: train bằng ms-swift rồi ghi train_config.json kèm sha256 của các file dữ liệu."""
    args = parse_args(argv)
    cfg_path = abs_path(args.config, repo_root)
    cfg = load_llm_train_config(cfg_path)
    out = abs_path(args.out_dir, repo_root) if args.out_dir else default_out_dir(args.env, cfg_path, repo_root)
    from lifemate.train import swift_sft  # swift_sft import module này; import cục bộ để tránh vòng import

    # ckpt_dir = train(cfg, repo_root=repo_root, out_dir=out, resume_from=args.resume_from)  # Unsloth trainer
    ckpt_dir = swift_sft.train(cfg, repo_root=repo_root, out_dir=out, resume=not args.no_resume)
    # sha256 cho mọi giá trị chuỗi trong [data] trỏ tới file có thật (train, eval, cache ASR).
    data_sha = {
        k: sha256_file(abs_path(v, repo_root))
        for k, v in dataclasses.asdict(cfg.data).items()
        if isinstance(v, str) and v and abs_path(v, repo_root).is_file()
    }
    write_train_config(
        out / "train_config.json",
        dataclasses.asdict(cfg),
        detect(repo_root),
        data_sha,
        argv=list(sys.argv),
        seed=cfg.train.seed,
        checkpoints=str(ckpt_dir),
    )
    print(f"checkpoints: {ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
