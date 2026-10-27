"""Khung fine-tune ASR: huấn luyện Seq2Seq LoRA hoặc toàn phần, chọn checkpoint theo tỉ lệ sót thực thể (SPEC 6.4.3).

Chưa chọn mô hình; code viết cho mô hình speech seq2seq của HF (kiểu Whisper: processor + generate). Các gói nặng
(torch, transformers, peft, datasets, soundfile) được import bên trong `_import_training_stack`.

Nội dung: dataclass config (`ASRTrainConfig` và các mục), hàm đo thuần (`normalize_text`, `wer`,
`score_checkpoint`, `choose_checkpoint`) chạy được không cần GPU, và phần GPU (`train`, `transcribe_rows`,
`merge_lora`, `select_checkpoint`). Luồng: file cặp jsonl từ `asr_data` → `train` lưu checkpoint qua
`CheckpointManager` → chép lời val_unseen bằng từng checkpoint → chọn checkpoint tốt nhất → gộp LoRA.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import sys
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import structlog

from lifemate.core.config import ConfigError, load_profile
from lifemate.core.envinfo import detect
from lifemate.core.paths import REPO_ROOT
from lifemate.train.common import (
    CheckpointInfo,
    CheckpointManager,
    MetricsLogger,
    abs_path,
    build_section,
    load_toml,
    seed_everything,
    sha256_file,
    write_train_config,
)
from lifemate.train.sft import read_jsonl

log = structlog.get_logger(__name__)

TRAIN_PACKAGES = ("torch", "transformers", "peft", "datasets", "soundfile")
EVAL_METRIC = "eval_loss"
_PUNCT_CATEGORIES = ("P", "S")


@dataclass(frozen=True, slots=True)
class ASRBase:
    """Mục [base]: mô hình gốc và revision."""

    model: str
    revision: str


@dataclass(frozen=True, slots=True)
class ASRData:
    """Mục [data]: đường dẫn file cặp huấn luyện."""

    train: str


@dataclass(frozen=True, slots=True)
class ASRMethod:
    """Mục [method]: kiểu huấn luyện (lora|full), đóng băng encoder, SpecAugment."""

    kind: str
    freeze_encoder: bool
    spec_augment: bool


@dataclass(frozen=True, slots=True)
class ASRTrain:
    """Mục [train]: siêu tham số và nhịp eval/lưu checkpoint."""

    lr: float
    epochs: int
    batch_size: int
    grad_accum: int
    seed: int
    eval_every_steps: int
    save_every_steps: int


@dataclass(frozen=True, slots=True)
class ASRTrainConfig:
    """Toàn bộ configs/train/asr_vN.toml."""

    base: ASRBase
    data: ASRData
    method: ASRMethod
    train: ASRTrain


def load_asr_train_config(path: Path) -> ASRTrainConfig:
    """Nạp và kiểm tra TOML config huấn luyện ASR (khóa chặt chẽ).

    Kiểm thêm: `method.kind` thuộc lora|full và `save_every_steps` là bội của `eval_every_steps`.
    """
    raw = load_toml(path, ("base", "data", "method", "train"))
    cfg = ASRTrainConfig(
        base=build_section(ASRBase, "base", raw["base"]),
        data=build_section(ASRData, "data", raw["data"]),
        method=build_section(ASRMethod, "method", raw["method"]),
        train=build_section(ASRTrain, "train", raw["train"]),
    )
    if cfg.method.kind not in {"lora", "full"}:
        raise ConfigError(f"method.kind: {cfg.method.kind!r} not in lora|full")
    if cfg.train.save_every_steps % cfg.train.eval_every_steps != 0:
        raise ConfigError("train.save_every_steps must be a multiple of train.eval_every_steps")
    return cfg


# ---------------------------------------------------------------- chỉ số (hàm thuần; dự phòng khi thiếu eval_asr)


def normalize_text(text: str) -> str:
    """Chữ thường, NFC, bỏ dấu câu, gộp khoảng trắng (chuẩn hóa WER theo SPEC 6.4.2)."""
    text = unicodedata.normalize("NFC", text.lower())
    # Ký tự thuộc nhóm Unicode P (dấu câu) hoặc S (ký hiệu) được thay bằng khoảng trắng.
    kept = "".join(" " if unicodedata.category(ch)[0] in _PUNCT_CATEGORIES else ch for ch in text)
    return " ".join(kept.split())


def _edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Khoảng cách Levenshtein theo từ (quy hoạch động, chỉ giữ hai hàng)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """WER trên toàn tập, tính trên văn bản đã chuẩn hóa."""
    errors = words = 0
    for ref, hyp in zip(refs, hyps, strict=True):
        r, h = normalize_text(ref).split(), normalize_text(hyp).split()
        errors += _edit_distance(r, h)
        words += len(r)
    return errors / words if words else 0.0


def score_checkpoint(rows: Sequence[Mapping[str, Any]], hyps: Sequence[str], samples: Any = None) -> dict[str, float]:
    """`entity_miss_rate` (lifemate.eval.eval_asr, lượt cuối của mỗi mẫu) và WER; không có samples thì chỉ có WER.

    Khi thiếu samples, `entity_miss_rate` lấy bằng WER và `entity_metric` = 0.0 để đánh dấu đây là giá trị dự phòng.
    """
    refs = [str(r["text"]) for r in rows]
    w = wer(refs, hyps)
    if samples is None:
        log.warning("eval_asr_unavailable", fallback="wer")
        return {"entity_miss_rate": w, "wer": w, "entity_metric": 0.0}
    from lifemate.core.types import Transcript
    from lifemate.eval.eval_asr import entity_miss_rate  # tầng eval; import tại đây để train vẫn import độc lập được

    # Chỉ lượt cuối được chấm thực thể; các lượt lịch sử chỉ góp vào WER.
    final_hyp = {str(r["id"]): h for r, h in zip(rows, hyps, strict=True) if r.get("turn") == "final"}
    by_id = {s.id: s for s in samples}
    ids = [i for i in final_hyp if i in by_id]
    ordered = [by_id[i] for i in ids]
    transcripts = [Transcript(final=final_hyp[i], history=(), seconds=0.0) for i in ids]
    _, _, rate, _ = entity_miss_rate(ordered, transcripts)
    return {"entity_miss_rate": float(rate), "wer": w, "entity_metric": 1.0}


def choose_checkpoint(scored: Sequence[tuple[CheckpointInfo, Mapping[str, float]]]) -> CheckpointInfo:
    """Tỉ lệ sót thực thể thấp nhất trên val_unseen, rồi WER thấp nhất, rồi step sớm nhất (SPEC 6.4.3)."""
    if not scored:
        raise ValueError("no checkpoints to choose from")
    return min(scored, key=lambda cs: (cs[1]["entity_miss_rate"], cs[1]["wer"], cs[0].step))[0]


# ---------------------------------------------------------------- phần GPU (không chạy trong test)


def _import_training_stack() -> SimpleNamespace:
    """Import các gói phụ huấn luyện ASR; thiếu gói nào thì RuntimeError nêu tên gói đó."""
    mods: dict[str, ModuleType] = {}
    missing: list[str] = []
    for name in TRAIN_PACKAGES:
        try:
            mods[name] = importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise RuntimeError(f"ASR fine-tune needs packages {', '.join(missing)} (GPU machine, SPEC 6.4.3)")
    return SimpleNamespace(**mods)


def load_audio(stack: SimpleNamespace, path: Path, sampling_rate: int) -> Any:
    """Mảng float mono ở `sampling_rate`; sai tần số lấy mẫu thì báo lỗi, cần resample dữ liệu trước."""
    data, sr = stack.soundfile.read(str(path), dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    if sr != sampling_rate:
        raise ValueError(f"{path}: sample rate {sr} != {sampling_rate}; resample the data first")
    return data


def train(
    cfg: ASRTrainConfig, *, repo_root: Path, data_root: Path, split: str, out_dir: Path, resume_from: Path | None = None
) -> Path:
    """Fine-tune mô hình ASR trên `cfg.data.train`; checkpoint nằm dưới `out_dir/checkpoints`.

    Trainer của HF không tự lưu (`save_strategy="no"`); callback lưu qua `CheckpointManager` mỗi
    `save_every_steps` và ghi log vào metrics.jsonl. Có `resume_from` thì khôi phục RNG rồi tiếp tục.
    Trả về thư mục checkpoint.
    """
    if not cfg.base.model:
        raise ConfigError("base.model: required to train (ASR model not chosen yet, SPEC 6.0)")
    stack = _import_training_stack()
    tf, torch = stack.transformers, stack.torch
    seed_everything(cfg.train.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    mgr = CheckpointManager(out_dir / "checkpoints", metric_name=EVAL_METRIC, lower_is_better=True)
    metrics = MetricsLogger(out_dir / "metrics.jsonl")
    revision = cfg.base.revision or None
    processor = tf.AutoProcessor.from_pretrained(cfg.base.model, revision=revision)
    model = tf.AutoModelForSpeechSeq2Seq.from_pretrained(cfg.base.model, revision=revision)
    if hasattr(model.config, "apply_spec_augment"):
        model.config.apply_spec_augment = cfg.method.spec_augment
    if cfg.method.freeze_encoder and hasattr(model, "freeze_encoder"):
        model.freeze_encoder()
    if cfg.method.kind == "lora":
        lora = stack.peft.LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
        model = stack.peft.get_peft_model(model, lora)
    sr = int(processor.feature_extractor.sampling_rate)
    split_dir = data_root / split

    # Mỗi dòng: đặc trưng âm thanh làm input, token của text nguyên văn làm nhãn.
    def encode(row: dict[str, Any]) -> dict[str, Any]:
        audio = load_audio(stack, split_dir / row["audio"], sr)
        feats = processor.feature_extractor(audio, sampling_rate=sr).input_features[0]
        return {"input_features": feats, "labels": processor.tokenizer(row["text"]).input_ids}

    rows = read_jsonl(abs_path(cfg.data.train, repo_root))
    train_ds = stack.datasets.Dataset.from_list(rows).map(encode, remove_columns=list(rows[0].keys()))

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        feats = processor.feature_extractor.pad(
            [{"input_features": b["input_features"]} for b in batch], return_tensors="pt"
        )
        labels = processor.tokenizer.pad([{"input_ids": b["labels"]} for b in batch], return_tensors="pt")
        # Vị trí padding của nhãn đặt -100 để loss bỏ qua.
        ids = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        return {"input_features": feats["input_features"], "labels": ids}

    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    args = tf.Seq2SeqTrainingArguments(
        output_dir=str(out_dir / "hf"),
        learning_rate=cfg.train.lr,
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        save_strategy="no",  # CheckpointManager lo việc lưu (SPEC 6.5)
        logging_steps=10,
        seed=cfg.train.seed,
        data_seed=cfg.train.seed,
        fp16=not bf16,
        bf16=bf16,
        report_to=[],
        remove_unused_columns=False,
    )
    base_cb: Any = tf.TrainerCallback

    class _Callback(base_cb):  # type: ignore[misc]
        trainer: Any = None

        def on_log(self, a: Any, state: Any, control: Any, logs: Any = None, **kw: Any) -> None:
            metrics.log({"step": state.global_step, **(logs or {})})

        def on_step_end(self, a: Any, state: Any, control: Any, **kw: Any) -> None:
            if state.global_step % cfg.train.save_every_steps == 0:
                # Loss gần nhất trong log_history làm metric của checkpoint.
                loss = next((h["loss"] for h in reversed(state.log_history) if "loss" in h), None)
                mgr.save(state.global_step, loss, lambda d: self.trainer.model.save_pretrained(str(d)))

    callback = _Callback()
    trainer = tf.Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_ds, data_collator=collate, callbacks=[callback]
    )
    callback.trainer = trainer
    if resume_from is not None:
        mgr.load(resume_from)
    trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    # Luôn lưu checkpoint cuối, kể cả khi step cuối không rơi vào nhịp lưu.
    mgr.save(trainer.state.global_step, None, lambda d: trainer.model.save_pretrained(str(d)))
    processor.save_pretrained(str(out_dir / "processor"))
    return mgr.directory


def transcribe_rows(
    model_dir: Path, processor_dir: Path, rows: Sequence[Mapping[str, Any]], split_dir: Path
) -> list[str]:
    """Chép lời `rows` bằng giải mã tham lam với một checkpoint (đã gộp); dùng để chọn checkpoint."""
    stack = _import_training_stack()
    tf, torch = stack.transformers, stack.torch
    processor = tf.AutoProcessor.from_pretrained(str(processor_dir))
    model = tf.AutoModelForSpeechSeq2Seq.from_pretrained(str(model_dir)).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    sr = int(processor.feature_extractor.sampling_rate)
    out: list[str] = []
    with torch.no_grad():
        for row in rows:
            audio = load_audio(stack, split_dir / str(row["audio"]), sr)
            feats = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
            ids = model.generate(feats, num_beams=1, do_sample=False)
            out.append(processor.batch_decode(ids, skip_special_tokens=True)[0])
    return out


def merge_lora(base: str, adapter_dir: Path, out_dir: Path, *, revision: str | None = None) -> Path:
    """Gộp adapter LoRA vào mô hình ASR gốc và lưu safetensors (chạy CPU cũng được)."""
    stack = _import_training_stack()
    model = stack.transformers.AutoModelForSpeechSeq2Seq.from_pretrained(base, revision=revision)
    merged = stack.peft.PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    stack.transformers.AutoProcessor.from_pretrained(base, revision=revision).save_pretrained(str(out_dir))
    return out_dir


def select_checkpoint(
    ckpts: Sequence[CheckpointInfo],
    transcribe: Callable[[CheckpointInfo], list[str]],
    rows: Sequence[Mapping[str, Any]],
    samples: Any = None,
) -> tuple[CheckpointInfo, list[dict[str, Any]]]:
    """Chấm mọi checkpoint trên `rows` (cặp val_unseen), chọn theo tỉ lệ sót thực thể rồi WER; trả về kèm bảng điểm."""
    scored = [(c, score_checkpoint(rows, transcribe(c), samples)) for c in ckpts]
    table = [{"path": str(c.path), "step": c.step, **s} for c, s in scored]
    return choose_checkpoint(scored), table


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Tham số dòng lệnh."""
    ap = argparse.ArgumentParser(prog="python -m lifemate.train.asr", description="ASR fine-tune (SPEC 6.4.3).")
    ap.add_argument("--config", required=True, type=Path, help="configs/train/asr_v1.toml")
    ap.add_argument("--env", default="local", help="env profile")
    ap.add_argument("--split", default="train", help="data split folder holding the audio")
    ap.add_argument("--out-dir", type=Path, help="default <runs_dir>/train/<config stem>")
    ap.add_argument("--resume-from", type=Path)
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None, repo_root: Path = REPO_ROOT) -> int:
    """Điểm vào CLI: huấn luyện rồi ghi train_config.json."""
    args = parse_args(argv)
    cfg_path = abs_path(args.config, repo_root)
    cfg = load_asr_train_config(cfg_path)
    profile = load_profile(args.env, repo_root)
    out = (
        abs_path(args.out_dir, repo_root)
        if args.out_dir
        else abs_path(profile.runs_dir, repo_root) / "train" / cfg_path.stem
    )
    ckpts = train(
        cfg,
        repo_root=repo_root,
        data_root=abs_path(profile.data_root, repo_root),
        split=args.split,
        out_dir=out,
        resume_from=args.resume_from,
    )
    data = abs_path(cfg.data.train, repo_root)
    write_train_config(
        out / "train_config.json",
        dataclasses.asdict(cfg),
        detect(repo_root),
        {"train": sha256_file(data)} if data.is_file() else {},
        argv=list(sys.argv),
    )
    print(f"checkpoints: {ckpts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
