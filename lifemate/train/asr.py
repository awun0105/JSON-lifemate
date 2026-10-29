"""Fine-tune model ASR (LoRA hoặc toàn phần, kiểu Seq2Seq) và chọn checkpoint theo tỉ lệ bỏ sót thực thể (SPEC 6.4.3).

Nội dung chính:
- `ASRTrainConfig` và các mục `ASRBase`, `ASRData`, `ASRMethod`, `ASRTrain`: config train đọc từ
  `configs/train/asr_vN.toml`; `load_asr_train_config` đọc và kiểm tra file này.
- `normalize_text`, `wer`, `score_checkpoint`, `choose_checkpoint`: hàm đo thuần, chạy được không cần GPU.
- `train`, `transcribe_rows`, `merge_lora`, `select_checkpoint`: phần chạy trên GPU.
- `parse_args`, `main`: CLI train rồi ghi `train_config.json`.

Cách hoạt động:
1. File cặp jsonl do `asr_data` tạo.
2. `train` fine-tune và lưu checkpoint qua `CheckpointManager`.
3. Chép lời `val_unseen` bằng từng checkpoint, chọn checkpoint tốt nhất.
4. Gộp LoRA vào model gốc.

Lưu ý:
- Chưa chọn model. Mã viết cho model speech seq2seq của Hugging Face (kiểu Whisper: processor + `generate`).
- Các gói nặng (torch, transformers, peft, datasets, soundfile) chỉ được import trong `_import_training_stack`.
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
from lifemate.core.seeding import seed_everything
from lifemate.train.common import (
    CheckpointInfo,
    CheckpointManager,
    MetricsLogger,
    abs_path,
    build_section,
    load_toml,
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
    """Mục `[base]`: model gốc và revision.

    Thuộc tính:
        model: tên model trên Hugging Face; rỗng khi chưa chọn.
        revision: commit cố định của model.
    """

    model: str
    revision: str


@dataclass(frozen=True, slots=True)
class ASRData:
    """Mục `[data]`: đường dẫn file cặp dữ liệu train.

    Thuộc tính:
        train: file jsonl do `lifemate.train.asr_data` tạo.
    """

    train: str


@dataclass(frozen=True, slots=True)
class ASRMethod:
    """Mục `[method]`: kiểu train và các tuỳ chọn đi kèm.

    Thuộc tính:
        kind: `"lora"` hoặc `"full"` (train toàn bộ trọng số).
        freeze_encoder: `True` để đóng băng encoder, chỉ train decoder.
        spec_augment: `True` để bật SpecAugment (che ngẫu nhiên một phần phổ âm thanh khi train).
    """

    kind: str
    freeze_encoder: bool
    spec_augment: bool


@dataclass(frozen=True, slots=True)
class ASRTrain:
    """Mục `[train]`: siêu tham số và nhịp eval, nhịp lưu checkpoint.

    Thuộc tính:
        lr: learning rate.
        epochs: số epoch.
        batch_size: batch size trên mỗi GPU.
        grad_accum: số bước cộng dồn gradient.
        seed: seed ngẫu nhiên.
        eval_every_steps: số step giữa hai lần eval.
        save_every_steps: số step giữa hai lần lưu checkpoint; phải là bội của `eval_every_steps`.
    """

    lr: float
    epochs: int
    batch_size: int
    grad_accum: int
    seed: int
    eval_every_steps: int
    save_every_steps: int


@dataclass(frozen=True, slots=True)
class ASRTrainConfig:
    """Toàn bộ nội dung `configs/train/asr_vN.toml`.

    Thuộc tính:
        base: mục `[base]`.
        data: mục `[data]`.
        method: mục `[method]`.
        train: mục `[train]`.
    """

    base: ASRBase
    data: ASRData
    method: ASRMethod
    train: ASRTrain


def load_asr_train_config(path: Path) -> ASRTrainConfig:
    """Đọc và kiểm tra file TOML config train ASR, từ chối khoá lạ hoặc thiếu.

    Ngoài kiểm tra khoá, hàm còn kiểm tra `method.kind` thuộc `lora|full` và `save_every_steps` là bội của
    `eval_every_steps`.

    Lỗi:
        ConfigError: khi config sai.
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


# ---------------------------------------------------------------- Chỉ số (hàm thuần; dự phòng khi thiếu `eval_asr`)


def normalize_text(text: str) -> str:
    """Chuẩn hoá văn bản để tính WER theo SPEC 6.4.2: chữ thường, NFC, bỏ dấu câu, gộp khoảng trắng."""
    # `<unk>` là token không biết của model, không phải từ: xoá trước, kẻo bỏ dấu `<>` xong còn lại từ "unk".
    text = unicodedata.normalize("NFC", text.lower()).replace("<unk>", " ")
    # Ký tự thuộc nhóm Unicode P (dấu câu) hoặc S (ký hiệu) được thay bằng khoảng trắng.
    kept = "".join(" " if unicodedata.category(ch)[0] in _PUNCT_CATEGORIES else ch for ch in text)
    return " ".join(kept.split())


def _edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Tính khoảng cách Levenshtein theo từ bằng quy hoạch động, chỉ giữ hai hàng trong bộ nhớ."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Tính WER trên toàn tập (tổng lỗi chia tổng số từ tham chiếu), trên văn bản đã chuẩn hoá."""
    errors = words = 0
    for ref, hyp in zip(refs, hyps, strict=True):
        r, h = normalize_text(ref).split(), normalize_text(hyp).split()
        errors += _edit_distance(r, h)
        words += len(r)
    return errors / words if words else 0.0


def score_checkpoint(rows: Sequence[Mapping[str, Any]], hyps: Sequence[str], samples: Any = None) -> dict[str, float]:
    """Chấm bản chép lời của một checkpoint: `entity_miss_rate` và WER.

    `entity_miss_rate` tính bằng `lifemate.eval.eval_asr`, chỉ trên lượt cuối của mỗi mẫu. Khi không có `samples`,
    `entity_miss_rate` lấy bằng WER và `entity_metric` = 0.0 để đánh dấu đây là giá trị dự phòng.

    Tham số:
        rows: các cặp dữ liệu đã chép lời.
        hyps: văn bản chép lời, cùng thứ tự với `rows`.
        samples: các `Sample` có nhãn để tính thực thể; `None` thì chỉ có WER.

    Trả về:
        Dict gồm `entity_miss_rate`, `wer` và `entity_metric` (1.0 khi tính thật, 0.0 khi dự phòng).
    """
    refs = [str(r["text"]) for r in rows]
    w = wer(refs, hyps)
    if samples is None:
        log.warning("eval_asr_unavailable", fallback="wer")
        return {"entity_miss_rate": w, "wer": w, "entity_metric": 0.0}
    from lifemate.core.types import Transcript
    from lifemate.eval.eval_asr import entity_miss_rate  # tầng eval; import ở đây để `train` vẫn import được độc lập

    # Chỉ lượt cuối được chấm thực thể; các lượt lịch sử chỉ góp vào WER.
    final_hyp = {str(r["id"]): h for r, h in zip(rows, hyps, strict=True) if r.get("turn") == "final"}
    by_id = {s.id: s for s in samples}
    ids = [i for i in final_hyp if i in by_id]
    ordered = [by_id[i] for i in ids]
    transcripts = [Transcript(final=final_hyp[i], history=(), seconds=0.0) for i in ids]
    _, _, rate, _ = entity_miss_rate(ordered, transcripts)
    return {"entity_miss_rate": float(rate), "wer": w, "entity_metric": 1.0}


def choose_checkpoint(scored: Sequence[tuple[CheckpointInfo, Mapping[str, float]]]) -> CheckpointInfo:
    """Chọn checkpoint có tỉ lệ bỏ sót thực thể thấp nhất trên `val_unseen`, hoà thì WER thấp nhất, rồi step sớm nhất.

    Lỗi:
        ValueError: khi không có checkpoint nào.
    """
    if not scored:
        raise ValueError("no checkpoints to choose from")
    return min(scored, key=lambda cs: (cs[1]["entity_miss_rate"], cs[1]["wer"], cs[0].step))[0]


# ---------------------------------------------------------------- Phần chạy trên GPU (test không chạy phần này)


def _import_training_stack() -> SimpleNamespace:
    """Import các gói cần cho fine-tune ASR.

    Trả về:
        Namespace chứa các module, truy cập theo tên (ví dụ `stack.torch`).

    Lỗi:
        RuntimeError: khi thiếu gói, kèm tên các gói còn thiếu.
    """
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
    """Đọc file audio thành mảng float mono ở `sampling_rate`.

    Lỗi:
        ValueError: khi tần số lấy mẫu của file khác `sampling_rate`; cần resample dữ liệu trước.
    """
    data, sr = stack.soundfile.read(str(path), dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    if sr != sampling_rate:
        raise ValueError(f"{path}: sample rate {sr} != {sampling_rate}; resample the data first")
    return data


def train(
    cfg: ASRTrainConfig, *, repo_root: Path, data_root: Path, split: str, out_dir: Path, resume_from: Path | None = None
) -> Path:
    """Fine-tune model ASR trên `cfg.data.train`; checkpoint nằm trong `out_dir/checkpoints`.

    Trainer của HF không tự lưu (`save_strategy="no"`). Một callback lưu qua `CheckpointManager` mỗi
    `save_every_steps` step và ghi log vào `metrics.jsonl`. Có `resume_from` thì khôi phục RNG rồi train tiếp.

    Tham số:
        cfg: config train.
        repo_root: thư mục gốc repo.
        data_root: thư mục dữ liệu.
        split: split chứa file audio.
        out_dir: thư mục ghi kết quả.
        resume_from: checkpoint để train tiếp; `None` là train từ đầu.

    Trả về:
        Thư mục chứa các checkpoint.

    Lỗi:
        ConfigError: khi chưa chọn `base.model`.
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
        # LoRA chỉ gắn vào q_proj và v_proj của attention; `lora_alpha = 2r` là cặp giá trị phổ biến.
        lora = stack.peft.LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
        model = stack.peft.get_peft_model(model, lora)
    sr = int(processor.feature_extractor.sampling_rate)
    split_dir = data_root / split

    # Mỗi dòng: đặc trưng âm thanh làm input, token của text nguyên văn làm nhãn.
    def encode(row: dict[str, Any]) -> dict[str, Any]:
        """Đổi một dòng dữ liệu thành đặc trưng âm thanh (input) và token của văn bản (nhãn)."""
        audio = load_audio(stack, split_dir / row["audio"], sr)
        feats = processor.feature_extractor(audio, sampling_rate=sr).input_features[0]
        return {"input_features": feats, "labels": processor.tokenizer(row["text"]).input_ids}

    rows = read_jsonl(abs_path(cfg.data.train, repo_root))
    train_ds = stack.datasets.Dataset.from_list(rows).map(encode, remove_columns=list(rows[0].keys()))

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Gộp một batch: đệm đặc trưng âm thanh và nhãn cho cùng độ dài."""
        feats = processor.feature_extractor.pad(
            [{"input_features": b["input_features"]} for b in batch], return_tensors="pt"
        )
        labels = processor.tokenizer.pad([{"input_ids": b["labels"]} for b in batch], return_tensors="pt")
        # Vị trí padding của nhãn đặt -100 để loss bỏ qua.
        ids = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        return {"input_features": feats["input_features"], "labels": ids}

    # GPU hỗ trợ bf16 thì dùng bf16, không thì dùng fp16.
    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    args = tf.Seq2SeqTrainingArguments(
        output_dir=str(out_dir / "hf"),
        learning_rate=cfg.train.lr,
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        save_strategy="no",  # `CheckpointManager` lo việc lưu (SPEC 6.5)
        logging_steps=10,
        seed=cfg.train.seed,
        data_seed=cfg.train.seed,
        fp16=not bf16,
        bf16=bf16,
        report_to=[],
        remove_unused_columns=False,
    )
    # Gán qua biến kiểu `Any` để mypy chấp nhận lớp con của một lớp nạp động.
    base_cb: Any = tf.TrainerCallback

    class _Callback(base_cb):  # type: ignore[misc]
        """Callback của Trainer: ghi log vào `metrics.jsonl` và lưu checkpoint theo nhịp `save_every_steps`."""

        trainer: Any = None

        def on_log(self, a: Any, state: Any, control: Any, logs: Any = None, **kw: Any) -> None:
            """Ghi mỗi lần Trainer log ra một dòng trong `metrics.jsonl`."""
            metrics.log({"step": state.global_step, **(logs or {})})

        def on_step_end(self, a: Any, state: Any, control: Any, **kw: Any) -> None:
            """Lưu checkpoint khi step hiện tại là bội của `save_every_steps`."""
            if state.global_step % cfg.train.save_every_steps == 0:
                # Loss gần nhất trong `log_history` làm metric của checkpoint.
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
    # Luôn lưu checkpoint cuối, kể cả khi step cuối không rơi đúng nhịp lưu.
    mgr.save(trainer.state.global_step, None, lambda d: trainer.model.save_pretrained(str(d)))
    processor.save_pretrained(str(out_dir / "processor"))
    return mgr.directory


def transcribe_rows(
    model_dir: Path, processor_dir: Path, rows: Sequence[Mapping[str, Any]], split_dir: Path
) -> list[str]:
    """Chép lời `rows` bằng một checkpoint (đã gộp), giải mã tham lam; dùng khi chọn checkpoint.

    Tham số:
        model_dir: thư mục model đã gộp.
        processor_dir: thư mục processor.
        rows: các cặp dữ liệu cần chép lời.
        split_dir: thư mục split chứa file audio.

    Trả về:
        Văn bản chép lời, cùng thứ tự với `rows`.
    """
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
    """Gộp adapter LoRA vào model ASR gốc và lưu dạng safetensors; chạy trên CPU cũng được.

    Tham số:
        base: tên model gốc.
        adapter_dir: thư mục adapter LoRA.
        out_dir: thư mục ghi model đã gộp.
        revision: commit của model gốc.

    Trả về:
        `out_dir`.
    """
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
    """Chấm mọi checkpoint trên `rows` (cặp `val_unseen`) và chọn theo tỉ lệ bỏ sót thực thể, rồi WER.

    Tham số:
        ckpts: các checkpoint cần chấm.
        transcribe: hàm chép lời `rows` bằng một checkpoint.
        rows: các cặp dữ liệu dùng để chấm.
        samples: các `Sample` có nhãn để tính thực thể; `None` thì chỉ dùng WER.

    Trả về:
        Cặp (checkpoint được chọn, bảng điểm của mọi checkpoint).
    """
    scored = [(c, score_checkpoint(rows, transcribe(c), samples)) for c in ckpts]
    table = [{"path": str(c.path), "step": c.step, **s} for c, s in scored]
    return choose_checkpoint(scored), table


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    ap = argparse.ArgumentParser(prog="python -m lifemate.train.asr", description="ASR fine-tune (SPEC 6.4.3).")
    ap.add_argument("--config", required=True, type=Path, help="configs/train/asr_v1.toml")
    ap.add_argument("--env", default="local", help="env profile")
    ap.add_argument("--split", default="train", help="data split folder holding the audio")
    ap.add_argument("--out-dir", type=Path, help="default <runs_dir>/train/<config stem>")
    ap.add_argument("--resume-from", type=Path)
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None, repo_root: Path = REPO_ROOT) -> int:
    """Chạy CLI: train rồi ghi `train_config.json`; trả về mã thoát 0."""
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
