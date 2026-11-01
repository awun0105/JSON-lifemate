"""Chuẩn bị PhoWhisper-large-CS (`rinhoooo/phowhisper-large-vien-cs-asr`) cho engine `faster_whisper` (mục 6e).

Cách dùng (trên máy có GPU, vì cần tải khoảng 6,4 GB và bước kiểm tra chạy mô hình):
  python -m lifemate.eval.phowhisper_cs_prepare --out <thư mục mô hình> --check-wavs <wav> <wav> ...

Nội dung chính:
- `prepare`: tải PhoWhisper-large và bộ LoRA ở đúng commit, gộp LoRA vào mô hình gốc (`merge_and_unload`), lưu
  `merged/` (safetensors float16), rồi chuyển sang CTranslate2 float16 (`ct2/`) kèm `tokenizer.json` và
  `preprocessor_config.json` (80 kênh mel; faster-whisper đọc file này, large-v3 dùng 128).
- `check`: chép các file WAV bằng bản gộp chạy trong transformers (giải mã tham lam, ép `<|vi|>`, như cách dùng trong
  model card nhưng có ép ngôn ngữ) và bằng bản CTranslate2 qua `FasterWhisperASR`, in chữ thô để thấy thẻ ngôn ngữ nếu
  có, và đếm số câu hai bản khác nhau. Hai bản phải gần như trùng thì việc chuyển đổi mới đúng.

Lưu ý:
- Model card không nói dạng chữ lúc fine-tune (dấu câu, số, thẻ ngôn ngữ; chỉ ghi "language tags stripped" khi chấm),
  nên `check` in chữ thô để xem trước khi benchmark.
- `generation_config` gốc không ép ngôn ngữ (`forced_decoder_ids` để trống ngôn ngữ): không ép thì Whisper tự đoán và
  có thể chuyển sang tiếng Anh ở câu chêm nhiều từ tiếng Anh; ở đây luôn ép `vi`, như cấu hình large-v3 đang dùng.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

BASE = "vinai/PhoWhisper-large"
BASE_REVISION = "b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb"
ADAPTER = "rinhoooo/phowhisper-large-vien-cs-asr"
ADAPTER_REVISION = "a98f55e0f42b2c4f1e71b3348a2b917fac0a7328"
COPY_FILES = ("tokenizer.json", "preprocessor_config.json")


def _numel(shape: Sequence[int]) -> int:
    """Số phần tử của một tensor có kích thước `shape`."""
    n = 1
    for d in shape:
        n *= int(d)
    return n


def prepare(out: Path) -> dict[str, Any]:
    """Gộp LoRA và chuyển sang CTranslate2; trả số tham số và đường dẫn."""
    import ctranslate2
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    base_dir = Path(snapshot_download(BASE, revision=BASE_REVISION))
    merged = out / "merged"
    if not (merged / "model.safetensors").is_file():  # bước gộp chạy lại được: bỏ qua khi đã có bản gộp
        adapter_dir = snapshot_download(ADAPTER, revision=ADAPTER_REVISION)
        model = WhisperForConditionalGeneration.from_pretrained(base_dir, torch_dtype=torch.float32)
        model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
        model.to(torch.float16).save_pretrained(merged, safe_serialization=True)
        WhisperProcessor.from_pretrained(base_dir).save_pretrained(merged)
    # transformers 5 lưu cấu hình processor thành `processor_config.json`; CTranslate2 và faster-whisper cần đúng file
    # `preprocessor_config.json` (80 kênh mel) và `tokenizer.json`, nên chép nguyên bản từ mô hình gốc.
    for name in COPY_FILES:
        shutil.copyfile(base_dir / name, merged / name)
    from safetensors import safe_open

    with safe_open(str(merged / "model.safetensors"), framework="pt") as f:
        n_params = sum(_numel(f.get_slice(k).get_shape()) for k in f.keys())  # noqa: SIM118 (safe_open không phải dict)
    ct2 = out / "ct2"
    if ct2.exists():
        shutil.rmtree(ct2)
    ctranslate2.converters.TransformersConverter(str(merged), copy_files=list(COPY_FILES)).convert(
        str(ct2), quantization="float16"
    )
    pre = json.loads((ct2 / "preprocessor_config.json").read_text())
    return {"merged": str(merged), "ct2": str(ct2), "n_params": n_params, "feature_size": pre.get("feature_size")}


def check(out: Path, wavs: Sequence[Path]) -> dict[str, Any]:
    """Chép `wavs` bằng bản gộp (transformers) và bản CTranslate2 (faster-whisper), so chữ thô."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from lifemate.backends.asr import FasterWhisperASR, read_wav
    from lifemate.core.config import ASRConfig

    merged = out / "merged"
    proc = WhisperProcessor.from_pretrained(merged)
    hf = WhisperForConditionalGeneration.from_pretrained(merged, torch_dtype=torch.float16).to("cuda").eval()
    fw = FasterWhisperASR(ASRConfig(backend="asr", engine="faster_whisper", model=str(out / "ct2"), revision="",
                                    generate_kwargs={"language": "vi", "task": "transcribe", "temperature": 0.0,
                                                     "beam_size": 1, "condition_on_previous_text": False}))  # fmt: skip
    rows, same = [], 0
    for p in wavs:
        x = read_wav(p, 16000)
        feats = proc(x, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
        with torch.no_grad():
            ids = hf.generate(feats, language="vi", task="transcribe", num_beams=1, max_new_tokens=440)
        h_hf = proc.batch_decode(ids, skip_special_tokens=False)[0]
        h_hf_clean = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        h_fw = fw.transcribe_one(x, "")[0]
        same += " ".join(h_hf_clean.split()) == " ".join(h_fw.split())
        rows.append({"wav": str(p), "hf_raw": h_hf, "hf": h_hf_clean, "faster_whisper": h_fw})
    res = {"n": len(rows), "hf_equals_ct2": same, "rows": rows}
    (out / "check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: xem docstring module."""
    ap = argparse.ArgumentParser(prog="lifemate.eval.phowhisper_cs_prepare", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check-wavs", type=Path, nargs="*", default=[])
    ap.add_argument("--skip-prepare", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.skip_prepare:
        print(json.dumps(prepare(args.out), ensure_ascii=False))
    if args.check_wavs:
        res = check(args.out, args.check_wavs)
        print(json.dumps({k: res[k] for k in ("n", "hf_equals_ct2")}, ensure_ascii=False))
        for r in res["rows"][:10]:
            print("HF raw:", r["hf_raw"][:200], "\n   CT2:", r["faster_whisper"][:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
