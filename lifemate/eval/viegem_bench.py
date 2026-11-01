"""Benchmark VieGem-ASR dùng nguyên bản so với ChunkFormer RNNT ở vai trò mô hình chính (benchmark ASR lần 4 mục 6d).

Cách dùng (từ thư mục gốc repo):
  python -m lifemate.eval.viegem_bench check --out <thư mục>                                   # GPU, 10 file mẫu
  python -m lifemate.eval.viegem_bench dump --env colabcli --ids configs/splits/asr_tune.txt --out <thư mục>   # GPU
  python -m lifemate.eval.viegem_bench decode --dump <thư mục> --split asr_tune --weights 0 2 4 8   # CPU, cần kenlm
  python -m lifemate.eval.viegem_bench score --dump <thư mục> --split asr_tune --rnnt <rnnt.hyps.jsonl>

Nội dung chính:
- `stage_check`: tải 10 file mẫu kèm lời chuẩn từ Space của tác giả (ở đúng commit), nạp mô hình, báo số khoá encoder
  lấy từ mô hình gốc và số tham số, so chữ float32 với chữ kiểu số thấp (bf16 trên sm80, fp16 trên GPU cũ), tính WER
  so với lời chuẩn theo đúng bộ chuẩn hoá của tác giả (`text.normalize`), có và không có KenLM.
- `stage_dump`: chép mọi câu người dùng của một tập bằng giải mã tham lam, lưu chữ (`greedy.hyps.jsonl`), 32 token log
  xác suất cao nhất mỗi khung (`top32.npz`), thời gian từng mẫu và bộ nhớ GPU đỉnh. Chỉ chạy mô hình ở bước này.
- `stage_decode`: giải mã lại bằng KenLM của tác giả (alpha 0,35, beta 1,5, chùm 32, như app) trên CPU từ `top32.npz`,
  với danh sách từ gợi ý của từng mẫu (`bias_words`: từ tiếng Anh có trong ngữ cảnh mẫu) và các trọng số cho trước.
- `stage_score`: so các cấu hình với RNNT bằng `asr_compare.compare` (bootstrap ghép cặp theo mẫu) và áp tiêu chí đặt
  trước của mục 6d.

Cách hoạt động:
- KenLM là mô hình n-gram, không phải mạng nơ-ron; giải mã lại trên CPU không chạy mô hình ASR. Bộ giải mã chỉ xét
  token có log xác suất ≥ −5 ở mỗi khung, nên 32 token cao nhất đủ thay cho toàn bộ 21 033 token.
- Từ gợi ý lấy từ đúng nguồn của từ điển ngữ cảnh mà H-Anh dùng (`asr_merge.sample_context_words`: catalog của app đã
  cài, ngữ cảnh người dùng, lượt trợ lý trong lịch sử), chỉ giữ từ tiếng Anh (`find_foreign_words`), viết thường.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SPACE = "baohuynhbk14/VieGem-ASR"
SPACE_REVISION = "60eea95b3ffeb2c517a6f3bd4a79a5ca138b17c1"
MODEL = "5CD-AI/VieGem-ASR"
MODEL_REVISION = "95eae70b21fb0ef640e3ac4ea8cfa2380872ed20"
TOP_K = 32
CANDIDATE_ORDER = ("greedy", "lm_w0", "lm_w2", "lm_w4", "lm_w8")  # từ đơn giản tới phức tạp (tiêu chí mục 6d)
METRICS = ("wer", "vi_only_wer", "wer_nonum", "en_word_recall", "en_word_precision", "en_word_f1", "entity_miss_rate")


def _cfg(decoder: str = "greedy", dtype: str = "auto") -> Any:
    from lifemate.core.config import ASRConfig

    return ASRConfig(
        backend="asr", engine="viegem", model=MODEL, revision=MODEL_REVISION, dtype=dtype,
        generate_kwargs={"decoder": decoder},
    )  # fmt: skip


def _wer(ref: str, hyp: str) -> tuple[int, int]:
    from lifemate.eval.eval_asr import _edit_distance

    r, h = ref.split(), hyp.split()
    return _edit_distance(r, h), len(r)


def stage_check(args: argparse.Namespace) -> dict[str, Any]:
    """Ba bước kiểm tra trước benchmark (asr_v4.md mục 6d)."""
    import importlib

    import numpy as np
    import soundfile as sf

    from lifemate.backends.asr_viegem import ViegemASR

    out: Path = args.out
    ex_dir = out / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/spaces/{SPACE}/resolve/{SPACE_REVISION}/examples"
    meta = json.loads(urllib.request.urlopen(f"{base}/examples.json").read().decode("utf-8"))
    audio: dict[str, Any] = {}
    for e in meta:
        p = ex_dir / e["file"]
        if not p.is_file():
            p.write_bytes(urllib.request.urlopen(f"{base}/{e['file']}").read())
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != 16000:
            import soxr

            x = soxr.resample(x, sr, 16000).astype(np.float32)
        audio[e["file"]] = x
    res: dict[str, Any] = {"examples": len(meta)}
    fp32 = ViegemASR(_cfg("greedy", "float32"))
    text = importlib.import_module("viegem_asr_code.text")
    res["n_params"] = fp32.n_params
    import torch

    low = "bfloat16" if torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 0) else "float16"
    lowp = ViegemASR(_cfg("greedy", low))
    rows = []
    tot = {"fp32": [0, 0], low: [0, 0]}
    same = 0
    for e in meta:
        ref = text.normalize(e["ref"])
        h32, hlo = fp32.transcribe_audio(audio[e["file"]]), lowp.transcribe_audio(audio[e["file"]])
        same += h32 == hlo
        for name, h in (("fp32", h32), (low, hlo)):
            err, n = _wer(ref, h)
            tot[name][0] += err
            tot[name][1] += n
        rows.append({"file": e["file"], "ref": ref, "fp32": h32, low: hlo})
    res["greedy_wer"] = {k: v[0] / max(1, v[1]) for k, v in tot.items()}
    res["fp32_equals_" + low] = f"{same}/{len(meta)}"
    try:
        lm = ViegemASR(_cfg("lm", "float32"))
        e_lm = [0, 0]
        for r, e in zip(rows, meta, strict=True):
            r["lm"] = lm.transcribe_audio(audio[e["file"]])
            err, n = _wer(r["ref"], r["lm"])
            e_lm[0] += err
            e_lm[1] += n
        res["lm_wer"] = e_lm[0] / max(1, e_lm[1])
    except ImportError as exc:  # kenlm chưa cài
        res["lm_wer"] = f"skipped: {exc}"
    (out / "check.json").write_text(json.dumps({**res, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def _utterances(sample: Any) -> list[tuple[str, str | None]]:
    keys: list[tuple[str, str | None]] = [
        (str(i), t.audio_text) for i, t in enumerate(sample.history) if t.role == "user"
    ]
    keys.append(("final", sample.audio_text))
    return keys


def stage_dump(args: argparse.Namespace) -> dict[str, Any]:
    """Chép mọi câu của `args.ids` (giải mã tham lam), lưu chữ, top-32 log xác suất, thời gian, bộ nhớ."""
    import numpy as np

    from lifemate.backends.asr import ASRTranscriber, read_wav
    from lifemate.backends.asr_viegem import ViegemASR
    from lifemate.core.config import load_profile
    from lifemate.core.paths import REPO_ROOT, repo_path
    from lifemate.eval.gpu_memory import GpuMemoryMonitor
    from lifemate.io.data import load_split, read_ids

    split = Path(args.ids).stem
    out: Path = args.out / split
    out.mkdir(parents=True, exist_ok=True)
    root = repo_path(load_profile(args.env, REPO_ROOT).with_overrides(root=args.root).data_root, REPO_ROOT)
    samples = load_split(root, "train", read_ids(repo_path(args.ids, REPO_ROOT)))
    with GpuMemoryMonitor(out / "gpu_memory.csv") as mon:
        t0 = time.perf_counter()
        eng = ViegemASR(_cfg("greedy", args.dtype))
        eng.warmup()
        load_s = time.perf_counter() - t0
        locator = ASRTranscriber(eng._cfg, data_root=root, split="train", backend=eng)
        rows, timing, arrays = [], [], {}
        for n, s in enumerate(samples, 1):
            start = time.perf_counter()
            for (turn, ref), path in zip(_utterances(s), locator.wav_paths(s), strict=True):
                segs = eng.top_logprobs(read_wav(path, 16000), TOP_K)
                hyp = " ".join(x["greedy"] for x in segs if x["greedy"]).strip()
                rows.append({"id": s.id, "turn": turn, "ref": ref, "hyp": hyp, "raw": hyp})
                for j, x in enumerate(segs):
                    arrays[f"{s.id}|{turn}|{j}|ids"] = x["ids"].astype(np.int32)
                    arrays[f"{s.id}|{turn}|{j}|lp"] = x["lp"].astype(np.float16)
            timing.append({"id": s.id, "seconds": time.perf_counter() - start})
            if n % 50 == 0:
                print(f"[viegem] {n}/{len(samples)}", flush=True)
    (out / "greedy.hyps.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    (out / "timing.jsonl").write_text("".join(json.dumps(t) + "\n" for t in timing), "utf-8")
    np.savez_compressed(out / "top32.npz", **arrays)
    summary = {"split": split, "n_samples": len(samples), "load_s": load_s, "dtype": eng.dtype,
               "n_params": eng.n_params, "gpu_peak_mb": mon.peak_mb}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
    return summary


def bias_words(sample: Any, catalog: Mapping[str, Any]) -> list[str]:
    """Từ tiếng Anh (viết thường) trong ngữ cảnh mẫu, theo đúng nguồn của từ điển ngữ cảnh của H-Anh."""
    from lifemate.eval.asr_merge import _strings
    from lifemate.eval.foreign_words import find_foreign_words

    context = dict(sample.context)
    installed = context.pop("installed_apps", None) or {}
    texts: list[str] = []
    for app in installed:
        texts.append(str(app).replace("_", " "))
        texts.extend(_strings(catalog.get(app, {})))
    texts.extend(_strings(context))
    for turn in sample.history:
        if turn.role == "assistant":
            texts.extend(_strings([turn.text, turn.tool_calls, turn.tool_results]))
    return sorted({w.folded for w in find_foreign_words(" ".join(texts))})


def stage_decode(args: argparse.Namespace) -> dict[str, Any]:
    """Giải mã lại từ `top32.npz` bằng KenLM của tác giả với từng trọng số gợi ý; ghi `lm_w<trọng số>.hyps.jsonl`."""
    import importlib

    import numpy as np
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from lifemate.backends.asr_viegem import ENCODER_ID, ENCODER_REVISION, _import_code
    from lifemate.eval.asr_v3_report import _data

    d: Path = args.dump / args.split
    repo = Path(snapshot_download(MODEL, revision=MODEL_REVISION, allow_patterns=["asr/*", "lm/*", "viegem_asr.pt"]))
    _import_code(repo)
    lm_mod = importlib.import_module("viegem_asr_code.lm_decode")
    text = importlib.import_module("viegem_asr_code.text")
    import torch

    ck = torch.load(repo / "viegem_asr.pt", map_location="cpu", weights_only=True)
    units = text.Units(ck["gemma_ids"])
    tok = AutoTokenizer.from_pretrained(snapshot_download(ENCODER_ID, revision=ENCODER_REVISION))
    pieces = ["", "", *tok.convert_ids_to_tokens(units.gemma_ids)]
    words = (repo / "lm" / "viegem_lm.unigrams.txt").read_text(encoding="utf-8").split()
    samples, catalog = _data(f"configs/splits/{args.split}.txt")
    by_id = {s.id: s for s in samples}
    arrays = np.load(d / "top32.npz")
    greedy = [json.loads(x) for x in (d / "greedy.hyps.jsonl").read_text("utf-8").splitlines() if x.strip()]
    n_units = len(pieces)
    out: dict[str, Any] = {}
    # Nạp KenLM một lần; mỗi mẫu chỉ thay tập từ gợi ý (và thêm chúng vào từ vựng, như hàm khởi tạo của tác giả).
    dec = lm_mod.LMDecoder(str(repo / "lm" / "viegem_lm.bin"), unigrams=words, alpha=0.35, beta=1.5)
    base_vocab, base_prefixes = set(dec.vocab), set(dec.prefixes)
    bias_cache: dict[str, list[str]] = {}
    for w in args.weights:
        rows, secs = [], []
        dec.bias_weight = float(w)
        for r in greedy:
            s = by_id[r["id"]]
            if s.id not in bias_cache:
                bias_cache[s.id] = bias_words(s, catalog)
            bias = bias_cache[s.id] if w > 0 else []
            dec.bias_words = set(bias)
            dec.vocab = base_vocab | dec.bias_words
            dec.prefixes = base_prefixes | {b[:i] for b in dec.bias_words for i in range(1, len(b) + 1)}
            texts, j = [], 0
            t0 = time.perf_counter()
            while f"{r['id']}|{r['turn']}|{j}|ids" in arrays:
                ids, lp = arrays[f"{r['id']}|{r['turn']}|{j}|ids"], arrays[f"{r['id']}|{r['turn']}|{j}|lp"]
                full = np.full((ids.shape[0], n_units), -1e4, dtype=np.float32)
                np.put_along_axis(full, ids, lp.astype(np.float32), axis=1)
                texts.append(units.decode(tok, dec.decode(full, pieces, beam_width=32)))
                j += 1
            secs.append(time.perf_counter() - t0)
            hyp = " ".join(t for t in texts if t).strip()
            rows.append({**r, "hyp": hyp, "raw": hyp})
        name = f"lm_w{w:g}"
        (d / f"{name}.hyps.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), "utf-8")
        secs.sort()
        out[name] = {
            "utterances": len(rows),
            "decode_s_p50": secs[len(secs) // 2],
            "decode_s_p95": secs[int(0.95 * len(secs))],
        }
        print(name, out[name], flush=True)
    (d / "decode.json").write_text(json.dumps(out, indent=2), "utf-8")
    return out


def stage_score(args: argparse.Namespace) -> dict[str, Any]:
    """So RNNT (mốc) với các cấu hình VieGem đã có trong `--dump/<tập>`; áp tiêu chí chọn của mục 6d."""
    from lifemate.eval.asr_compare import compare
    from lifemate.eval.asr_v3_report import _data, _fixed_prompt, _rows

    d: Path = args.dump / args.split
    prompt_text, _ = _fixed_prompt()
    samples, _ = _data(f"configs/splits/{args.split}.txt")
    systems = {"rnnt": (_rows(args.rnnt), prompt_text)}
    if getattr(args, "merged", None):
        systems["merged_v2"] = (_rows(args.merged), prompt_text)
    for name in CANDIDATE_ORDER:
        p = d / f"{name}.hyps.jsonl"
        if p.is_file():
            systems[name] = (_rows(p), prompt_text)
    cmp = compare(samples, systems)
    table = {n: {m: {"value": cmp[n]["report"][m], "diff_ci": cmp[n]["diff_ci"][m]} for m in METRICS} for n in cmp}
    res = {"split": args.split, "baseline": "rnnt", "table": table}
    (d / "score.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    return res


def _cell(name: str, v: Mapping[str, Any]) -> str:
    lo, hi = v["diff_ci"]
    return f"{name} {100 * v['value']:.2f} [{100 * lo:+.2f}; {100 * hi:+.2f}]"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: xem docstring module."""
    ap = argparse.ArgumentParser(prog="lifemate.eval.viegem_bench", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("check")
    pc.add_argument("--out", type=Path, required=True)
    pd = sub.add_parser("dump")
    pd.add_argument("--env", required=True)
    pd.add_argument("--root", default=None)
    pd.add_argument("--ids", required=True)
    pd.add_argument("--dtype", default="auto")
    pd.add_argument("--out", type=Path, required=True)
    pe = sub.add_parser("decode")
    pe.add_argument("--dump", type=Path, required=True)
    pe.add_argument("--split", required=True)
    pe.add_argument("--weights", type=float, nargs="+", default=[0.0, 2.0, 4.0, 8.0])
    ps = sub.add_parser("score")
    ps.add_argument("--dump", type=Path, required=True)
    ps.add_argument("--split", required=True)
    ps.add_argument("--rnnt", type=Path, required=True)
    ps.add_argument("--merged", type=Path, help="chữ của pipeline ghép bản 2 (`merge_v2_report measure`), để so thêm")
    args = ap.parse_args(argv)
    if getattr(args, "split", "") == "val_unseen" or "val_unseen" in str(getattr(args, "ids", "")):
        raise SystemExit("val_unseen bị khoá (D53)")
    if args.cmd == "check":
        args.out.mkdir(parents=True, exist_ok=True)
        print(json.dumps(stage_check(args), ensure_ascii=False, indent=1))
    elif args.cmd == "dump":
        print(json.dumps(stage_dump(args), ensure_ascii=False))
    elif args.cmd == "decode":
        stage_decode(args)
    else:
        res = stage_score(args)
        for n, t in res["table"].items():
            cells = " | ".join(_cell(m, t[m]) for m in METRICS)
            print(f"{n}: {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
