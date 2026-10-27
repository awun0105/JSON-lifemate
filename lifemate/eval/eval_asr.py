"""Đo chất lượng ASR: WER, CER, tỷ lệ bỏ sót thực thể và độ trễ (SPEC 6.4.2).

Nội dung: các hàm chuẩn hóa văn bản, khoảng cách chỉnh sửa, WER/CER, trích thực thể từ lời gọi gold, phân vị độ trễ,
lớp báo cáo `AsrReport` và CLI chạy một transcriber trên một split.

Cách xây dựng:
  - WER/CER tính trên cả kho (corpus): cộng tổng số lỗi chỉnh sửa (Levenshtein) rồi chia cho tổng độ dài tham chiếu,
    trên mọi lượt người dùng trong lịch sử và câu cuối, sau khi chuẩn hóa (chữ thường, NFC, bỏ dấu câu).
  - Thực thể: giá trị chuỗi và số trong tham số của lời gọi gold, giữ lại những giá trị có xuất hiện trong audio_text
    sau khi gập (bỏ dấu tiếng Việt). Một thực thể bị bỏ sót khi không xuất hiện, theo ranh giới từ, trong văn bản ASR
    cuối đã gập.
  - Độ trễ: phân vị p50/p95 theo hạng gần nhất và giá trị lớn nhất của thời gian chép lời từng mẫu.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from lifemate.core.paths import REPO_ROOT
from lifemate.core.types import Sample, Transcriber, Transcript

_PUNCT = re.compile(r"[^\w\s]|_")


def _fold(s: str) -> str:
    """Gập văn bản: chữ thường, bỏ dấu tiếng Việt, gộp khoảng trắng.

    Copied from `_fold` in the competition package zaic_tools (train/zaic_tools/utils.py) so lifemate never imports
    zaic_tools; keep in sync with that file.
    (Sao chép từ `_fold` của zaic_tools để lifemate không phải import gói đó; cần giữ đồng bộ với tệp gốc.)
    """
    # NFD tách dấu thành ký tự tổ hợp để bỏ đi; "đ" không tách được nên đổi tay thành "d".
    t = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())


def _folded_tokens(s: str) -> str:
    """Văn bản đã gập dưới dạng các từ có khoảng trắng hai đầu, để phép so khớp tôn trọng ranh giới từ.

    Ví dụ "huy" không khớp bên trong "chuyen".
    """
    return " " + " ".join(re.findall(r"[^\W_]+", _fold(s))) + " "


def normalize_text(s: str) -> str:
    """Chuẩn hóa để tính WER/CER: chữ thường, NFC, bỏ dấu câu, gộp khoảng trắng."""
    t = unicodedata.normalize("NFC", s.lower())
    return " ".join(_PUNCT.sub(" ", t).split())


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Khoảng cách Levenshtein giữa hai dãy (từ hoặc ký tự), quy hoạch động chỉ giữ một hàng."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            # Xóa, chèn, hoặc thay thế (chi phí 0 khi hai phần tử bằng nhau).
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _rate(errors: int, n: int) -> float:
    """Tỷ lệ lỗi errors / n; khi n = 0 trả 0.0 nếu không có lỗi, ngược lại 1.0."""
    if n == 0:
        return 0.0 if errors == 0 else 1.0
    return errors / n


def wer(ref: str, hyp: str) -> float:
    """Tỷ lệ lỗi từ (WER) trên văn bản đã chuẩn hóa."""
    r, h = normalize_text(ref).split(), normalize_text(hyp).split()
    return _rate(_edit_distance(r, h), len(r))


def cer(ref: str, hyp: str) -> float:
    """Tỷ lệ lỗi ký tự (CER) trên văn bản đã chuẩn hóa (khoảng trắng cũng tính là ký tự)."""
    r, h = normalize_text(ref), normalize_text(hyp)
    return _rate(_edit_distance(list(r), list(h)), len(r))


def _values(obj: Any) -> Iterator[str]:
    """Duyệt đệ quy và sinh mọi giá trị chuỗi khác rỗng và giá trị số (dạng chuỗi) trong một cấu trúc JSON.

    Bool và None bị bỏ qua; số thực nguyên như 3.0 được ghi thành "3".
    """
    # bool là lớp con của int nên phải loại trước nhánh số.
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, str):
        if obj.strip():
            yield obj
    elif isinstance(obj, int | float):
        yield str(int(obj)) if isinstance(obj, float) and obj.is_integer() else str(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _values(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            yield from _values(v)


def entity_values(sample: Sample) -> list[str]:
    """Các giá trị tham số (chuỗi, số) của lời gọi gold có xuất hiện (sau khi gập) trong văn bản cuối gold.

    Bỏ qua `sys.ask_missing_info`; kết quả đã gập, không trùng lặp, theo thứ tự xuất hiện.
    """
    if sample.label is None or sample.audio_text is None:
        return []
    gold = _folded_tokens(sample.audio_text)
    out: list[str] = []
    for call in sample.label.get("tool_calls") or []:
        if call.get("name") == "sys.ask_missing_info":
            continue
        for v in _values(call.get("arguments") or {}):
            fv = _folded_tokens(v)
            # Chỉ giữ thực thể thực sự được nói ra, nếu không thì ASR không thể nghe được.
            if fv.strip() and fv in gold and fv.strip() not in out:
                out.append(fv.strip())
    return out


@dataclass(frozen=True, slots=True)
class EntityDetail:
    """Kết quả kiểm tra thực thể của một mẫu: id, các thực thể cần nghe và các thực thể bị bỏ sót."""

    id: str
    entities: tuple[str, ...]
    missed: tuple[str, ...]


def entity_miss_rate(
    samples: Sequence[Sample], transcripts: Sequence[Transcript]
) -> tuple[int, int, float, list[EntityDetail]]:
    """Trả về (missed, total, rate, details): thực thể gold có trong audio_text nhưng vắng trong văn bản ASR cuối.

    Văn bản ASR được gập trước khi so, và thực thể phải khớp nguyên từ.
    """
    missed = total = 0
    details: list[EntityDetail] = []
    for s, t in zip(samples, transcripts, strict=True):
        ents = entity_values(s)
        hyp = _folded_tokens(t.final)
        miss = tuple(e for e in ents if f" {e} " not in hyp)
        missed += len(miss)
        total += len(ents)
        details.append(EntityDetail(s.id, tuple(ents), miss))
    return missed, total, _rate(missed, total), details


def percentile(values: Sequence[float], q: float) -> float:
    """Phân vị theo hạng gần nhất (q trong khoảng 0..100); trả 0.0 khi không có giá trị."""
    if not values:
        return 0.0
    xs = sorted(values)
    k = max(1, math.ceil(q / 100 * len(xs)))
    return xs[k - 1]


@dataclass(slots=True)
class AsrReport:
    """Các chỉ số ASR đã tổng hợp: số mẫu, số câu, WER, CER, thực thể bỏ sót và độ trễ (giây)."""

    n_samples: int
    n_utterances: int
    wer: float
    cer: float
    entity_missed: int
    entity_total: int
    entity_miss_rate: float
    p50_s: float
    p95_s: float
    max_s: float
    entity_details: list[EntityDetail] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Dict sẵn sàng ghi JSON; phần chi tiết chỉ gồm các mẫu có thực thể bị bỏ sót."""
        d: dict[str, Any] = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "entity_details"}
        d["entity_misses"] = [
            {"id": e.id, "entities": list(e.entities), "missed": list(e.missed)}
            for e in self.entity_details
            if e.missed
        ]
        return d

    def to_text(self) -> str:
        """Bản tóm tắt ngắn cho người đọc."""
        return (
            f"samples={self.n_samples} utterances={self.n_utterances}\n"
            f"WER={self.wer:.4f} CER={self.cer:.4f}\n"
            f"entity_miss_rate={self.entity_miss_rate:.4f} ({self.entity_missed}/{self.entity_total})\n"
            f"seconds p50={self.p50_s:.3f} p95={self.p95_s:.3f} max={self.max_s:.3f}"
        )


def evaluate(samples: Sequence[Sample], transcriber: Transcriber) -> AsrReport:
    """Chép lời mọi mẫu rồi tính WER/CER trên cả kho, gồm câu cuối và các lượt người dùng trong lịch sử.

    Chỉ những câu có văn bản gold mới được tính. Transcriber được khởi động (warmup) trước để độ trễ không gồm thời
    gian nạp mô hình.
    """
    transcriber.warmup()
    transcripts = [transcriber.transcribe(s) for s in samples]
    w_err = w_n = c_err = c_n = n_utt = 0
    for s, t in zip(samples, transcripts, strict=True):
        # Ghép (tham chiếu, giả thuyết) cho mọi câu có cả hai phía.
        pairs: list[tuple[str, str]] = []
        for turn, hyp in zip(s.history, t.history, strict=True):
            if turn.role == "user" and turn.audio_text is not None and hyp is not None:
                pairs.append((turn.audio_text, hyp))
        if s.audio_text is not None:
            pairs.append((s.audio_text, t.final))
        # Cộng dồn lỗi và độ dài để ra tỷ lệ trên cả kho, không lấy trung bình theo câu.
        for ref, hyp in pairs:
            r, h = normalize_text(ref), normalize_text(hyp)
            w_err += _edit_distance(r.split(), h.split())
            w_n += len(r.split())
            c_err += _edit_distance(list(r), list(h))
            c_n += len(r)
            n_utt += 1
    missed, total, rate, details = entity_miss_rate(samples, transcripts)
    secs = [t.seconds for t in transcripts]
    return AsrReport(
        n_samples=len(samples),
        n_utterances=n_utt,
        wer=_rate(w_err, w_n),
        cer=_rate(c_err, c_n),
        entity_missed=missed,
        entity_total=total,
        entity_miss_rate=rate,
        p50_s=percentile(secs, 50),
        p95_s=percentile(secs, 95),
        max_s=max(secs, default=0.0),
        entity_details=details,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Điểm vào dòng lệnh.

    CLI: python -m lifemate.eval.eval_asr --config ... --env ... --split train --ids <ids.txt> --out x.json.
    """
    from lifemate.backends.asr import ASRTranscriber, GoldTranscriber
    from lifemate.core.config import load_config, load_profile
    from lifemate.io.data import load_split

    ap = argparse.ArgumentParser(prog="lifemate.eval.eval_asr")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--ids", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    repo_root = REPO_ROOT
    cfg = load_config(args.config)
    profile = load_profile(args.env, repo_root).with_overrides(root=args.root)
    root = Path(profile.data_root)
    ids = None
    if args.ids is not None:
        ids = [x.strip() for x in args.ids.read_text(encoding="utf-8").splitlines() if x.strip()]
    samples = load_split(root, args.split, ids)
    # factory.TRANSCRIBERS chỉ nhận Config (không có data_root/split), nên dựng trực tiếp tại đây.
    transcriber: Transcriber
    if cfg.asr.backend == "gold":
        transcriber = GoldTranscriber()
    else:
        transcriber = ASRTranscriber(cfg.asr, data_root=root, split=args.split)
    report = evaluate(samples, transcriber)
    print(report.to_text())
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
