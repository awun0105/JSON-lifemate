"""Đo chất lượng ASR trên một split: WER, CER, tỉ lệ bỏ sót thực thể và độ trễ (SPEC 6.4.2).

Nội dung chính:
- `evaluate`: chạy transcriber trên mọi mẫu và tính các chỉ số.
- `AsrReport`: các chỉ số đã tổng hợp, xuất được ra dict hoặc văn bản.
- `wer`, `cer`, `normalize_text`, `strip_numbers`: tính tỉ lệ lỗi trên văn bản đã chuẩn hoá.
- `entity_values`, `entity_miss_rate`: trích thực thể từ lời gọi gold và đếm số thực thể ASR bỏ sót.
- `percentile`: phân vị của độ trễ.
- `main`: CLI chạy một transcriber trên một split.

Cách hoạt động:
- WER/CER (tỉ lệ lỗi từ / ký tự) tính trên cả kho: cộng tổng số lỗi chỉnh sửa (Levenshtein) rồi chia cho tổng độ
  dài tham chiếu. Tính trên mọi lượt user trong lịch sử và câu cuối, sau khi chuẩn hoá (chữ thường, NFC, bỏ dấu câu).
- Thực thể: giá trị chuỗi và số trong tham số của lời gọi gold, chỉ giữ những giá trị có xuất hiện trong
  `audio_text` sau khi gập (bỏ dấu tiếng Việt). Thực thể bị coi là bỏ sót khi không xuất hiện, theo ranh giới từ,
  trong văn bản ASR cuối đã gập.
- Độ trễ: p50, p95 theo hạng gần nhất và giá trị lớn nhất của thời gian chép lời từng mẫu.
- `wer_nonum`: WER sau khi bỏ mọi số (chữ số và từ đọc số) ở cả hai phía. Đáp án viết số bằng chữ ("tám giờ"),
  còn whisper-large-v3 viết bằng chữ số ("8 giờ"). WER thường vì vậy phạt sai cách viết chứ không phải sai nghe.
  Độ đúng của số được đo riêng qua thực thể.
- `n_hallucinated`: số câu mà chữ ASR trước khi lọc (`Transcript.raw`) khớp một mẫu phụ đề bịa.
- `terms`: với mỗi tên riêng trong `TERMS`, cặp [số câu đáp án có tên đó, số câu ASR cũng viết ra tên đó].
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

# Từ đọc số tiếng Việt; bị bỏ cùng chữ số khi tính `wer_nonum`. Một số từ như "một", "năm", "ba" còn có nghĩa
# khác, nhưng bị bỏ ở cả hai phía nên không thiên vị model nào.
_NUM_WORDS = frozenset(
    {
        "không",
        "một",
        "mốt",
        "hai",
        "ba",
        "bốn",
        "tư",
        "năm",
        "lăm",
        "nhăm",
        "sáu",
        "bảy",
        "bẩy",
        "tám",
        "chín",
        "mười",
        "mươi",
        "trăm",
        "nghìn",
        "ngàn",
        "triệu",
        "tỷ",
        "tỉ",
        "linh",
        "lẻ",
        "rưỡi",
        "chục",
    }
)
# Tên riêng hay bị nghe sai (tên app, tên kênh); so trên văn bản đã gập (bỏ dấu, chữ thường).
TERMS: dict[str, str] = {
    "zalo": r"\bzalo\b",
    "vtv": r"\bvtv",
    "htv": r"\bhtv",
    "voucher": r"\bvoucher\b",
    "momo": r"\bmomo\b",
    "grab": r"\bgrab\b",
    "shopee": r"\bshopee\b",
    "youtube": r"\byoutube\b",
    "spotify": r"\bspotify\b",
}


def _fold(s: str) -> str:
    """Gập văn bản: chữ thường, bỏ dấu tiếng Việt, gộp khoảng trắng.

    Copied from `_fold` in the competition package zaic_tools (train/zaic_tools/utils.py) so lifemate never imports
    zaic_tools; keep in sync with that file. Nói cách khác: hàm được chép từ `zaic_tools` để lifemate không phải import
    gói đó, nên cần giữ đồng bộ với file gốc.
    """
    # NFD tách dấu thành ký tự tổ hợp để bỏ đi; "đ" không tách được nên phải đổi tay thành "d".
    t = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())


def _folded_tokens(s: str) -> str:
    """Gập văn bản thành các từ cách nhau một dấu cách, có thêm dấu cách ở hai đầu.

    Dấu cách hai đầu giúp phép so khớp chuỗi con tôn trọng ranh giới từ: tìm `" huy "` sẽ không khớp bên trong
    `" chuyen "`.
    """
    return " " + " ".join(re.findall(r"[^\W_]+", _fold(s))) + " "


def normalize_text(s: str) -> str:
    """Chuẩn hoá văn bản để tính WER/CER: chữ thường, NFC, bỏ dấu câu, gộp khoảng trắng."""
    t = unicodedata.normalize("NFC", s.lower())
    return " ".join(_PUNCT.sub(" ", t).split())


def strip_numbers(normalized: str) -> list[str]:
    """Tách văn bản đã chuẩn hoá thành từ, bỏ chữ số và từ đọc số (dùng cho `wer_nonum`)."""
    return [w for w in normalized.split() if not any(c.isdigit() for c in w) and w not in _NUM_WORDS]


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Tính khoảng cách Levenshtein giữa hai dãy (từ hoặc ký tự) bằng quy hoạch động, chỉ giữ một hàng trong bộ nhớ."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            # Xoá, chèn, hoặc thay thế (chi phí 0 khi hai phần tử bằng nhau).
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _rate(errors: int, n: int) -> float:
    """Tính tỉ lệ `errors / n`; khi `n = 0` thì trả 0.0 nếu không có lỗi, ngược lại 1.0."""
    if n == 0:
        return 0.0 if errors == 0 else 1.0
    return errors / n


def wer(ref: str, hyp: str) -> float:
    """Tính tỉ lệ lỗi từ (WER) của `hyp` so với `ref`, trên văn bản đã chuẩn hoá."""
    r, h = normalize_text(ref).split(), normalize_text(hyp).split()
    return _rate(_edit_distance(r, h), len(r))


def cer(ref: str, hyp: str) -> float:
    """Tính tỉ lệ lỗi ký tự (CER) của `hyp` so với `ref`, trên văn bản đã chuẩn hoá; khoảng trắng cũng tính là ký tự."""
    r, h = normalize_text(ref), normalize_text(hyp)
    return _rate(_edit_distance(list(r), list(h)), len(r))


def _values(obj: Any) -> Iterator[str]:
    """Duyệt đệ quy một cấu trúc JSON, sinh mọi giá trị chuỗi khác rỗng và giá trị số (dạng chuỗi).

    Bool và `None` bị bỏ qua; số thực có giá trị nguyên như 3.0 được ghi thành "3".
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
    """Lấy các giá trị tham số (chuỗi, số) của lời gọi gold có xuất hiện trong văn bản gold của lượt cuối.

    Bỏ qua `sys.ask_missing_info`. So khớp trên văn bản đã gập.

    Tham số:
        sample: mẫu có nhãn.

    Trả về:
        Các thực thể đã gập, không trùng, theo thứ tự xuất hiện; danh sách rỗng khi mẫu không có nhãn hoặc
        `audio_text`.
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
            # Chỉ giữ thực thể thật sự được nói ra; thực thể không được nói thì ASR không thể nghe thấy.
            if fv.strip() and fv in gold and fv.strip() not in out:
                out.append(fv.strip())
    return out


@dataclass(frozen=True, slots=True)
class EntityDetail:
    """Kết quả kiểm tra thực thể của một mẫu.

    Thuộc tính:
        id: id mẫu.
        entities: các thực thể cần nghe ra.
        missed: các thực thể bị bỏ sót.
    """

    id: str
    entities: tuple[str, ...]
    missed: tuple[str, ...]


def entity_miss_rate(
    samples: Sequence[Sample], transcripts: Sequence[Transcript]
) -> tuple[int, int, float, list[EntityDetail]]:
    """Đếm thực thể gold có trong `audio_text` nhưng vắng trong văn bản ASR cuối.

    Văn bản ASR được gập trước khi so, và thực thể phải khớp nguyên từ.

    Tham số:
        samples: các mẫu có nhãn.
        transcripts: transcript tương ứng của từng mẫu.

    Trả về:
        Bộ bốn (số bỏ sót, tổng số thực thể, tỉ lệ bỏ sót, chi tiết từng mẫu).
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
    """Tính phân vị theo hạng gần nhất, với `q` trong khoảng 0..100; trả về 0.0 khi không có giá trị."""
    if not values:
        return 0.0
    xs = sorted(values)
    k = max(1, math.ceil(q / 100 * len(xs)))
    return xs[k - 1]


@dataclass(slots=True)
class AsrReport:
    """Các chỉ số ASR đã tổng hợp trên một tập mẫu.

    Thuộc tính:
        n_samples: số mẫu.
        n_utterances: số câu đã chấm (câu cuối và các lượt user có văn bản gold).
        n_empty: số câu có văn bản gold nhưng ASR trả về chuỗi rỗng.
        wer: tỉ lệ lỗi từ trên cả kho.
        cer: tỉ lệ lỗi ký tự trên cả kho.
        entity_missed: số thực thể bị bỏ sót.
        entity_total: tổng số thực thể cần nghe.
        entity_miss_rate: tỉ lệ thực thể bị bỏ sót.
        p50_s: trung vị thời gian chép lời một mẫu, tính bằng giây.
        p95_s: phân vị 95 của thời gian chép lời, tính bằng giây.
        max_s: thời gian chép lời lâu nhất, tính bằng giây.
        entity_details: chi tiết thực thể theo từng mẫu.
        wer_nonum: WER sau khi bỏ mọi số.
        n_hallucinated: số câu mà chữ trước khi lọc khớp mẫu phụ đề bịa.
        terms: tên riêng tới [số câu đáp án có tên đó, số câu ASR viết đúng].
    """

    n_samples: int
    n_utterances: int
    n_empty: int  # câu có văn bản gold nhưng ASR trả chuỗi rỗng (VAD cắt, bộ lọc xoá, hoặc model không ra chữ)
    wer: float
    cer: float
    entity_missed: int
    entity_total: int
    entity_miss_rate: float
    p50_s: float
    p95_s: float
    max_s: float
    entity_details: list[EntityDetail] = field(default_factory=list)
    wer_nonum: float = 0.0
    n_hallucinated: int = 0  # câu mà chữ trước khi lọc khớp mẫu phụ đề bịa (chỉ đếm khi transcript có `raw`)
    terms: dict[str, list[int]] = field(default_factory=dict)  # tên -> [số câu đáp án có tên, số câu ASR viết đúng]

    def to_dict(self) -> dict[str, Any]:
        """Chuyển thành dict để ghi JSON; phần chi tiết thực thể chỉ gồm các mẫu có thực thể bị bỏ sót."""
        d: dict[str, Any] = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "entity_details"}
        d["entity_misses"] = [
            {"id": e.id, "entities": list(e.entities), "missed": list(e.missed)}
            for e in self.entity_details
            if e.missed
        ]
        return d

    def to_text(self) -> str:
        """Trả về bản tóm tắt ngắn, dễ đọc."""
        return (
            f"samples={self.n_samples} utterances={self.n_utterances} empty={self.n_empty}\n"
            f"WER={self.wer:.4f} WER_nonum={self.wer_nonum:.4f} CER={self.cer:.4f} hallucinated={self.n_hallucinated}\n"
            f"entity_miss_rate={self.entity_miss_rate:.4f} ({self.entity_missed}/{self.entity_total})\n"
            f"seconds p50={self.p50_s:.3f} p95={self.p95_s:.3f} max={self.max_s:.3f}"
        )


def evaluate(
    samples: Sequence[Sample], transcriber: Transcriber, hyps: list[dict[str, Any]] | None = None
) -> AsrReport:
    """Chép lời mọi mẫu rồi tính WER/CER trên cả kho, gồm câu cuối và các lượt user trong lịch sử.

    Chỉ những câu có văn bản gold mới được tính. Transcriber được warmup trước, để độ trễ không gồm thời gian nạp
    model.

    Tham số:
        samples: các mẫu cần đo.
        transcriber: transcriber cần đo.
        hyps: nếu truyền vào, mỗi câu được thêm vào list này dạng `{"id", "turn", "ref", "hyp"}` (`turn` là "final"
            hoặc chỉ số lượt trong lịch sử), để soi lỗi từng câu.

    Trả về:
        `AsrReport` của tập mẫu.
    """
    from lifemate.backends.asr_sanitizer import strip_hallucinations

    transcriber.warmup()
    transcripts = [transcriber.transcribe(s) for s in samples]
    w_err = w_n = c_err = c_n = n_utt = n_empty = nn_err = nn_n = n_hall = 0
    terms = {name: [0, 0] for name in TERMS}
    for s, t in zip(samples, transcripts, strict=True):
        # Ghép (tham chiếu, giả thuyết, chữ trước khi lọc) cho mọi câu có cả hai phía. `raw` theo thứ tự file: các lượt
        # user trong lịch sử trước, lượt cuối sau. Vì vậy `k` chỉ tăng ở lượt user.
        raw = list(t.raw)
        pairs: list[tuple[str, str, str, str | None]] = []
        k = 0
        for i, (turn, hyp) in enumerate(zip(s.history, t.history, strict=True)):
            if turn.role != "user":
                continue
            raw_i = raw[k] if k < len(raw) else None
            k += 1
            if turn.audio_text is not None and hyp is not None:
                pairs.append((str(i), turn.audio_text, hyp, raw_i))
        if s.audio_text is not None:
            pairs.append(("final", s.audio_text, t.final, raw[-1] if raw else None))
        # Cộng dồn lỗi và độ dài để ra tỉ lệ trên cả kho, không lấy trung bình theo câu.
        for turn_key, ref, hyp, raw_text in pairs:
            if hyps is not None:
                row: dict[str, Any] = {"id": s.id, "turn": turn_key, "ref": ref, "hyp": hyp}
                if raw_text is not None:
                    row["raw"] = raw_text
                hyps.append(row)
            r, h = normalize_text(ref), normalize_text(hyp)
            n_empty += int(bool(r) and not h)
            w_err += _edit_distance(r.split(), h.split())
            w_n += len(r.split())
            rn, hn = strip_numbers(r), strip_numbers(h)
            nn_err += _edit_distance(rn, hn)
            nn_n += len(rn)
            c_err += _edit_distance(list(r), list(h))
            c_n += len(r)
            n_utt += 1
            # Câu bịa: bộ lọc tìm thấy cụm phụ đề trong chữ trước khi lọc.
            if raw_text is not None and strip_hallucinations(raw_text) != raw_text.strip():
                n_hall += 1
            fr, fh = _fold(ref), _fold(hyp)
            for name, pat in TERMS.items():
                if re.search(pat, fr):
                    terms[name][0] += 1
                    terms[name][1] += int(bool(re.search(pat, fh)))
    missed, total, rate, details = entity_miss_rate(samples, transcripts)
    secs = [t.seconds for t in transcripts]
    return AsrReport(
        n_samples=len(samples),
        n_utterances=n_utt,
        n_empty=n_empty,
        wer=_rate(w_err, w_n),
        cer=_rate(c_err, c_n),
        entity_missed=missed,
        entity_total=total,
        entity_miss_rate=rate,
        p50_s=percentile(secs, 50),
        p95_s=percentile(secs, 95),
        max_s=max(secs, default=0.0),
        entity_details=details,
        wer_nonum=_rate(nn_err, nn_n),
        n_hallucinated=n_hall,
        terms={k: v for k, v in terms.items() if v[0]},
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy CLI đo ASR và in bản tóm tắt; trả về mã thoát 0.

    Cách dùng: `python -m lifemate.eval.eval_asr --config ... --env ... --split train --ids <ids.txt> --out x.json`.
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
    ap.add_argument("--hyps", type=Path, default=None, help="jsonl: mỗi câu một dòng {id, turn, ref, hyp, raw}")
    args = ap.parse_args(argv)
    repo_root = REPO_ROOT
    cfg = load_config(args.config)
    profile = load_profile(args.env, repo_root).with_overrides(root=args.root)
    root = Path(profile.data_root)
    ids = None
    if args.ids is not None:
        ids = [x.strip() for x in args.ids.read_text(encoding="utf-8").splitlines() if x.strip()]
    samples = load_split(root, args.split, ids)
    # Dựng transcriber giống hệt hai mục `gold` / `asr` trong `factory.TRANSCRIBERS` (registry nhận `Config, Path, str`).
    transcriber: Transcriber
    if cfg.asr.backend == "gold":
        transcriber = GoldTranscriber()
    else:
        transcriber = ASRTranscriber(cfg.asr, data_root=root, split=args.split)
    hyps: list[dict[str, Any]] | None = [] if args.hyps is not None else None
    report = evaluate(samples, transcriber, hyps)
    if args.hyps is not None and hyps is not None:
        args.hyps.parent.mkdir(parents=True, exist_ok=True)
        args.hyps.write_text("".join(json.dumps(h, ensure_ascii=False) + "\n" for h in hyps), encoding="utf-8")
    print(report.to_text())
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
