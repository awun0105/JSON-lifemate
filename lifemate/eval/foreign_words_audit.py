"""Đo độ chính xác của quy tắc tìm từ ngoại (`foreign_words.find_foreign_words`) bằng nhãn rà tay, và chấm lại tỉ lệ
chép đúng từ tiếng Anh của các hệ thống ASR theo nhãn rà tay (benchmark ASR lần 2, báo cáo mục 2.1).

Cách dùng (chạy từ thư mục gốc repo):
  python -m lifemate.eval.foreign_words_audit --env local --labels data/labels/foreign_words_hand_v1.jsonl \\
      --ids configs/splits/val_seen.txt --system rnnt=<run>/chunkformer-rnnt__report_val_seen.hyps.jsonl

Nội dung chính:
- `load_hand_labels`: đọc file nhãn rà tay.
- `labelled_words`: các từ ngoại của một câu đáp án theo nhãn rà tay.
- `audit_rule`: so quy tắc với nhãn rà tay trên các câu đáp án: số lần đúng, nhận nhầm, bỏ sót, và độ khớp nhóm từ.
- `english_recall`: tỉ lệ chép đúng từ tiếng Anh của một hệ thống, theo quy tắc và theo nhãn rà tay.
- `main`: CLI, in kết quả dạng JSON.

Cách hoạt động:
- File nhãn là JSON Lines, mỗi dòng `{id, turn, word_index, group}`: mã mẫu, lượt ("final" hoặc chỉ số lượt trong
  lịch sử, như `hyps.jsonl`), thứ tự của từ trong câu đáp án theo cách tách từ của `foreign_words` (dãy chữ cái liền
  nhau, tính cả từ một chữ cái, sau khi chuẩn hoá NFC), và nhóm từ (`WORD_GROUPS`). Câu không có dòng nào là câu không
  có từ ngoại.
- Nhãn chỉ ghi vị trí, không chép chữ của dữ liệu, và nằm trong `data/` (git bỏ qua).
- `english_recall` đếm như `eval_asr.evaluate`: mỗi lần xuất hiện của từ ngoại trong đáp án là một lượt, đúng khi chữ
  ASR của cùng câu có từ đó (`match_words_in_transcript`). Khác `evaluate` ở chỗ chấm trên mọi câu của tập, không chỉ
  câu chêm tiếng Anh, để hai cách gán nhãn chấm trên cùng các câu.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from lifemate.core.types import Sample
from lifemate.eval.foreign_words import (
    _LETTER_RUN,
    ForeignWord,
    find_foreign_words,
    fold_text,
    match_words_in_transcript,
)

# (mã mẫu, lượt) -> thứ tự từ trong câu -> nhóm từ.
HandLabels = dict[tuple[str, str], dict[int, str]]


def load_hand_labels(path: Path) -> HandLabels:
    """Đọc file nhãn rà tay (JSON Lines `{id, turn, word_index, group}`)."""
    labels: HandLabels = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            labels.setdefault((str(row["id"]), str(row["turn"])), {})[int(row["word_index"])] = str(row["group"])
    return labels


def reference_turns(sample: Sample) -> Iterator[tuple[str, str]]:
    """Các câu đáp án của một mẫu: (lượt, chữ chuẩn), lượt cuối là "final", lượt cũ là chỉ số trong lịch sử."""
    yield "final", str(sample.audio_text)
    for i, turn in enumerate(sample.history):
        if turn.role == "user":
            yield str(i), str(turn.audio_text)


def labelled_words(reference: str, positions: Mapping[int, str]) -> list[ForeignWord]:
    """Các từ ngoại của câu đáp án `reference` theo nhãn rà tay `positions` (thứ tự từ -> nhóm), theo thứ tự câu."""
    tokens = [m.group() for m in _LETTER_RUN.finditer(unicodedata.normalize("NFC", reference))]
    return [ForeignWord(tokens[k], fold_text(tokens[k]), positions[k]) for k in sorted(positions)]


def _rule_positions(reference: str, lowercase_vocabulary: set[str]) -> dict[int, str]:
    """Vị trí (thứ tự từ -> nhóm) của các từ ngoại mà quy tắc tìm ra trong `reference`."""
    text = unicodedata.normalize("NFC", reference)
    found = iter(find_foreign_words(text, lowercase_vocabulary))
    current = next(found, None)
    positions: dict[int, str] = {}
    for k, match in enumerate(_LETTER_RUN.finditer(text)):
        if current is not None and match.group() == current.text:
            positions[k] = current.group
            current = next(found, None)
    return positions


def audit_rule(samples: Sequence[Sample], labels: HandLabels) -> dict[str, Any]:
    """So quy tắc với nhãn rà tay trên mọi câu đáp án của `samples`.

    Trả về:
        Dict gồm `rule_words`, `hand_words` (số lần xuất hiện theo từng cách), `both`, `false_positives` (quy tắc
        nhận nhưng nhãn không), `misses` (nhãn có nhưng quy tắc bỏ sót), `precision`, `recall`, `group_agreement`
        (tỉ lệ cùng nhóm trong `both`), `false_positive_words`, `missed_words` (từ đã chuẩn hoá -> số lần), và
        `group_mismatches` ("từ: nhóm theo quy tắc -> nhóm theo nhãn" -> số lần).
    """
    references = [(sample.id, turn, text) for sample in samples for turn, text in reference_turns(sample)]
    # Cùng cách xếp nhóm từ viết hoa đầu câu như `eval_asr.evaluate`.
    lowercase_vocabulary = {
        w.folded for _, _, text in references for w in find_foreign_words(text) if w.text.islower()
    }
    counts: Counter[str] = Counter()
    false_positive_words: Counter[str] = Counter()
    missed_words: Counter[str] = Counter()
    group_mismatches: Counter[str] = Counter()
    for sample_id, turn, text in references:
        rule = _rule_positions(text, lowercase_vocabulary)
        hand = labels.get((sample_id, turn), {})
        tokens = [m.group() for m in _LETTER_RUN.finditer(unicodedata.normalize("NFC", text))]
        counts["rule_words"] += len(rule)
        counts["hand_words"] += len(hand)
        for k in rule.keys() & hand.keys():
            counts["both"] += 1
            counts["same_group"] += rule[k] == hand[k]
            if rule[k] != hand[k]:
                group_mismatches[f"{fold_text(tokens[k])}: {rule[k]} -> {hand[k]}"] += 1
        for k in rule.keys() - hand.keys():
            false_positive_words[fold_text(tokens[k])] += 1
        for k in hand.keys() - rule.keys():
            missed_words[fold_text(tokens[k])] += 1
    both = counts["both"]
    return {
        "rule_words": counts["rule_words"],
        "hand_words": counts["hand_words"],
        "both": both,
        "false_positives": counts["rule_words"] - both,
        "misses": counts["hand_words"] - both,
        "precision": both / counts["rule_words"] if counts["rule_words"] else 0.0,
        "recall": both / counts["hand_words"] if counts["hand_words"] else 0.0,
        "group_agreement": counts["same_group"] / both if both else 0.0,
        "false_positive_words": dict(false_positive_words.most_common()),
        "missed_words": dict(missed_words.most_common()),
        "group_mismatches": dict(group_mismatches.most_common()),
    }


def english_recall(
    samples: Sequence[Sample], hyp_rows: Sequence[Mapping[str, Any]], labels: HandLabels
) -> dict[str, dict[str, float]]:
    """Tỉ lệ chép đúng từ tiếng Anh của một hệ thống trên mọi câu của `samples`, theo quy tắc và theo nhãn rà tay.

    Tham số:
        samples: các mẫu của tập.
        hyp_rows: các dòng `hyps.jsonl` của hệ thống (`{id, turn, hyp}`); câu thiếu dòng coi như chữ rỗng.
        labels: nhãn rà tay.

    Trả về:
        `{"rule": {...}, "hand": {...}}`, mỗi phần gồm `words`, `hits`, `recall`, `hits_lenient`, `recall_lenient`.
    """
    hyps = {(str(r["id"]), str(r["turn"])): str(r["hyp"]) for r in hyp_rows}
    references = [(sample.id, turn, text) for sample in samples for turn, text in reference_turns(sample)]
    lowercase_vocabulary = {
        w.folded for _, _, text in references for w in find_foreign_words(text) if w.text.islower()
    }
    totals = {name: Counter[str]() for name in ("rule", "hand")}
    for sample_id, turn, text in references:
        hyp = hyps.get((sample_id, turn), "")
        words_by_labelling = {
            "rule": find_foreign_words(text, lowercase_vocabulary),
            "hand": labelled_words(text, labels.get((sample_id, turn), {})),
        }
        for name, words in words_by_labelling.items():
            totals[name]["words"] += len(words)
            totals[name]["hits"] += sum(match_words_in_transcript(words, hyp))
            totals[name]["hits_lenient"] += sum(match_words_in_transcript(words, hyp, lenient=True))
    return {
        name: {
            "words": t["words"],
            "hits": t["hits"],
            "recall": t["hits"] / t["words"] if t["words"] else 0.0,
            "hits_lenient": t["hits_lenient"],
            "recall_lenient": t["hits_lenient"] / t["words"] if t["words"] else 0.0,
        }
        for name, t in totals.items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy CLI: in độ chính xác của quy tắc và tỉ lệ chép đúng của từng hệ thống (JSON); trả về mã thoát 0."""
    from lifemate.core.config import load_profile
    from lifemate.core.paths import REPO_ROOT, repo_path
    from lifemate.io.data import load_split, read_ids

    ap = argparse.ArgumentParser(prog="lifemate.eval.foreign_words_audit", description=__doc__)
    ap.add_argument("--env", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--system", action="append", default=[], metavar="NAME=HYPS")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    profile = load_profile(args.env, REPO_ROOT).with_overrides(root=args.root)
    root = repo_path(profile.data_root, REPO_ROOT)
    samples = load_split(root, args.split, read_ids(repo_path(str(args.ids), REPO_ROOT)))
    labels = load_hand_labels(repo_path(str(args.labels), REPO_ROOT))
    result: dict[str, Any] = {"rule_accuracy": audit_rule(samples, labels), "systems": {}}
    for spec in args.system:
        name, _, path = spec.partition("=")
        rows = [json.loads(x) for x in repo_path(path, REPO_ROOT).read_text(encoding="utf-8").splitlines() if x]
        result["systems"][name] = english_recall(samples, rows, labels)
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
