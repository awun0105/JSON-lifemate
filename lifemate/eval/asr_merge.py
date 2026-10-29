"""Ghép chữ của hai ASR: giữ chữ của model chính, chỉ lấy từ model phụ các đoạn có từ tiếng Anh mới.

Cách dùng (chạy từ thư mục gốc repo):
  python -m lifemate.eval.asr_merge --primary <rnnt>.hyps.jsonl --secondary <whisper>.hyps.jsonl --out merged.hyps.jsonl
  python -m lifemate.eval.asr_merge ... --policy context --env local --prompt-file configs/asr_prompts/en_full_v1.txt

Nội dung chính:
- `align_words`: gióng hàng hai dãy từ theo khoảng cách Levenshtein (số từ phải thêm, xoá, thay), có truy vết.
- `Region`, `split_regions`, `candidate_regions`, `assemble`, `rule_decision`: tách câu thành từ khớp và vùng khác
  nhau, chọn vùng cần quyết định, ghép lại theo một hàm quyết định bất kỳ (luật, điểm âm thanh, bộ phân loại).
- `merge_transcripts`: ghép hai câu theo quy tắc ở dưới.
- `merge_hyp_rows`: ghép hai danh sách dòng `hyps.jsonl` (mỗi dòng là chữ ASR của một câu) theo từng câu.
- `CONTEXT_POLICIES`, `sample_context_words`, `english_filter`: bộ lọc ngữ cảnh của mẫu (bước 3b).
- `main`: CLI ghép hai file `hyps.jsonl`; file ra chấm được bằng `lifemate.eval.rescore_asr`.

Cách hoạt động (`merge_transcripts`):
1. Tách hai câu thành từ (giữ nguyên cách viết để ghép lại), gióng hàng trên dạng đã chuẩn hoá (chữ thường, bỏ
   dấu, bỏ dấu câu).
2. Gom các cặp không khớp nằm giữa hai từ khớp thành một "vùng khác nhau".
3. Ở mỗi vùng: nếu phía model phụ có từ tiếng Anh (`foreign_words.find_foreign_words`) mà phía model chính không
   có, và vùng phía phụ không dài quá `2 x (số từ phía chính) + 2` (chặn câu bịa), thì lấy chữ của model phụ cho cả
   vùng; còn lại giữ model chính.
   3b. Khi có bộ lọc (`english_filter`), chỉ tính các từ tiếng Anh mới mà bộ lọc nhận. Bộ lọc dựa vào **từ điển ngữ
   cảnh** của mẫu (`sample_context_words`): các từ có sẵn dạng chữ trong đầu vào lúc thi, gồm catalog của các app đã
   cài, ngữ cảnh người dùng, và lượt trợ lý trong lịch sử (câu trả lời, lời gọi tool, kết quả tool). Từ bị chèn nhầm
   ("Mail" cho "mai") thường không có trong ngữ cảnh; từ được nói thật ("Zalo" khi máy có cài Zalo) thường có.
4. Câu của model phụ bị bỏ hẳn khi chữ trước lọc khác chữ sau lọc (bộ lọc ảo giác đã cắt) hoặc rỗng.

Lưu ý:
- Model chính là model có WER thấp và không bịa câu (ChunkFormer RNNT); model phụ là model chép từ tiếng Anh tốt hơn
  (whisper-large-v3 hoặc PhoWhisper). Quy tắc chỉ sửa từ tiếng Anh, không sửa phần tiếng Việt.
- Tất định: không có ngẫu nhiên, hoà khi truy vết ưu tiên thay thế rồi xoá rồi chèn.
- Từ điển ngữ cảnh chỉ dùng đầu vào dạng chữ của mẫu, không đọc dữ liệu app của môi trường (thể lệ cấm), và không
  dùng lượt người dùng trong lịch sử (lượt đó chỉ có âm thanh lúc thi).
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lifemate.core.types import Sample
from lifemate.eval.foreign_words import find_foreign_words, fold_text, folded_words

# Dấu câu dính ở hai đầu một từ ("ví," hay "(Zalo)").
_EDGE_PUNCTUATION = re.compile(r"^[^\w]+|[^\w]+$")


def _match_key(word: str) -> str:
    """Dạng so khớp của một từ: bỏ dấu câu hai đầu rồi chuẩn hoá chữ (chữ thường, bỏ dấu tiếng Việt)."""
    return fold_text(_EDGE_PUNCTUATION.sub("", word))


def align_words(a: Sequence[str], b: Sequence[str]) -> list[tuple[int | None, int | None]]:
    """Gióng hàng hai dãy khoá từ; trả về các cặp (chỉ số trong `a` hoặc None, chỉ số trong `b` hoặc None).

    Cặp có cả hai chỉ số là hai từ đứng cùng chỗ (khớp hoặc bị thay); `None` ở một bên là từ chỉ có ở bên kia.
    """
    n, m = len(a), len(b)
    # cost[i][j]: số từ ít nhất phải thêm, xoá hoặc thay để biến `a[:i]` thành `b[:j]`.
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i - 1][j - 1] + (a[i - 1] != b[j - 1]), cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return pairs[::-1]


# Cách lọc từ tiếng Anh mới ở bước 3b (tên -> ý nghĩa):
# "all": không lọc; "context": mọi từ phải có trong từ điển ngữ cảnh; "context_prompt": chỉ từ của câu
# mồi phải có trong ngữ cảnh, từ khác nhận luôn; "context_other": chỉ từ ngoài câu mồi phải có trong ngữ cảnh.
CONTEXT_POLICIES = ("all", "context", "context_prompt", "context_other")


def _strings(value: Any) -> Iterator[str]:
    """Mọi chuỗi giá trị trong một giá trị JSON lồng nhau (giá trị của dict, phần tử của list), theo thứ tự.

    Không lấy khoá của dict: khoá là tên trường do người viết schema đặt (`type`, `properties`, `content`...), không
    phải nội dung người dùng có thể nhắc tới (benchmark ASR lần 3, mục 2.1).
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def sample_context_words(sample: Sample, catalog_apps: Mapping[str, Mapping[str, Any]]) -> frozenset[str]:
    """Từ điển ngữ cảnh của một mẫu: tập từ (đã chuẩn hoá, `folded_words`) có sẵn dạng chữ trong đầu vào lúc thi.

    Tham số:
        sample: mẫu cần lấy ngữ cảnh.
        catalog_apps: id app -> mục của app trong catalog tool (tên, mô tả, các tool và schema tham số).

    Trả về:
        Tập từ lấy từ (chỉ giá trị, không lấy khoá JSON): id và mục catalog của mọi app trong
        `installed_apps`; các trường khác của `context` (giờ, vị trí,
        hồ sơ người dùng); lượt trợ lý trong lịch sử (câu trả lời, lời gọi tool, kết quả tool).
    """
    context = dict(sample.context)
    installed = context.pop("installed_apps", None) or {}
    texts: list[str] = []
    for app in installed:
        texts.append(str(app).replace("_", " "))
        texts.extend(_strings(catalog_apps.get(app, {})))
    texts.extend(_strings(context))
    for turn in sample.history:
        if turn.role == "assistant":
            texts.extend(_strings([turn.text, turn.tool_calls, turn.tool_results]))
    return frozenset(folded_words(" ".join(texts)))


def english_filter(
    policy: str, context_words: frozenset[str], prompt_words: frozenset[str]
) -> Callable[[str], bool] | None:
    """Bộ lọc bước 3b cho một mẫu: nhận một từ tiếng Anh (đã chuẩn hoá) hay không; `None` khi `policy` là "all".

    Tham số:
        policy: một tên trong `CONTEXT_POLICIES`.
        context_words: từ điển ngữ cảnh của mẫu (`sample_context_words`).
        prompt_words: các từ ngoại (đã chuẩn hoá) của câu mồi mà model phụ đã dùng.
    """
    if policy == "all":
        return None
    if policy == "context":
        return lambda word: word in context_words
    if policy == "context_prompt":
        return lambda word: word not in prompt_words or word in context_words
    if policy == "context_other":
        return lambda word: word in prompt_words or word in context_words
    raise ValueError(f"unknown policy {policy!r}; expected one of {CONTEXT_POLICIES}")


@dataclass(frozen=True)
class Region:
    """Một vùng khác nhau giữa chữ của model chính và chữ của model phụ, nằm giữa hai từ khớp.

    Thuộc tính:
        primary: các từ của model chính trong vùng (giữ cách viết gốc).
        secondary: các từ của model phụ trong vùng.
        new_english: từ tiếng Anh (đã chuẩn hoá) có ở phía phụ mà phía chính không có, chưa qua bộ lọc nào.
    """

    primary: tuple[str, ...]
    secondary: tuple[str, ...]
    new_english: frozenset[str]

    @property
    def not_too_long(self) -> bool:
        """Điều kiện độ dài của lần 2: phía phụ không dài quá `2 x (số từ phía chính) + 2`."""
        return len(self.secondary) <= 2 * len(self.primary) + 2


def split_regions(primary: str, secondary: str) -> list[str | Region]:
    """Đặt hai câu cạnh nhau từ ứng với từ; trả về dãy gồm từ khớp (chuỗi, cách viết của model chính) và `Region`.

    Câu của model phụ rỗng thì cả câu chính là các từ khớp (không có vùng nào).
    """
    primary_words, secondary_words = primary.split(), secondary.split()
    if not secondary_words:
        return list(primary_words)
    primary_keys = [_match_key(w) for w in primary_words]
    secondary_keys = [_match_key(w) for w in secondary_words]
    out: list[str | Region] = []
    # Vùng khác nhau đang gom: các từ của mỗi bên nằm giữa hai từ khớp.
    primary_region: list[str] = []
    secondary_region: list[str] = []

    def close_region() -> None:
        """Ghi vùng đang gom (nếu có) vào `out` rồi xoá vùng."""
        if not primary_region and not secondary_region:
            return
        # Phía phụ mang tới từ tiếng Anh mà phía chính không có. Không đòi phía chính "không có từ ngoại", vì chữ rác
        # của model chính ("s tp") cũng không ghép được âm tiết Việt nên bị coi là từ ngoại.
        new_english = {w.folded for w in find_foreign_words(" ".join(secondary_region))} - {
            w.folded for w in find_foreign_words(" ".join(primary_region))
        }
        out.append(Region(tuple(primary_region), tuple(secondary_region), frozenset(new_english)))
        primary_region.clear()
        secondary_region.clear()

    for i, j in align_words(primary_keys, secondary_keys):
        if i is not None and j is not None and primary_keys[i] == secondary_keys[j]:
            close_region()
            out.append(primary_words[i])
            continue
        if i is not None:
            primary_region.append(primary_words[i])
        if j is not None:
            secondary_region.append(secondary_words[j])
    close_region()
    return out


def candidate_regions(parts: Sequence[str | Region]) -> list[int]:
    """Chỉ số (trong `parts`) của các vùng cần quyết định: phía phụ có từ tiếng Anh mới, chưa qua bộ lọc."""
    return [k for k, part in enumerate(parts) if isinstance(part, Region) and part.secondary and part.new_english]


def assemble(parts: Sequence[str | Region], take_secondary: Callable[[int, Region], bool]) -> str:
    """Ghép lại câu từ `parts`; mỗi vùng lấy chữ phía phụ khi `take_secondary(chỉ số, vùng)` trả `True`."""
    words: list[str] = []
    for k, part in enumerate(parts):
        if isinstance(part, Region):
            words.extend(part.secondary if take_secondary(k, part) else part.primary)
        else:
            words.append(part)
    return " ".join(words)


def rule_decision(region: Region, accept_english: Callable[[str], bool] | None = None) -> bool:
    """Quyết định của luật lần 2 cho một vùng: còn từ tiếng Anh mới sau bộ lọc, và vùng phía phụ không quá dài."""
    words = region.new_english if accept_english is None else {w for w in region.new_english if accept_english(w)}
    return bool(region.secondary) and bool(words) and region.not_too_long


def merge_transcripts(primary: str, secondary: str, accept_english: Callable[[str], bool] | None = None) -> str:
    """Ghép câu của model chính với câu của model phụ theo quy tắc ở docstring module.

    Tham số:
        primary: chữ của model chính.
        secondary: chữ của model phụ cho cùng câu.
        accept_english: bộ lọc bước 3b (`english_filter`); `None` là nhận mọi từ tiếng Anh mới.
    """
    if not secondary.split():
        return primary
    return assemble(split_regions(primary, secondary), lambda _k, region: rule_decision(region, accept_english))


def merge_hyp_rows(
    primary: Sequence[dict[str, Any]],
    secondary: Sequence[dict[str, Any]],
    filters: Mapping[str, Callable[[str], bool] | None] | None = None,
) -> list[dict[str, Any]]:
    """Ghép hai danh sách dòng `hyps.jsonl` theo (id, turn); câu không có ở model phụ thì giữ model chính.

    Mỗi dòng là `{id, turn, ref, hyp, raw}`: id mẫu, lượt ("final" hoặc chỉ số lượt trong lịch sử), đáp án, chữ ASR
    sau và trước bộ lọc ảo giác. Câu của model phụ bị bỏ khi rỗng hoặc khi bộ lọc ảo giác đã cắt (`raw` khác `hyp` sau
    khi gộp khoảng trắng). `filters` là id mẫu -> bộ lọc bước 3b (`english_filter`); thiếu id thì không lọc.
    """
    secondary_by_turn = {(str(r["id"]), str(r["turn"])): r for r in secondary}
    merged_rows = []
    for row in primary:
        other = secondary_by_turn.get((str(row["id"]), str(row["turn"])))
        hyp = str(row["hyp"])
        if other is not None:
            other_hyp = str(other["hyp"])
            filtered = " ".join(str(other.get("raw", other_hyp)).split()) != " ".join(other_hyp.split())
            if other_hyp.strip() and not filtered:
                accept = (filters or {}).get(str(row["id"]))
                hyp = merge_transcripts(hyp, other_hyp, accept)
        merged_rows.append({**row, "hyp": hyp, "raw": hyp})
    return merged_rows


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy CLI ghép hai file `hyps.jsonl`; trả về mã thoát 0."""
    ap = argparse.ArgumentParser(prog="lifemate.eval.asr_merge", description=__doc__)
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--secondary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policy", choices=CONTEXT_POLICIES, default="all", help="bộ lọc bước 3b")
    ap.add_argument("--prompt-file", type=Path, default=None, help="câu mồi của model phụ (policy khác 'all')")
    ap.add_argument("--env", default="local", help="profile máy, để tìm dữ liệu (policy khác 'all')")
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="train")
    args = ap.parse_args(argv)

    def load(path: Path) -> list[dict[str, Any]]:
        """Đọc một file `hyps.jsonl`."""
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

    primary_rows = load(args.primary)
    filters = None
    if args.policy != "all":
        filters = _load_filters(args, {str(r["id"]) for r in primary_rows})
    rows = merge_hyp_rows(primary_rows, load(args.secondary), filters)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"merged {len(rows)} utterances -> {args.out}")
    return 0


def _load_filters(args: argparse.Namespace, ids: set[str]) -> dict[str, Callable[[str], bool] | None]:
    """Dựng bộ lọc bước 3b cho từng mẫu trong `ids` từ dữ liệu của split và câu mồi của model phụ."""
    from lifemate.core.config import load_profile
    from lifemate.core.paths import REPO_ROOT, repo_path
    from lifemate.io.data import load_split
    from lifemate.io.zaic import load_catalog

    if args.prompt_file is None:
        raise SystemExit("--prompt-file is required when --policy is not 'all'")
    profile = load_profile(args.env, REPO_ROOT).with_overrides(root=args.root)
    root = repo_path(profile.data_root, REPO_ROOT)
    catalog_apps = {str(app["id"]): app for app in load_catalog(root, args.split)["apps"]}
    prompt_text = repo_path(str(args.prompt_file), REPO_ROOT).read_text(encoding="utf-8")
    prompt_words = frozenset(w.folded for w in find_foreign_words(prompt_text))
    return {
        sample.id: english_filter(args.policy, sample_context_words(sample, catalog_apps), prompt_words)
        for sample in load_split(root, args.split, sorted(ids))
    }


if __name__ == "__main__":
    raise SystemExit(main())
