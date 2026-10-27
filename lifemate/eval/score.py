"""Bộ chấm cục bộ cho LifeMate (Zalo AI Challenge 2026), theo đúng luật chấm chính thức.

Nội dung: hàm so khớp giá trị và lời gọi, ghép cặp cực đại, phát lại lời gọi trên simulator, chấm từng mẫu, in báo
cáo phân tích lỗi và CLI.

Cách xây dựng:
  - Phát lại (replay): lời gọi gold và lời gọi nộp lần lượt được chạy qua simulator `ToolEnv` của chính split, từ
    trạng thái ban đầu của mẫu; mỗi lần cho ra state hash cuối và tập lời gọi bị từ chối (lỗi schema).
  - Hash trạng thái: hash của bài nộp khác hash của gold thì mẫu được 0 điểm.
  - F1 bằng ghép cặp hai phía (bipartite matching): cạnh nối lời gọi nộp với lời gọi gold khớp nhau; số cặp cực đại
    M tìm bằng đường tăng (augmenting path, thuật toán Kuhn). Từ đó P = M / (M + U), R = M / |G|.
  - Báo cáo: ma trận nhầm lẫn quyết định, loại lỗi và điểm theo app.
  - Không sửa công thức hay chuỗi nào ở đây: tệp này mô phỏng công thức chấm chính thức.

Chi tiết luật chấm (tiếng Anh, giữ nguyên):

Each submitted sample is replayed on its initial state with the split's own zaic_tools, then scored:
  - gold list empty      -> 1 if the submitted list is also empty, else 0
  - otherwise            -> 0 if the state hash differs from the gold one or no call matches (M = 0),
                            else F1 = 2PR / (P + R), with P = M / (M + U) and R = M / |G|
  Final score = 100 / N * sum(score_i)

A submitted call matches a gold call when the tool name is the same, every non-fuzzy argument is identical and every
fuzzy argument (x-match: fuzzy in the catalog) has token-F1 > 0.6 after lowercasing and Unicode NFC. Each call matches
at most one call and order is ignored. Rejected calls (schema errors) never match, so they count in U.

Besides the score, the report shows where points are lost: a confusion matrix of the decision (execute / ask / out of
scope), the error types of the samples that should act, and the score per app.

Usage (from the repo root):
  python -m lifemate.eval.score submission.jsonl                          # score against all train labels
  python -m lifemate.eval.score submission.jsonl --ids configs/splits/val_seen.txt  # one split (missing ids = 0)
  python -m lifemate.eval.score submission.jsonl --partial                # score only the ids present in the submission
  python -m lifemate.eval.score submission.jsonl --details out.jsonl      # also write the per-sample breakdown
  python -m lifemate.eval.score --make-gold gold.jsonl                    # gold answers as a submission (sanity)
  python -m lifemate.eval.score submission.jsonl --match both             # print the strict and the canon score

Matching modes (--match): `strict` (default) compares non-fuzzy arguments with type-aware equality (bool only equals
bool, int equals float when numerically equal); `canon` additionally normalises strings like zaic_tools'
_canon_scalar (lower case, collapsed whitespace, '700 cm' == '700cm'). `both` prints both scores and details use strict.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

from lifemate.core.paths import REPO_ROOT as REPO


def default_data_root() -> Path:
    """Trả về thư mục dữ liệu của profile môi trường `local` (configs/env/local.toml).

    Đường dẫn tương đối được tính từ gốc repo; `~` được mở rộng.
    """
    import tomllib

    with open(REPO / "configs" / "env" / "local.toml", "rb") as f:
        raw = tomllib.load(f)
    p = Path(raw["data_root"]).expanduser()
    return p if p.is_absolute() else REPO / p


DEFAULT_ROOT = default_data_root()
FUZZY_THRESHOLD = 0.6
CATEGORIES = ["execute", "ask_missing_info", "out_of_scope"]
SHORT = {"execute": "execute", "ask_missing_info": "ask", "out_of_scope": "[]"}
ERRORS = [
    ("hash_fail", "wrong final state (state_hash)"),
    ("missing_tool", "missing a needed tool"),
    ("wrong_args", "right tool, wrong arguments"),
    ("extra_tool", "extra / wrong tool calls"),
    ("rejected", "calls rejected (schema error)"),
]


def load_jsonl(path: Path) -> list[dict]:
    """Đọc tệp jsonl thành danh sách dict, bỏ qua dòng trống."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


_NUM_UNIT = re.compile(r"(\d)\s+(?=[^\W\d_])")  # '700 cm' -> '700cm', '2 kg' -> '2kg' (bỏ khoảng trắng số-đơn vị)


def _local_canon_scalar(v):
    """Bản sao dự phòng của zaic_tools.utils._canon_scalar, chỉ dùng khi zaic_tools của split thiếu thuộc tính đó.

    Chuỗi được đưa về chữ thường, gộp khoảng trắng và bỏ khoảng trắng giữa số và đơn vị; kiểu khác giữ nguyên.
    """
    if isinstance(v, str):
        return _NUM_UNIT.sub(r"\1", " ".join(v.lower().split()))
    return v


_canon_scalar = _local_canon_scalar


def load_canon_scalar() -> None:
    """Dùng _canon_scalar của chính split (zaic_tools phải import được); nếu không thì giữ bản sao cục bộ."""
    global _canon_scalar  # noqa: PLW0603
    try:
        from zaic_tools import utils as zutils
    except ImportError:
        return
    _canon_scalar = getattr(zutils, "_canon_scalar", _local_canon_scalar)


def values_equal(a, b, canon: bool = False) -> bool:
    """So bằng có xét kiểu dữ liệu.

    bool chỉ bằng bool; int/float bằng nhau khi giá trị số bằng nhau; str bằng str (chuẩn hóa bằng _canon_scalar
    khi `canon`); list và dict so đệ quy; mọi tổ hợp kiểu khác đều không bằng.
    """
    # bool là lớp con của int nên phải xét trước nhánh số.
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return a == b
    if isinstance(a, str) and isinstance(b, str):
        return _canon_scalar(a) == _canon_scalar(b) if canon else a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(values_equal(x, y, canon) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(values_equal(a[k], b[k], canon) for k in a)
    if a is None or b is None:
        return a is None and b is None
    return False


def fuzzy_params(catalog: dict) -> dict[str, set[str]]:
    """Ánh xạ {tên tool: tập tên tham số có `x-match: fuzzy`} lấy từ catalog."""
    out = {}
    for app in catalog["apps"]:
        for tool in app["tools"]:
            props = tool.get("parameters", {}).get("properties", {})
            out[tool["name"]] = {k for k, v in props.items() if v.get("x-match") == "fuzzy"}
    return out


def tokens(text) -> list[str]:
    """Tách token chữ-số sau khi đưa về chữ thường và chuẩn hóa Unicode NFC."""
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", str(text).lower()))


def token_f1(a, b) -> float:
    """F1 trên túi token (multiset) của hai văn bản; hai văn bản đều rỗng thì bằng 1.0."""
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return 1.0
    common = sum((collections.Counter(ta) & collections.Counter(tb)).values())
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def call_matches(sub: dict, gold: dict, fuzzy: dict[str, set[str]], canon: bool = False) -> bool:
    """Kiểm tra một lời gọi nộp có khớp một lời gọi gold hay không.

    Cần cùng tên tool và cùng tập khóa tham số; tham số fuzzy cần token-F1 lớn hơn ngưỡng, tham số khác cần bằng
    nhau theo `values_equal`.
    """
    if sub.get("name") != gold["name"]:
        return False
    sa, ga = sub.get("arguments") or {}, gold.get("arguments") or {}
    if set(sa) != set(ga):
        return False
    fz = fuzzy.get(gold["name"], set())
    for k, gv in ga.items():
        if k in fz:
            if token_f1(sa[k], gv) <= FUZZY_THRESHOLD:
                return False
        elif not values_equal(sa[k], gv, canon):
            return False
    return True


def max_matching(sub: list[dict], gold: list[dict], fuzzy, rejected: set[int], canon: bool = False) -> int:
    """Kích thước ghép cặp một-một cực đại giữa lời gọi nộp và lời gọi gold (bỏ qua thứ tự).

    Đồ thị hai phía có cạnh i -> j khi lời gọi nộp i (không bị từ chối) khớp lời gọi gold j; cặp cực đại tìm bằng
    đường tăng (thuật toán Kuhn).
    """
    edges = [
        [j for j, g in enumerate(gold) if i not in rejected and call_matches(s, g, fuzzy, canon)]
        for i, s in enumerate(sub)
    ]
    # owner[j]: chỉ số lời gọi nộp đang được ghép với gold j, -1 nếu chưa ghép.
    owner = [-1] * len(gold)

    def augment(i, seen):
        """Tìm đường tăng từ lời gọi nộp i bằng DFS; nhường gold j nếu chủ cũ ghép được sang gold khác."""
        for j in edges[i]:
            if j not in seen:
                seen.add(j)
                if owner[j] == -1 or augment(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    return sum(augment(i, set()) for i in range(len(sub)))


def category(calls: list[dict]) -> str:
    """Loại câu trả lời: "out_of_scope" (không gọi gì), "ask_missing_info" (có hỏi lại) hoặc "execute"."""
    names = [c.get("name") for c in calls]
    if not names:
        return "out_of_scope"
    if "sys.ask_missing_info" in names:
        return "ask_missing_info"
    return "execute"


def replay(ToolEnv, split: str, root: Path, sample_id: str, calls: list[dict]):
    """Phát lại các lời gọi trên simulator từ trạng thái ban đầu của mẫu.

    Trả về (state hash cuối, tập chỉ số lời gọi bị từ chối, tức kết quả không có `ok`).
    """
    env = ToolEnv(split, sample_id, root=root)
    rejected = set()
    for i, c in enumerate(calls):
        if not env.call(c.get("name", ""), c.get("arguments") or {}).get("ok"):
            rejected.add(i)
    return env.get_state_hash(), rejected


def score_sample(item: dict, sub: dict | None, ToolEnv, split, root, fuzzy, canon: bool = False) -> dict:
    """Chấm một mẫu và trả về dict chi tiết (điểm, hash, M, U, P, R, các loại lỗi).

    Mẫu thiếu trong bài nộp được 0 điểm. Gold rỗng: 1 điểm khi bài nộp cũng rỗng. Ngược lại điểm là F1 khi hash
    trạng thái khớp và M > 0, nếu không là 0.
    """
    gold = item["label"]["tool_calls"]
    apps = sorted({c["name"].split(".", 1)[0] for c in gold} - {"sys"})
    d = {"id": item["id"], "category": category(gold), "gold_calls": len(gold), "apps": apps}
    if sub is None:
        return {**d, "score": 0.0, "missing": True, "predicted": None}
    calls = sub.get("tool_calls") or []
    d["sub_calls"] = len(calls)
    d["predicted"] = category(calls)
    if not gold:
        return {**d, "score": 1.0 if not calls else 0.0}

    gold_hash, _ = replay(ToolEnv, split, root, item["id"], gold)
    sub_hash, rejected = replay(ToolEnv, split, root, item["id"], calls)
    m = max_matching(calls, gold, fuzzy, rejected, canon)
    u = len(calls) - m
    p = m / (m + u) if calls else 0.0
    r = m / len(gold)
    hash_ok = sub_hash == gold_hash
    f1 = 2 * p * r / (p + r) if m else 0.0

    gold_names = collections.Counter(c["name"] for c in gold)
    sub_names = collections.Counter(c.get("name") for c in calls)
    # Phân loại lỗi để báo cáo (không ảnh hưởng điểm): thiếu tool là gold cần mà bài nộp không gọi đủ số lần;
    # lời gọi gold chưa khớp còn lại sau khi trừ thiếu tool được tính là sai tham số.
    missing_tool = sum((gold_names - sub_names).values())
    errors = {
        "hash_fail": not hash_ok,
        "missing_tool": missing_tool > 0,
        "wrong_args": len(gold) - m - missing_tool > 0,
        "extra_tool": sum((sub_names - gold_names).values()) > 0,
        "rejected": bool(rejected),
    }
    return {
        **d,
        "score": f1 if hash_ok and m else 0.0,
        "hash_ok": hash_ok,
        "M": m,
        "U": u,
        "P": p,
        "R": r,
        "rejected": len(rejected),
        "errors": [k for k, v in errors.items() if v],
        "submitted_hash_ok": sub.get("state_hash") == sub_hash,
    }


def pct(rows: list[dict]) -> float:
    """Điểm trung bình theo thang 100 của các dòng; NaN khi rỗng."""
    return 100 * sum(r["score"] for r in rows) / len(rows) if rows else float("nan")


def report(rows: list[dict], min_app: int, extra: dict[str, list[dict]] | None = None) -> None:
    """In báo cáo: điểm cuối, điểm theo loại câu trả lời, ma trận nhầm lẫn, loại lỗi, điểm theo app.

    `extra` (chế độ --match both) chứa điểm cuối của từng chế độ; các bảng chi tiết dùng `rows` (strict).
    """
    n = len(rows)
    if extra:
        for mode, mrows in extra.items():
            print(f"Final score [{mode}]: {pct(mrows):.2f}  (N = {len(mrows)})")
        print("(details below use strict)")
    else:
        print(f"Final score: {pct(rows):.2f}  (N = {n})")
    missing = sum(r.get("missing", False) for r in rows)
    if missing:
        print(f"  missing ids in submission: {missing} (scored 0)")

    print(f"\n== Score by answer type\n{'category':<18}{'N':>6}{'score':>9}{'hash ok':>9}{'P':>7}{'R':>7}")
    for name in CATEGORIES:
        g = [r for r in rows if r["category"] == name]
        if not g:
            continue
        scored = [r for r in g if "hash_ok" in r]
        mean = lambda k: sum(r[k] for r in scored) / len(scored) if scored else float("nan")  # noqa: B023
        print(f"{name:<18}{len(g):>6}{pct(g):>9.2f}{100 * mean('hash_ok'):>9.1f}{mean('P'):>7.3f}{mean('R'):>7.3f}")

    # a) agent đã quyết định gì so với quyết định đúng
    print("\n== Decision confusion matrix (rows: gold, columns: submitted)")
    cols = CATEGORIES + ([None] if missing else [])
    print("gold \\ submitted".ljust(18) + "".join(f"{SHORT.get(c, 'missing'):>10}" for c in cols))
    for g in CATEGORIES:
        cnt = collections.Counter(r["predicted"] for r in rows if r["category"] == g)
        if cnt:
            print(f"{SHORT[g]:<18}" + "".join(f"{cnt.get(c, 0):>10}" for c in cols))
    agree = sum(r["predicted"] == r["category"] for r in rows)
    print(f"decision accuracy: {100 * agree / n:.1f}%")

    # b) vì sao các mẫu cần hành động bị mất điểm
    acting = [r for r in rows if r["category"] != "out_of_scope" and "errors" in r]
    imperfect = [r for r in acting if r["score"] < 1]
    print(
        f"\n== Error types ({len(imperfect)} of {len(acting)} execute/ask samples below a perfect score;"
        " a sample can have several)"
    )
    for key, label in ERRORS:
        k = sum(key in r["errors"] for r in imperfect)
        print(f"  {label:<34}{k:>6}  ({100 * k / len(acting):.1f}% of samples)" if acting else f"  {label:<34}{k:>6}")

    # c) mất điểm ở đâu: điểm của các mẫu có gold dùng từng app
    per_app = collections.defaultdict(list)
    for r in rows:
        for a in r["apps"]:
            per_app[a].append(r)
    shown = sorted((a for a in per_app if len(per_app[a]) >= min_app), key=lambda a: (pct(per_app[a]), a))
    print(f"\n== Score by app (samples whose gold uses the app, N >= {min_app}; weakest first)")
    for a in shown:
        g = per_app[a]
        print(f"  {a:<20}{len(g):>5}{pct(g):>9.2f}")

    rejected = sum(r.get("rejected", 0) for r in rows)
    bad_hash = sum(1 for r in rows if r.get("submitted_hash_ok") is False)
    print(f"\nrejected calls (schema errors): {rejected}")
    if bad_hash:
        print(
            f"WARNING: {bad_hash} submitted state_hash values differ from the replayed hash; the replayed one is used"
        )


def main() -> None:
    """Điểm vào dòng lệnh: nạp split và bài nộp, chấm từng mẫu, in báo cáo, ghi chi tiết (tùy chọn)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission", nargs="?", type=Path, help="submission.jsonl to score")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="folder holding the split folders")
    ap.add_argument("--split", default="train", help="labelled split to score against (default: train)")
    ap.add_argument(
        "--ids", type=Path, help="file with one sample id per line (e.g. configs/splits/val_seen.txt); only these"
    )
    ap.add_argument("--partial", action="store_true", help="score only the ids present in the submission")
    ap.add_argument("--min-app", type=int, default=5, help="hide apps with fewer samples in the per-app table")
    ap.add_argument("--details", type=Path, help="write per-sample results to this jsonl file")
    ap.add_argument(
        "--match",
        choices=["strict", "canon", "both"],
        default="strict",
        help="non-fuzzy argument matching: strict, canon (zaic_tools string normalisation) or both",
    )
    ap.add_argument("--make-gold", type=Path, metavar="OUT", help="write the gold answers as a submission and exit")
    args = ap.parse_args()

    # zaic_tools nằm trong thư mục split; mỗi split mang simulator riêng.
    sys.path.insert(0, str(args.root / args.split))
    from zaic_tools import ToolEnv

    load_canon_scalar()

    items = load_jsonl(args.root / args.split / f"{args.split}.jsonl")
    if "label" not in items[0]:
        sys.exit(f"split {args.split!r} has no labels; only a labelled split can be scored locally")
    if args.ids:
        with open(args.ids, encoding="utf-8") as f:
            wanted = {line.strip() for line in f if line.strip()}
        unknown = wanted - {it["id"] for it in items}
        if unknown:
            sys.exit(f"{len(unknown)} ids in {args.ids} are not in {args.split}, e.g. {sorted(unknown)[:3]}")
        items = [it for it in items if it["id"] in wanted]

    if args.make_gold:
        with open(args.make_gold, "w", encoding="utf-8") as f:
            for it in items:
                h, _ = replay(ToolEnv, args.split, args.root, it["id"], it["label"]["tool_calls"])
                f.write(
                    json.dumps(
                        {"id": it["id"], "tool_calls": it["label"]["tool_calls"], "state_hash": h}, ensure_ascii=False
                    )
                    + "\n"
                )
        print(f"{len(items)} lines -> {args.make_gold}")
        return
    if not args.submission:
        ap.error("a submission file is required (or use --make-gold)")

    subs = {}
    for s in load_jsonl(args.submission):
        if s["id"] in subs:
            print(f"WARNING: duplicate id {s['id']}; the last line is used")
        subs[s["id"]] = s
    if not args.ids:
        unknown = set(subs) - {it["id"] for it in items}
        if unknown:
            print(f"WARNING: {len(unknown)} ids are not in {args.split} and are ignored, e.g. {sorted(unknown)[:3]}")
    if args.partial:
        items = [it for it in items if it["id"] in subs]
        if not items:
            sys.exit("no submission id belongs to the scored ids")

    with open(args.root / args.split / f"{args.split}_tools_catalog.json", encoding="utf-8") as f:
        catalog = json.load(f)
    fuzzy = fuzzy_params(catalog)

    def run(canon: bool) -> list[dict]:
        """Chấm mọi mẫu với một chế độ so khớp."""
        return [score_sample(it, subs.get(it["id"]), ToolEnv, args.split, args.root, fuzzy, canon) for it in items]

    if args.match == "both":
        rows = run(False)
        report(rows, args.min_app, extra={"strict": rows, "canon": run(True)})
    else:
        rows = run(args.match == "canon")
        report(rows, args.min_app)
    if args.details:
        with open(args.details, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
