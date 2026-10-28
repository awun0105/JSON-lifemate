"""Chia tập train có nhãn thành các danh sách id `train`, `val_seen`, `val_unseen`, kèm k fold phân tầng.

Cách dùng (chạy từ thư mục gốc repo):
  python -m lifemate.eval.make_splits --smoke 30
  python -m lifemate.eval.make_splits --held-out medication fitness --folds 5 --seed 7

Nội dung chính:
- `stratified_group_folds`: chia các cụm mẫu vào k fold, cân bằng kích thước và nhãn.
- `smoke_sample`: lấy một tập nhỏ phân tầng (cho `smoke.txt` và `asr_tune.txt`).
- `call_bucket`, `category`, `gold_apps`: phân loại mẫu theo số lời gọi, loại câu trả lời và app được dùng.
- `UNSEEN_OUT_OF_SCOPE`: các mẫu gold rỗng được gắn tay vào app held-out.
- `main`: CLI đọc dữ liệu, chia tập và ghi file.

Các tập được tạo:
- `val_unseen`: mọi mẫu có lời gọi gold dùng một trong các app held-out (giả lập các app mới của tập private
  test), cộng các mẫu gold rỗng gắn tay liên quan tới những app đó (`UNSEEN_OUT_OF_SCOPE`).
- Các fold: phần còn lại, gom thành cụm các yêu cầu gần trùng (độ giống `audio_text` từ `--dup-threshold` trở
  lên), rồi chia vào k fold sao cho mỗi fold có tỉ lệ số lời gọi (0, 1, 2-3, >= 4) và tỉ lệ `ask_missing_info`
  như nhau. Một cụm không bao giờ nằm ở hai fold.
- `val_seen`: fold 0.
- `train`: các fold 1..k-1. Khi dựng prompt cho các mẫu này, phải bỏ app held-out khỏi `installed_apps`.

File ghi ra: `configs/splits/{train,val_seen,val_unseen}.txt`, `configs/splits/folds/fold_<i>.txt` (i = 0..k-1,
mỗi dòng một id) và `configs/splits/meta.json`.

Cách hoạt động:
1. Tách app held-out trước: mẫu có lời gọi gold dùng app held-out, hoặc nằm trong `UNSEEN_OUT_OF_SCOPE`, vào
   `val_unseen`. Chỉ phần còn lại mới được chia, nên `val_unseen` không thể lọt vào train.
2. Gom cụm gần trùng: `cluster_near_duplicates` nhóm các mẫu có `audio_text` giống nhau từ `--dup-threshold`
   trở lên.
3. Stratified group k-fold: gán nguyên cụm vào fold bằng thuật toán tham lam, cân bằng kích thước fold và nhãn
   chính của cụm (nhóm số lời gọi, thêm hậu tố "_ask"). Nhãn hiếm và cụm lớn được xếp trước.
4. Kiểm tra rò rỉ: `meta.json` ghi số id trùng giữa train và hai tập val (phải bằng 0).
5. Tuỳ chọn `--smoke`: lấy N id của `val_seen`, phân tầng theo (loại câu trả lời, có lịch sử hay không).
   Tuỳ chọn `--asr-tune`: lấy N id của train theo cùng cách phân tầng, làm tập dò config ASR.

Lưu ý:
- Nguồn: Adapted from the team lead's repo ThanhDanh1510/ZaloAI2026 src/preprocessing/dataset/split.py@7033c9f
  (gom cụm gần trùng, iterative stratified group k-fold, 6 app held-out).
- Khác bản gốc ở một điểm: train chỉ gồm mẫu không đụng tới app held-out, nên `val_unseen` không bao giờ lọt vào
  dữ liệu train.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from lifemate.core.config import load_profile
from lifemate.core.paths import REPO_ROOT as REPO
from lifemate.core.paths import local_data_root, repo_path
from lifemate.core.textsim import cluster_near_duplicates

DEFAULT_ROOT = local_data_root()
# Lựa chọn của trưởng nhóm: mỗi miền một app (sức khoẻ, thể thao, du lịch, di chuyển, cá nhân, tài chính).
DEFAULT_HELD_OUT = ["medication", "fitness", "flights", "bike_rental", "library", "subscriptions"]

# Các mẫu gold rỗng có yêu cầu liên quan tới app held-out (đã huỷ, phụ thuộc điều kiện, hoặc app chưa cài).
# Gold không gọi tool nào nên `gold_apps()` không phân loại được; thiếu danh sách này thì `val_unseen` không có ca
# `out_of_scope`. Gắn tay dựa trên `audio_text`.
# `train_1606` (giữ chỗ sách thư viện, gold []) có vẻ bị gán nhãn sai nên cố ý bỏ ra. `train_0736` là đăng ký khoá
# học (learning), không phải subscription.
UNSEEN_OUT_OF_SCOPE = {
    "train_0856": "flights",  # đặt hai vé máy bay, flights chưa cài
    "train_1114": "library",  # tìm sách, sau đó hủy
    "train_1172": "flights",  # làm thủ tục chuyến bay, flights chưa cài
    "train_1185": "medication",  # đánh dấu đã uống thuốc, sau đó hủy
    "train_1655": "library",  # giữ chỗ sách thư viện, library chưa cài
}


def call_bucket(gold_calls: list[dict]) -> str:
    """Xếp số lời gọi gold vào nhóm "0", "1", "2-3" hoặc ">=4" để phân tầng fold."""
    n = len(gold_calls)
    return "0" if n == 0 else "1" if n == 1 else "2-3" if n <= 3 else ">=4"


def stratified_group_folds(clusters: list[list[str]], by_id: dict[str, dict], k: int, seed: int) -> list[list[str]]:
    """Gán nguyên từng cụm vào k fold, cân bằng kích thước fold và nhãn chính của cụm.

    Nhãn chính là nhóm số lời gọi gặp nhiều nhất trong cụm, thêm hậu tố "_ask" khi có mẫu hỏi lại thông tin.

    Thuật toán tham lam: nhãn hiếm nhất và cụm lớn nhất được xếp trước. Mỗi cụm vào fold có điểm thấp nhất, với
    điểm = 2 × số cụm cùng nhãn đã có trong fold + 3 × độ vượt kích thước mục tiêu (tính tương đối).

    Tham số:
        clusters: các cụm id gần trùng.
        by_id: id tới mẫu thô có nhãn.
        k: số fold.
        seed: seed quyết định thứ tự giữa các cụm cùng khoá sắp xếp.

    Trả về:
        k danh sách id, mỗi danh sách đã sắp xếp.
    """
    rng = random.Random(seed)
    records = []
    for ids in clusters:
        calls = [by_id[i]["label"]["tool_calls"] for i in ids]
        buckets = collections.Counter(call_bucket(c) for c in calls)
        # Nhóm xuất hiện nhiều nhất; sắp xếp trước để khi hoà, kết quả vẫn cố định.
        label = max(sorted(buckets.items()), key=lambda kv: kv[1])[0]
        if any(any(c["name"] == "sys.ask_missing_info" for c in cs) for cs in calls):
            label += "_ask"
        records.append((ids, label))
    # Xáo trộn rồi sắp xếp ổn định (stable sort): seed quyết định thứ tự giữa các cụm cùng khoá.
    rng.shuffle(records)
    freq = collections.Counter(label for _, label in records)
    records.sort(key=lambda r: (freq[r[1]], -len(r[0])))
    target = sum(len(ids) for ids, _ in records) / k
    folds: list[list[str]] = [[] for _ in range(k)]
    sizes = [0] * k
    label_counts = [collections.Counter() for _ in range(k)]
    for ids, label in records:
        score = [label_counts[f][label] * 2.0 + 3.0 * (sizes[f] + len(ids) - target) / target for f in range(k)]
        # Hoà điểm thì chọn fold có chỉ số nhỏ hơn.
        best = min(range(k), key=lambda f: (score[f], f))
        folds[best] += ids
        sizes[best] += len(ids)
        label_counts[best][label] += 1
    return [sorted(f) for f in folds]


def category(gold_calls: list[dict]) -> str:
    """Phân loại câu trả lời: "out_of_scope" (không gọi gì), "ask_missing_info" (có hỏi lại) hoặc "execute"."""
    names = [c["name"] for c in gold_calls]
    if not names:
        return "out_of_scope"
    if "sys.ask_missing_info" in names:
        return "ask_missing_info"
    return "execute"


def gold_apps(item: dict) -> set[str]:
    """Trả về tập app mà lời gọi gold dùng (phần trước dấu chấm của tên tool), bỏ app hệ thống "sys"."""
    return {c["name"].split(".", 1)[0] for c in item["label"]["tool_calls"]} - {"sys"}


def smoke_sample(val_seen: list[dict], n: int, seed: int) -> list[str]:
    """Lấy n id của `val_seen`, phân tầng theo (loại câu trả lời, có lịch sử hay không), cố định theo seed.

    Hàm dùng bộ sinh số ngẫu nhiên riêng nên không ảnh hưởng các split chính. Mỗi tầng nhận phần tỉ lệ với kích
    thước của tầng (phương pháp phần dư lớn nhất), và ít nhất một mẫu khi n đủ lớn.

    Tham số:
        val_seen: các mẫu thô để lấy (cũng dùng cho tập train khi tạo `asr_tune.txt`).
        n: số id cần lấy.
        seed: seed của bộ sinh số ngẫu nhiên.

    Trả về:
        Danh sách id đã sắp xếp.
    """
    strata = collections.defaultdict(list)
    for it in sorted(val_seen, key=lambda x: x["id"]):
        strata[(category(it["label"]["tool_calls"]), bool(it["history"]))].append(it["id"])
    keys = sorted(strata)
    n = min(n, len(val_seen))
    total = len(val_seen)
    # Phần nguyên của hạn mức theo tỉ lệ; tối thiểu 1 khi n không nhỏ hơn số tầng; không vượt kích thước tầng.
    quota = {k: min(len(strata[k]), max(1 if n >= len(keys) else 0, n * len(strata[k]) // total)) for k in keys}
    # Còn thiếu thì cộng thêm cho tầng có phần dư lớn nhất trước.
    by_rem = sorted(keys, key=lambda k: (-(n * len(strata[k]) % total), k))
    while sum(quota.values()) < n:
        for k in by_rem:
            if sum(quota.values()) < n and quota[k] < len(strata[k]):
                quota[k] += 1
    # Bị thừa (do mức tối thiểu 1) thì bớt ở tầng có hạn mức lớn nhất.
    while sum(quota.values()) > n:
        k = max(keys, key=lambda k: (quota[k], k))
        quota[k] -= 1
    rng = random.Random(seed)
    out = []
    for k in keys:
        out += rng.sample(strata[k], quota[k])
    return sorted(out)


def main() -> None:
    """Chạy CLI: tạo các split, các fold, `smoke.txt` và `asr_tune.txt` (tuỳ chọn), `meta.json`, rồi in thống kê."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, help="folder holding the split folders (default: local data folder)")
    ap.add_argument("--env", help="take the data folder from env profile configs/env/<name>.toml (colab, kaggle, ...)")
    ap.add_argument("--out", type=Path, default=REPO / "configs" / "splits", help="output folder")
    ap.add_argument(
        "--held-out", nargs="+", default=DEFAULT_HELD_OUT, metavar="APP", help="apps reserved for val_unseen"
    )
    ap.add_argument("--folds", type=int, default=5, help="k stratified group folds; fold 0 is val_seen")
    ap.add_argument("--dup-threshold", type=float, default=0.85, help="audio_text similarity that makes a cluster")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--smoke",
        type=int,
        default=0,
        metavar="N",
        help="also write smoke.txt: N val_seen ids stratified by (answer type, has history); 0 = skip",
    )
    ap.add_argument(
        "--asr-tune",
        type=int,
        default=0,
        metavar="N",
        help="also write asr_tune.txt: N train ids (same stratification) for choosing the ASR config; 0 = skip",
    )
    args = ap.parse_args()
    # Thứ tự ưu tiên: `--root`, rồi `data_root` của profile `--env`, rồi thư mục dữ liệu mặc định trên máy.
    if args.root is None:
        args.root = repo_path(load_profile(args.env, REPO).data_root, REPO) if args.env else DEFAULT_ROOT

    with open(args.root / "train" / "train.jsonl", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    with open(args.root / "train" / "train_tools_catalog.json", encoding="utf-8") as f:
        catalog = json.load(f)
    unknown = set(args.held_out) - {a["id"] for a in catalog["apps"]}
    if unknown:
        ap.error(f"apps not in the train catalog: {sorted(unknown)}")
    if args.folds < 2:
        ap.error("--folds must be at least 2")
    held_out = set(args.held_out)

    def is_unseen(it: dict) -> bool:
        """Kiểm tra mẫu thuộc `val_unseen`: gold dùng app held-out, hoặc được gắn tay trong `UNSEEN_OUT_OF_SCOPE`."""
        return bool(gold_apps(it) & held_out) or UNSEEN_OUT_OF_SCOPE.get(it["id"]) in held_out

    # Bước 1: tách `val_unseen` trước; mọi bước sau chỉ làm việc trên phần còn lại.
    val_unseen = [it for it in items if is_unseen(it)]
    rest = [it for it in items if not is_unseen(it)]
    by_id = {it["id"]: it for it in rest}
    # Bước 2 và 3: gom cụm gần trùng rồi chia cụm vào fold; fold 0 là `val_seen`, các fold còn lại là train.
    clusters = cluster_near_duplicates(rest, args.dup_threshold)
    folds = stratified_group_folds(clusters, by_id, args.folds, args.seed)
    val_seen = [by_id[i] for i in folds[0]]
    train = [by_id[i] for f in folds[1:] for i in f]

    args.out.mkdir(parents=True, exist_ok=True)
    splits = {"train": train, "val_seen": val_seen, "val_unseen": val_unseen}
    for name, part in splits.items():
        (args.out / f"{name}.txt").write_text("".join(f"{it['id']}\n" for it in sorted(part, key=lambda x: x["id"])))
    fold_dir = args.out / "folds"
    fold_dir.mkdir(exist_ok=True)
    # Xoá fold cũ, để không sót file thừa khi k nhỏ hơn lần chạy trước.
    for old in fold_dir.glob("fold_*.txt"):
        old.unlink()
    for i, fold in enumerate(folds):
        (fold_dir / f"fold_{i}.txt").write_text("".join(f"{x}\n" for x in fold))

    if args.smoke:
        smoke = smoke_sample(val_seen, args.smoke, args.seed)
        (args.out / "smoke.txt").write_text("".join(f"{i}\n" for i in smoke))
        print(f"  smoke      {len(smoke):>5}  -> {args.out / 'smoke.txt'}")
    # Tập dò config ASR: lấy từ train (ASR không học trên train nên không bị lộ đáp án); split chính không đổi.
    # Nếu sau này fine-tune ASR trên train thì phải loại các id này khỏi dữ liệu train ASR.
    if args.asr_tune:
        tune = smoke_sample(train, args.asr_tune, args.seed)
        (args.out / "asr_tune.txt").write_text("".join(f"{i}\n" for i in tune))
        print(f"  asr_tune   {len(tune):>5}  -> {args.out / 'asr_tune.txt'}")

    def mix(part):
        """Đếm số mẫu theo loại câu trả lời trong một split."""
        return dict(sorted(collections.Counter(category(it["label"]["tool_calls"]) for it in part).items()))

    def buckets(ids):
        """Đếm số mẫu theo nhóm số lời gọi trong một fold."""
        return dict(sorted(collections.Counter(call_bucket(by_id[i]["label"]["tool_calls"]) for i in ids).items()))

    multi = [c for c in clusters if len(c) > 1]
    meta = {
        "held_out_apps": sorted(held_out),
        "folds": args.folds,
        "dup_threshold": args.dup_threshold,
        "seed": args.seed,
        "counts": {name: len(part) for name, part in splits.items()},
        "categories": {name: mix(part) for name, part in splits.items()},
        "fold_sizes": [len(f) for f in folds],
        "fold_call_buckets": [buckets(f) for f in folds],
        "near_duplicate_clusters": {"clusters": len(multi), "samples": sum(len(c) for c in multi)},
        "train_samples_with_held_out_installed": sum(
            bool(held_out & set(it["context"]["installed_apps"])) for it in train
        ),
        # Bước 4: kiểm tra rò rỉ; cả hai số phải bằng 0.
        "leak_check": {
            "train_and_val_unseen": len({it["id"] for it in train} & {it["id"] for it in val_unseen}),
            "train_and_val_seen": len({it["id"] for it in train} & {it["id"] for it in val_seen}),
        },
        "note": "For train samples, remove held_out_apps from context.installed_apps when building prompts.",
        "source": "adapted from ThanhDanh1510/ZaloAI2026 src/preprocessing/dataset/split.py@7033c9f",
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    print(f"held-out apps: {', '.join(sorted(held_out))}")
    print(f"near-duplicate clusters: {meta['near_duplicate_clusters']}")
    for name, part in splits.items():
        print(f"  {name:<11}{len(part):>5}  {mix(part)}")
    print(f"  folds      {meta['fold_sizes']}")
    print(f"leak check: {meta['leak_check']}")
    print(f"-> {args.out}/")


if __name__ == "__main__":
    main()
