"""Chia tập train có nhãn thành các danh sách id train / val_seen / val_unseen, kèm k fold phân tầng.

Nội dung: script đọc tập train và catalog, tách mẫu theo app giữ lại (held-out), gom mẫu gần trùng thành cụm, chia
cụm vào k fold, rồi ghi danh sách id và tệp meta.

Cách xây dựng, theo thứ tự:
  1. Loại app giữ lại trước: mẫu có lời gọi gold dùng app giữ lại, hoặc nằm trong UNSEEN_OUT_OF_SCOPE, vào
     val_unseen. Phần còn lại mới được chia, nên val_unseen không thể rò vào train.
  2. Gom cụm gần trùng: `cluster_near_duplicates` nhóm các mẫu có audio_text giống nhau từ `--dup-threshold` trở lên.
  3. Stratified group k-fold: gán nguyên cụm vào fold bằng thuật toán tham lam, cân bằng kích thước fold và nhãn
     chính của cụm (nhóm số lời gọi, cộng hậu tố "_ask"). Nhãn hiếm và cụm lớn được xếp trước.
  4. Kiểm tra rò rỉ: meta.json ghi số id trùng giữa train và hai tập val (phải bằng 0).
  5. Tùy chọn `--smoke`: lấy N id của val_seen, phân tầng theo (loại câu trả lời, có lịch sử hay không).
     Tùy chọn `--asr-tune`: lấy N id của train cùng cách phân tầng, làm tập chỉnh cấu hình ASR.

Adapted from the team lead's repo ThanhDanh1510/ZaloAI2026 src/preprocessing/dataset/split.py@7033c9f
(near-duplicate clustering, iterative stratified group k-fold, 6 held-out apps), with one fix: train is built
only from samples that do not touch a held-out app, so val_unseen never leaks into training.
(Chuyển thể từ repo của trưởng nhóm, với một sửa đổi: train chỉ gồm mẫu không đụng tới app giữ lại.)

  val_unseen: every sample whose gold calls use one of the held-out apps (simulates the new apps of the private
              test), plus the hand-tagged empty-gold samples about those apps (UNSEEN_OUT_OF_SCOPE)
  folds:      the remaining samples, grouped into clusters of near-duplicate requests (audio_text similarity
              >= --dup-threshold) and spread over k folds so each fold keeps the same mix of call counts
              (0, 1, 2-3, >= 4) and of ask_missing_info; a cluster never spans two folds
  val_seen:   fold 0
  train:      folds 1..k-1; when building prompts for these samples, drop the held-out apps from installed_apps

Writes configs/splits/{train,val_seen,val_unseen}.txt, configs/splits/folds/fold_<i>.txt (i = 0..k-1, one id per
line) and configs/splits/meta.json.

Usage (from the repo root):
  python -m lifemate.eval.make_splits --smoke 30
  python -m lifemate.eval.make_splits --held-out medication fitness --folds 5 --seed 7
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from lifemate.core.config import load_profile
from lifemate.core.paths import REPO_ROOT as REPO
from lifemate.core.textsim import cluster_near_duplicates


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
# Lựa chọn của trưởng nhóm: mỗi miền một app (sức khỏe, thể thao, du lịch, di chuyển, cá nhân, tài chính).
DEFAULT_HELD_OUT = ["medication", "fitness", "flights", "bike_rental", "library", "subscriptions"]

# Các mẫu gold rỗng có yêu cầu liên quan tới app giữ lại (đã hủy, phụ thuộc điều kiện, hoặc app chưa cài).
# Gold không gọi tool nào nên gold_apps() không định tuyến được; thiếu danh sách này thì val_unseen không có ca
# out_of_scope. Gắn tay dựa trên audio_text. train_1606 (giữ chỗ sách thư viện, gold []) có vẻ gán nhãn sai nên cố ý
# bỏ ra; train_0736 là đăng ký khóa học (learning), không phải subscription.
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

    Nhãn chính là nhóm số lời gọi phổ biến nhất trong cụm, thêm hậu tố "_ask" khi có thành viên hỏi thêm thông tin.
    Thuật toán tham lam: nhãn hiếm nhất và cụm lớn nhất được xếp trước; mỗi cụm vào fold có điểm thấp nhất, với điểm
    gồm số cụm cùng nhãn đã có trong fold (trọng số 2) cộng độ vượt kích thước mục tiêu tương đối (trọng số 3).
    """
    rng = random.Random(seed)
    records = []
    for ids in clusters:
        calls = [by_id[i]["label"]["tool_calls"] for i in ids]
        buckets = collections.Counter(call_bucket(c) for c in calls)
        # Nhóm xuất hiện nhiều nhất; sắp xếp trước để hòa thì kết quả vẫn xác định.
        label = max(sorted(buckets.items()), key=lambda kv: kv[1])[0]
        if any(any(c["name"] == "sys.ask_missing_info" for c in cs) for cs in calls):
            label += "_ask"
        records.append((ids, label))
    # Xáo trộn rồi sắp xếp ổn định: seed quyết định thứ tự giữa các cụm cùng khóa.
    rng.shuffle(records)
    freq = collections.Counter(label for _, label in records)
    records.sort(key=lambda r: (freq[r[1]], -len(r[0])))
    target = sum(len(ids) for ids, _ in records) / k
    folds: list[list[str]] = [[] for _ in range(k)]
    sizes = [0] * k
    label_counts = [collections.Counter() for _ in range(k)]
    for ids, label in records:
        score = [label_counts[f][label] * 2.0 + 3.0 * (sizes[f] + len(ids) - target) / target for f in range(k)]
        # Hòa điểm thì chọn fold có chỉ số nhỏ hơn.
        best = min(range(k), key=lambda f: (score[f], f))
        folds[best] += ids
        sizes[best] += len(ids)
        label_counts[best][label] += 1
    return [sorted(f) for f in folds]


def category(gold_calls: list[dict]) -> str:
    """Loại câu trả lời: "out_of_scope" (không gọi gì), "ask_missing_info" (có hỏi lại) hoặc "execute"."""
    names = [c["name"] for c in gold_calls]
    if not names:
        return "out_of_scope"
    if "sys.ask_missing_info" in names:
        return "ask_missing_info"
    return "execute"


def gold_apps(item: dict) -> set[str]:
    """Tập app mà lời gọi gold dùng (tiền tố trước dấu chấm của tên tool), bỏ app hệ thống "sys"."""
    return {c["name"].split(".", 1)[0] for c in item["label"]["tool_calls"]} - {"sys"}


def smoke_sample(val_seen: list[dict], n: int, seed: int) -> list[str]:
    """Lấy n id của val_seen, phân tầng theo (loại câu trả lời, có lịch sử hay không), cố định theo seed.

    Hàm dùng RNG riêng nên không ảnh hưởng các split chính. Mỗi tầng nhận phần tỷ lệ với kích thước (phương pháp
    phần dư lớn nhất), và ít nhất một mẫu khi n đủ lớn.
    """
    strata = collections.defaultdict(list)
    for it in sorted(val_seen, key=lambda x: x["id"]):
        strata[(category(it["label"]["tool_calls"]), bool(it["history"]))].append(it["id"])
    keys = sorted(strata)
    n = min(n, len(val_seen))
    total = len(val_seen)
    # Phần nguyên của hạn mức tỷ lệ, tối thiểu 1 khi n không nhỏ hơn số tầng, không vượt kích thước tầng.
    quota = {k: min(len(strata[k]), max(1 if n >= len(keys) else 0, n * len(strata[k]) // total)) for k in keys}
    # Thiếu thì cộng thêm cho tầng có phần dư lớn nhất trước.
    by_rem = sorted(keys, key=lambda k: (-(n * len(strata[k]) % total), k))
    while sum(quota.values()) < n:
        for k in by_rem:
            if sum(quota.values()) < n and quota[k] < len(strata[k]):
                quota[k] += 1
    # Thừa (do mức tối thiểu 1) thì bớt ở tầng có hạn mức lớn nhất.
    while sum(quota.values()) > n:
        k = max(keys, key=lambda k: (quota[k], k))
        quota[k] -= 1
    rng = random.Random(seed)
    out = []
    for k in keys:
        out += rng.sample(strata[k], quota[k])
    return sorted(out)


def main() -> None:
    """Điểm vào dòng lệnh: tạo các split, các fold, smoke.txt (tùy chọn) và meta.json, rồi in thống kê."""
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
    # Thứ tự ưu tiên: --root, rồi data_root của profile --env, rồi thư mục dữ liệu cục bộ mặc định.
    if args.root is None:
        args.root = Path(load_profile(args.env, REPO).data_root).expanduser() if args.env else DEFAULT_ROOT

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
        """Mẫu thuộc val_unseen: gold dùng app giữ lại, hoặc được gắn tay trong UNSEEN_OUT_OF_SCOPE."""
        return bool(gold_apps(it) & held_out) or UNSEEN_OUT_OF_SCOPE.get(it["id"]) in held_out

    # Bước 1: tách val_unseen trước, mọi bước sau chỉ làm việc trên phần còn lại.
    val_unseen = [it for it in items if is_unseen(it)]
    rest = [it for it in items if not is_unseen(it)]
    by_id = {it["id"]: it for it in rest}
    # Bước 2 và 3: gom cụm gần trùng rồi chia cụm vào fold; fold 0 là val_seen, các fold còn lại là train.
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
    # Xóa fold cũ để không sót tệp khi k nhỏ hơn lần chạy trước.
    for old in fold_dir.glob("fold_*.txt"):
        old.unlink()
    for i, fold in enumerate(folds):
        (fold_dir / f"fold_{i}.txt").write_text("".join(f"{x}\n" for x in fold))

    if args.smoke:
        smoke = smoke_sample(val_seen, args.smoke, args.seed)
        (args.out / "smoke.txt").write_text("".join(f"{i}\n" for i in smoke))
        print(f"  smoke      {len(smoke):>5}  -> {args.out / 'smoke.txt'}")
    # Tập chỉnh cấu hình ASR: lấy từ train (ASR không học trên train nên không rò), split chính không đổi.
    # Nếu sau này fine-tune ASR trên train thì phải loại các id này khỏi dữ liệu train ASR.
    if args.asr_tune:
        tune = smoke_sample(train, args.asr_tune, args.seed)
        (args.out / "asr_tune.txt").write_text("".join(f"{i}\n" for i in tune))
        print(f"  asr_tune   {len(tune):>5}  -> {args.out / 'asr_tune.txt'}")

    def mix(part):
        """Số mẫu theo loại câu trả lời trong một split."""
        return dict(sorted(collections.Counter(category(it["label"]["tool_calls"]) for it in part).items()))

    def buckets(ids):
        """Số mẫu theo nhóm số lời gọi trong một fold."""
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
        # Bước 4: kiểm tra rò rỉ, cả hai số phải bằng 0.
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
