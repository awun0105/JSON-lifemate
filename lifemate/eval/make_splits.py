"""Split the labelled train set into train / val_seen / val_unseen id lists, plus k stratified folds.

Adapted from the team lead's repo ThanhDanh1510/ZaloAI2026 src/preprocessing/dataset/split.py@7033c9f
(near-duplicate clustering, iterative stratified group k-fold, 6 held-out apps), with one fix: train is built
only from samples that do not touch a held-out app, so val_unseen never leaks into training.

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

from lifemate.core.paths import REPO_ROOT as REPO
from lifemate.core.textsim import cluster_near_duplicates


def default_data_root() -> Path:
    """Data root of the `local` env profile (configs/env/local.toml), resolved from the repo root."""
    import tomllib

    with open(REPO / "configs" / "env" / "local.toml", "rb") as f:
        raw = tomllib.load(f)
    p = Path(raw["data_root"]).expanduser()
    return p if p.is_absolute() else REPO / p


DEFAULT_ROOT = default_data_root()
# Team lead's choice: one app per domain (health, sport, travel, mobility, personal, finance).
DEFAULT_HELD_OUT = ["medication", "fitness", "flights", "bike_rental", "library", "subscriptions"]

# Empty-gold samples whose request is about a held-out app (cancelled, condition-dependent, or the app is missing).
# Their gold calls nothing, so gold_apps() cannot route them; without this list val_unseen has no out_of_scope case.
# Tagged by hand from audio_text. train_1606 (library hold, gold []) looks mislabelled and is left out on purpose;
# train_0736 is a course sign-up (learning), not a subscription.
UNSEEN_OUT_OF_SCOPE = {
    "train_0856": "flights",  # book two plane tickets, flights not installed
    "train_1114": "library",  # find a book, then cancelled
    "train_1172": "flights",  # check in for a flight, flights not installed
    "train_1185": "medication",  # mark pill taken, then cancelled
    "train_1655": "library",  # place a library hold, library not installed
}


def call_bucket(gold_calls: list[dict]) -> str:
    n = len(gold_calls)
    return "0" if n == 0 else "1" if n == 1 else "2-3" if n <= 3 else ">=4"


def stratified_group_folds(clusters: list[list[str]], by_id: dict[str, dict], k: int, seed: int) -> list[list[str]]:
    """Assign whole clusters to k folds, balancing fold size and the cluster's main label (call bucket, + "_ask" when
    any member asks for missing info). Rarest labels and biggest clusters are placed first (greedy)."""
    rng = random.Random(seed)
    records = []
    for ids in clusters:
        calls = [by_id[i]["label"]["tool_calls"] for i in ids]
        buckets = collections.Counter(call_bucket(c) for c in calls)
        label = max(sorted(buckets.items()), key=lambda kv: kv[1])[0]
        if any(any(c["name"] == "sys.ask_missing_info" for c in cs) for cs in calls):
            label += "_ask"
        records.append((ids, label))
    rng.shuffle(records)
    freq = collections.Counter(label for _, label in records)
    records.sort(key=lambda r: (freq[r[1]], -len(r[0])))
    target = sum(len(ids) for ids, _ in records) / k
    folds: list[list[str]] = [[] for _ in range(k)]
    sizes = [0] * k
    label_counts = [collections.Counter() for _ in range(k)]
    for ids, label in records:
        score = [label_counts[f][label] * 2.0 + 3.0 * (sizes[f] + len(ids) - target) / target for f in range(k)]
        best = min(range(k), key=lambda f: (score[f], f))
        folds[best] += ids
        sizes[best] += len(ids)
        label_counts[best][label] += 1
    return [sorted(f) for f in folds]


def category(gold_calls: list[dict]) -> str:
    names = [c["name"] for c in gold_calls]
    if not names:
        return "out_of_scope"
    if "sys.ask_missing_info" in names:
        return "ask_missing_info"
    return "execute"


def gold_apps(item: dict) -> set[str]:
    return {c["name"].split(".", 1)[0] for c in item["label"]["tool_calls"]} - {"sys"}


def smoke_sample(val_seen: list[dict], n: int, seed: int) -> list[str]:
    """n val_seen ids stratified by (answer type, has history), deterministic for a seed (its own RNG, so the main
    splits are unaffected). Each stratum gets a proportional share (largest remainder), at least one when n allows."""
    strata = collections.defaultdict(list)
    for it in sorted(val_seen, key=lambda x: x["id"]):
        strata[(category(it["label"]["tool_calls"]), bool(it["history"]))].append(it["id"])
    keys = sorted(strata)
    n = min(n, len(val_seen))
    total = len(val_seen)
    quota = {k: min(len(strata[k]), max(1 if n >= len(keys) else 0, n * len(strata[k]) // total)) for k in keys}
    by_rem = sorted(keys, key=lambda k: (-(n * len(strata[k]) % total), k))
    while sum(quota.values()) < n:
        for k in by_rem:
            if sum(quota.values()) < n and quota[k] < len(strata[k]):
                quota[k] += 1
    while sum(quota.values()) > n:
        k = max(keys, key=lambda k: (quota[k], k))
        quota[k] -= 1
    rng = random.Random(seed)
    out = []
    for k in keys:
        out += rng.sample(strata[k], quota[k])
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="folder holding the split folders")
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
    args = ap.parse_args()

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
        return bool(gold_apps(it) & held_out) or UNSEEN_OUT_OF_SCOPE.get(it["id"]) in held_out

    val_unseen = [it for it in items if is_unseen(it)]
    rest = [it for it in items if not is_unseen(it)]
    by_id = {it["id"]: it for it in rest}
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
    for old in fold_dir.glob("fold_*.txt"):
        old.unlink()
    for i, fold in enumerate(folds):
        (fold_dir / f"fold_{i}.txt").write_text("".join(f"{x}\n" for x in fold))

    if args.smoke:
        smoke = smoke_sample(val_seen, args.smoke, args.seed)
        (args.out / "smoke.txt").write_text("".join(f"{i}\n" for i in smoke))
        print(f"  smoke      {len(smoke):>5}  -> {args.out / 'smoke.txt'}")

    def mix(part):
        return dict(sorted(collections.Counter(category(it["label"]["tool_calls"]) for it in part).items()))

    def buckets(ids):
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
