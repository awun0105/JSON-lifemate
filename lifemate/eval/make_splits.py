"""Split the labelled train set into train / val_seen / val_unseen id lists.

  val_unseen: every sample whose gold calls use one of the held-out apps (simulates apps never seen in training),
              plus the hand-tagged empty-gold samples about those apps (UNSEEN_OUT_OF_SCOPE)
  val_seen:   a stratified share of the remaining samples (by answer type and by whether there is history)
  train:      the rest; when building prompts for these samples, drop the held-out apps from installed_apps
              (their gold never calls them, so the label stays correct and the model never sees those apps)

Writes configs/splits/{train,val_seen,val_unseen}.txt (one id per line) and configs/splits/meta.json.

Usage (from the repo root):
  python -m lifemate.eval.make_splits
  python -m lifemate.eval.make_splits --held-out medication fitness --val-frac 0.2 --seed 7
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from lifemate.core.paths import REPO_ROOT as REPO


def default_data_root() -> Path:
    """Data root of the `local` env profile (configs/env/local.toml), resolved from the repo root."""
    import tomllib

    with open(REPO / "configs" / "env" / "local.toml", "rb") as f:
        raw = tomllib.load(f)
    p = Path(raw["data_root"]).expanduser()
    return p if p.is_absolute() else REPO / p


DEFAULT_ROOT = default_data_root()
DEFAULT_HELD_OUT = [
    "medication",
    "fitness",
    "door_lock",
    "home_security",
    "robot_vacuum",
    "library",
    "loyalty",
    "learning",
]

# Empty-gold samples whose request is about a held-out app (cancelled, condition-dependent, or the app is missing).
# Their gold calls nothing, so gold_apps() cannot route them; without this list val_unseen has no out_of_scope case.
# Tagged by hand from audio_text. train_1606 (library hold, gold []) looks mislabelled and is left out on purpose.
UNSEEN_OUT_OF_SCOPE = {
    "train_0863": "home_security",  # camera recognition, home_security not installed
    "train_0926": "door_lock",  # gate code, then cancelled
    "train_0982": "home_security",  # away-mode alarm, home_security not installed
    "train_1114": "library",  # find a book, then cancelled
    "train_1185": "medication",  # mark pill taken, then cancelled
    "train_1286": "door_lock",  # door code, then cancelled
    "train_1360": "loyalty",  # redeem seat upgrade only if enough points
    "train_1446": "learning",  # book an English lesson only if not booked yet
    "train_1655": "library",  # place a library hold, library not installed
    "train_1707": "home_security",  # check offline camera, then cancelled
}


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
    ap.add_argument("--val-frac", type=float, default=0.15, help="share of the remaining samples put in val_seen")
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
    held_out = set(args.held_out)

    def is_unseen(it: dict) -> bool:
        return bool(gold_apps(it) & held_out) or UNSEEN_OUT_OF_SCOPE.get(it["id"]) in held_out

    val_unseen = [it for it in items if is_unseen(it)]
    rest = [it for it in items if not is_unseen(it)]

    # stratify val_seen by (answer type, has history) so it mirrors the remaining data
    rng = random.Random(args.seed)
    strata = collections.defaultdict(list)
    for it in rest:
        strata[(category(it["label"]["tool_calls"]), bool(it["history"]))].append(it)
    val_seen = []
    for key in sorted(strata):
        group = strata[key]
        rng.shuffle(group)
        val_seen += group[: round(len(group) * args.val_frac)]
    seen_ids = {it["id"] for it in val_seen}
    train = [it for it in rest if it["id"] not in seen_ids]

    args.out.mkdir(parents=True, exist_ok=True)
    splits = {"train": train, "val_seen": val_seen, "val_unseen": val_unseen}
    for name, part in splits.items():
        (args.out / f"{name}.txt").write_text("".join(f"{it['id']}\n" for it in sorted(part, key=lambda x: x["id"])))

    if args.smoke:
        smoke = smoke_sample(val_seen, args.smoke, args.seed)
        (args.out / "smoke.txt").write_text("".join(f"{i}\n" for i in smoke))
        print(f"  smoke      {len(smoke):>5}  -> {args.out / 'smoke.txt'}")

    def mix(part):
        return dict(collections.Counter(category(it["label"]["tool_calls"]) for it in part))

    meta = {
        "held_out_apps": sorted(held_out),
        "val_frac": args.val_frac,
        "seed": args.seed,
        "counts": {name: len(part) for name, part in splits.items()},
        "categories": {name: mix(part) for name, part in splits.items()},
        "train_samples_with_held_out_installed": sum(
            bool(held_out & set(it["context"]["installed_apps"])) for it in train
        ),
        "note": "For train samples, remove held_out_apps from context.installed_apps when building prompts.",
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    print(f"held-out apps: {', '.join(sorted(held_out))}")
    for name, part in splits.items():
        print(f"  {name:<11}{len(part):>5}  {mix(part)}")
    print(
        "train samples that have a held-out app installed (strip it from the prompt): "
        f"{meta['train_samples_with_held_out_installed']}"
    )
    print(f"-> {args.out}/")


if __name__ == "__main__":
    main()
