"""Build SFT rows exactly as the agent sees them at run time (SPEC 6.2.1).

Prompt from the run's PromptBuilder (train mode strips held-out apps), gold transcript, gold calls replayed through
a fresh zaic_tools env per sample for byte-exact `KẾT QUẢ` turns.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import sys
import tomllib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from lifemate.agent.agent import BLOCK_PREFIX
from lifemate.agent.prompt import HeldOutLeak, PromptBuilder, truncate_result
from lifemate.agent.protocol import DONE_TEXT, call_text
from lifemate.agent.tooldocs import ASK_TOOL, CatalogIndex
from lifemate.backends.asr import GoldTranscriber
from lifemate.backends.asr_sanitizer import entities_preserved
from lifemate.core.config import AgentConfig, ConfigError, LLMConfig, load_profile
from lifemate.core.envinfo import code_version
from lifemate.core.jsonutil import canonical_json
from lifemate.core.paths import REPO_ROOT
from lifemate.core.textsim import cluster_near_duplicates
from lifemate.core.tokens import HFTokenCounter, SimpleTokenCounter
from lifemate.core.types import Sample, TokenCounter, Transcriber, Transcript
from lifemate.io.data import load_split, read_ids
from lifemate.io.zaic import EnvRegistry, ZaicLoader, load_catalog
from lifemate.train.asr_cache import CachedTranscriber
from lifemate.train.common import abs_path, build_section, read_held_out, sha256_file
from lifemate.train.sft import LLMTrainConfig, load_llm_train_config

log = structlog.get_logger(__name__)

RESULT_PREFIX = "KẾT QUẢ: "
EVAL_ID_FILES = ("configs/splits/val_seen.txt", "configs/splits/val_unseen.txt")


@dataclass(slots=True)
class BuildResult:
    """Rows kept, samples dropped with reasons, replay mismatches and shrink levels."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[tuple[str, str]] = field(default_factory=list)
    replay_warnings: list[dict[str, Any]] = field(default_factory=list)
    shrink_levels: Counter[int] = field(default_factory=Counter)


@dataclass(frozen=True, slots=True)
class RunPromptConfig:
    """The [llm] and [agent] sections of the run config: everything that shapes the prompt."""

    llm: LLMConfig
    agent: AgentConfig


def load_run_config(path: Path) -> RunPromptConfig:
    """[llm] and [agent] of a run config; unlike load_config it accepts an empty llm.model (SPEC 6.0)."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    for key in ("llm", "agent"):
        if key not in raw:
            raise ConfigError(f"missing key {key}")
    return RunPromptConfig(
        build_section(LLMConfig, "llm", raw["llm"]), build_section(AgentConfig, "agent", raw["agent"])
    )


def make_counter(model: str, revision: str) -> TokenCounter:
    """SimpleTokenCounter while no model is chosen, else the base model's HFTokenCounter."""
    if not model:
        return SimpleTokenCounter()
    return HFTokenCounter(model, revision or None)


def make_builder(
    run_cfg: RunPromptConfig, repo_root: Path, index: CatalogIndex, counter: TokenCounter
) -> PromptBuilder:
    """PromptBuilder wired exactly like lifemate.run.factory.build."""
    template = (repo_root / run_cfg.agent.system_prompt).read_text(encoding="utf-8")
    return PromptBuilder(
        template,
        index,
        counter,
        max_model_len=run_cfg.llm.max_model_len,
        max_tokens=run_cfg.llm.max_tokens,
        max_result_chars=run_cfg.agent.max_result_chars,
        held_out_apps=read_held_out(repo_root),
    )


TRANSCRIPT_SOURCES = ("gold", "asr", "mix")


def _frac(seed: int, tag: str, sample_id: str) -> float:
    """Stable number in [0, 1) from (seed, tag, id): the same choice on every rebuild."""
    h = hashlib.sha256(f"{seed}:{tag}:{sample_id}".encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32


def category(sample: Sample) -> str:
    """`execute`, `ask_missing_info` or `out_of_scope` (empty gold)."""
    calls = (sample.label or {}).get("tool_calls") or []
    if not calls:
        return "out_of_scope"
    return "ask_missing_info" if any(c.get("name") == ASK_TOOL for c in calls) else "execute"


@dataclass(frozen=True, slots=True)
class Augment:
    """Train-only data augmentation, adapted from the team lead's repo ThanhDanh1510/ZaloAI2026
    (src/preprocessing/dataset/builder.py and docs/llm_finetuning.md@c703bd9). Every choice is a hash of
    (seed, id), so a rebuild gives the same file. All of it keeps the prompt built by the run-time PromptBuilder.

    - error_recovery_ratio: share of samples that get one failed call first (a required argument dropped, the real
      guard error text from `env.validate`), then the correct call. The failed call has `"loss": false`: the model
      learns to recover from the error, never to make it.
    - oversample_hard: copies of `ask_missing_info` and `out_of_scope` rows (1.0 = none, 1.5 = half get a 2nd copy).
    - distractor_apps: extra installed apps (never held-out ones) added to the context so the model has to read
      the schema. Not applied to `out_of_scope` samples: an added app could serve the request and make the empty
      label wrong.
    - dedup_max_k: keep at most k samples per near-duplicate cluster (0 = keep all)."""

    error_recovery_ratio: float = 0.0
    oversample_hard: float = 1.0
    distractor_apps: int = 0
    dedup_max_k: int = 0
    seed: int = 0


def dedup_clusters(samples: Sequence[Sample], max_k: int, threshold: float = 0.85) -> list[Sample]:
    """At most `max_k` samples (lowest ids first) per near-duplicate cluster; order kept."""
    if max_k <= 0:
        return list(samples)
    items = [{"id": s.id, "audio_text": s.audio_text or ""} for s in samples]
    keep = {sid for group in cluster_near_duplicates(items, threshold) for sid in sorted(group)[:max_k]}
    return [s for s in samples if s.id in keep]


def with_distractors(
    sample: Sample, index: CatalogIndex, held_out: frozenset[str], k: int, seed: int
) -> tuple[Sample, list[str]]:
    """`sample` with `k` extra installed apps picked by hash among apps that are neither installed nor held out."""
    if k <= 0 or category(sample) == "out_of_scope":
        return sample, []
    installed: dict[str, Any] = dict(sample.context.get("installed_apps") or {})
    pool = [a for a in index.apps if a not in installed and a not in held_out]
    picked = sorted(pool, key=lambda a: _frac(seed, "distractor:" + a, sample.id))[:k]
    for app in picked:
        installed[app] = {"tools": [t["name"] for t in index.app_spec(app)["tools"]]}
    return dataclasses.replace(sample, context={**sample.context, "installed_apps": installed}), picked


def hard_copies(sample: Sample, factor: float, seed: int) -> int:
    """Extra copies of a hard (`ask_missing_info`, `out_of_scope`) sample for `oversample_hard = factor`."""
    if factor <= 1.0 or category(sample) == "execute":
        return 0
    extra = factor - 1.0
    return math.floor(extra) + (1 if _frac(seed, "oversample", sample.id) < extra - math.floor(extra) else 0)


class MixedTranscriber:
    """Team lead's data mix (ThanhDanh1510/ZaloAI2026 builder.py@7033c9f): a fixed share of samples use the saved
    ASR transcript, the rest the gold `audio_text`. A sample falls back to gold when its ASR text is empty or lost a
    gold entity (a name, amount or phone number the calls need), so the model is never taught to call tools with
    arguments the transcript does not contain. The choice per sample is a hash of (seed, id): stable across rebuilds.
    `chosen` records what each sample got."""

    def __init__(self, asr: CachedTranscriber, ratio: float, seed: int = 0) -> None:
        self._asr, self._gold = asr, GoldTranscriber()
        self._ratio, self._seed = ratio, seed
        self.chosen: dict[str, str] = {}

    def _picked(self, sample_id: str) -> bool:
        h = hashlib.sha256(f"{self._seed}:{sample_id}".encode()).digest()
        return int.from_bytes(h[:4], "big") / 2**32 < self._ratio

    def transcribe(self, sample: Sample) -> Transcript:
        """ASR transcript for picked samples that keep every gold entity, gold otherwise."""
        if self._ratio > 0 and self._asr.has(sample.id) and self._picked(sample.id):
            t = self._asr.transcribe(sample)
            calls = (sample.label or {}).get("tool_calls") or []
            if t.final.strip() and entities_preserved(calls, t.final):
                self.chosen[sample.id] = "asr"
                return t
            self.chosen[sample.id] = "gold_fallback"
            return self._gold.transcribe(sample)
        self.chosen[sample.id] = "gold"
        return self._gold.transcribe(sample)

    def warmup(self) -> None:
        """Nothing to warm up."""


def _transcriber(source: str, asr_cache: Path | None = None, asr_ratio: float = 0.7, seed: int = 0) -> Transcriber:
    if source == "gold":
        return GoldTranscriber()
    if source not in TRANSCRIPT_SOURCES:
        raise ValueError(f"transcript_source: {source!r} not in {'|'.join(TRANSCRIPT_SOURCES)}")
    if asr_cache is None or not asr_cache.is_file():
        raise FileNotFoundError(
            f"transcript_source={source!r} needs an ASR cache ({asr_cache}); build it on a GPU box with "
            "python -m lifemate.train.asr_cache (SPEC 6.2.1 item 2)"
        )
    cache = CachedTranscriber.from_file(asr_cache)
    return cache if source == "asr" else MixedTranscriber(cache, asr_ratio, seed)


def build_sample(
    sample: Sample,
    builder: PromptBuilder,
    loader: ZaicLoader,
    counter: TokenCounter,
    transcriber: Transcriber,
    *,
    max_model_len: int,
    max_result_chars: int,
    train_mode: bool,
    source: str,
    result: BuildResult,
    inject_error: bool = False,
) -> None:
    """Append one row (or one dropped entry) for `sample` to `result`. With `inject_error`, the first call that has
    a required argument is preceded by the same call without it and the guard's error turn (see `Augment`)."""
    label = sample.label or {}
    calls: list[dict[str, Any]] = list(label.get("tool_calls") or [])
    gold_results: list[Any] = list(label.get("tool_results") or [])
    try:
        conv, level = builder.start(sample, transcriber.transcribe(sample), train_mode=train_mode)
    except HeldOutLeak as exc:
        result.dropped.append((sample.id, f"held_out_leak:{exc.app}"))
        return
    ask_at = [i for i, c in enumerate(calls) if c.get("name") == ASK_TOOL]
    if ask_at and ask_at[0] != len(calls) - 1:
        result.dropped.append((sample.id, "ask_missing_info_not_last"))
        return
    env = loader.open_env(sample.id)
    no_loss: list[int] = []
    for i, call in enumerate(calls):
        name, arguments = str(call["name"]), dict(call.get("arguments") or {})
        if inject_error and not no_loss and name != ASK_TOOL:
            for key in arguments:
                faulty = {k: v for k, v in arguments.items() if k != key}
                verdict = env.validate(name, copy.deepcopy(faulty))
                if not verdict.get("ok"):
                    no_loss.append(len(conv.as_list()))
                    conv.append_assistant(call_text(name, faulty), 0)
                    conv.append_user(BLOCK_PREFIX + str(verdict.get("error")))
                    break
        conv.append_assistant(call_text(name, arguments), 0)
        replayed = env.call(name, copy.deepcopy(arguments))
        if i >= len(gold_results) or canonical_json(replayed) != canonical_json(gold_results[i]):
            log.warning("sft_replay_mismatch", id=sample.id, call=i, name=name)
            result.replay_warnings.append({"id": sample.id, "call": i, "name": name})
        if name == ASK_TOOL:
            break  # run time stops right after ask_missing_info (StopReason.ASKED): no result turn, no done
        conv.append_user(RESULT_PREFIX + truncate_result(replayed, max_result_chars))
    else:
        conv.append_assistant(DONE_TEXT, 0)
    messages = conv.as_list()
    for k in no_loss:
        messages[k]["loss"] = False  # sft.build_labels trains no token of this turn
    n_tokens = counter.count(messages)
    if n_tokens > max_model_len:
        result.dropped.append((sample.id, f"too_long:{n_tokens}>{max_model_len}"))
        return
    result.shrink_levels[level] += 1
    result.rows.append(
        {
            "id": sample.id,
            "source": source,
            "category": category(sample),
            "error_recovery": bool(no_loss),
            "messages": messages,
            "n_tokens": n_tokens,
            "n_assistant_turns": sum(1 for m in messages if m["role"] == "assistant"),
        }
    )


def build_rows(
    samples: Sequence[Sample],
    builder: PromptBuilder,
    loader: ZaicLoader,
    counter: TokenCounter,
    *,
    max_model_len: int,
    max_result_chars: int,
    train_mode: bool,
    source: str = "train",
    transcript_source: str = "gold",
    transcriber: Transcriber | None = None,
    augment: Augment | None = None,
    index: CatalogIndex | None = None,
    held_out: frozenset[str] = frozenset(),
) -> BuildResult:
    """Rows for every sample in order; one fresh env per sample from `loader`. `augment` applies only in train
    mode (eval files stay plain); distractors need `index`."""
    transcriber = transcriber if transcriber is not None else _transcriber(transcript_source)
    aug = augment if (augment is not None and train_mode) else Augment()
    result = BuildResult()
    for original in dedup_clusters(samples, aug.dedup_max_k):
        sample = original
        if aug.distractor_apps > 0 and index is not None:
            sample, _ = with_distractors(original, index, held_out, aug.distractor_apps, aug.seed)
        n_before = len(result.rows)
        build_sample(
            sample,
            builder,
            loader,
            counter,
            transcriber,
            max_model_len=max_model_len,
            max_result_chars=max_result_chars,
            train_mode=train_mode,
            source=source,
            result=result,
            inject_error=_frac(aug.seed, "error", sample.id) < aug.error_recovery_ratio,
        )
        if len(result.rows) > n_before:
            row = result.rows[-1]
            result.rows.extend(
                {**row, "copy": c + 1} for c in range(hard_copies(sample, aug.oversample_hard, aug.seed))
            )
    return result


def eval_ids(repo_root: Path) -> set[str]:
    """Ids of val_seen and val_unseen."""
    out: set[str] = set()
    for rel in EVAL_ID_FILES:
        p = repo_root / rel
        if p.is_file():
            out |= set(read_ids(p))
    return out


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write rows as jsonl via a temp file, fsync and rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_outputs(
    out: Path, result: BuildResult, *, ids: Sequence[str], argv: Sequence[str], repo_root: Path, extra: dict[str, Any]
) -> dict[str, Any]:
    """Write `<out>`, `<out>.dropped.txt` and `<out>.meta.json`; return the meta dict."""
    write_jsonl(out, result.rows)
    dropped_path = Path(f"{out}.dropped.txt")
    dropped_path.write_text("".join(f"{i}\t{r}\n" for i, r in result.dropped), encoding="utf-8")
    version, dirty = code_version(repo_root)
    meta = {
        "file": out.name,
        "sha256": sha256_file(out),
        "argv": list(argv),
        "code_version": version,
        "code_dirty": dirty,
        "ids": list(ids),
        "n_rows": len(result.rows),
        "n_dropped": len(result.dropped),
        "dropped_reasons": dict(Counter(r.split(":")[0] for _, r in result.dropped)),
        "n_replay_warnings": len(result.replay_warnings),
        "replay_warnings": result.replay_warnings,
        "shrink_levels": {str(k): v for k, v in sorted(result.shrink_levels.items())},
        **extra,
    }
    Path(f"{out}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI arguments."""
    ap = argparse.ArgumentParser(prog="python -m lifemate.train.sft_data", description="Build SFT data (SPEC 6.2.1).")
    ap.add_argument("--config", required=True, type=Path, help="train config TOML (configs/train/llm_v1.toml)")
    ap.add_argument("--env", default="local", help="env profile name or path")
    ap.add_argument("--ids", required=True, type=Path, help="file with one sample id per line")
    ap.add_argument("--out", required=True, type=Path, help="output jsonl (e.g. data/sft/v1.jsonl)")
    ap.add_argument("--split", default="train", help="data split folder holding the ids")
    ap.add_argument("--root", help="data root override")
    ap.add_argument("--allow-eval", action="store_true", help="allow val ids (eval-loss files only; no train mode)")
    ap.add_argument("--transcript-source", choices=TRANSCRIPT_SOURCES, default="gold")
    ap.add_argument("--asr-cache", type=Path, help="ASR cache from lifemate.train.asr_cache (for asr and mix)")
    ap.add_argument("--asr-ratio", type=float, default=0.7, help="mix: share of samples that use the ASR transcript")
    ap.add_argument("--mix-seed", type=int, default=0, help="mix: seed of the per-sample choice (and of augment)")
    ap.add_argument("--error-recovery", type=float, default=0.0, help="train: share of samples with a failed call")
    ap.add_argument("--oversample-hard", type=float, default=1.0, help="train: copies of ask/out_of_scope rows")
    ap.add_argument("--distractor-apps", type=int, default=0, help="train: extra installed apps per sample")
    ap.add_argument("--dedup-max-k", type=int, default=0, help="train: max samples per near-duplicate cluster")
    return ap.parse_args(argv)


def run(args: argparse.Namespace, repo_root: Path = REPO_ROOT, argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Build one SFT file from parsed arguments; returns the meta dict."""
    cfg: LLMTrainConfig = load_llm_train_config(abs_path(args.config, repo_root))
    run_cfg = load_run_config(abs_path(cfg.base.run_config, repo_root))
    profile = load_profile(args.env, repo_root).with_overrides(root=args.root)
    root = abs_path(profile.data_root, repo_root)
    ids = read_ids(abs_path(args.ids, repo_root))
    leaked = sorted(set(ids) & eval_ids(repo_root))
    if leaked and not args.allow_eval:
        raise SystemExit(f"refusing: {len(leaked)} ids are in val_seen/val_unseen (e.g. {leaked[:3]}); --allow-eval")
    train_mode = not args.allow_eval  # SPEC 4.5: held-out rule only for train ids
    asr_cache = abs_path(args.asr_cache, repo_root) if args.asr_cache else None
    transcriber = _transcriber(args.transcript_source, asr_cache, args.asr_ratio, args.mix_seed)  # fail fast
    counter = make_counter(cfg.base.model, cfg.base.revision)
    index = CatalogIndex(load_catalog(root, args.split))
    builder = make_builder(run_cfg, repo_root, index, counter)
    samples = load_split(root, args.split, ids)
    order = {sid: i for i, sid in enumerate(ids)}
    samples.sort(key=lambda s: order[s.id])
    loader = ZaicLoader(root, args.split, EnvRegistry())
    result = build_rows(
        samples,
        builder,
        loader,
        counter,
        max_model_len=run_cfg.llm.max_model_len,
        max_result_chars=run_cfg.agent.max_result_chars,
        train_mode=train_mode,
        source="train" if train_mode else "eval",
        transcript_source=args.transcript_source,
        transcriber=transcriber,
        augment=Augment(
            args.error_recovery, args.oversample_hard, args.distractor_apps, args.dedup_max_k, args.mix_seed
        ),
        index=index,
        held_out=read_held_out(repo_root),
    )
    template = (repo_root / run_cfg.agent.system_prompt).read_bytes()
    meta = write_outputs(
        abs_path(args.out, repo_root),
        result,
        ids=ids,
        argv=list(argv if argv is not None else sys.argv),
        repo_root=repo_root,
        extra={
            "prompt_sha256": hashlib.sha256(template).hexdigest(),
            "split": args.split,
            "train_mode": train_mode,
            "transcript_source": args.transcript_source,
            "asr_cache": str(args.asr_cache) if args.asr_cache else None,
            "asr_ratio": args.asr_ratio if args.transcript_source == "mix" else None,
            "transcript_choice": dict(sorted(Counter(transcriber.chosen.values()).items()))
            if isinstance(transcriber, MixedTranscriber)
            else None,
            "augment": dataclasses.asdict(
                Augment(
                    args.error_recovery, args.oversample_hard, args.distractor_apps, args.dedup_max_k, args.mix_seed
                )
            )
            if train_mode
            else None,
            "rows_by_category": dict(sorted(Counter(r["category"] for r in result.rows).items())),
            "n_error_recovery_rows": sum(1 for r in result.rows if r["error_recovery"] and "copy" not in r),
            "n_oversample_copies": sum(1 for r in result.rows if "copy" in r),
            "token_counter": type(counter).__name__,
            "base_model": cfg.base.model,
            "base_revision": cfg.base.revision,
        },
    )
    log.info("sft_data_done", rows=meta["n_rows"], dropped=meta["n_dropped"], warnings=meta["n_replay_warnings"])
    return meta


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    meta = run(parse_args(argv), argv=argv)
    print(f"rows={meta['n_rows']} dropped={meta['n_dropped']} replay_warnings={meta['n_replay_warnings']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
