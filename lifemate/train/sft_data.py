"""Build SFT rows exactly as the agent sees them at run time (SPEC 6.2.1).

Prompt from the run's PromptBuilder (train mode strips held-out apps), gold transcript, gold calls replayed through
a fresh zaic_tools env per sample for byte-exact `KẾT QUẢ` turns.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tomllib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from lifemate.agent.prompt import HeldOutLeak, PromptBuilder, truncate_result
from lifemate.agent.protocol import DONE_TEXT, call_text
from lifemate.agent.tooldocs import ASK_TOOL, CatalogIndex
from lifemate.backends.asr import GoldTranscriber
from lifemate.core.config import AgentConfig, ConfigError, LLMConfig, load_profile
from lifemate.core.envinfo import code_version
from lifemate.core.jsonutil import canonical_json
from lifemate.core.paths import REPO_ROOT
from lifemate.core.tokens import HFTokenCounter, SimpleTokenCounter
from lifemate.core.types import Sample, TokenCounter, Transcriber
from lifemate.io.data import load_split, read_ids
from lifemate.io.zaic import EnvRegistry, ZaicLoader, load_catalog
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


def _transcriber(source: str) -> Transcriber:
    if source == "gold":
        return GoldTranscriber()
    raise NotImplementedError("transcript_source='asr' needs the ASR model (SPEC 6.1, 6.2.1 item 2)")


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
) -> None:
    """Append one row (or one dropped entry) for `sample` to `result`."""
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
    for i, call in enumerate(calls):
        name, arguments = str(call["name"]), dict(call.get("arguments") or {})
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
    n_tokens = counter.count(messages)
    if n_tokens > max_model_len:
        result.dropped.append((sample.id, f"too_long:{n_tokens}>{max_model_len}"))
        return
    result.shrink_levels[level] += 1
    result.rows.append(
        {
            "id": sample.id,
            "source": source,
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
) -> BuildResult:
    """Rows for every sample in order; one fresh env per sample from `loader`."""
    transcriber = _transcriber(transcript_source)
    result = BuildResult()
    for sample in samples:
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
    ap.add_argument("--transcript-source", choices=("gold", "asr"), default="gold")
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
    _transcriber(args.transcript_source)  # fail fast for "asr"
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
