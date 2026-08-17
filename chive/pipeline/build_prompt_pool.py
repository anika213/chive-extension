"""
Build the canonical model-understanding prompt pool from WildChat-4.8M.

Streams WildChat, applies all filters + template dedup, and writes a single
line-delimited JSONL pool (one prompt per line). The pool is **model-agnostic**:
it stores `messages` only (no tokenization), so the same pool drives any target
model. Tokenization, if ever needed, happens at generation time in Stage 1.

This is "part 1 of Stage 0": run it ONCE to (re)build the pool. Per-run Stage 0
just slices the first N rows of the pool — see run_pipeline.run_stage0.

Sourcing/quality design:
  - Source: WildChat-4.8M. Its parquet shards are TIME-ORDERED, and ~half the
    corpus is automated/template API traffic (Midjourney generators, data-
    labeling bots, recruiter bots, etc.) concentrated in particular periods.
  - SHARD-SPREAD sampling: we read a quota from N evenly-spaced shards (not a
    sequential prefix of shard 0), so the sample spans the full 2023->2025 range
    instead of being biased to one period.
  - TEMPLATE DEDUP: a global cap of <= PREFIX_CAP prompts per normalized
    PREFIX_CHARS-char prefix. The bot/template families share prefixes, so this
    collapses them to a few representatives each (removing ~50% template spam)
    while keeping all genuinely-unique human prompts.

Supports both single-turn and multi-turn conversations, filtered by total
character count. The char filter (e.g. 400-8000) also bounds prompt length, so
there is no separate token-length cap (that would require binding to a model).

Usage:
    .venv/bin/python chive/pipeline/build_prompt_pool.py \
        --n-prompts 200000 --min-chars 400 --max-chars 8000 --max-turns 5 \
        --prefix-cap 5 \
        --jsonl-out data/runs/wildchat48m_clean_172k/prompts.jsonl
"""

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset

from chive.pipeline.pool_common import content_hash, read_jsonl, write_jsonl

CHAT_DATASET = "allenai/WildChat-4.8M"
SEED = 42

# Sourcing/dedup defaults (experiment-semantic — see module docstring).
STREAM_OVERSHOOT = 1.3    # collect this x target then sample down (covers
                          # low-yield shards)
PREFIX_CHARS = 100        # prefix length for template dedup
PREFIX_CAP = 5            # max prompts kept per normalized prefix

_WS = re.compile(r"\s+")


def prefix_key(messages: list[dict], prefix_chars: int) -> str:
    """Normalized prefix of the concatenated user text, for template dedup."""
    txt = " ".join(m["content"] for m in messages if m["role"] == "user")
    return _WS.sub(" ", txt).strip()[:prefix_chars]


def list_shard_files() -> list[str]:
    """List the dataset's train parquet shard paths, sorted by index."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(CHAT_DATASET, repo_type="dataset")
    parquet = sorted(f for f in files if f.endswith(".parquet") and "/train-" in f)
    if not parquet:
        raise RuntimeError(f"No train parquet shards found in {CHAT_DATASET}")
    return parquet


def spread_indices(n_total: int, k: int) -> list[int]:
    """k evenly-spaced indices across range(n_total)."""
    if k <= 1:
        return [0]
    if k >= n_total:
        return list(range(n_total))
    return sorted({round(i * (n_total - 1) / (k - 1)) for i in range(k)})


def load_exclude_hashes(exclude_files: list[str]) -> tuple[set[str], set[str]]:
    """Load conversation_hashes and content hashes to exclude.

    Returns (conversation_hashes, content_hashes). Content hashes catch
    WildChat duplicates that share identical messages but have different
    conversation_hash values. Accepts both completions JSON
    ({"prompts": [...]}) and a prompts JSONL slice (one record per line).
    """
    conv_hashes = set()
    content_hashes = set()
    for path in exclude_files:
        p = Path(path)
        if p.suffix == ".jsonl":
            records = read_jsonl(p)
        else:
            records = json.loads(p.read_text())["prompts"]
        for rec in records:
            h = rec.get("conversation_hash", "")
            if h:
                conv_hashes.add(h)
            msgs = rec.get("messages")
            if msgs:
                content_hashes.add(content_hash(msgs))
    print(f"Loaded {len(conv_hashes)} conversation_hashes and "
          f"{len(content_hashes)} content hashes to exclude "
          f"from {len(exclude_files)} file(s)")
    return conv_hashes, content_hashes


def stream_prompts(
    min_chars: int,
    max_chars: int,
    max_turns: int,
    target: int,
    n_sample_shards: int | None = None,
    prefix_cap: int = PREFIX_CAP,
    prefix_chars: int = PREFIX_CHARS,
    exclude_conv_hashes: set[str] | None = None,
    exclude_content_hashes: set[str] | None = None,
):
    """Yield eligible, template-deduped prompts spread across the corpus.

    Spreads across `n_sample_shards` evenly-spaced parquet shards (default: ALL
    shards, since they are time-ordered) and pulls a per-shard quota sized so the
    quotas sum to ~`target` x an overshoot. This both hits the target and forces
    coverage of every sampled shard before stopping — set the target and forget
    it. Applies Stage-0 eligibility filters and a global per-prefix cap so no
    template family contributes more than `prefix_cap` prompts.
    """
    all_files = list_shard_files()
    n_total = len(all_files)
    k = n_sample_shards if n_sample_shards else n_total
    idx = spread_indices(n_total, k)
    shard_files = [all_files[i] for i in idx]
    per_shard_quota = max(1, math.ceil(target * STREAM_OVERSHOOT / len(shard_files)))
    print(f"Sampling up to {per_shard_quota}/shard across {len(shard_files)} of "
          f"{n_total} shards (target {target}); cap {prefix_cap}/{prefix_chars}-char prefix")

    prefix_counts: dict[str, int] = {}
    n_excluded_conv = 0
    n_excluded_content = 0
    n_capped = 0
    n_yielded = 0

    for sf in shard_files:
        got = 0
        for row in load_dataset(CHAT_DATASET, split="train", streaming=True, data_files=sf):
            if row.get("language") != "English":
                continue
            if row.get("redacted", False):
                continue

            conv = row["conversation"]
            if not conv or conv[0]["role"] != "user":
                continue

            # Check conversation_hash exclusion early (before expensive processing)
            conv_hash = row.get("conversation_hash", "")
            if exclude_conv_hashes and conv_hash in exclude_conv_hashes:
                n_excluded_conv += 1
                continue

            # Build messages, ensure ends with user turn
            messages = [{"role": m["role"], "content": m["content"]} for m in conv]
            while messages and messages[-1]["role"] != "user":
                messages.pop()
            if not messages:
                continue

            # Filter redacted placeholders
            if "NAME_" in " ".join(m["content"] for m in messages):
                continue

            total_chars = sum(len(m["content"]) for m in messages)
            if total_chars < min_chars or total_chars > max_chars:
                continue

            n_user = sum(1 for m in messages if m["role"] == "user")
            if n_user > max_turns:
                continue

            # Check content hash exclusion (catches WildChat dupes with different
            # conversation_hash but identical messages)
            if exclude_content_hashes and content_hash(messages) in exclude_content_hashes:
                n_excluded_content += 1
                continue

            # Template dedup: cap occurrences per normalized prefix (global)
            pk = prefix_key(messages, prefix_chars)
            prefix_counts[pk] = prefix_counts.get(pk, 0) + 1
            if prefix_counts[pk] > prefix_cap:
                n_capped += 1
                continue

            yield {
                "conversation_hash": conv_hash,
                "messages": messages,
                "user_message": messages[-1]["content"],
                "n_turns": n_user,
                "total_chars": total_chars,
            }
            got += 1
            n_yielded += 1
            if got >= per_shard_quota:
                break  # shard quota met; move to next shard

    print(f"Yielded {n_yielded} prompts; dropped {n_capped} by prefix cap "
          f"({len(prefix_counts)} distinct prefixes)")
    if exclude_conv_hashes or exclude_content_hashes:
        print(f"Excluded {n_excluded_conv} by conversation_hash, "
              f"{n_excluded_content} by content hash")


def build_pool(
    n_prompts: int,
    jsonl_out: str,
    min_chars: int,
    max_chars: int,
    max_turns: int,
    n_sample_shards: int | None = None,
    prefix_cap: int = PREFIX_CAP,
    prefix_chars: int = PREFIX_CHARS,
    exclude_files: list[str] | None = None,
):
    """Stream WildChat, dedup, sample down to ~n_prompts, write a JSONL pool."""
    random.seed(SEED)

    exclude_conv_hashes = None
    exclude_content_hashes = None
    if exclude_files:
        exclude_conv_hashes, exclude_content_hashes = load_exclude_hashes(exclude_files)

    print(f"Streaming {CHAT_DATASET} (target {n_prompts} prompts)...")
    print(f"Filters: {min_chars}-{max_chars} total chars, <= {max_turns} turns")

    all_candidates = list(stream_prompts(
        min_chars, max_chars, max_turns,
        target=n_prompts,
        n_sample_shards=n_sample_shards,
        prefix_cap=prefix_cap,
        prefix_chars=prefix_chars,
        exclude_conv_hashes=exclude_conv_hashes,
        exclude_content_hashes=exclude_content_hashes,
    ))
    print(f"Found {len(all_candidates)} eligible prompts")

    # random.sample returns a shuffled selection (any sub-slice is itself a valid
    # random sample), so the first N rows of the pool are a representative sample.
    n_sample = min(n_prompts, len(all_candidates))
    if n_sample < n_prompts:
        print(f"WARNING: only {n_sample} prompts available (< requested {n_prompts}). "
              f"Corpus exhausted at these filters/cap — loosen --max-chars/--min-chars "
              f"or accept fewer (raising --prefix-cap re-adds near-duplicate templates).")
    selected = random.sample(all_candidates, n_sample)

    for i, c in enumerate(selected):
        c["id"] = f"prompt_{i:05d}"

    from collections import Counter
    turn_dist = Counter(c["n_turns"] for c in selected)
    print(f"Selected {n_sample} prompts:")
    for k in sorted(turn_dist):
        print(f"  {k}-turn: {turn_dist[k]}")

    write_jsonl(selected, jsonl_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the model-understanding prompt pool")
    parser.add_argument("--n-prompts", type=int, required=True)
    parser.add_argument("--jsonl-out", type=str, required=True,
                        help="Output JSONL pool path (one prompt per line, head-able).")
    parser.add_argument("--min-chars", type=int, required=True)
    parser.add_argument("--max-chars", type=int, required=True)
    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--n-sample-shards", type=int, default=None,
                        help="how many evenly-spaced corpus shards to spread across "
                             "(default: all shards; lower it for fast small dev runs)")
    parser.add_argument("--prefix-cap", type=int, default=PREFIX_CAP,
                        help="max prompts kept per normalized prefix (template dedup)")
    parser.add_argument("--prefix-chars", type=int, default=PREFIX_CHARS,
                        help="prefix length (chars) used for template dedup")
    parser.add_argument(
        "--exclude", nargs="+", default=None,
        help="Completions JSON / prompts JSONL file(s) to deduplicate against",
    )
    args = parser.parse_args()
    build_pool(
        args.n_prompts, args.jsonl_out,
        args.min_chars, args.max_chars, args.max_turns,
        n_sample_shards=args.n_sample_shards,
        prefix_cap=args.prefix_cap,
        prefix_chars=args.prefix_chars,
        exclude_files=args.exclude,
    )
    # datasets streaming (pyarrow + aiohttp across many shards) can raise a fatal
    # error during interpreter finalization (a GIL teardown race), making this
    # subprocess exit non-zero even though the run succeeded. Flush and hard-exit
    # to guarantee a clean 0. (Only reached on success; real errors still raise.)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
