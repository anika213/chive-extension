"""
Build agentic / system-prompt prompt records and merge them into the WildChat pool.

Produces three per-source JSONLs in the SAME record schema as the WildChat pool
(build_prompt_pool.py), then shuffles all four sources (these three + the
existing WildChat pool) into one big JSONL.

Each agentic record is `messages = [{role: system}, {role: user}]`, ending at the
user turn (so the target model generates a completion exactly as in Stage 1). The
system prompt carries the persona / behavioral instructions and, for the tool
datasets, the tool definitions as TEXT inside the system message. We deliberately
do NOT pass structured `tools=` to the backend: with a text tool spec the model
emits its `<tool_call>{...}</tool_call>` as plain text in the completion (verified
against the live vLLM server), so the whole existing pipeline handles these with
no code changes (Stage 1 sends `messages` straight through; stages 2-4 already
render any role as `[ROLE]:`).

Sources (all length-filtered to [MIN_CHARS, MAX_CHARS] total chars):
  - hermes      : NousResearch/hermes-function-calling-v1 (func_calling_singleturn
                  + func_calling) -- rich, realistic multi-step tool tasks.
  - toolace     : Team-ACE/ToolACE -- kept ONLY when the system prompt's tool list
                  is a parseable JSON array (the HTML-<table> / bespoke-format rows
                  are skipped).
  - systemchat2 : cognitivecomputations/SystemChat-2.0 -- diverse behavioral /
                  persona system prompts (often conflicting with the request).

Usage (CPU-only, no GPU/model; set HF_HOME first):
    export HF_HOME=/workspace-vast/$USER/hf_cache
    .venv/bin/python chive/pipeline/build_agentic_pool.py \
        --wildchat data/runs/wildchat48m_clean_172k/prompts.jsonl \
        --out-dir data/runs/wildchat_plus_agentic
"""

import argparse
import json
import hashlib
import random
from pathlib import Path

from datasets import load_dataset

from chive import paths
from chive.pipeline.pool_common import read_jsonl, write_jsonl

SEED = 42
MIN_CHARS = 400
MAX_CHARS = 8000

# Train/eval-leakage blocklist (see scripts/build_eval_blocklist.py): conversations
# that appeared in ANY training dataset are removed from the pool as the FINAL build
# step — after dedup, shuffle, and id assignment — so a rebuild reproduces the shipped
# filtered pool exactly (surviving records keep their original, now-gapped ids).
BLOCKLIST_PATH = paths.DATA_ROOT / "eval_blocklist.json"


def _content_hash(messages: list) -> str:
    """Same normalization as scripts/build_eval_blocklist.py and the overlap audit."""
    normalized = [(m["role"], m["content"]) for m in messages]
    blob = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def apply_eval_blocklist(records: list[dict]) -> list[dict]:
    """Drop records whose conversation content-hash is blocklisted. Fails loudly if
    the blocklist file is missing — build the pool WITH the exclusion or not at all."""
    blocklist = json.loads(BLOCKLIST_PATH.read_text())
    hashes = set(blocklist["content_hashes"])
    kept = [r for r in records if _content_hash(r["messages"]) not in hashes]
    print(f"  eval blocklist: dropped {len(records) - len(kept)} of {len(records)} records "
          f"({len(hashes)} blocklisted conversations, {BLOCKLIST_PATH})")
    return kept


def _role(msg: dict) -> str:
    return msg.get("from") or msg.get("role")


def _val(msg: dict) -> str:
    return msg.get("value") or msg.get("content") or ""


def make_record(source: str, system: str, user: str) -> dict | None:
    """Build a pool record from a system prompt + first user message.

    Returns None if it falls outside the length band. Schema matches the WildChat
    pool (build_prompt_pool.py) plus a `source` field; `conversation_hash` is empty
    (agentic rows have no WildChat hash -- dedup, if ever needed, uses content).
    """
    system = system.strip()
    user = user.strip()
    assert system and user, f"{source}: empty system or user"
    total = len(system) + len(user)
    if total < MIN_CHARS or total > MAX_CHARS:
        return None
    return {
        "source": source,
        "conversation_hash": "",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "user_message": user,
        "n_turns": 1,
        "total_chars": total,
    }


def _first(conv: list[dict], roles: tuple[str, ...]) -> str | None:
    for m in conv:
        if _role(m) in roles:
            return _val(m)
    return None


def extract_json_tool_array(system: str) -> list | None:
    """Return the first balanced JSON array in `system` if it parses, else None.

    ToolACE stores its tools as a JSON array inside the system text for most rows,
    but some rows use an HTML <table> spec / single-quoted Python-repr instead.
    Those don't parse as JSON, so this is exactly the "keep only JSON tool list"
    filter the row needs.
    """
    start = system.find("[")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(system)):
        ch = system[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(system[start : j + 1])
                    except json.JSONDecodeError:
                        return None
                    return parsed if isinstance(parsed, list) else None
    return None


def build_hermes() -> list[dict]:
    records = []
    for cfg in ["func_calling_singleturn", "func_calling"]:
        ds = load_dataset("NousResearch/hermes-function-calling-v1", cfg, split="train")
        for row in ds:
            conv = row["conversations"]
            system = _first(conv, ("system",))
            user = _first(conv, ("human", "user"))
            if not system or not user:
                continue
            rec = make_record("hermes", system, user)
            if rec is not None:
                records.append(rec)
    return records


def build_toolace() -> tuple[list[dict], int]:
    records = []
    skipped_non_json = 0
    ds = load_dataset("Team-ACE/ToolACE", split="train")
    for row in ds:
        system = row["system"]
        user = _first(row["conversations"], ("human", "user"))
        if not system or not user:
            continue
        tools = extract_json_tool_array(system)
        if not tools or not all(isinstance(t, dict) and "name" in t for t in tools):
            skipped_non_json += 1
            continue  # HTML-table / bespoke-format row -- skip per spec
        rec = make_record("toolace", system, user)
        if rec is not None:
            records.append(rec)
    return records, skipped_non_json


def build_systemchat2() -> list[dict]:
    records = []
    ds = load_dataset("cognitivecomputations/SystemChat-2.0", split="train")
    for row in ds:
        msgs = row["messages"]
        system = _first(msgs, ("system",))
        user = _first(msgs, ("user", "human"))
        if not system or not user:
            continue
        rec = make_record("systemchat2", system, user)
        if rec is not None:
            records.append(rec)
    return records


def validate(records: list[dict], source: str) -> None:
    """Loud invariant checks; the pipeline assumes all of these."""
    for r in records:
        msgs = r["messages"]
        assert msgs[0]["role"] == "system", f"{source}: first msg not system"
        assert msgs[-1]["role"] == "user", f"{source}: does not end with user"
        assert r["user_message"] == msgs[-1]["content"], f"{source}: user_message mismatch"
        assert MIN_CHARS <= r["total_chars"] <= MAX_CHARS, f"{source}: out of length band"


def merge_and_shuffle(source_files: list[Path], wildchat: Path, out_path: Path) -> None:
    """Load WildChat + the agentic JSONLs, shuffle, reassign ids, write one pool."""
    random.seed(SEED)
    merged: list[dict] = []

    wc_lines = read_jsonl(wildchat)
    for r in wc_lines:
        r.setdefault("source", "wildchat")
    merged.extend(wc_lines)
    print(f"  loaded {len(wc_lines):>7} from {wildchat} (source=wildchat)")

    for f in source_files:
        recs = read_jsonl(f)
        merged.extend(recs)
        print(f"  loaded {len(recs):>7} from {f}")

    random.shuffle(merged)
    for i, r in enumerate(merged):
        r["id"] = f"prompt_{i:06d}"

    # AFTER id assignment: ids stay stable (with gaps) across the exclusion.
    merged = apply_eval_blocklist(merged)

    write_jsonl(merged, out_path)

    from collections import Counter
    comp = Counter(r["source"] for r in merged)
    n = len(merged)
    agentic = n - comp["wildchat"]
    print(f"\nMerged pool: {n} records")
    for src, c in comp.most_common():
        print(f"  {src:14s} {c:>7} ({c / n:.1%})")
    print(f"  agentic total  {agentic:>7} ({agentic / n:.1%})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build agentic pool records + merge with WildChat")
    ap.add_argument("--wildchat", required=True, help="Existing WildChat pool JSONL to merge in")
    ap.add_argument("--out-dir", required=True, help="Directory for the per-source JSONLs + merged pool")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    wildchat = Path(args.wildchat)
    assert wildchat.exists(), f"WildChat pool not found: {wildchat}"

    print("Building hermes...")
    hermes = build_hermes()
    validate(hermes, "hermes")
    print("Building toolace (JSON-tool-list rows only)...")
    toolace, skipped = build_toolace()
    validate(toolace, "toolace")
    print(f"  skipped {skipped} ToolACE rows without a parseable JSON tool list")
    print("Building systemchat2...")
    systemchat2 = build_systemchat2()
    validate(systemchat2, "systemchat2")

    print("\nWriting per-source JSONLs:")
    f_hermes = out_dir / "hermes.jsonl"
    f_toolace = out_dir / "toolace.jsonl"
    f_systemchat2 = out_dir / "systemchat2.jsonl"
    write_jsonl(hermes, f_hermes)
    write_jsonl(toolace, f_toolace)
    write_jsonl(systemchat2, f_systemchat2)

    print("\nMerging + shuffling all four sources:")
    merge_and_shuffle([f_hermes, f_toolace, f_systemchat2], wildchat, out_dir / "prompts.jsonl")


if __name__ == "__main__":
    main()
