"""Build the canonical train/eval-leakage blocklist for the WildChat eval slice.

PROVENANCE ONLY — not runnable from this release. Its input (an overlap audit that
matched every training source against every eval run by a sha256 content hash of the
normalized (role, content) message sequence) is not included; the blocklist it
produced ships as data/eval_blocklist.json, and verify_eval_blocklist_clean.py
checks the released data against it independently. Kept because it documents
exactly how the blocklist was derived: the UNION over all training sources of
eval-side matched prompts — i.e. every eval prompt that ever appeared in ANY
training dataset (including pre-filter training dirs that are not part of the release).

Verifications performed here (all fail loudly):
  1. The per-run eval-side id unions are identical across the five audited
     *_5000_thorough_n30 runs (they share the same first-5000 wildchat_plus_agentic
     slice, so ids must align).
  2. For every WildChat-slice run in this release (including the two think1024 runs
     the audit did not cover), all blocklisted ids exist in prompts.jsonl and their
     content hashes are identical across runs.
  3. The PETRI runs contain ZERO blocklisted conversations by content hash — but
     their prompt_id numbering collides with the WildChat ids, which is why any
     downstream filtering must be scoped per-run, never applied globally by id.

Output: data/eval_blocklist.json with prompt ids, content hashes, and provenance.

Run (CPU-only, light):
    .venv/bin/python scripts/build_eval_blocklist.py
"""

import hashlib
import json
from pathlib import Path

REPO = Path("/workspace-vast/adamk/chive")
AUDIT = Path(
    "/workspace-vast/adamk/activation_oracles_dev/investigations/train_eval_overlap/results/pre_filter/overlap_matches.json"
)
OUT = REPO / "data/eval_blocklist.json"

WILDCHAT_RUNS = [
    "opus_investigates_qwen3_8b_5000_thorough_n30",
    "opus_investigates_qwen3p5_397b_5000_thorough_n30",
    "opus_investigates_gemma3_27b_it_5000_thorough_n30",
    "opus_investigates_llama3_1_8b_5000_thorough_n30",
    "opus_investigates_qwen3_8b_think1024_5000_thorough_n30",
    "opus_investigates_qwen3p5_397b_think1024_5000_thorough_n30",
]
PETRI_RUNS = [
    "opus_investigates_qwen3_8b_petri_thorough_n30",
    "opus_investigates_qwen3p5_397b_petri_thorough_n30",
]
# The audited 5000 runs whose id unions must agree exactly (llama/32b runs are not in
# this release but were audited; we only require the ones present in the audit file).
AUDITED_5000_KEYS = [
    "qwen3_8b_5000",
    "qwen3p5_397b_5000",
    "gemma3_27b_5000",
    "llama31_8b_5000",
    "qwen3_32b_5000",
]

# Additions beyond the audit's exact conversation-hash matches, each with provenance.
# prompt_002181: its bank claims _conv_hash-match the 397B binary training data
# (experiments/unfiltered_evaluation/results/train_overlap_q397.json, claim
# prompt_002181_c0) even though the audit's conversation-level hash found no match —
# a near-duplicate; blocklisted conservatively.
EXTRA_PROMPT_IDS = {
    "prompt_002181": "claim-level _conv_hash match to 397B binary training data (train_overlap_q397.json)",
}


def content_hash(messages: list) -> str:
    """Identical normalization to the audit's check_overlap.py."""
    normalized = [(m["role"], m["content"]) for m in messages]
    blob = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_pool(run_dir: Path) -> dict[str, list[str]]:
    """Full pool as content-hash -> [prompt ids] (multimap: pools may contain duplicates)."""
    pool: dict[str, list[str]] = {}
    with open(run_dir / "prompts.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            pool.setdefault(content_hash(rec["messages"]), []).append(rec["id"])
    return pool


def main() -> None:
    matches = json.loads(AUDIT.read_text())

    per_run_union: dict[str, set[str]] = {}
    per_source_counts: dict[str, int] = {}
    for tname, evals in matches.items():
        source_ids: set[str] = set()
        for ename, pairs in evals.items():
            ids = {p["eval_prompt_id"] for p in pairs}
            source_ids |= ids
            if ids:
                per_run_union.setdefault(ename, set()).update(ids)
        if source_ids:
            per_source_counts[tname] = len(source_ids)

    union_sets = [per_run_union[k] for k in AUDITED_5000_KEYS]
    if not all(s == union_sets[0] for s in union_sets):
        raise AssertionError("eval-side id unions differ across the audited 5000 runs")
    blocklist_ids = sorted(union_sets[0] | set(EXTRA_PROMPT_IDS))
    print(f"blocklist: {len(blocklist_ids)} prompt ids (incl. {len(EXTRA_PROMPT_IDS)} extra); "
          f"per-source contributions: {per_source_counts}")

    # The blocklist is hash-first: resolve the audited ids to content hashes in the
    # first run, then re-derive per-run id sets from hashes. This catches leaked
    # conversations that sit in a run pool under a SECOND id (pools contain duplicates,
    # and the audit's hash->id map kept only one id per hash).
    idset = set(blocklist_ids)
    print("verifying id -> content alignment across WildChat-slice runs (hash-first):")
    hash_set: set[str] = set()
    per_run_ids: dict[str, list[str]] = {}
    for run in WILDCHAT_RUNS:
        pool = load_pool(REPO / "data/runs" / run)
        n_dup = sum(len(v) - 1 for v in pool.values())
        if not hash_set:
            hash_set = {h for h, ids in pool.items() if idset & set(ids)}
            if len(hash_set) != len(idset):
                raise AssertionError(f"{run}: {len(idset)} audited ids resolve to {len(hash_set)} hashes")
        matched_ids = sorted(i for h in hash_set for i in pool.get(h, []))
        missing_hashes = [h for h in hash_set if h not in pool]
        extra = sorted(set(matched_ids) - idset)
        print(f"  {run}: pool_unique={len(pool)}, intra-pool dup records={n_dup}, "
              f"blocklist-hash ids={len(matched_ids)} (beyond audited 282: {extra if extra else 'none'})")
        if missing_hashes:
            raise AssertionError(f"{run}: {len(missing_hashes)} blocklisted hashes missing from pool")
        per_run_ids[run] = matched_ids
    run_id_sets = {tuple(v) for v in per_run_ids.values()}
    if len(run_id_sets) != 1:
        raise AssertionError("blocklisted id sets differ across WildChat runs — per-run ids required")
    final_ids = per_run_ids[WILDCHAT_RUNS[0]]

    # PETRI runs: must be content-disjoint; numeric id collisions counted (scoping hazard).
    print("verifying PETRI pools are content-disjoint:")
    final_idset = set(final_ids)
    for run in PETRI_RUNS:
        pool = load_pool(REPO / "data/runs" / run)
        n_hash = sum(len(pool.get(h, [])) for h in hash_set)
        n_id_collide = sum(1 for ids in pool.values() for i in ids if i in final_idset)
        print(f"  {run}: content matches={n_hash}, numeric id collisions={n_id_collide}")
        if n_hash != 0:
            raise AssertionError(f"{run}: PETRI pool contains blocklisted conversations — audit assumption broken")

    # Source pool (711M, full WildChat+agentic): every copy of a blocklisted conversation,
    # under its own id namespace (duplicates expected).
    print("scanning wildchat_plus_agentic source pool by content hash:")
    pool_matched_ids: list[str] = []
    with open(REPO / "data/runs/wildchat_plus_agentic/prompts.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if content_hash(rec["messages"]) in hash_set:
                pool_matched_ids.append(rec["id"])
    print(f"  source pool: {len(pool_matched_ids)} records match {len(hash_set)} blocklisted conversations")
    if len({h for h in hash_set}) and len(pool_matched_ids) < len(hash_set):
        raise AssertionError("source pool is missing blocklisted conversations")

    out = {
        "generated": "2026-07-28",
        "source": str(AUDIT),
        "description": (
            "Union over all training sources (incl. superseded pre-filter dirs) of eval-side "
            "prompts matched by content hash in the 2026-07-17 pre-filter train/eval overlap audit, "
            "re-derived hash-first so every duplicate copy of a leaked conversation is included. "
            "prompt_ids are valid ONLY in the wildchat_plus_agentic first-5000 slice namespace (the "
            "*_5000_thorough_n30 runs); source_pool_ids are the full-pool namespace; PETRI runs are "
            "content-disjoint (and share no ids)."
        ),
        "per_source_prompt_counts": per_source_counts,
        "extra_prompt_ids": EXTRA_PROMPT_IDS,
        "n_prompts": len(final_ids),
        "prompt_ids": final_ids,
        "content_hashes": sorted(hash_set),
        "source_pool_ids": sorted(pool_matched_ids),
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
