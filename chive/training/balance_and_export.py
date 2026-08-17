"""Balance the cf7 claim bank to equal cells and export the trainer's synthetic-data format.

Input: the cf7 claim bank (`bank.jsonl`, one unsigned claim per factor) from
chive/binary_eval/build_claims.py — the SAME builder the eval uses, so train and eval
claims are byte-identical in format (no separate cf generator).
Output: a `{metadata, results:[...]}` JSON whose rows are `example_type="counterfactual_prediction"`
with the fields chive/training/model_understanding.py reads (messages, chosen_completion,
chosen_completion_index, question, answer). Point a training config's `synthetic_data_paths` at it.

Baseline floor (--min-baseline-rate, optional): drop claims whose behavior barely fires originally
(k_baseline/n_baseline below the floor) — near-noise behaviors. Use 0.30 to match the 397B screening.

Disjointness (REQUIRED --eval-set): drop any claim whose ORIGINAL conversation (the `messages`
prompt) exactly content-hashes to an eval-set conversation, so no eval transcript leaks into the
training data. Exact hash only (the train and eval runs have independent prompt_id namespaces but
can draw the same underlying prompt, so an id-based exclusion would miss the overlap). Reported.

Balance: downsample the two cf7 cells (supported = TRUE, inert = FALSE) to the smaller (seeded) so
the set is 50/50 and the claim text — identical in form for both labels — carries no tell. Also
reports the rendered token-length distribution so max_seq_len can be set before training.

  python -m chive.training.balance_and_export \
      --in data/runs/qwen3_8b_50k_train_structured/binary_counterfactual_questions.jsonl \
      --eval-set data/runs/opus_investigates_qwen3_8b_5000_thorough_n30/eval_set_conc3_n1000.jsonl \
      --out data/training_data/binary_counterfactual_training_data.json --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Canonical answer-format suffix appended to a claim's `statement` to make the train/eval turn. The
# structured-claim eval (run_structured_claim_eval.py) imports this from here so train and eval phrase
# the claim identically (matters: a wording mismatch silently degrades the trained model's scores).
QUESTION_INSTRUCTION = '\n\nIs the CLAIM above true? Answer with a single word: "Yes" or "No".'

# cf7: one unsigned claim per factor, claim_type == factor_status. Balance the two cells
# (supported = TRUE, inert = FALSE) 50/50.
CELLS = ("supported", "inert")


def _conv_hash(messages: list[dict]) -> str:
    """Exact content hash of a conversation (whitespace/case-normalized), for train/eval disjointness."""
    text = " ".join(m["content"] for m in messages if isinstance(m.get("content"), str))
    return hashlib.md5(re.sub(r"\s+", " ", text.strip().lower()).encode()).hexdigest()


def to_synth_record(r: dict) -> dict:
    return {
        "example_type": "counterfactual_prediction",
        "prompt_id": r["prompt_id"],
        # claim_index makes the trainer's per-example id (f"{prompt_id}__cf_{experiment_index}") unique
        # across the multiple claims one investigation emits.
        "experiment_index": r["claim_index"],
        "messages": r["messages"],
        "chosen_completion": r["chosen_completion"],
        "chosen_completion_index": r["chosen_completion_index"],
        "question": r["statement"] + QUESTION_INSTRUCTION,
        "answer": "Yes" if r["is_true"] else "No",
        "verification_score": r["verification_score"],
        # provenance for audit
        "claim_id": r["claim_id"], "claim_type": r["claim_type"], "is_true": r["is_true"],
        "effect_pp": r["effect_pp"], "k_baseline": r["k_baseline"],
        "k_counterfactual": r["k_counterfactual"],
    }


def check_lengths(records: list[dict], model_name: str, max_seq_len: int) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    lens = []
    for rec in records:
        convo = (list(rec["messages"])
                 + [{"role": "assistant", "content": rec["chosen_completion"]}]
                 + [{"role": "user", "content": rec["question"]}]
                 + [{"role": "assistant", "content": rec["answer"]}])
        ids = tok.apply_chat_template(convo, tokenize=True, enable_thinking=False)
        lens.append(len(ids))
    lens.sort()
    n = len(lens)
    pct = lambda p: lens[min(n - 1, int(p * n))]
    over = sum(1 for x in lens if x > max_seq_len)
    print(f"[lengths] n={n}  p50={pct(.50)}  p90={pct(.90)}  p99={pct(.99)}  max={lens[-1]}")
    print(f"[lengths] > max_seq_len({max_seq_len}): {over} ({over/n:.1%}) — these truncate the claim off the end")
    if over / n > 0.02:
        print(f"*** WARNING: {over/n:.1%} of examples exceed max_seq_len={max_seq_len}; raise it before training. ***")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, required=True,
                   help="cf7 claim bank (bank.jsonl) from build_claims.py")
    p.add_argument("--eval-set", type=Path, required=True,
                   help="eval jsonl; any training claim whose conversation exactly content-hashes "
                        "to an eval conversation is dropped (train/eval disjointness)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-baseline-rate", type=float, default=0.0,
                   help="drop claims whose behavior occurs in < this fraction of baseline completions "
                        "(k_baseline/n_baseline). Use 0.30 to match the 397B screening floor and "
                        "exclude near-noise behaviors that only fire ~1/10 originally.")
    p.add_argument("--max-per-cell", type=int, default=None, help="optional cap; else use the smallest cell")
    p.add_argument("--model-name", default="Qwen/Qwen3-8B")
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--no-length-check", action="store_true")
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.inp)]
    print(f"[load] {len(rows)} claims from {args.inp}")

    # Baseline floor: drop claims whose behavior barely happens originally (sampling noise). For
    # consistency with the 397B screening, which required the behavior to occur >=30% of the time.
    if args.min_baseline_rate > 0:
        before = len(rows)
        rows = [r for r in rows if r["k_baseline"] / r["n_baseline"] >= args.min_baseline_rate]
        dropped = before - len(rows)
        print(f"[baseline] dropped {dropped}/{before} claims with baseline < "
              f"{args.min_baseline_rate:.0%} ({dropped / before:.1%}); {len(rows)} remain")

    # Disjointness: drop claims whose ORIGINAL conversation matches an eval conversation (exact hash).
    eval_hashes = {_conv_hash(r["messages"]) for r in (json.loads(l) for l in open(args.eval_set))
                   if r.get("messages")}
    before = len(rows)
    rows = [r for r in rows if _conv_hash(r["messages"]) not in eval_hashes]
    n_overlap = before - len(rows)
    print(f"[disjoint] {len(eval_hashes)} eval conversations; dropped {n_overlap}/{before} "
          f"eval-overlapping claims ({n_overlap / before:.1%}); {len(rows)} remain")

    by_cell: dict[str, list] = defaultdict(list)
    for r in rows:
        by_cell[r["claim_type"]].append(r)
    counts = {c: len(by_cell[c]) for c in CELLS}
    print(f"[balance] raw cells: {counts}")
    for c in CELLS:
        assert counts[c] > 0, f"cell {c} is EMPTY — cannot balance (need both supported and inert)"

    per_cell = min(counts.values())
    if args.max_per_cell is not None:
        per_cell = min(per_cell, args.max_per_cell)
    rng = random.Random(args.seed)
    kept = []
    for c in CELLS:
        pool = by_cell[c][:]
        rng.shuffle(pool)
        kept.extend(pool[:per_cell])
    rng.shuffle(kept)
    print(f"[balance] kept {per_cell}/cell -> {len(kept)} total; "
          f"is_true {sum(1 for r in kept if r['is_true'])}/{len(kept)}; "
          f"final cells {dict(Counter(r['claim_type'] for r in kept))}")

    records = [to_synth_record(r) for r in kept]
    if not args.no_length_check:
        check_lengths(records, args.model_name, args.max_seq_len)

    out = {
        "metadata": {
            "source": str(args.inp), "eval_set": str(args.eval_set),
            "min_baseline_rate": args.min_baseline_rate,
            "n_eval_overlap_dropped": n_overlap, "n_examples": len(records), "per_cell": per_cell,
            "seed": args.seed, "cells": CELLS,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[balance] wrote {len(records)} -> {args.out}")


if __name__ == "__main__":
    main()
