"""Manual QC / inspection tool — not part of the eval pipeline and not needed
to reproduce the paper; run by hand after precompute_sae_cache.py.

QC + spot-read for a precomputed SAE cache (CPU, login-side).

Two halves, per the mandatory qualitative-review convention:
  1. Aggregate QC: L0/FVU distributions, attention-sink stats (batch-top-k caches), and
     the Option-B equivalence check — every empty topk list should BE a sink position
     (the eval-time tool renders empty as "attention sink / unreadable", so non-sink
     empties would be mislabeled to the agent; expect ~0).
  2. Spot-read: N random rows printed end-to-end — every position, token text, and the
     top DISPLAY_TOP_K feature labels (with activations, which the agent never sees) —
     NO truncation, for manual review against the transcript content.

Usage (labels path = data/sae_labels/<np_model>_<np_source>.jsonl):
    python -m chive.binary_eval.agent.SAE.qc_sae_cache \
        --cache <run_dir>/sae_cache/<slug>.jsonl --labels data/sae_labels/....jsonl \
        --n-spot 12 > /path/to/spot_read.txt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from chive.binary_eval.agent.sae_tool import DISPLAY_TOP_K, load_sae_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--n-spot", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.cache)]
    labels = load_sae_labels(Path(args.labels))
    assert rows, f"empty cache {args.cache}"
    base_models = {r["base_model"] for r in rows}
    assert len(base_models) == 1, base_models
    tokenizer = AutoTokenizer.from_pretrained(base_models.pop())

    # ---- aggregate QC ----
    l0s = sorted(r["median_l0"] for r in rows)
    fvus = sorted(r["fvu"] for r in rows)
    print(f"[qc] {len(rows)} rows from {args.cache}")
    print(f"[qc] sae={rows[0]['sae']}  top_k={rows[0]['top_k']}  "
          f"capture_layer={rows[0]['capture_layer']}")
    print(f"[qc] median-of-medians L0={l0s[len(l0s)//2]:.1f}  "
          f"(p10={l0s[len(l0s)//10]:.1f} p90={l0s[9*len(l0s)//10]:.1f})")
    print(f"[qc] FVU median={fvus[len(fvus)//2]:.4f}  p90={fvus[9*len(fvus)//10]:.4f}  "
          f"max={fvus[-1]:.4f}")

    if "sink_positions" in rows[0]:
        n_sink = n_pos = n_pos0_masked = n_empty_nonsink = n_sink_nonempty = 0
        for r in rows:
            sinks = set(r["sink_positions"])
            n_sink += len(sinks - {0})
            n_pos += r["n_tokens"] - 1
            n_pos0_masked += 0 in sinks
            for pos, feats in r["topk"].items():
                if not feats and int(pos) not in sinks:
                    n_empty_nonsink += 1
                if feats and int(pos) in sinks:
                    n_sink_nonempty += 1
        print(f"[qc] sink rate (excl. position 0): {n_sink}/{n_pos} = {n_sink/n_pos:.4%} "
              f"(expected ~0.1%)")
        print(f"[qc] position 0 masked in {n_pos0_masked}/{len(rows)} rows "
              f"(expected ~all: first token is an attention sink)")
        print(f"[qc] empty-topk positions NOT sink-masked: {n_empty_nonsink} (expect ~0 — "
              f"the tool renders empty as 'attention sink')")
        print(f"[qc] sink-masked positions with non-empty topk: {n_sink_nonempty} (expect 0)")

    # ---- spot-read ----
    sample = random.Random(args.seed).sample(rows, min(args.n_spot, len(rows)))
    for r in sample:
        sinks = set(r.get("sink_positions", []))
        print("\n" + "#" * 100)
        print(f"# {r['prompt_id']}  n_tokens={r['n_tokens']}  median_l0={r['median_l0']:.1f}  "
              f"fvu={r['fvu']:.4f}  sink_positions={sorted(sinks)}")
        print("#" * 100)
        for pos in range(r["n_tokens"]):
            piece = tokenizer.decode([r["token_ids"][pos]])
            feats = r["topk"][str(pos)][:DISPLAY_TOP_K]
            mark = "  << SINK" if pos in sinks else ""
            print(f"\n[{pos}] token={piece!r}{mark}")
            if not feats:
                print("  (no features stored)")
            for rank, (feat, act) in enumerate(feats, start=1):
                print(f"  {rank}. [{act:8.2f}] {labels.get(feat, f'feature {feat} (unlabeled)')}")


if __name__ == "__main__":
    main()
