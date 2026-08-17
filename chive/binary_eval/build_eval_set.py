"""Build the canonical eval set: concreteness-filter -> segmentation check -> balance
-> cap, deterministic.

This is the SINGLE auditable definition of which claims the plain/AO/NLA/SAE answerers
score. It consumes the cf7 bank + mechanism_concreteness.jsonl (both produced
upstream); verified>=8 is already guaranteed by the bank build. It filters to
concreteness>=min, then ASSERTS every surviving transcript segments cleanly over the
target tokenizer (a hard requirement, not a filter — the agent arms read activations at
token positions selected over the segmented [S#] view, so an unsegmentable transcript
can't be scored by the tools at all; this is 0% on every current bank, so a failure means
something upstream changed — template/length/tokenizer — and the build STOPS rather than
silently shrinking the set). Then balances 50/50 supported/inert (TRUE/FALSE) — cf7 claims
are unsigned and identical in form, so there is no polarity to balance within — then caps
at --n.

Both the plain and agent runs point --bank at this one file, so "same claims" is
structural, not something to reproduce twice (on-the-fly balancing in each arm could
drift and silently compare different claim sets).

Usage:
    python -m chive.binary_eval.build_eval_set \
        --run-dir <run> --base-model <target model> --min-concreteness 3 --n 1000 --seed 0
    # -> <run>/eval_set_conc3_n1000.jsonl
"""

import argparse
import json
import random
from pathlib import Path

from chive.binary_eval.agent.transcript_segments import segment_transcript


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--base-model", required=True,
                   help="the run's target model (tokenizer for the segmentability filter — "
                        "MUST be the model that produced the bank's transcripts)")
    p.add_argument("--bank", type=Path, default=None,
                   help="claim bank (default <run>/binary_counterfactual_questions.jsonl)")
    p.add_argument("--concreteness", type=Path, default=None,
                   help="default <run>/mechanism_concreteness.jsonl")
    p.add_argument("--min-concreteness", type=int, default=3)
    p.add_argument("--n", type=int, default=1000, help="max total claims (50/50 TRUE/FALSE)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--max-unsegmentable-rate", type=float, default=0.0,
                   help="tolerate DROPPING up to this fraction of investigations that fail "
                        "segmentation, instead of raising (default 0.0 = strict). A failure rate "
                        "above this still raises — the guard's point is to catch systemic upstream "
                        "changes, not a rare pathological transcript.")
    args = p.parse_args()

    from transformers import AutoTokenizer

    run = args.run_dir
    bank_path = args.bank or run / "binary_counterfactual_questions.jsonl"
    conc_path = args.concreteness or run / "mechanism_concreteness.jsonl"
    out_path = args.out or run / f"eval_set_conc{args.min_concreteness}_n{args.n}.jsonl"

    conc = {json.loads(l)["prompt_id"]: json.loads(l)["mechanism_concreteness"]
            for l in open(conc_path) if l.strip()}
    rows = [json.loads(l) for l in open(bank_path)]
    n_missing = sum(1 for r in rows if r["prompt_id"] not in conc)
    if n_missing:
        print(f"[eval_set] WARNING: {n_missing}/{len(rows)} bank claims have no concreteness "
              f"score (treated as below threshold)")
    kept = [r for r in rows if conc.get(r["prompt_id"], 0) >= args.min_concreteness]
    print(f"[eval_set] {len(rows)} bank claims -> {len(kept)} at concreteness>={args.min_concreteness}")

    # Segmentation is a HARD REQUIREMENT, not a filter: the agent arms read activations at
    # token positions the model selects over the segmented [S#] view, so a transcript that
    # can't segment can't be scored by the tools at all (and would silently desync the arms'
    # sets). This is 0% on every current bank, so rather than quietly dropping rows we RAISE:
    # a failure means something upstream changed (chat template, prompt length, tokenizer) and
    # the build should stop so it gets investigated, not ship a shrunken set. Checked once per
    # prompt_id (claims share transcripts). Add an explicit --max-unsegmentable-rate escape
    # hatch only if a model ever legitimately needs to tolerate drops.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    failed: dict[str, str] = {}
    checked: set[str] = set()
    for r in kept:
        pid = r["prompt_id"]
        if pid in checked:
            continue
        checked.add(pid)
        try:
            segment_transcript(tokenizer, r["messages"], r["chosen_completion"])
        except AssertionError as e:
            failed[pid] = str(e)
    if failed:
        rate = len(failed) / len(checked)
        sample = "\n".join(f"  {pid}: {msg}" for pid, msg in list(failed.items())[:5])
        if rate <= args.max_unsegmentable_rate:
            print(f"[eval_set] WARNING: DROPPING {len(failed)}/{len(checked)} investigations "
                  f"({rate:.1%}) that failed segmentation (<= --max-unsegmentable-rate "
                  f"{args.max_unsegmentable_rate:.1%}) — these can't be scored by the agent arms; "
                  f"the remaining set stays aligned across arms.\n{sample}")
            kept = [r for r in kept if r["prompt_id"] not in failed]
        else:
            raise RuntimeError(
                f"segmentation failed for {len(failed)}/{len(checked)} investigations "
                f"({rate:.1%}) over {args.base_model}, exceeding --max-unsegmentable-rate "
                f"({args.max_unsegmentable_rate:.1%}) — these transcripts cannot be scored by the "
                f"agent arms. Something upstream changed (template/length/tokenizer); investigate "
                f"before building the eval set.\n{sample}")
    else:
        print(f"[eval_set] segmentation check ({args.base_model}): all {len(checked)} "
              f"investigations OK")

    # 50/50 true/false (supported vs inert), capped to n//2 per label. cf7 claims are unsigned and
    # identical in form, so there is no polarity to balance within — just the label.
    rng = random.Random(args.seed)
    per_label = args.n // 2
    T = [r for r in kept if r["is_true"]]
    F = [r for r in kept if not r["is_true"]]
    k = min(per_label, len(T), len(F))
    out = rng.sample(T, k) + rng.sample(F, k)
    print(f"  take {k}T/{k}F  (available {len(T)}T/{len(F)}F; cap/label {per_label})")
    rng.shuffle(out)

    with open(out_path, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    nt = sum(r["is_true"] for r in out)
    print(f"[eval_set] wrote {len(out)} claims ({nt}T/{len(out) - nt}F, 50/50) -> {out_path}")


if __name__ == "__main__":
    main()
