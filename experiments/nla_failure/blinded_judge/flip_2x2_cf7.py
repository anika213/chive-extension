"""The 2x2 flip grid for the paper: NLA-vs-plain answer flips split by factor_status.

Pure post hoc on the cf7 final-eval result files (the run the paper reports). A "flip"
is a claim where the NLA arm's thresholded answer (prob_true >= 0.5) differs from the
plain (transcript-only) arm's. Each flip either fixes a plain error (w->r) or breaks a
plain success (r->w); we split those by whether the claim's edited factor really drives
the behavior (supported) or not (inert).

Run: python flip_2x2_cf7.py --plain <plain.jsonl> --nla <nla.jsonl> --eval-set <eval.jsonl>
"""

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    return {json.loads(l)["claim_id"]: json.loads(l) for l in open(path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", type=Path, required=True)
    ap.add_argument("--nla", type=Path, required=True)
    ap.add_argument("--eval-set", type=Path, required=True)
    ap.add_argument("--quality-slice-run-dir", type=Path, default=None,
                    help="restrict to the paper's primary population (concreteness ∩ "
                         "cf-correct ∩ not-confounded) for this run dir, via "
                         "experiments.common.gold_slice.quality_ids")
    args = ap.parse_args()

    plain, nla, ev = load(args.plain), load(args.nla), load(args.eval_set)
    common = set(plain) & set(nla) & set(ev)
    if args.quality_slice_run_dir:
        from experiments.common.gold_slice import quality_ids
        common &= quality_ids(str(args.quality_slice_run_dir))
    common = sorted(common)
    print(f"n={len(common)} common claims"
          + (" (quality slice)" if args.quality_slice_run_dir else "") + "\n")

    print(f"{'':12s} {'n':>5s} {'flips':>6s} {'flip%':>6s} {'w->r':>5s} {'r->w':>5s} "
          f"{'net':>5s} {'->no-effect':>12s} {'plain acc':>10s} {'nla acc':>8s}")
    for fs in ("supported", "inert", None):
        ids = [c for c in common if fs is None or ev[c]["factor_status"] == fs]
        wr = rw = toward_none = 0
        for c in ids:
            # every claim in this eval set asserts "changes", so pred_true=False means
            # the arm predicts the edit has no effect
            p_ok, n_ok = plain[c]["correct"], nla[c]["correct"]
            if plain[c]["pred_true"] != nla[c]["pred_true"]:
                wr += n_ok
                rw += p_ok
                toward_none += not nla[c]["pred_true"]
        flips = wr + rw
        pacc = sum(plain[c]["correct"] for c in ids) / len(ids)
        nacc = sum(nla[c]["correct"] for c in ids) / len(ids)
        name = fs or "ALL"
        print(f"{name:12s} {len(ids):5d} {flips:6d} {flips / len(ids):6.1%} {wr:5d} {rw:5d} "
              f"{wr - rw:+5d} {toward_none / flips:12.1%} {pacc:10.1%} {nacc:8.1%}")


if __name__ == "__main__":
    main()
