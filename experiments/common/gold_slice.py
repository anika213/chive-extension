"""Canonical dataset-slice definitions for the binary causal-claim eval (cf7).

SINGLE SOURCE OF TRUTH for the dataset filters described in the paper's evaluation-dataset
appendix (paper/main.tex, app:eval_dataset). Import the slice you want from
here; do NOT re-implement the filters elsewhere and do NOT cache the result to disk — the slice
is recomputed from the source files on every call, so it can never go stale.

Two nested slices are defined:

  QUALITY SLICE (`quality_ids` / `quality_funnel`) — the PAPER'S PRIMARY population, used
  consistently across the interpretability, direct-prediction, and self-explanation figures.
  Three data-quality filters:
    1. mechanism concreteness >= 3  — applied UPSTREAM when the eval set is built
       (eval_set_conc3_n*.jsonl), so every claim_id in that file already passes it.
    2. counterfactual reproducibility — the `cf` (counterfactual-tool) arm answered correctly,
       i.e. the labeled effect reproduces when an agent re-runs the intervention.
    3. single-factor / unconfounded — the eval-set `edit_cleanliness` rating != 'confounded'.

  GOLD / HARD SLICE (`gold_ids` / `gold_funnel`) — the quality slice PLUS a fourth,
  comparator-dependent gate, kept only for secondary stress-test analyses
  (privileged_access). NOT used for the main paper figures:
    4. difficulty — the transcript-only baseline (`plain`) is unsure (prob_true in [lo, hi]).
  This gate conditions on the Opus transcript reader's uncertainty, so it handicaps that
  comparator; it is a difficulty/comparator gate, not a data-quality filter. That is why the
  headline figures use the quality slice and exclude it.

SCOPE. These slices are for the IN-DISTRIBUTION cf7 evals only (main tool eval, self-explanation,
direct-counterfactual). They do NOT apply to the sycophancy AITA OOD eval — that eval set has
no cf arm / concreteness / edit_cleanliness, so report it on its own terms (AUROC over flip
labels). No leakage exclusion is applied here: the eval data itself excludes
training-overlapping prompts (data/eval_blocklist.json), so the slice is the only filter.

Usage in an analysis script:
    from experiments.common.gold_slice import quality_ids
    keep = quality_ids(run_dir)                   # set of claim_ids
    ids = [c for c in my_scored_ids if c in keep]

CLI (audit only — prints the funnel, writes nothing):
    PYTHONPATH=. python experiments/common/gold_slice.py
"""

import argparse
import glob
import json
import os

from chive import paths

RUNS_DIR = str(paths.RUNS_DIR)
SUFFIX = "_claude-opus-4-8_high_conc3.jsonl"

# slug -> (display run name, run dir) — must match main_counterfactual_eval/analyze.py RUNS.
RUNS = {
    "gemma": ("gemma-3-27b-it", "opus_investigates_gemma3_27b_it_5000_thorough_n30"),
    "qwen8b": ("qwen3-8b", "opus_investigates_qwen3_8b_5000_thorough_n30"),
    "q397_5000": ("qwen3.5-397b (5000)", "opus_investigates_qwen3p5_397b_5000_thorough_n30"),
    "q397_petri": ("qwen3.5-397b (petri)", "opus_investigates_qwen3p5_397b_petri_thorough_n30"),
    "qwen8b_petri": ("qwen3-8b (petri)", "opus_investigates_qwen3_8b_petri_thorough_n30"),
}


def _load_rows(path):
    """claim_id -> row, failing loudly if the file is missing or a claim_id repeats."""
    rows = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            cid = r["claim_id"]
            assert cid not in rows, f"duplicate claim_id {cid} in {path}"
            rows[cid] = r
    return rows


def _eval_set_path(run_dir):
    hits = sorted(glob.glob(os.path.join(run_dir, "eval_set_conc3_n*.jsonl")))
    assert len(hits) == 1, f"expected exactly one eval_set_conc3_n*.jsonl in {run_dir}, got {hits}"
    return hits[0]


def _quality_core(run_dir, restrict_to=None):
    """Shared core: apply the three data-quality filters (concreteness, cf-correct,
    not-confounded). Returns (sub_list, funnel_dict, plain_rows) so gold_funnel can add
    the fourth (baseline-unsure) gate without re-reading the source files.

    Filter 1 (concreteness) is implicit: we start from the conc3 eval set's claim_ids.
    Filters 2-3 are applied against the canonical `cf` and `plain` arm result files.

    `restrict_to`: if given, intersect the starting set with these claim_ids first. Use this
    to respect a caller's own common-claim set (e.g. an analysis paired across AO/NLA/SAE arms),
    so the returned set is guaranteed to be scored by every arm the caller reports.
    """
    eval_set = _load_rows(_eval_set_path(run_dir))                       # concreteness-passed claims
    cf = _load_rows(os.path.join(run_dir, "binary_evals", "cf" + SUFFIX))
    plain = _load_rows(os.path.join(run_dir, "binary_evals", "plain" + SUFFIX))

    # start = claims that passed concreteness AND were scored by both the cf and plain arms
    base = set(eval_set) & set(cf) & set(plain)
    if restrict_to is not None:
        base &= set(restrict_to)
    start = sorted(base)
    funnel = {"concreteness": len(eval_set), "scored": len(start)}

    sub = [c for c in start if cf[c]["correct"]]                        # 2. cf reproduces the label
    funnel["cf_correct"] = len(sub)
    sub = [c for c in sub if eval_set[c]["edit_cleanliness"] != "confounded"]   # 3. single-factor
    funnel["not_confounded"] = len(sub)
    return sub, funnel, plain


def quality_funnel(run_dir, restrict_to=None):
    """Return (sorted quality-slice claim_ids, funnel dict), recomputed from source each call.

    The quality slice = concreteness>=3 ∩ cf-correct ∩ not-confounded. This is the paper's
    PRIMARY population; it excludes the comparator-dependent baseline-unsure gate (see gold_funnel).
    """
    sub, funnel, _ = _quality_core(run_dir, restrict_to)
    return sorted(sub), funnel


def quality_ids(run_dir, restrict_to=None):
    """The quality-slice claim_id set for a run, recomputed from source (the common case)."""
    ids, _ = quality_funnel(run_dir, restrict_to)
    return set(ids)


def gold_funnel(run_dir, lo=0.25, hi=0.75, restrict_to=None):
    """Return (sorted gold/hard claim_ids, funnel dict): the quality slice PLUS the fourth
    baseline-unsure gate (transcript-only `plain` prob_true in [lo, hi]).

    HARD STRESS SLICE for secondary analyses only (privileged_access). The main
    paper figures use quality_funnel — the baseline-unsure gate conditions on the Opus comparator.
    """
    sub, funnel, plain = _quality_core(run_dir, restrict_to)
    sub = [c for c in sub if lo <= plain[c]["prob_true"] <= hi]         # 4. baseline unsure
    funnel["gold"] = len(sub)
    return sorted(sub), funnel


def gold_ids(run_dir, lo=0.25, hi=0.75, restrict_to=None):
    """The gold/hard claim_id set for a run, recomputed from source."""
    ids, _ = gold_funnel(run_dir, lo, hi, restrict_to)
    return set(ids)


def main():
    ap = argparse.ArgumentParser(description="Print the gold-subset funnel for every run (audit only).")
    ap.add_argument("--lo", type=float, default=0.25, help="transcript-only prob_true lower bound (Filter 4)")
    ap.add_argument("--hi", type=float, default=0.75, help="transcript-only prob_true upper bound (Filter 4)")
    args = ap.parse_args()
    hdr = (f"{'run':24s} {'concrete':>9s} {'scored':>7s} {'cf_ok':>7s} {'QUALITY':>8s} "
           f"{'gold':>6s} {'gold/scored':>12s}   (QUALITY=not-confounded, the paper slice)")
    print(hdr); print("-" * len(hdr))
    for slug, (run_name, d) in RUNS.items():
        run_dir = os.path.join(RUNS_DIR, d)
        if not os.path.isdir(run_dir):
            print(f"[skip] {slug}: no run dir {run_dir}")
            continue
        _, fn = gold_funnel(run_dir, args.lo, args.hi)
        pct = fn["gold"] / fn["scored"] if fn["scored"] else float("nan")
        print(f"{run_name:24s} {fn['concreteness']:9d} {fn['scored']:7d} {fn['cf_correct']:7d} "
              f"{fn['not_confounded']:8d} {fn['gold']:6d} {pct:11.1%}")


if __name__ == "__main__":
    main()
