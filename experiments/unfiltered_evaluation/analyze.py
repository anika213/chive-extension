"""Analysis for the truly-unfiltered evaluation (see README.md).

Two populations per run, recomputed from source on every call (no persisted slice state):
  NO FILTERS  = conc3 eval set ∪ unfiltered complement, minus train-overlap claim_ids
                (the full bank minus claims the trained models saw; gemma has no overlap).
  ALL FILTERS = gold_slice.quality_ids (concreteness>=3 ∩ cf-correct ∩ not-confounded),
                minus train-overlap claim_ids (on 8B this removes 8 claims, so the numbers
                differ from the published SUMMARYs by <=0.4% of the set — reported below).

Scored rows per arm = published *_conc3.jsonl ∪ this folder's *_unfiltered.jsonl, merged by
claim_id (claims are scored independently; the two banks are disjoint).

Tables:
  1. Gemma tool arms (plain/AO/NLA/SAE) on both populations.
  2. Trained direct-counterfactual arms (base / Opus plain / trained) for 8B + 397B.
  3. Concreteness breakdown (conc 1-2 vs 3-4) within NO FILTERS.
  4. Consistency check vs the published all-filters numbers + Yes/No adherence rates.

Run: PYTHONPATH=. python experiments/unfiltered_evaluation/analyze.py
Writes: SUMMARY.md (this folder).
"""

import json
import os
from pathlib import Path

import numpy as np

from chive import paths
from experiments.common.gold_slice import quality_ids

HERE = Path(__file__).resolve().parent
DATA = paths.experiment_data_dir("unfiltered_evaluation")
RUNS_DIR = paths.RUNS_DIR
CANON = paths.experiment_data_dir("direct_counterfactual_evaluation") / "results" / "canonical"
SUFFIX = "_claude-opus-4-8_high_conc3.jsonl"
UNF_SUFFIX = "_claude-opus-4-8_high_unfiltered.jsonl"

N_BOOT = 2000
RNG = np.random.default_rng(0)

RUNS = {
    "gemma": ("gemma-3-27b-it", RUNS_DIR / "opus_investigates_gemma3_27b_it_5000_thorough_n30"),
    "qwen8b": ("qwen3-8b", RUNS_DIR / "opus_investigates_qwen3_8b_5000_thorough_n30"),
    "q397": ("qwen3.5-397b", RUNS_DIR / "opus_investigates_qwen3p5_397b_5000_thorough_n30"),
}
# Published all-filters AUROCs (main_counterfactual_eval / direct_counterfactual_evaluation
# SUMMARY.md) for the consistency check. Tolerance is loose only where the population differs
# (8B: -8 train-overlap claims).
PUBLISHED = {
    ("gemma", "plain"): 0.817, ("gemma", "ao"): 0.814, ("gemma", "nla"): 0.807,
    ("gemma", "sae"): 0.813,
    ("qwen8b", "plain"): 0.818, ("qwen8b", "base"): 0.547, ("qwen8b", "trained"): 0.818,
    ("q397", "plain"): 0.837, ("q397", "base"): 0.722, ("q397", "trained"): 0.852,
}

GEMMA_TOOL_ARMS = {  # cat -> filename stem (identical for conc3 and unfiltered)
    "plain": "plain",
    "ao": "ao_latentqa_cls_past_lens_gemma-3-27b-it",
    "nla": "nla_gemma3-27b-L41",
    "sae": "sae_gemma-scope-2-27b-it_layer_40_width_65k_l0_medium",
}


# ---------- stats (mirrors main_counterfactual_eval/analyze.py) ----------
def rankdata_avg(a):
    a = np.asarray(a, float); n = len(a); order = np.argsort(a, kind="mergesort")
    sv = a[order]; ranks = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1; i = j + 1
    return ranks


def auroc(y, p):
    y = np.asarray(y); p = np.asarray(p); r = rankdata_avg(p)
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def auroc_ci(y, p):
    y = np.asarray(y); p = np.asarray(p); n = len(y); boots = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n); yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        boots.append(auroc(yb, p[idx]))
    return tuple(np.percentile(boots, [2.5, 97.5]))


def paired_delta_ci(y, p_tool, p_plain):
    y = np.asarray(y); n = len(y); boots = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n); yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        boots.append(auroc(yb, p_tool[idx]) - auroc(yb, p_plain[idx]))
    return tuple(np.percentile(boots, [2.5, 97.5]))


# ---------- loading ----------
def load_rows(path: Path) -> dict:
    rows = {}
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        assert r["claim_id"] not in rows, f"duplicate claim_id {r['claim_id']} in {path}"
        rows[r["claim_id"]] = r
    return rows


def merged(conc3_path: Path, unf_path: Path) -> dict:
    a, b = load_rows(conc3_path), load_rows(unf_path)
    dup = set(a) & set(b)
    assert not dup, f"{len(dup)} claim_ids scored in both {conc3_path.name} and {unf_path.name}"
    return {**a, **b}


def populations(slug: str, run_dir: Path):
    """(no_filters_ids, all_filters_ids, conc_by_claim) for a run."""
    eval_ids = set(load_rows(run_dir / "eval_set_conc3_n2000.jsonl"))
    comp = load_rows(DATA / "eval_sets" / f"{slug}_complement.jsonl")
    overlap = set(json.load(open(DATA / "results" / f"train_overlap_{slug}.json"))["claim_ids"])
    no_filters = (eval_ids | set(comp)) - overlap
    all_filters = quality_ids(str(run_dir)) - overlap
    assert all_filters <= no_filters, f"{slug}: quality slice not nested in no-filters"
    conc_by_prompt = {r["prompt_id"]: r["mechanism_concreteness"]
                      for r in map(json.loads, open(run_dir / "mechanism_concreteness.jsonl"))}
    conc_by_claim = {}
    for src in (load_rows(run_dir / "eval_set_conc3_n2000.jsonl"), comp):
        for cid, r in src.items():
            conc_by_claim[cid] = conc_by_prompt[r["prompt_id"]]
    return no_filters, all_filters, conc_by_claim


def arm_files(slug: str, run_dir: Path) -> dict:
    """cat -> merged rows for every arm this run reports."""
    arms = {}
    if slug == "gemma":
        for cat, stem in GEMMA_TOOL_ARMS.items():
            arms[cat] = merged(run_dir / "binary_evals" / f"{stem}{SUFFIX}",
                               DATA / "results" / "gemma" / f"{stem}{UNF_SUFFIX}")
    else:
        arms["plain"] = merged(run_dir / "binary_evals" / f"plain{SUFFIX}",
                               DATA / "results" / slug / f"plain{UNF_SUFFIX}")
        model = "8b" if slug == "qwen8b" else "397b"
        for cat in ("base", "trained"):
            arms[cat] = merged(CANON / f"{model}_{cat}_regular.jsonl",
                               DATA / "results" / f"{model}_{cat}_unfiltered.jsonl")
    return arms


# ---------- reporting ----------
def fmt(x):
    return f"{x:+.3f}" if isinstance(x, float) and abs(x) < 0.5 else f"{x:.3f}"


def metrics_row(rows, ids, plain_rows=None):
    y = np.array([rows[c]["is_true"] for c in ids]).astype(int)
    p = np.array([rows[c]["prob_true"] for c in ids], dtype=float)
    a = auroc(y, p); lo, hi = auroc_ci(y, p)
    acc = float(np.mean([rows[c]["correct"] for c in ids]))
    brier = float(np.mean((p - y) ** 2))
    out = {"n": len(ids), "base_rate": float(y.mean()), "auroc": a, "ci": (lo, hi),
           "acc": acc, "brier": brier}
    if plain_rows is not None:
        pp = np.array([plain_rows[c]["prob_true"] for c in ids], dtype=float)
        out["delta"] = a - auroc(y, pp)
        out["delta_ci"] = paired_delta_ci(y, p, pp)
    return out


def adherence(path: Path):
    """Yes/No-prefix rate of the argmax token (canonical convention: 'Yes.', 'No\\n\\n...'
    count as adherent — the trailing text rides along in the token string; the yes/no
    logprobs and prob_true are well-defined regardless)."""
    rows = load_rows(path)
    ok = sum(1 for r in rows.values()
             if str(r.get("argmax_token", "")).strip().lower().startswith(("yes", "no")))
    return ok, len(rows)


def main():
    md = ["# Truly unfiltered evaluation (concreteness 1-2 included) — results\n",
          "\n**Takeaways (2026-07-17 run).** (1) The interpretability-tool null is "
          "Filter-1-invariant: on the full unfiltered Gemma bank (n=4,814) AO/NLA/SAE sit at "
          "ΔAUROC +0.001/−0.003/+0.004 vs the transcript-only baseline, every CI straddling "
          "zero — same on the conc1-2-only slice. (2) The trained direct-counterfactual "
          "models EXCEED the Opus reader on the unfiltered set (8B +0.019 [+0.007, +0.032]; "
          "397B +0.035 [+0.024, +0.045]), driven by the vague-mechanism conc1-2 claims that "
          "Filter 1 removes (8B trained 0.848 vs plain 0.818 there) — on the filtered set the "
          "gap was ~0. (3) Absolute difficulty is slightly higher unfiltered (plain drops "
          "~0.01–0.04) but conclusions are unchanged.\n",
          "Populations recomputed from source each run of this script (see module docstring). "
          "AUROC via rank-sum; 95% bootstrap CIs (2000 resamples, seed 0); ΔAUROC paired vs "
          "the transcript-only Opus baseline on the identical claim set. `acc` at the 0.5 "
          "threshold with the TRUE base rate shown (the no-filters population is the full "
          "bank, not re-balanced).\n"]
    consistency = []
    for slug, (name, run_dir) in RUNS.items():
        no_f, all_f, conc = populations(slug, run_dir)
        arms = arm_files(slug, run_dir)
        # restrict to claims scored by every arm; report any shortfall loudly
        scored = set.intersection(*[set(r) for r in arms.values()])
        pops = {"no filters": sorted(no_f & scored), "all filters": sorted(all_f & scored)}
        missing = {k: len((no_f if k == "no filters" else all_f) - scored) for k in pops}
        md.append(f"\n## {name}\n")
        for pop_name, ids in pops.items():
            if missing[pop_name]:
                md.append(f"\n> WARNING: {missing[pop_name]} {pop_name} claims not scored "
                          f"by every arm (excluded from the paired analysis below).\n")
            y = np.array([arms["plain"][c]["is_true"] for c in ids]).astype(int)
            md.append(f"\n### {pop_name}  —  n={len(ids)}  ({y.sum()}T/{len(y) - y.sum()}F)\n")
            md.append("\n| arm | AUROC [95% CI] | acc | Brier | ΔAUROC vs plain [95% CI] |\n"
                      "|---|---|---|---|---|\n")
            for cat, rows in arms.items():
                m = metrics_row(rows, ids, arms["plain"] if cat != "plain" else None)
                d = (f"{fmt(m['delta'])} [{fmt(m['delta_ci'][0])}, {fmt(m['delta_ci'][1])}]"
                     if cat != "plain" else "—")
                md.append(f"| {cat} | {m['auroc']:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}] | "
                          f"{m['acc']:.1%} | {m['brier']:.3f} | {d} |\n")
                if pop_name == "all filters" and (slug, cat) in PUBLISHED:
                    consistency.append((slug, cat, PUBLISHED[(slug, cat)], m["auroc"]))
        # concreteness breakdown inside no-filters
        md.append("\n### Concreteness breakdown (within no filters)\n\n"
                  "| arm | conc 1-2 AUROC (n) | conc 3-4 AUROC (n) | Δ vs plain (1-2) |\n"
                  "|---|---|---|---|\n")
        ids = pops["no filters"]
        lo_ids = [c for c in ids if conc[c] <= 2]
        hi_ids = [c for c in ids if conc[c] >= 3]
        for cat, rows in arms.items():
            def _a(sub):
                yy = np.array([rows[c]["is_true"] for c in sub]).astype(int)
                return auroc(yy, np.array([rows[c]["prob_true"] for c in sub], dtype=float))
            d12 = ""
            if cat != "plain":
                yy = np.array([rows[c]["is_true"] for c in lo_ids]).astype(int)
                pt = np.array([rows[c]["prob_true"] for c in lo_ids], dtype=float)
                pl = np.array([arms["plain"][c]["prob_true"] for c in lo_ids], dtype=float)
                d12 = fmt(auroc(yy, pt) - auroc(yy, pl))
            md.append(f"| {cat} | {_a(lo_ids):.3f} ({len(lo_ids)}) | "
                      f"{_a(hi_ids):.3f} ({len(hi_ids)}) | {d12} |\n")

    md.append("\n## Consistency check vs published all-filters numbers\n\n"
              "Published numbers use the population WITHOUT the train-overlap exclusion "
              "(8B: 8 extra claims), so small diffs are expected there and near-zero "
              "elsewhere.\n\n| run | arm | published | recomputed | diff |\n|---|---|---|---|---|\n")
    for slug, cat, pub, ours in consistency:
        md.append(f"| {slug} | {cat} | {pub:.3f} | {ours:.3f} | {ours - pub:+.3f} |\n")

    md.append("\n## Yes/No format adherence (first-token argmax), base + trained on the "
              "complement\n\n| file | adherent | rate |\n|---|---|---|\n")
    for f in sorted((DATA / "results").glob("*_unfiltered.jsonl")):
        ok, n = adherence(f)
        md.append(f"| {f.name} | {ok}/{n} | {ok / n:.1%} |\n")
        if ok / n < 0.99:
            md.append(f"\n> WARNING: {f.name} adherence {ok / n:.1%} < 99% — the base "
                      f"comparison on this population may be invalid; inspect outputs.\n")

    md.append("\n### Known data notes\n"
              "- gemma `prompt_002230_c0` is in the conc3 eval set but absent from the current "
              "bank (bank regenerated after the eval set was frozen); it is kept in both "
              "populations via the eval-set file.\n"
              "- 8 claims in the published 8B conc3 eval set overlap the 8B training data "
              "(the training build deduped against the old n1000 set); they are excluded from "
              "BOTH populations here, and the published canonical numbers are left untouched.\n")

    (HERE / "SUMMARY.md").write_text("".join(md))
    print(f"wrote {HERE / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
