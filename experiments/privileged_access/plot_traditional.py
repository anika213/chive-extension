"""Paper figure: the traditional (Binder et al.) privileged-access test.

Two self-contained instances of the standard test, one per target model. For target T,
both predictors (T itself and the other model) are trained on T's claim data — size-matched
at 23,944 examples, identical hyperparameters — and evaluated on held-out quality-slice claims
about T. Shown: self-explanation bar, cross-explanation bar, and the Opus transcript-reader
reference, all with 95% CIs. The quality slice is defined by three claim-level filters
(concreteness>=3, cf-correct, not-confounded) via quality_funnel — independent of any model's
predictions. Base (untrained) arms are still loaded (so the common claim set is one every arm
scored) and reported in stdout, but not drawn — they sit at chance
(AUROC 0.51-0.55); state that in the figure caption.

Arm inputs (see arm_path): the Opus reference comes from each target run's canonical
transcript-only scoring; the Qwen-bank Qwen3-8B base arm reuses the canonical
direct_counterfactual_evaluation scoring (identical scorer + token-ids protocol); every other
arm is scored by jobs/eval_traditional.sh into results/scored/{bank}__{arm}.jsonl.

Writes plots/privileged_access_traditional.png next to this script, and prints the
per-arm table + paired self−cross deltas. The figure is regenerated directly from the
scored result JSONL each run; no intermediate figure-data file is kept.
Run (CPU, from repo root):
  python experiments/privileged_access/plot_traditional.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chive import paths
from experiments.common.gold_slice import (
    quality_funnel,
)

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 19, "axes.labelsize": 17,
    "xtick.labelsize": 16, "ytick.labelsize": 15, "legend.fontsize": 15,
    "figure.titlesize": 21,
})

HERE = Path(__file__).resolve().parent
DATA = paths.experiment_data_dir("privileged_access")
RUNS_ROOT = paths.RUNS_DIR
DCE = paths.experiment_data_dir("direct_counterfactual_evaluation")
OPUS_PLAIN = "binary_evals/plain_claude-opus-4-8_high_conc3.jsonl"

BANKS = ["qwen_regular", "llama_regular"]
RUN_DIR = {
    "qwen_regular": "opus_investigates_qwen3_8b_5000_thorough_n30",
    "llama_regular": "opus_investigates_llama3_1_8b_5000_thorough_n30",
}

PRED_DISPLAY = {"qwen3_8b": "Qwen3-8B", "llama31_8b": "Llama-3.1-8B"}
# per bank: target model, (self predictor, cross predictor), and the arm suffix that means
# "trained on the target's data" (Qwen data = the seeded size-matched 24k subsample)
SETTING = {
    "qwen_regular": {"target": "Qwen3-8B", "self": "qwen3_8b", "cross": "llama31_8b",
                     "trained_suffix": "on_qwen3_8b_sub24k"},
    "llama_regular": {"target": "Llama-3.1-8B", "self": "llama31_8b", "cross": "qwen3_8b",
                      "trained_suffix": "on_llama31_8b"},
}

RNG = np.random.default_rng(0)
N_BOOT = 2000


# ---------- metrics (same definitions as direct_counterfactual_evaluation/analyze.py) ----------
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


def paired_dauroc_ci(y, p_a, p_b):
    """Bootstrap CI for AUROC(a) − AUROC(b), resampling claims jointly (paired)."""
    y = np.asarray(y); p_a = np.asarray(p_a); p_b = np.asarray(p_b); n = len(y); boots = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n); yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        boots.append(auroc(yb, p_a[idx]) - auroc(yb, p_b[idx]))
    return tuple(np.percentile(boots, [2.5, 97.5]))


def wilson_ci(k, n):
    z = 1.96; p = k / n; denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z / denom * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return center - half, center + half


# ---------- loading ----------
def load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r["claim_id"]] = r
    return rows


def arm_path(bank: str, arm: str) -> Path:
    if arm == "opus":
        return RUNS_ROOT / RUN_DIR[bank] / OPUS_PLAIN
    # the Qwen-bank Qwen3-8B base arm reuses the canonical direct_counterfactual_evaluation
    # scoring (identical scorer + token-ids protocol)
    if bank == "qwen_regular" and arm == "qwen3_8b_base":
        return DCE / "results" / "canonical" / "8b_base_regular.jsonl"
    return DATA / "results" / "scored" / f"{bank}__{arm}.jsonl"


def arm_metrics(rows: dict[str, dict], ids: list[str]) -> dict:
    y = np.array([rows[c]["is_true"] for c in ids]).astype(int)
    p = np.array([rows[c]["prob_true"] for c in ids])
    pred = (p >= 0.5).astype(int)
    k = int((pred == y).sum()); n = len(ids)
    au_lo, au_hi = auroc_ci(y, p)
    acc_lo, acc_hi = wilson_ci(k, n)
    return {"n": n, "n_true": int(y.sum()),
            "auroc": float(auroc(y, p)), "auroc_ci": [float(au_lo), float(au_hi)],
            "acc": k / n, "acc_ci": [float(acc_lo), float(acc_hi)],
            "brier": float(np.mean((p - y) ** 2))}


def main():
    fig_data = {"banks": {}, "deltas": {}}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, bank in zip(axes, BANKS):
        cfg = SETTING[bank]
        preds = [cfg["self"], cfg["cross"]]  # self predictor always leftmost
        arms = {"opus": "opus"}
        for p in preds:
            arms[f"{p}_base"] = f"{p}_base"
            arms[f"{p}_trained"] = f"{p}_{cfg['trained_suffix']}"
        arms_rows = {k: load_rows(arm_path(bank, a)) for k, a in arms.items()}
        common = set.intersection(*[set(r) for r in arms_rows.values()])
        ids_q, funnel = quality_funnel(str(RUNS_ROOT / RUN_DIR[bank]), restrict_to=common)
        ids = sorted(ids_q)

        metrics = {k: arm_metrics(r, ids) for k, r in arms_rows.items()}
        fig_data["banks"][bank] = {"n_quality": len(ids), "funnel": funnel,
                                   "arms": {k: metrics[k] for k in metrics},
                                   "arm_files": {k: str(arm_path(bank, a))
                                                 for k, a in arms.items()}}

        # paired self−cross ΔAUROC on the trained arms (the privileged-access statistic)
        y = np.array([arms_rows["opus"][c]["is_true"] for c in ids]).astype(int)
        p_self = np.array([arms_rows[f"{cfg['self']}_trained"][c]["prob_true"] for c in ids])
        p_cross = np.array([arms_rows[f"{cfg['cross']}_trained"][c]["prob_true"] for c in ids])
        d = float(auroc(y, p_self) - auroc(y, p_cross))
        lo, hi = paired_dauroc_ci(y, p_self, p_cross)
        fig_data["deltas"][bank] = {"self_minus_cross_trained": d,
                                    "ci": [float(lo), float(hi)]}

        # three bars: self-explanation, cross-explanation, Opus transcript reader.
        # Colors: trained-model blue #3182bd (self), a
        # lighter CVD-safe blue for cross, Opus orange #e6550d hatched (different method).
        bars = [(f"{cfg['self']}_trained", f"{PRED_DISPLAY[cfg['self']]}\n(self-explanation)",
                 "#3182bd", None),
                (f"{cfg['cross']}_trained", f"{PRED_DISPLAY[cfg['cross']]}\n(cross-explanation)",
                 "#56B4E9", None),
                ("opus", "Opus 4.8\n(transcript-only)", "#e6550d", "//")]
        xs = np.arange(len(bars))
        for x, (key, _, color, hatch) in zip(xs, bars):
            m = metrics[key]
            err = np.array([[m["auroc"] - m["auroc_ci"][0]],
                            [m["auroc_ci"][1] - m["auroc"]]])
            ax.bar(x, m["auroc"], 0.55, yerr=err, color=color, hatch=hatch,
                   edgecolor="black", linewidth=0.8, capsize=5,
                   error_kw={"lw": 1.5, "ecolor": "black"})
            ax.annotate(f"{m['auroc']:.3f}", xy=(x, m["auroc_ci"][1]), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=12)

        ax.set_xticks(xs)
        ax.set_xticklabels([lab for _, lab, _, _ in bars], fontsize=12)
        ax.set_title(f"Target model: {cfg['target']}", fontsize=16)
        # y-axis floor at 0.5 = chance, so bar height reads as skill above chance
        ax.set_ylim(0.5, 0.92)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("AUROC")
    fig.tight_layout()
    (HERE / "plots").mkdir(exist_ok=True)
    out_fig = HERE / "plots" / "privileged_access_traditional.png"
    fig.savefig(out_fig, dpi=200, bbox_inches="tight")

    for bank in BANKS:
        b = fig_data["banks"][bank]
        print(f"\n{bank} (quality n={b['n_quality']}):")
        for k, m in b["arms"].items():
            print(f"  {k}: AUROC {m['auroc']:.3f} [{m['auroc_ci'][0]:.3f}, "
                  f"{m['auroc_ci'][1]:.3f}]  acc {m['acc']:.1%}")
        d = fig_data["deltas"][bank]
        print(f"  paired Δ self−cross (trained): {d['self_minus_cross_trained']:+.3f} "
              f"[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]")
    print(f"\nwrote {out_fig}")


if __name__ == "__main__":
    main()
