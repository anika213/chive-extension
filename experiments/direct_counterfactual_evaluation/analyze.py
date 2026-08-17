"""Thread A — direct counterfactual prediction: base vs trained vs Opus, on the gold subset.

Self-contained (no heavy cross-folder imports beyond the canonical gold-slice definition).
Reads:
  - results/canonical/{8b,397b}_{base,trained}_{regular,petri}.jsonl  (our base/trained arms,
    written by run_structured_claim_eval.py via run_all.sh)
  - <run_dir>/binary_evals/plain_<answerer>_conc3.jsonl                (Opus transcript-reader)
  - the quality subset via experiments/common/gold_slice.quality_ids (recomputed fresh from source
    every call — no persisted state; the single source of truth for the data-quality filters)

Writes, next to this script:
  - SUMMARY.md       — per-run table, full-set base/Opus/trained AUROC, and the QUALITY-subset table
  - plots/direct_counterfactual_evaluation_fig_auroc.png    — main figure: AUROC,
    base/Opus/trained, both models, in-distribution (WildChat) + PETRI panels
  - plots/direct_counterfactual_evaluation_fig_accuracy.png — same, accuracy

The quality subset is the mainline reported set: concreteness≥3 ∩ cf-correct ∩ not-confounded.
The comparator-dependent baseline-unsure gate is deliberately excluded (see gold_slice.py). The
cf arm is NOT plotted: the slice is defined as cf-correct, so cf would be ~100% by construction.

Run (CPU only, from anywhere):
  python analyze.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from chive import paths
from experiments.common.gold_slice import (
    quality_funnel,
)

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 22, "axes.labelsize": 19,
    "xtick.labelsize": 17, "ytick.labelsize": 16, "legend.fontsize": 18,
    "figure.titlesize": 22,
})

HERE = Path(__file__).resolve().parent
DATA = paths.experiment_data_dir("direct_counterfactual_evaluation")
RUNS_ROOT = paths.RUNS_DIR
OPUS_PLAIN = "binary_evals/plain_claude-opus-4-8_high_conc3.jsonl"

MODELS = ["8b", "397b"]
SETS = ["regular", "petri"]
MODEL_DISPLAY = {"8b": "Qwen3-8B", "397b": "Qwen3.5-397B"}
RUN_DIR = {
    ("8b", "regular"): "opus_investigates_qwen3_8b_5000_thorough_n30",
    ("8b", "petri"): "opus_investigates_qwen3_8b_petri_thorough_n30",
    ("397b", "regular"): "opus_investigates_qwen3p5_397b_5000_thorough_n30",
    ("397b", "petri"): "opus_investigates_qwen3p5_397b_petri_thorough_n30",
}

# figure arms (left -> right): base, trained, then Opus as the external transcript-reader
# reference on the far right (base|trained = training gain; trained|opus = introspection signal)
ARM_ORDER = ["base", "trained", "opus"]
ARM_DISPLAY = {"base": "Base", "opus": "Opus", "trained": "Trained"}
ARM_COLOR = {"base": "#999999", "opus": "#e6550d", "trained": "#3182bd"}
# two-line titles: at the enlarged title font the single-line versions collided between panels
DIST_TITLE = {"regular": "In distribution\n(WildChat)", "petri": "Out of distribution\n(PETRI)"}
CHANCE_COLOR = "#e10600"

RNG = np.random.default_rng(0)
N_BOOT = 2000


# ---------- metrics (self-contained, same definitions as main_counterfactual_eval) ----------
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


def wilson_ci(k, n):
    z = 1.96; p = k / n; denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z / denom * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return center - half, center + half


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------- loading ----------
def load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            rows[r["claim_id"]] = r
    return rows


def canonical_path(model: str, arm: str, eset: str) -> Path:
    return DATA / "results" / "canonical" / f"{model}_{arm}_{eset}.jsonl"


def arm_metrics(rows: dict[str, dict], ids: list[str]) -> dict:
    y = np.array([rows[c]["is_true"] for c in ids]).astype(int)
    p = np.array([rows[c]["prob_true"] for c in ids])
    pred = np.array([rows[c]["pred_true"] for c in ids]).astype(int)
    k = int((pred == y).sum()); n = len(ids)
    au_lo, au_hi = auroc_ci(y, p)
    acc_lo, acc_hi = wilson_ci(k, n)
    return {
        "n": n,
        "auroc": float(auroc(y, p)), "auroc_ci": [float(au_lo), float(au_hi)],
        "acc": k / n, "acc_ci": [float(acc_lo), float(acc_hi)],
        "brier": float(np.mean((p - y) ** 2)),
    }


def gold_cell(model: str, eset: str):
    """Return (arms_dict, slice_ids, funnel) for one (model, eval set) on the quality subset.

    arms_dict maps arm -> metrics dict. None if any input arm is missing.
    """
    run_dir = RUNS_ROOT / RUN_DIR[(model, eset)]
    paths = {
        "base": canonical_path(model, "base", eset),
        "trained": canonical_path(model, "trained", eset),
        "opus": run_dir / OPUS_PLAIN,
    }
    if not all(p.exists() for p in paths.values()):
        return None, [], {}
    rows = {arm: load_rows(p) for arm, p in paths.items()}
    common = sorted(set(rows["base"]) & set(rows["trained"]) & set(rows["opus"]))
    qids, funnel = quality_funnel(str(run_dir), restrict_to=common)  # fresh recompute, no state
    arms = {arm: arm_metrics(rows[arm], qids) for arm in ARM_ORDER}
    return arms, qids, funnel


# ---------- plots ----------
def _panel(ax, cells: dict[str, dict], metric: str, ci_key: str, title: str):
    """Draw one distribution's grouped bars into ax. cells: model -> {arm -> metrics}."""
    models = [m for m in MODELS if m in cells]
    x = np.arange(len(models)); w = 0.8 / len(ARM_ORDER)
    for j, arm in enumerate(ARM_ORDER):
        vals = np.array([cells[m][arm][metric] for m in models])
        cis = [cells[m][arm][ci_key] for m in models]
        err = np.clip(np.vstack([vals - [c[0] for c in cis], [c[1] for c in cis] - vals]), 0, None)
        off = (j - (len(ARM_ORDER) - 1) / 2) * w
        ax.bar(x + off, vals, w, yerr=err, color=ARM_COLOR[arm], edgecolor="black",
               linewidth=0.9, error_kw=dict(capsize=4, linewidth=1.4, ecolor="black"))
        for xi, v, hi in zip(x + off, vals, [c[1] for c in cis]):
            ax.text(xi, hi + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=13)
    ax.axhline(0.5, ls="--", color=CHANCE_COLOR, lw=2.0, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([MODEL_DISPLAY[m] for m in models])
    ax.set_title(title); style(ax)


def two_panel(gold_cells, metric: str, ci_key: str, ylabel: str, out_path: Path):
    """One figure, two panels: In distribution (regular) | Out of distribution (petri)."""
    dists = [d for d in ("regular", "petri")
             if any((m, d) in gold_cells for m in MODELS)]
    # narrower canvas => less downscaling at 0.9\linewidth => larger effective legend/title fonts
    fig, axes = plt.subplots(1, len(dists), figsize=(4.4 * len(dists), 4.6), sharey=True)
    if len(dists) == 1:
        axes = [axes]
    for ax, d in zip(axes, dists):
        cells = {m: gold_cells[(m, d)] for m in MODELS if (m, d) in gold_cells}
        _panel(ax, cells, metric, ci_key, DIST_TITLE[d])
    axes[0].set_ylabel(ylabel); axes[0].set_ylim(0, 1.05)
    handles = [Patch(facecolor=ARM_COLOR[a], edgecolor="black", label=ARM_DISPLAY[a])
               for a in ARM_ORDER]
    handles.append(Line2D([0], [0], color=CHANCE_COLOR, ls="--", lw=2.0, label="Chance"))
    fig.tight_layout(rect=(0, 0.06, 1, 1))  # reserve a bottom band so the legend clears the x labels
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.02))
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", out_path)


# ---------- summary ----------
def fmt(x):
    return " nan" if x != x else f"{x:.3f}"


def write_summary(found, gold_cells, gold_funnels, out: Path):
    L = ["# Thread A — direct counterfactual prediction (base vs trained vs Opus)\n",
         "Canonical arm: format A (claim as a follow-up turn on the transcript), no-think, "
         "first-token Yes/No logprob. Base and trained share this exact prompt; Opus `plain` is the "
         "high-effort transcript-reader reference.\n",
         "**Main-body numbers are the QUALITY subset**: concreteness≥3 ∩ cf-correct ∩ "
         "not-confounded, via `experiments/common/gold_slice.quality_funnel` (recomputed from source). "
         "The comparator-dependent baseline-unsure gate is deliberately excluded. The cf arm is "
         "omitted from the figures because the slice is defined as cf-correct.\n",
         "> Regenerate with `analyze.py` (writes SUMMARY.md, plots/). Reproduce "
         "the eval from scratch with `run_all.sh`.\n",
         "## Per-run (full scored set)", "",
         "| model | arm | eval set | n | acc | AUROC | Brier |",
         "|---|---|---|---|---|---|---|"]
    for model in MODELS:
        for eset in SETS:
            for arm in ["base", "trained"]:
                key = (model, arm, eset)
                if key not in found:
                    L.append(f"| {model} | {arm} | {eset} | — | (missing) | | |")
                    continue
                m = found[key]
                L.append(f"| {model} | {arm} | {eset} | {m['n']} | {m['acc']:.1%} | "
                         f"{fmt(m['auroc'])} | {fmt(m['brier'])} |")

    L += ["", "## Quality subset — base vs Opus vs trained (AUROC [95% CI] and acc)", "",
          "AUROC with 95% bootstrap CI (2000 resamples, seed 0); acc with Wilson 95% CI. Δ columns "
          "are trained minus Opus on the identical quality claim set.", "",
          "| model | eval set | quality n | base AUROC | Opus AUROC | trained AUROC | trained−Opus | "
          "base acc | Opus acc | trained acc |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for model in MODELS:
        for eset in SETS:
            cells = gold_cells.get((model, eset))
            if cells is None:
                L.append(f"| {model} | {eset} | — | (missing arm) | | | | | | |")
                continue
            b, o, t = cells["base"], cells["opus"], cells["trained"]
            fn = gold_funnels[(model, eset)]
            L.append(
                f"| {model} | {eset} | {b['n']} "
                f"| {b['auroc']:.3f} [{b['auroc_ci'][0]:.3f}, {b['auroc_ci'][1]:.3f}] "
                f"| {o['auroc']:.3f} [{o['auroc_ci'][0]:.3f}, {o['auroc_ci'][1]:.3f}] "
                f"| {t['auroc']:.3f} [{t['auroc_ci'][0]:.3f}, {t['auroc_ci'][1]:.3f}] "
                f"| {t['auroc'] - o['auroc']:+.3f} "
                f"| {b['acc']:.1%} | {o['acc']:.1%} | {t['acc']:.1%} |")
            L.append(f"|  |  | _funnel: scored {fn['scored']} → cf-correct {fn['cf_correct']} → "
                     f"not-confounded (quality) {fn['not_confounded']}_ | | | | | | | | |")
    out.write_text("\n".join(L) + "\n")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE)
    args = ap.parse_args()
    plots_dir = args.out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # full-set per-run metrics (for the SUMMARY per-run table)
    found = {}
    for p in sorted(glob.glob(str(DATA / "results" / "canonical" / "*.jsonl"))):
        stem = Path(p).stem
        model = "397b" if stem.startswith("397b") else "8b"
        arm = "trained" if "trained" in stem else "base"
        eset = "petri" if "petri" in stem else "regular"
        rows = load_rows(Path(p))
        found[(model, arm, eset)] = arm_metrics(rows, list(rows))

    # gold-subset metrics per (model, eval set)
    gold_cells, gold_funnels = {}, {}
    for model in MODELS:
        for eset in SETS:
            arms, gold, funnel = gold_cell(model, eset)
            if arms is None:
                print(f"[skip gold] {model}/{eset}: missing an arm")
                continue
            gold_cells[(model, eset)] = arms
            gold_funnels[(model, eset)] = funnel
            print(f"quality {model}/{eset}: n={funnel['not_confounded']}  "
                  f"base {arms['base']['auroc']:.3f} / Opus {arms['opus']['auroc']:.3f} / "
                  f"trained {arms['trained']['auroc']:.3f}")

    # main figures: AUROC (headline) and accuracy, each a two-panel in/out-of-distribution figure.
    # Filenames are prefixed with the folder name so figures stay self-identifying once collected.
    prefix = HERE.name
    if gold_cells:
        two_panel(gold_cells, "auroc", "auroc_ci", "AUROC", plots_dir / f"{prefix}_fig_auroc.png")
        two_panel(gold_cells, "acc", "acc_ci", "Accuracy", plots_dir / f"{prefix}_fig_accuracy.png")

    write_summary(found, gold_cells, gold_funnels, args.out_dir / "SUMMARY.md")


if __name__ == "__main__":
    main()
