"""Figures for the unfiltered-evaluation appendix, plotted with the SAME code as the main body.

- Gemma interpretability tools: `plot_overall` copied verbatim from
  main_counterfactual_eval/analyze.py (accuracy + AUROC grouped bars, chance line, fonts).
  Two PNGs: all-filters (quality slice) and no-filters (full bank).
- Trained direct-counterfactual models (8B + 397B): `two_panel` copied from
  direct_counterfactual_evaluation/analyze.py (base/trained/opus AUROC bars). One PNG,
  two panels: All filters (quality) | No filters (full bank).

Data comes from analyze.py's loaders so the populations match SUMMARY.md exactly.

Run: PYTHONPATH=. python experiments/unfiltered_evaluation/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from experiments.unfiltered_evaluation.analyze import (  # noqa: E402
    RUNS, arm_files, auroc, auroc_ci, populations)

HERE = Path(__file__).resolve().parent

OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)
Z = 1.96

# ---- interp-tool styling (main_counterfactual_eval/analyze.py) ----
TOOL_DISPLAY = {"plain": "Transcript Baseline", "ao": "AO", "nla": "NLA", "sae": "SAE"}
TOOL_PALETTE = {"plain": "#cccccc", "ao": "#4363d8", "nla": "#f58231", "sae": "#3cb44b"}
TOOL_ORDER = ["plain", "ao", "nla", "sae"]

# ---- trained-model styling (direct_counterfactual_evaluation/analyze.py) ----
MODEL_DISPLAY = {"qwen8b": "Qwen3-8B", "q397": "Qwen3.5-397B"}
ARM_ORDER = ["base", "trained", "opus"]
ARM_DISPLAY = {"base": "Base", "opus": "Opus", "trained": "Trained"}
ARM_COLOR = {"base": "#999999", "opus": "#e6550d", "trained": "#3182bd"}
CHANCE_COLOR = "#e10600"


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def wilson_ci(k, n):
    p = k / n; denom = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denom
    half = Z / denom * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return center - half, center + half


# ---------- interp tools: plot_overall (verbatim style) ----------
def plot_overall(arms, ids, out_name):
    import textwrap
    accs, aus = [], []
    for a in arms:
        y = np.array([a["rows"][c]["is_true"] for c in ids]).astype(int)
        p = np.array([a["rows"][c]["prob_true"] for c in ids])
        pred = np.array([a["rows"][c]["pred_true"] for c in ids]).astype(int)
        k, n = int((pred == y).sum()), len(y)
        accs.append((k / n, *wilson_ci(k, n)))
        aus.append((auroc(y, p), *auroc_ci(y, p)))
    fig, ax = plt.subplots(figsize=(1.4 + 1.15 * len(arms), 4.6))
    x = np.arange(len(arms)); w = 0.38
    for metric, label, off, col in [(accs, "Accuracy", -w / 2, "#4c72b0"),
                                    (aus, "AUROC", w / 2, "#dd8452")]:
        vals = np.array([m[0] for m in metric])
        err = np.clip(np.vstack([vals - [m[1] for m in metric], [m[2] for m in metric] - vals]), 0, None)
        ax.bar(x + off, vals, w, yerr=err, label=label, color=col, edgecolor="black",
               linewidth=0.8, error_kw=dict(capsize=4, linewidth=1.3, ecolor="black"))
        for xi, v, hi in zip(x + off, vals, [m[2] for m in metric]):
            ax.text(xi, hi + 0.014, f"{v * 100:.1f}", ha="center", va="bottom", fontsize=11)
    ax.axhline(0.5, ls="--", color="#e10600", lw=1.8, zorder=5)
    labels = [textwrap.fill(TOOL_DISPLAY[a["cat"]], 15) for a in arms]
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=15)
    ax.tick_params(axis="y", labelsize=17)
    ax.set_ylabel("Score", fontsize=20); ax.set_ylim(0, 1.08)
    chance = Line2D([0], [0], color="#e10600", ls="--", lw=1.8, label="Chance")
    bar_h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=bar_h + [chance], frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=16)
    style(ax); fig.tight_layout()
    fig.savefig(OUT / out_name, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", OUT / out_name)


# ---------- trained models: two_panel (verbatim style) ----------
def _panel(ax, cells, title):
    models = [m for m in ("qwen8b", "q397") if m in cells]
    x = np.arange(len(models)); w = 0.8 / len(ARM_ORDER)
    for j, arm in enumerate(ARM_ORDER):
        vals = np.array([cells[m][arm][0] for m in models])
        cis = [cells[m][arm][1:] for m in models]
        err = np.clip(np.vstack([vals - [c[0] for c in cis], [c[1] for c in cis] - vals]), 0, None)
        off = (j - (len(ARM_ORDER) - 1) / 2) * w
        ax.bar(x + off, vals, w, yerr=err, color=ARM_COLOR[arm], edgecolor="black",
               linewidth=0.9, error_kw=dict(capsize=4, linewidth=1.4, ecolor="black"))
        for xi, v, hi in zip(x + off, vals, [c[1] for c in cis]):
            ax.text(xi, hi + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=13)
    ax.axhline(0.5, ls="--", color=CHANCE_COLOR, lw=2.0, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([MODEL_DISPLAY[m] for m in models])
    ax.set_title(title); style(ax)


def trained_panel(cells, out_name):
    """Single panel (one filter state): base/trained/opus AUROC for both models."""
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    _panel(ax, cells, "")
    ax.set_ylabel("AUROC"); ax.set_ylim(0, 1.05)
    handles = [Patch(facecolor=ARM_COLOR[a], edgecolor="black", label=ARM_DISPLAY[a]) for a in ARM_ORDER]
    handles.append(Line2D([0], [0], color=CHANCE_COLOR, ls="--", lw=2.0, label="Chance"))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.02))
    fig.savefig(OUT / out_name, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", OUT / out_name)


def main():
    # ---- Gemma interp tools: unfiltered (no filters) only ----
    run_dir = RUNS["gemma"][1]
    no_f, _all_f, _ = populations("gemma", run_dir)
    arms_rows = arm_files("gemma", run_dir)
    scored = set.intersection(*[set(r) for r in arms_rows.values()])
    arms = [{"cat": c, "rows": arms_rows[c]} for c in TOOL_ORDER]
    plot_overall(arms, sorted(no_f & scored), "gemma_unfiltered_overall_score.png")

    # ---- Trained models 8B + 397B: unfiltered (no filters) only, single panel ----
    cells = {}
    for slug in ("qwen8b", "q397"):
        rd = RUNS[slug][1]
        nf, _af, _ = populations(slug, rd)
        a = arm_files(slug, rd)  # plain(=opus)/base/trained
        ids = sorted(nf & set.intersection(*[set(r) for r in a.values()]))
        cell = {}
        for arm, key in (("base", "base"), ("trained", "trained"), ("opus", "plain")):
            y = np.array([a[key][c]["is_true"] for c in ids]).astype(int)
            p = np.array([a[key][c]["prob_true"] for c in ids])
            cell[arm] = (auroc(y, p), *auroc_ci(y, p))
        cells[slug] = cell
    trained_panel(cells, "unfiltered_direct_cf_auroc.png")


if __name__ == "__main__":
    main()
