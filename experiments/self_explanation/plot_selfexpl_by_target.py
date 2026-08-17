"""Counterfactual-simulatability bar chart for the Qwen3-8B target (appendix).

Grouped bars (was a line plot; reviewer asked for bars): x = simulator (weak -> strong),
one bar per arm (transcript-only / base self-explanation / trained self-explanation) with
bootstrap 95% CIs. Same underlying analysis as analyze.py (imported).

plot_selfexpl_397b.py reuses bar_figure() for the companion Qwen3.5-397B figure.

Run: PYTHONPATH=. python experiments/self_explanation/plot_selfexpl_by_target.py
Writes plots/figure_selfexpl_by_target.png (plots/ to match the other experiments dirs).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
from experiments.self_explanation.analyze import analyze_run, SIMS  # noqa: E402

OUT = HERE / "plots" / "figure_selfexpl_by_target.png"
# Arm order/colors follow the paper's base -> trained convention: gray baseline,
# then the two explanation arms.
ARMS = [
    ("plain", "Transcript Only", "#999999"),
    ("selfexpl_base", "Base Self-Explanation", "#2ca02c"),
    ("selfexpl", "Trained Self-Explanation", "#3182bd"),
]


def bar_figure(recs: list[dict], out: Path) -> None:
    """Grouped absolute-AUROC bars: x = simulator (weak -> strong), one bar per arm."""
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 17, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 14})
    by_sim = {r["sim"]: r for r in recs if "error" not in r}
    sims = sorted(by_sim, key=lambda s: by_sim[s]["plain_auroc"][0])
    labels = [SIMS.get(s, (99, s, "?"))[1] for s in sims]
    # wrap "Model-size" at the last hyphen onto two lines, so the labels can stay horizontal
    labels = [f"{l.rsplit('-', 1)[0]}-\n{l.rsplit('-', 1)[1]}" if "-" in l else l for l in labels]
    x = np.arange(len(sims))

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    w = 0.8 / len(ARMS)
    for j, (arm, label, color) in enumerate(ARMS):
        vals, errs = [], [[], []]
        for s in sims:
            a, lo, hi = by_sim[s][f"{arm}_auroc"]
            vals.append(a); errs[0].append(a - lo); errs[1].append(hi - a)
        off = (j - (len(ARMS) - 1) / 2) * w
        ax.bar(x + off, vals, w, yerr=errs, color=color, edgecolor="black", linewidth=0.9,
               label=label, error_kw=dict(capsize=3, linewidth=1.2, ecolor="black"))
        for xi, v, e_hi in zip(x + off, vals, errs[1]):
            ax.text(xi, v + e_hi + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=11)
    ax.axhline(0.5, ls="--", color="#e10600", lw=2.0, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Simulator (Weak → Strong)")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 0.92)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    bar_figure(analyze_run("qwen8b"), OUT)


if __name__ == "__main__":
    main()
