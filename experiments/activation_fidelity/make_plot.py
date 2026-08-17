"""Plot the activation-fidelity summary.

Three grouped-bar panels, KL0 vs KL1, transcript vs post-transcript:
  (1) cosine similarity to base   (higher = more faithful)
  (2) relative L2 drift            (lower  = more faithful)
  (3) top-K KL(trained || base)    (lower  = more faithful)
Activation panels show both captured layers (16, 24=NLA); KL is layer-independent (output dist).

Usage:  python make_plot.py --summary results/fidelity_summary.json --out results/fidelity.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REGIONS = ("transcript", "post")
MODELS = ("kl0", "kl1")
MODEL_LABEL = {"kl0": "KL0 (no reg)", "kl1": "KL1 (WildChat KL)"}
COLORS = {"kl0": "#d1495b", "kl1": "#2e86ab"}


def grouped_bars(ax, layers, summary, metric, title, ylabel, zoom=False, fmt="%.3f"):
    """x-axis = region x layer groups; two bars (kl0, kl1) per group."""
    groups = [(r, l) for r in REGIONS for l in layers]
    x = np.arange(len(groups))
    width = 0.38
    allvals = []
    for i, m in enumerate(MODELS):
        vals = [summary["activation"][f"{m}|L{l}|{r}"][metric] for (r, l) in groups]
        allvals += vals
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=MODEL_LABEL[m], color=COLORS[m])
        ax.bar_label(bars, fmt=fmt, fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\nL{l}" for (r, l) in groups])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    if zoom:  # cosine clusters near 1.0 — zoom so KL0 vs KL1 / transcript vs post is visible
        lo = min(allvals)
        ax.set_ylim(lo - (1.0 - lo) * 0.25, 1.0 + (1.0 - lo) * 0.05)


def kl_bars(ax, summary):
    x = np.arange(len(REGIONS))
    width = 0.38
    for i, m in enumerate(MODELS):
        vals = [summary["kl"][f"{m}|{r}"]["kl"] for r in REGIONS]
        ax.bar(x + (i - 0.5) * width, vals, width, label=MODEL_LABEL[m], color=COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(list(REGIONS))
    topk = summary["config"].get("topk_logprobs")
    if topk is not None:  # HTTP/offline path: top-K approximation
        cov = np.mean([summary["kl"][f"{m}|{r}"]["coverage"] for m in MODELS for r in REGIONS])
        ax.set_title(f"top-{topk} KL(trained || base)\n(mean top-K coverage {cov:.2f})")
    else:  # HF path: exact full-vocab KL
        ax.set_title("Exact KL(trained || base)\n(full vocab)")
    ax.set_ylabel("KL (nats)")
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    layers = summary["config"]["layers"]
    n_proc = summary["config"]["n_processed"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    grouped_bars(axes[0], layers, summary, "cosine",
                 "Cosine similarity to base\n(higher = more faithful)", "cosine", zoom=True, fmt="%.4f")
    grouped_bars(axes[1], layers, summary, "rel_l2",
                 "Relative L2 drift  ||Δ||²/||base||²\n(lower = more faithful)", "relative L2", fmt="%.2f")
    kl_bars(axes[2], summary)
    axes[0].legend()
    fig.suptitle(
        f"Activation fidelity: trained 8B adapters vs original 8B  "
        f"(n={n_proc} transcripts; L24 = NLA layer)", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
