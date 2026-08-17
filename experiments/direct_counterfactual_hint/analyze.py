"""Method A — direct counterfactual: the model is asked whether removing the injected hint would
flip its answer, scored by first-token Yes/No logprob (prob_true).

Reads the scored arms written by run_structured_claim_eval.py (base/trained, local logprobs) and
run_binary_eval.py --answerer plain (Opus transcript-reader baseline) over the unified hint banks:
  results/scored/<setting>_<model>_<arm>.jsonl
    setting in {mcq_explain, sycophancy}; model in {qwen3_8b, qwen35_397b};
    arm in {base, opus, trained}

HEADLINE (the paper figure): the POOLED result over the two in-distribution settings —
300 sycophancy + 300 CoT-hint (MMLU, explain) = 600 claims per (model, arm), each half balanced
150 supported / 150 inert, with a single error bar (bootstrap 95% CI for AUROC, Wilson for
accuracy). Opus is an external transcript reader (no privileged access): trained > Opus is the
introspection signal. Per-setting cells are reported as tables.

Writes: SUMMARY.md, plots/direct_counterfactual_hint_pooled_fig.png.
Run (CPU): python experiments/direct_counterfactual_hint/analyze.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chive import paths

HERE = Path(__file__).resolve().parent
SCORED = paths.experiment_data_dir("direct_counterfactual_hint") / "results" / "scored"
BANKS = paths.DATA_ROOT / "hint_banks" / "cot_hint"

SETTINGS = ["mcq_explain", "sycophancy"]
POOLED_SETTINGS = ["sycophancy", "mcq_explain"]  # the in-distribution pair -> the paper figure
MODELS = ["qwen3_8b", "qwen35_397b"]
ARMS = ["base", "trained", "opus"]  # opus = external transcript-reader reference (rightmost)
SETTING_DISPLAY = {"mcq_explain": "MMLU (explain)",
                   "sycophancy": "Sycophancy (AITA)"}
MODEL_DISPLAY = {"qwen3_8b": "Qwen3-8B", "qwen35_397b": "Qwen3.5-397B"}
ARM_DISPLAY = {"base": "Base", "opus": "Opus", "trained": "Trained"}
ARM_COLOR = {"base": "#999999", "opus": "#e6550d", "trained": "#3182bd"}
SYCO_DIR = {"qwen3_8b": "Qwen3-8B", "qwen35_397b": "Qwen3.5-397B-A17B-FP8"}

N_BOOT = 2000


def bank_path(setting: str, model: str) -> Path:
    if setting == "sycophancy":
        return paths.DATA_ROOT / "hint_banks" / "sycophancy_aita" / SYCO_DIR[model] / "cf_eval_bank.jsonl"
    return BANKS / setting / f"cf_eval_{model}.jsonl"


def scored_path(setting: str, model: str, arm: str) -> Path:
    return SCORED / f"{setting}_{model}_{arm}.jsonl"


# ---------- metrics ----------
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
    """Bootstrap 95% CI; fresh seed-0 rng per call so every cell is reproducible in isolation."""
    rng = np.random.default_rng(0)
    y = np.asarray(y); p = np.asarray(p); n = len(y); boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n); yb = y[idx]
        if 0 < yb.sum() < n:
            boots.append(auroc(yb, p[idx]))
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (float("nan"), float("nan"))


def wilson_ci(k, n):
    if n == 0:
        return float("nan"), float("nan")
    z = 1.96; p = k / n; denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z / denom * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return center - half, center + half


# ---------- loading ----------
def load_scored(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()] if path.exists() else []


def metrics_of(rows: list[dict]) -> dict:
    y = np.array([int(r["is_true"]) for r in rows])
    p = np.array([float(r["prob_true"]) for r in rows])
    pred = np.array([bool(r["pred_true"]) for r in rows])
    k = int((pred == y).sum()); n = len(rows)
    return {"n": n, "n_true": int(y.sum()), "n_false": int(n - y.sum()),
            "auroc": float(auroc(y, p)), "auroc_ci": list(auroc_ci(y, p)),
            "acc": k / n, "acc_ci": [float(v) for v in wilson_ci(k, n)],
            "brier": float(np.mean((p - y) ** 2))}


def cell_metrics(setting: str, model: str, arm: str):
    """Per-setting metrics, restricted to the bank's claim set. None if the arm is missing."""
    rows = load_scored(scored_path(setting, model, arm))
    bp = bank_path(setting, model)
    if not rows or not bp.exists():
        return None
    meta = {json.loads(l)["claim_id"]: json.loads(l) for l in open(bp) if l.strip()}
    rows = [r for r in rows if r["claim_id"] in meta]
    if not rows:
        return None
    d = metrics_of(rows)
    # per-cue AUROC (descriptive)
    by_cue = {}
    for cue in sorted({meta[r["claim_id"]]["cue"] for r in rows}):
        crows = [r for r in rows if meta[r["claim_id"]]["cue"] == cue]
        yc = np.array([int(r["is_true"]) for r in crows])
        pc = np.array([float(r["prob_true"]) for r in crows])
        by_cue[cue] = {"n": len(crows), "auroc": float(auroc(yc, pc)),
                       "n_true": int(yc.sum()), "n_false": int(len(crows) - yc.sum())}
    d["by_cue"] = by_cue
    return d


def pooled(model: str, arm: str):
    rows = []
    for s in POOLED_SETTINGS:
        rows += load_scored(scored_path(s, model, arm))
    return metrics_of(rows) if rows else None


def fpr_tpr(setting: str, model: str, arm: str) -> tuple[float, float]:
    """Per-setting FPR/TPR of the Yes/No decisions, restricted to the bank's claim set.

    FPR = fraction of negative (inert-cue) cases answered Yes; TPR = the same on positives."""
    ids = {json.loads(l)["claim_id"] for l in open(bank_path(setting, model)) if l.strip()}
    rows = [r for r in load_scored(scored_path(setting, model, arm)) if r["claim_id"] in ids]
    assert len(rows) == 300, (setting, model, arm, len(rows))
    neg = [bool(r["pred_true"]) for r in rows if not r["is_true"]]
    pos = [bool(r["pred_true"]) for r in rows if r["is_true"]]
    return sum(neg) / len(neg), sum(pos) / len(pos)


def fmt(x):
    return " nan" if x != x else f"{x:.3f}"


# ---------- figure (the paper figure) ----------
def make_figure(cells, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 15, "axes.titlesize": 22, "axes.labelsize": 21,
                         "xtick.labelsize": 18, "ytick.labelsize": 17, "legend.fontsize": 16})

    def grouped_bar(ax, metric, ci_key, title, show_ylabel, ylabel="score"):
        x = np.arange(len(MODELS)); w = 0.8 / len(ARMS)
        for j, arm in enumerate(ARMS):
            vals, errs = [], [[], []]
            for m in MODELS:
                d = cells.get((m, arm))
                if d is None:
                    vals.append(0.0); errs[0].append(0.0); errs[1].append(0.0); continue
                v, (lo, hi) = d[metric], d[ci_key]
                vals.append(v); errs[0].append(max(0, v - lo)); errs[1].append(max(0, hi - v))
            off = (j - (len(ARMS) - 1) / 2) * w
            ax.bar(x + off, vals, w, yerr=errs, color=ARM_COLOR[arm], edgecolor="black", linewidth=0.9,
                   error_kw=dict(capsize=4, linewidth=1.4, ecolor="black"))
            for xi, v, e_hi in zip(x + off, vals, errs[1]):
                if v > 0:  # sit the label above the upper error-bar cap, not on it
                    # fontsize 14: the 397B opus/trained values (0.94/0.96) are close and their
                    # labels touch at larger sizes in this narrow panel
                    ax.text(xi, v + e_hi + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=14)
        ax.axhline(0.5, ls="--", color="#e10600", lw=2.5, zorder=1)
        # wrap the long model name onto two lines (horizontal) so labels don't collide or need angling
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY[m].replace("-397B", "-\n397B") for m in MODELS],
                           rotation=0, ha="center")
        if title:  # empty => no on-plot title (the panel is labelled by its subcaption in the paper)
            ax.set_title(title)
        ax.set_ylim(0, 1.08)
        if show_ylabel:
            ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Two panels, accuracy + AUROC: with freeform self-explanation moved to the paper's
    # appendix this figure stands alone in the hint section, so the accuracy panel is
    # restored. Yes rates stay in SUMMARY.md and the appendix table.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8))
    grouped_bar(axes[0], "acc", "acc_ci", "Accuracy", show_ylabel=True)
    grouped_bar(axes[1], "auroc", "auroc_ci", "AUROC", show_ylabel=False)
    handles = [Patch(facecolor=ARM_COLOR[a], edgecolor="black", label=ARM_DISPLAY[a]) for a in ARMS]
    handles.append(Line2D([0], [0], color="#e10600", ls="--", lw=2.5, label="Chance"))
    # single legend above both panels (their titles sit below it via bbox_inches="tight")
    fig.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.97),
               ncol=4, columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", out_path)


# ---------- main ----------
def main():
    pool = {(m, a): pooled(m, a) for m in MODELS for a in ARMS}
    cells = {}
    for s in SETTINGS:
        for m in MODELS:
            for a in ARMS:
                d = cell_metrics(s, m, a)
                if d is not None:
                    cells[(s, m, a)] = d

    plots = HERE / "plots"; plots.mkdir(exist_ok=True)
    make_figure(pool, plots / "direct_counterfactual_hint_pooled_fig.png")

    L = ["# Direct counterfactual (method A) — hint introspection\n",
         "**Headline = the pooled in-distribution result**: 600 claims per (model, arm) = 300 sycophancy "
         "+ 300 CoT-hint (MMLU, explain), each 150 supported / 150 inert. AUROC with 95% bootstrap CI "
         "(2000 resamples, seed 0); accuracy with Wilson 95% CI. Opus = external transcript reader "
         "(no privileged access) -> trained-Opus is the introspection signal.\n",
         "> Regenerate: `python "
         "experiments/direct_counterfactual_hint/analyze.py`\n",
         "## Headline — pooled sycophancy + mcq_explain (the paper figure)", "",
         "| model | base AUROC | Opus AUROC | trained AUROC | trained−base | trained−Opus | base/Opus/tr acc |",
         "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        b, o, t = pool[(m, "base")], pool[(m, "opus")], pool[(m, "trained")]
        def a(d):
            return f"{d['auroc']:.3f} [{d['auroc_ci'][0]:.2f},{d['auroc_ci'][1]:.2f}]" if d else "—"
        d_tb = f"{t['auroc'] - b['auroc']:+.3f}" if (b and t) else ""
        d_to = f"{t['auroc'] - o['auroc']:+.3f}" if (o and t) else ""
        accs = "/".join(f"{pool[(m, x)]['acc']:.0%}" if pool[(m, x)] else "—" for x in ARMS)
        L.append(f"| {MODEL_DISPLAY[m]} | {a(b)} | {a(o)} | {a(t)} | {d_tb} | {d_to} | {accs} |")

    L += ["", "## Paper table (tab:hint_accuracy) — pooled accuracy + per-setting FPR", "",
          "FPR = fraction of negative (inert-cue) cases answered Yes; TPR for completeness.", "",
          "| model | arm | pooled acc [95% CI] | syc FPR | syc TPR | mcq FPR | mcq TPR |",
          "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for a in ARMS:
            d = pool[(m, a)]
            if d is None:
                continue
            sf, st = fpr_tpr("sycophancy", m, a)
            qf, qt = fpr_tpr("mcq_explain", m, a)
            L.append(f"| {MODEL_DISPLAY[m]} | {ARM_DISPLAY[a]} | "
                     f"{d['acc']:.3f} [{d['acc_ci'][0]:.3f}, {d['acc_ci'][1]:.3f}] | "
                     f"{sf:.3f} | {st:.3f} | {qf:.3f} | {qt:.3f} |")

    L += ["", "## Per-setting cells", "",
          "| setting | model | arm | n | n_true/n_false | AUROC [95% CI] | acc | Brier |",
          "|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for s in SETTINGS:
            for a in ARMS:
                d = cells.get((s, m, a))
                if d is None:
                    L.append(f"| {s} | {m} | {a} | — | (missing) | | | |")
                    continue
                L.append(f"| {s} | {m} | {a} | {d['n']} | {d['n_true']}/{d['n_false']} | "
                         f"{fmt(d['auroc'])} [{fmt(d['auroc_ci'][0])}, {fmt(d['auroc_ci'][1])}] | "
                         f"{d['acc']:.1%} | {fmt(d['brier'])} |")

    L += ["", "## Per-cue AUROC (descriptive)", "",
          "| setting | model | arm | cue | n | n_true/n_false | AUROC |",
          "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for s in SETTINGS:
            for a in ARMS:
                d = cells.get((s, m, a))
                if d is None:
                    continue
                for cue, cd in d["by_cue"].items():
                    L.append(f"| {s} | {m} | {a} | {cue} | {cd['n']} | "
                             f"{cd['n_true']}/{cd['n_false']} | {fmt(cd['auroc'])} |")
    (HERE / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("wrote", HERE / "SUMMARY.md")


if __name__ == "__main__":
    main()
