"""Authoritative analysis for the MAIN counterfactual-tool eval (binary causal-claim eval, cf7).

Self-contained (no cross-folder imports): reads each run's scored arms from
<RUNS_DIR>/<run>/binary_evals/*_<answerer>_conc3.jsonl and writes, next to this
script, plots/ + a single SUMMARY.md.

cf7 claim format: one unsigned claim per factor, claim_type in {supported, inert},
is_true = (claim_type == "supported"). prob_true is P(factor is supported / matters). Arms are
auto-discovered: `cf` (counterfactual, the headline tool) vs AO/NLA/SAE (activation-readout tools)
vs `plain` (transcript-only baseline). Metrics use the per-run common-claim intersection so the
comparison is paired/apples-to-apples even if an arm is a few claims short.

SUMMARY.md has two parts: the standard per-run tables (AUROC/acc/Brier + paired ΔAUROC vs plain)
and the post-hoc "clean / cf-solvable / baseline-unsure" subset (the strongest test for the tools).

Run with the local venv (from anywhere):
    python analyze.py                 # both parts (default)
    python analyze.py --mode standard # tables/plots only
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from chive import paths
from experiments.common.gold_slice import (
    gold_funnel,
    quality_funnel,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = str(paths.RUNS_DIR)
OUT = os.path.join(HERE, "plots")          # self-contained: plots live next to this script
SUMMARY_MD = os.path.join(HERE, "SUMMARY.md")  # single merged summary (standard + posthoc)
SUFFIX = "_claude-opus-4-8_high_conc3.jsonl"

# slug -> (display run name, run dir)
RUNS = {
    "gemma": ("gemma-3-27b-it", "opus_investigates_gemma3_27b_it_5000_thorough_n30"),
    "qwen8b": ("qwen3-8b", "opus_investigates_qwen3_8b_5000_thorough_n30"),
    "q397_5000": ("qwen3.5-397b (5000)", "opus_investigates_qwen3p5_397b_5000_thorough_n30"),
    "q397_petri": ("qwen3.5-397b (petri)", "opus_investigates_qwen3p5_397b_petri_thorough_n30"),
    "qwen8b_petri": ("qwen3-8b (petri)", "opus_investigates_qwen3_8b_petri_thorough_n30"),
}

CAT_ORDER = ["plain", "cf", "ao", "nla", "sae"]
DISPLAY = {"plain": "Transcript Baseline", "cf": "Counterfactual", "ao": "AO", "nla": "NLA", "sae": "SAE"}
# plain = baseline (gray); cf = headline tool (purple); AO blue, NLA orange, SAE green.
PALETTE = {"plain": ["#cccccc"], "cf": ["#9b30ff"], "ao": ["#4363d8", "#42a5f5"],
           "nla": ["#f58231"], "sae": ["#3cb44b"]}

RNG = np.random.default_rng(0)
N_BOOT = 2000
Z = 1.96


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


def wilson_ci(k, n):
    p = k / n; denom = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denom
    half = Z / denom * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return center - half, center + half


def mean_ci(vals):
    vals = np.asarray(vals, float); n = len(vals)
    boots = [vals[RNG.integers(0, n, n)].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return vals.mean(), lo, hi


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------- loading ----------
def discover_arms(rd):
    by_cat = {}
    for f in sorted(glob.glob(os.path.join(rd, "binary_evals", "*" + SUFFIX))):
        if "toolcalls" in f:
            continue
        base = os.path.basename(f)[: -len(SUFFIX)]
        # Exclude the additional, non-released Qwen3-8B AO (trained on extra synthetic
        # data; the paper's tool-sources paragraph notes it gave no uplift). The paper
        # uses only the released oracle from Karvonen et al., so it never appears as a
        # bar/row; with one AO file left, the arm is labeled simply "AO".
        if base.startswith("ao_500k"):
            continue
        by_cat.setdefault(base.split("_")[0], []).append((base, f))
    arms = []
    cats = CAT_ORDER + [c for c in by_cat if c not in CAT_ORDER]
    for cat in cats:
        lst = by_cat.get(cat, [])
        for i, (base, f) in enumerate(lst):
            disp = DISPLAY.get(cat, cat.upper())
            if len(lst) > 1:
                tag = "multi" if "500k" in base else ("single" if "addition" in base else str(i + 1))
                disp = f"{disp} ({tag})"
            colors = PALETTE.get(cat, ["#777777"])
            rows = {json.loads(l)["claim_id"]: json.loads(l) for l in open(f)}
            arms.append({"key": base, "cat": cat, "display": disp,
                         "color": colors[i % len(colors)], "rows": rows})
    return arms


def common_ids(arms):
    return sorted(set.intersection(*[set(a["rows"]) for a in arms]))


def arr(arm, ids, field):
    return np.array([arm["rows"][c][field] for c in ids])


# ---------- plots ----------
def plot_overall(arms, ids, slug, run_name, suffix="_overall_score", title=None, drop_cats=()):
    import textwrap
    from matplotlib.lines import Line2D
    arms = [a for a in arms if a["cat"] not in drop_cats]  # e.g. drop the 'cf' arm from the paper fig
    accs, aus = [], []
    for a in arms:
        y = arr(a, ids, "is_true").astype(int)
        p = arr(a, ids, "prob_true")
        pred = arr(a, ids, "pred_true").astype(int)
        k, n = int((pred == y).sum()), len(y)
        accs.append((k / n, *wilson_ci(k, n)))
        aus.append((auroc(y, p), *auroc_ci(y, p)))
    # Compact canvas so the figure is only lightly downscaled in the paper (keeps fonts legible);
    # font sizes set per-element so this styling does NOT leak into plot_difficulty/plot_calibration.
    fig, ax = plt.subplots(figsize=(1.4 + 1.15 * len(arms), 4.6))
    x = np.arange(len(arms)); w = 0.38
    for i, (metric, label, off, col) in enumerate(
        [(accs, "Accuracy", -w / 2, "#4c72b0"), (aus, "AUROC", w / 2, "#dd8452")]
    ):
        vals = np.array([m[0] for m in metric])
        # clip to >=0: when a point estimate equals its CI bound (e.g. cf AUROC=1.0 on the
        # posthoc subset) float rounding can make val-lo or hi-val a tiny negative, which errorbar rejects
        err = np.clip(np.vstack([vals - [m[1] for m in metric], [m[2] for m in metric] - vals]), 0, None)
        ax.bar(x + off, vals, w, yerr=err, label=label, color=col, edgecolor="black",
               linewidth=0.8, error_kw=dict(capsize=4, linewidth=1.3, ecolor="black"))
        for xi, v, hi in zip(x + off, vals, [m[2] for m in metric]):
            ax.text(xi, hi + 0.014, f"{v * 100:.1f}", ha="center", va="bottom", fontsize=11)
    ax.axhline(0.5, ls="--", color="#e10600", lw=1.8, zorder=5)
    # long arm labels wrapped onto two lines (kept horizontal, not rotated/angled)
    labels = [textwrap.fill(a["display"], 15) for a in arms]
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=15)
    ax.tick_params(axis="y", labelsize=17)
    ax.set_ylabel("Score", fontsize=20); ax.set_ylim(0, 1.08)  # headroom so value labels have room
    if title:  # None (the per-run default) => no on-plot title; the model/n go in the paper caption
        ax.set_title(title)
    # legend above the axes; Chance is a legend entry (no inline "chance" text on the plot)
    chance = Line2D([0], [0], color="#e10600", ls="--", lw=1.8, label="Chance")
    bar_h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=bar_h + [chance], frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=16)
    style(ax)
    fig.tight_layout()
    path = os.path.join(OUT, f"{slug}{suffix}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig); print("wrote", path)


DIFF_BINS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5001)]
DIFF_LABELS = ["0.0-0.1\n(hardest)", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5\n(easiest)"]


def plot_difficulty(arms, ids, slug, run_name):
    plain = next(a for a in arms if a["cat"] == "plain")
    pp = {c: plain["rows"][c]["prob_true"] for c in ids}
    fig, ax = plt.subplots(figsize=(8, 5)); x = np.arange(len(DIFF_BINS))
    for a in arms:
        ys = []
        for lo, hi in DIFF_BINS:
            b = [c for c in ids if lo <= abs(pp[c] - 0.5) < hi]
            ys.append(auroc(arr(a, b, "is_true").astype(int), arr(a, b, "prob_true")) if b else np.nan)
        ax.plot(x, ys, marker="o", label=a["display"], color=a["color"],
                lw=2.2 if a["cat"] == "plain" else 1.8, ls="--" if a["cat"] == "plain" else "-")
    ns = [sum(1 for c in ids if lo <= abs(pp[c] - 0.5) < hi) for lo, hi in DIFF_BINS]
    ax.axhline(0.5, ls=":", color="gray", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f"{lab}\nn={n}" for lab, n in zip(DIFF_LABELS, ns)], fontsize=8)
    ax.set_xlabel("transcript-only confidence bin  |p - 0.5|"); ax.set_ylabel("AUROC")
    ax.set_ylim(0.45, 1.02)
    ax.set_title(f"{run_name} — AUROC by transcript-only difficulty")
    ax.legend(frameon=False, loc="lower left"); style(ax)
    fig.tight_layout()
    path = os.path.join(OUT, f"{slug}_difficulty_curve.png")
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig); print("wrote", path)


CELLS = ["inert", "supported"]  # ground truth P(supported): inert=0, supported=1


def plot_calibration(arms, ids, slug, run_name):
    fig, ax = plt.subplots(figsize=(2 + 1.3 * len(arms), 5)); x = np.arange(len(CELLS))
    w = 0.8 / max(len(arms), 1)
    for j, a in enumerate(arms):
        vals, los, his = [], [], []
        for ct in CELLS:
            pc = [a["rows"][c]["prob_true"] for c in ids if a["rows"][c]["claim_type"] == ct]
            m, lo, hi = mean_ci(pc)
            vals.append(m); los.append(lo); his.append(hi)
        off = (j - (len(arms) - 1) / 2) * w
        err = np.vstack([np.array(vals) - los, np.array(his) - vals])
        ax.bar(x + off, vals, w, yerr=err, label=a["display"], color=a["color"],
               edgecolor="black", linewidth=0.8, hatch="//" if a["cat"] == "plain" else None,
               error_kw=dict(capsize=3, linewidth=1.0, ecolor="black"))
    for i, ct in enumerate(CELLS):
        gt = 1.0 if ct == "supported" else 0.0
        ax.plot([i - 0.45, i + 0.45], [gt, gt], color="black", ls=":", lw=1.5)
    ax.plot([], [], color="black", ls=":", lw=1.5, label="ground truth")
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n(truth {1 if c=='supported' else 0})" for c in CELLS])
    ax.set_ylabel('mean P(agent concludes "factor is supported")'); ax.set_ylim(0, 1.0)
    ax.set_title(f"{run_name} — calibration by claim type")
    ax.legend(frameon=False, ncol=min(len(arms) + 1, 4), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    style(ax)
    fig.tight_layout()
    path = os.path.join(OUT, f"{slug}_calibration_cells.png")
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig); print("wrote", path)


# ---------- summary table ----------
def fmt(x):
    return " nan" if x != x else f"{x:.3f}"


def write_summary(all_runs):
    L = ["# Binary causal-claim eval — final results (cf7, QUALITY slice)\n",
         "Answerer: `claude-opus-4-8`, effort `high`. AUROC via rank-sum; 95% bootstrap CIs "
         "(2000 resamples, seed 0). `acc` = pred_true vs is_true. Brier = mean((prob_true − is_true)²). "
         "ΔAUROC is paired (tool − plain) over the identical claim set. cf7: one unsigned claim per "
         "factor, `is_true = supported`.\n",
         "> **Population: the QUALITY slice** (concreteness≥3 ∩ cf-correct ∩ not-confounded, via "
         "`gold_slice.quality_funnel`) — the paper's primary population, consistent with the "
         "direct-prediction and self-explanation figures. The comparator-dependent baseline-unsure "
         "gate is NOT applied here (see the post-hoc/gold section below for that stress slice). "
         "Every arm within a run scores the identical slice, so paired analysis uses the common "
         "set. ~3 dead-zone claims carry known common-mode label noise: "
         "<0.1% absolute AUROC, cancels in paired Δ.\n",
         "> Regenerate with `analyze.py` (also writes the plots/ figures); reproduce the whole "
         "eval from scratch with `run_all.sh`.\n"]
    for run_name, arms, ids in all_runs:
        plain = next(a for a in arms if a["cat"] == "plain")
        py = arr(plain, ids, "is_true").astype(int); pp = arr(plain, ids, "prob_true")
        nT = int(py.sum())
        L.append(f"\n## {run_name}  —  n={len(ids)} claims  ({nT} supported / {len(py) - nT} inert)\n")
        L.append("| arm | detail | AUROC [95% CI] | acc | Brier | ΔAUROC vs plain [95% CI] |")
        L.append("|---|---|---|---|---|---|")
        for a in arms:
            y = arr(a, ids, "is_true").astype(int); p = arr(a, ids, "prob_true")
            pred = arr(a, ids, "pred_true").astype(int)
            au = auroc(y, p); lo, hi = auroc_ci(y, p)
            acc = float((pred == y).mean()); brier = float(np.mean((p - y) ** 2))
            detail = a["key"][len(a["cat"]) + 1:] if "_" in a["key"] else "—"
            if a["cat"] == "plain":
                delta = "—"
            else:
                dlo, dhi = paired_delta_ci(py, p, pp)
                delta = f"{au - auroc(py, pp):+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
            L.append(f"| {DISPLAY.get(a['cat'], a['cat'])} | {detail} | {fmt(au)} [{fmt(lo)}, {fmt(hi)}] "
                     f"| {acc:.1%} | {fmt(brier)} | {delta} |")
    open(SUMMARY_MD, "w").write("\n".join(L) + "\n")
    print("wrote", SUMMARY_MD)


# ---------- post-hoc "solvable / clean / non-obvious" subset ----------
# The subset is the canonical gold slice (concreteness → cf-correct → not-confounded →
# baseline-unsure), defined once in experiments/common/gold_slice.py and recomputed from source —
# we pass restrict_to=ids so it's intersected with this run's all-arm common claim set.
def run_posthoc(all_runs, lo, hi):
    L = ["\n\n---\n\n# Post-hoc subset: clean, cf-solvable, baseline-unsure\n",
         f"Per run, the subset = claims where **(1)** the counterfactual arm is correct, **(2)** the "
         f"Sonnet `edit_cleanliness` rating is not `confounded`, and **(3)** the transcript-only "
         f"baseline is unsure (`prob_true` ∈ [{lo}, {hi}]). This is the strongest test for the "
         "activation tools: a clean, answerable claim where the transcript alone is ambiguous — so "
         "there is genuine headroom and a recoverable answer. If AO/NLA/SAE still don't beat plain "
         "here, they don't help even on the slice most favorable to them.\n",
         "> **Note:** the `counterfactual` row is tautologically AUROC 1.0 / acc 100% — the subset is "
         "*defined* by cf being correct. The meaningful comparison is **transcript-only vs AO/NLA/SAE**: "
         "on this clean, recoverable, baseline-unsure slice the activation tools still sit at/below plain.\n"]
    print("\n" + "=" * 72)
    print(f"POST-HOC SUBSET  (cf-correct & not-confounded & plain prob in [{lo:.2f},{hi:.2f}])")
    print("=" * 72)
    for slug, run_name, arms, _qids, common, run_dir in all_runs:
        sub, fn = gold_funnel(run_dir, lo, hi, restrict_to=common)
        plain = next(a for a in arms if a["cat"] == "plain")
        head = (f"\n## {run_name} — subset n={len(sub)}  "
                f"(funnel: scored {fn['scored']} → cf-correct {fn['cf_correct']} → "
                f"not-confounded {fn['not_confounded']} → gold {fn['gold']})")
        print(head); L.append(head + "\n")
        if len(sub) < 20:
            note = f"  ⚠ subset too small for stable metrics (n={len(sub)})"
            print(note); L.append(note + "\n")
        hdr = "| arm | detail | n | AUROC | acc | ΔAUROC vs plain |"; sep = "|---|---|---|---|---|---|"
        print(hdr); print(sep); L += [hdr, sep]
        py = arr(plain, sub, "is_true").astype(int) if sub else None
        pp = arr(plain, sub, "prob_true") if sub else None
        for a in arms:
            if not sub:
                row = f"| {DISPLAY.get(a['cat'], a['cat'])} | — | 0 | nan | nan | — |"
            else:
                y = arr(a, sub, "is_true").astype(int); p = arr(a, sub, "prob_true")
                pred = arr(a, sub, "pred_true").astype(int)
                au = auroc(y, p); acc = float((pred == y).mean())
                delta = "—" if a["cat"] == "plain" else f"{au - auroc(py, pp):+.3f}"
                detail = a["key"][len(a["cat"]) + 1:] if "_" in a["key"] else "—"
                row = (f"| {DISPLAY.get(a['cat'], a['cat'])} | {detail} | {len(sub)} | "
                       f"{fmt(au)} | {acc:.1%} | {delta} |")
            print(row); L.append(row)
        if len(sub) >= 20:
            plot_overall(arms, sub, slug, run_name, suffix="_posthoc_solvable",
                         title=f"{run_name} — clean, cf-solvable, baseline-unsure (n={len(sub)})")
    with open(SUMMARY_MD, "a") as fh:  # append the posthoc section to the single SUMMARY.md
        fh.write("\n".join(L) + "\n")
    print("\nappended post-hoc section to", SUMMARY_MD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["standard", "posthoc", "both"], default="both",
                    help="standard = per-run overall/difficulty/calibration + FINAL_SUMMARY; "
                         "posthoc = the clean/cf-solvable/baseline-unsure subset table + plots")
    ap.add_argument("--posthoc-lo", type=float, default=0.25, help="transcript-only prob_true lower bound")
    ap.add_argument("--posthoc-hi", type=float, default=0.75, help="transcript-only prob_true upper bound")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    all_runs = []
    for slug, (run_name, d) in RUNS.items():
        rd = os.path.join(RUNS_DIR, d)
        arms = discover_arms(rd)
        if not arms:
            print(f"[skip] {slug}: no result files"); continue
        common = common_ids(arms)
        # PAPER PRIMARY slice: concreteness ∩ cf-correct ∩ not-confounded (the comparator-dependent
        # baseline-unsure gate is excluded here — it lives in the posthoc/gold slice below).
        ids, qfn = quality_funnel(rd, restrict_to=common)
        print(f"\n=== {run_name}: {len(arms)} arms, {len(common)} common → {len(ids)} quality claims "
              f"(cf-correct {qfn['cf_correct']} → not-confounded {qfn['not_confounded']}) ===")
        if args.mode in ("standard", "both"):
            plot_overall(arms, ids, slug, run_name, drop_cats={"cf"})  # 'counterfactual' arm not shown in the tool-eval fig
            plot_difficulty(arms, ids, slug, run_name)
            plot_calibration(arms, ids, slug, run_name)
        all_runs.append((slug, run_name, arms, ids, common, rd))
    if args.mode in ("standard", "both"):
        write_summary([(rn, a, i) for (s, rn, a, i, c, rd) in all_runs])
    if args.mode in ("posthoc", "both"):
        run_posthoc(all_runs, args.posthoc_lo, args.posthoc_hi)


if __name__ == "__main__":
    main()
