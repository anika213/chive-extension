"""Analysis for the tool-hyperparameter sweep — writes SUMMARY.md + plots/.

Sweep arms (run by run_all.sh, Gemma-3-27B-IT cf7 eval set, answerer claude-opus-4-8/high):
  - tool calls per claim 2 / 10 (AO, NLA, SAE), positions fixed at 32
  - token positions per call 16 / 64 (NLA, SAE), calls fixed at 5
The 5-call / 32-position center points and the `plain` transcript-only baseline are READ BACK
from the canonical main-run result files (binary_evals/*_conc3.jsonl) — not re-run.

Population: the quality slice (gold_slice.quality_ids) ∩ claims scored by every arm, so all
numbers are paired on identical claims and sit on the paper's primary population (n≈1,380).

Run with the local venv (from anywhere):
    python analyze.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from chive import paths
from experiments.common.gold_slice import (
    quality_ids,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(str(paths.RUNS_DIR),
                       "opus_investigates_gemma3_27b_it_5000_thorough_n30")
RESULTS = str(paths.experiment_data_dir("tool_hyperparameter_sweep") / "results")
OUT = os.path.join(HERE, "plots")
SUMMARY_MD = os.path.join(HERE, "SUMMARY.md")

MODEL_EFFORT = "claude-opus-4-8_high"
TOOL_TAGS = {
    "ao": "ao_latentqa_cls_past_lens_gemma-3-27b-it",
    "nla": "nla_gemma3-27b-L41",
    "sae": "sae_gemma-scope-2-27b-it_layer_40_width_65k_l0_medium",
}
DISPLAY = {"ao": "AO", "nla": "NLA", "sae": "SAE"}
PALETTE = {"ao": "#4363d8", "nla": "#f58231", "sae": "#3cb44b"}

CALLS_SWEEP = [2, 5, 10]        # 5 = canonical readback
POS_SWEEP = [16, 32, 64]        # 32 = canonical readback; NLA/SAE only
CANONICAL = {("calls", 5), ("pos", 32)}

RNG = np.random.default_rng(0)
N_BOOT = 2000


# ---------- metrics (conventions of main_counterfactual_eval/analyze.py) ----------
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
    assert npos and nneg, "AUROC undefined: single-class labels"
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


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


# ---------- loading ----------
def load_rows(path):
    assert os.path.exists(path), f"missing arm result file: {path}"
    rows = {}
    for line in open(path):
        r = json.loads(line)
        assert r["claim_id"] not in rows, f"duplicate claim_id {r['claim_id']} in {path}"
        rows[r["claim_id"]] = r
    return rows


def counting_calls(row):
    """Budget-consuming tool calls (excludes show_tokens + the terminal submit), recomputed
    from the stored agent transcript so canonical arms (written before the n_counting_calls
    field existed) and sweep arms are counted identically."""
    return sum(1 for turn in row["agent_transcript"] for tc in turn["tool_calls"]
               if tc["name"] not in ("show_tokens", "submit_answer"))


def arm_path(tool, knob, value):
    if (knob, value) in CANONICAL:
        return os.path.join(RUN_DIR, "binary_evals", f"{TOOL_TAGS[tool]}_{MODEL_EFFORT}_conc3.jsonl")
    label = f"tc{value}" if knob == "calls" else f"pos{value}"
    return os.path.join(RESULTS, f"{TOOL_TAGS[tool]}_{MODEL_EFFORT}_{label}.jsonl")


def load_all():
    """-> (plain_rows, {(tool, knob, value): rows})  with knob in {calls, pos}."""
    plain = load_rows(os.path.join(RUN_DIR, "binary_evals", f"plain_{MODEL_EFFORT}_conc3.jsonl"))
    arms = {}
    for tool in TOOL_TAGS:
        for v in CALLS_SWEEP:
            arms[(tool, "calls", v)] = load_rows(arm_path(tool, "calls", v))
        if tool != "ao":  # the AO has no positions knob (it selects transcript segments)
            for v in POS_SWEEP:
                if v == 32:
                    arms[(tool, "pos", v)] = arms[(tool, "calls", 5)]  # same file/object
                else:
                    arms[(tool, "pos", v)] = load_rows(arm_path(tool, "pos", v))
    return plain, arms


# ---------- analysis ----------
def stats_for(rows, ids, plain, y):
    p = np.array([rows[c]["prob_true"] for c in ids])
    pp = np.array([plain[c]["prob_true"] for c in ids])
    au = auroc(y, p); lo, hi = auroc_ci(y, p)
    dlo, dhi = paired_delta_ci(y, p, pp)
    forced = np.mean([rows[c]["stop_reason"] == "submitted_final" for c in ids])
    ncalls = [counting_calls(rows[c]) for c in ids]
    return dict(auroc=au, ci=(lo, hi), delta=au - auroc(y, pp), delta_ci=(dlo, dhi),
                acc=float(np.mean([rows[c]["correct"] for c in ids])),
                brier=brier(y, p), forced=float(forced),
                med_calls=float(np.median(ncalls)), mean_calls=float(np.mean(ncalls)))


def main():
    os.makedirs(OUT, exist_ok=True)
    plain, arms = load_all()
    q = quality_ids(RUN_DIR)
    ids = sorted(q & set(plain) & set.intersection(*[set(r) for r in arms.values()]))
    y = np.array([plain[c]["is_true"] for c in ids]).astype(int)
    print(f"[analyze] quality slice ∩ all arms: n={len(ids)} ({y.sum()} T / {len(y) - y.sum()} F)")

    plain_p = np.array([plain[c]["prob_true"] for c in ids])
    plain_au = auroc(y, plain_p); plain_ci = auroc_ci(y, plain_p)
    stats = {k: stats_for(rows, ids, plain, y) for k, rows in arms.items()}

    # ----- figure: ΔAUROC vs plain, one panel per knob -----
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, knob, sweep, tools, xlabel in (
        (axes[0], "calls", CALLS_SWEEP, ["ao", "nla", "sae"], "max tool calls per claim"),
        (axes[1], "pos", POS_SWEEP, ["nla", "sae"], "token positions per call (NLA/SAE)"),
    ):
        for ti, tool in enumerate(tools):
            xs = np.array(sweep, float) * (1 + 0.035 * (ti - 1))  # jitter so CIs don't overlap
            s = [stats[(tool, knob, v)] for v in sweep]
            ys = [d["delta"] for d in s]
            yerr = np.array([[d["delta"] - d["delta_ci"][0] for d in s],
                             [d["delta_ci"][1] - d["delta"] for d in s]])
            ax.errorbar(xs, ys, yerr=yerr, marker="o", ms=4.5, capsize=3, lw=1.6,
                        color=PALETTE[tool], label=DISPLAY[tool])
        ax.axhline(0, color="#999999", lw=1, ls="--", zorder=0)
        ax.set_xscale("log"); ax.set_xticks(sweep); ax.set_xticklabels([str(v) for v in sweep])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())  # else log-scale 2x10^1 etc. labels collide
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_xlabel(xlabel)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("ΔAUROC vs transcript-only")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Tool-hyperparameter sweep, Gemma-3-27B-IT (n={len(ids)}, paired)", fontsize=11)
    fig.tight_layout()
    fig_path = os.path.join(OUT, "tool_hyperparameter_sweep.png")
    fig.savefig(fig_path, dpi=200)
    print(f"[analyze] wrote {fig_path}")

    # ----- SUMMARY.md -----
    def fmt_row(name, d):
        return (f"| {name} | {d['auroc']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}] "
                f"| {d['delta']:+.3f} [{d['delta_ci'][0]:+.3f}, {d['delta_ci'][1]:+.3f}] "
                f"| {d['acc']:.1%} | {d['brier']:.3f} | {d['forced']:.1%} "
                f"| {d['med_calls']:.0f} ({d['mean_calls']:.1f}) |")

    L = []
    L.append("# Tool-hyperparameter sweep — results\n")
    L.append("Auto-generated by `analyze.py`; do not edit numbers by hand. See README.md for design "
             "and `run_all.sh` for end-to-end reproduction.\n")
    L.append(f"**Setup.** Gemma-3-27B-IT cf7 eval set; answerer `claude-opus-4-8` effort `high` "
             f"(identical to the main run). Population: quality slice ∩ all arms, **n={len(ids)}** "
             f"({y.sum()} T / {len(y) - y.sum()} F), fully paired. 95% bootstrap CIs "
             f"({N_BOOT} resamples); ΔAUROC CIs are paired against `plain` on the same claims. "
             f"The 5-call / 32-position rows are the canonical main-run arms read back; the other "
             f"rows are freshly scored sweep arms. 'calls' counts budget-consuming tool reads "
             f"(`show_tokens` is free); positions = token positions returned per NLA/SAE call.\n")
    L.append(f"**Baseline.** `plain` (transcript-only): AUROC {plain_au:.3f} "
             f"[{plain_ci[0]:.3f}, {plain_ci[1]:.3f}].\n")
    L.append("**Caveat (interface version).** The canonical main-run arms (June) used the tools' "
             "earlier segment-based selection input (`start_segment`/`end_segment`); the fresh "
             "sweep arms use the current selection-string interface (segment runs, token ranges, "
             "scattered tokens). So the *(main run)* rows differ from the sweep rows in interface "
             "version as well as knob value. Empirically this is also a wash — every fresh arm "
             "sits within ~±0.01 AUROC of its canonical center point — and sweep-vs-sweep "
             "comparisons (2 vs 10 calls; 16 vs 64 positions) are interface-identical.\n")
    hdr = ("| arm | AUROC [95% CI] | ΔAUROC vs plain [95% CI] | acc@0.5 | Brier | forced submits "
           "| med (mean) calls used |\n|---|---|---|---|---|---|---|")
    L.append("## Tool calls per claim (positions fixed at 32)\n")
    L.append(hdr)
    for tool in ("ao", "nla", "sae"):
        for v in CALLS_SWEEP:
            name = f"{DISPLAY[tool]} — {v} calls" + (" *(main run)*" if v == 5 else "")
            L.append(fmt_row(name, stats[(tool, "calls", v)]))
    L.append("\n## Token positions per call (calls fixed at 5; NLA/SAE only — the AO selects "
             "transcript segments, not token positions)\n")
    L.append(hdr)
    for tool in ("nla", "sae"):
        for v in POS_SWEEP:
            name = f"{DISPLAY[tool]} — {v} positions" + (" *(main run)*" if v == 32 else "")
            L.append(fmt_row(name, stats[(tool, "pos", v)]))
    L.append(f"\n![sweep](plots/tool_hyperparameter_sweep.png)\n")
    L.append("## Additionally tried (not re-run on the current bank)\n")
    L.append("- **Full-dump / no-selection ceiling (NLA).** Every token's NLA verbalization dumped "
             "into a single Opus call (no agent loop, no selection step), on the ≤800-token-transcript "
             "slice of the OLD pre-cf7 n1000 bank (n=398): full dump AUROC 0.784 vs plain 0.806 vs "
             "the 5-call agent arm 0.801 — removing the targeting/selection bottleneck entirely does "
             "not help. Write-up: `NLA_FULL_DUMP_RESULTS.md` (raw result JSONLs were not "
             "retained; numbers are from that write-up).")
    L.append("- **Earlier ad-hoc 32→64-position NLA run** (old bank, n=500; not included in this "
             "release): a wash vs 32 — superseded by the 64-position arm above.\n")
    with open(SUMMARY_MD, "w") as f:
        f.write("\n".join(L))
    print(f"[analyze] wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
