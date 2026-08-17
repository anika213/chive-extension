"""Analyze cross-model behavior-similarity verdicts (see README.md).

Reads results/verdicts.jsonl (from grade_cross.py), computes rate agreement
between each subject model's stored ground-truth baseline rate and the other
model's cross-graded rate, and writes SUMMARY.md + plots/.
"""

import functools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

print = functools.partial(print, flush=True)

from chive import paths

HERE = Path(__file__).parent
VERDICTS_PATH = paths.experiment_data_dir("model_behavior_similarity") / "results" / "verdicts.jsonl"
SUMMARY_PATH = HERE / "SUMMARY.md"
PLOTS_DIR = HERE / "plots"

# Unit counts after the 2026-07-28 eval-blocklist filter (data/eval_blocklist.json)
# removed train/eval-leaked prompts (pre-filter: 698/643 cross, 50/50 self).
CROSS_DIRECTIONS = {
    "qwen_bank_on_llama": ("Qwen3-8B bank", "Llama-3.1-8B", 649),
    "llama_bank_on_qwen": ("Llama-3.1-8B bank", "Qwen3-8B", 602),
}
SELF_DIRECTIONS = {
    "self_regrade_qwen_bank": ("Qwen3-8B bank self re-grade", 47),
    "self_regrade_llama_bank": ("Llama-3.1-8B bank self re-grade", 43),
}
# "behavior present" if k >= threshold (out of n=30)
PRESENCE_THRESHOLDS = {">0% (k>=1)": 1, ">=10% (k>=3)": 3, ">=30% (k>=9)": 9, ">=50% (k>=15)": 15}


def cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float(np.mean(a == b))
    pe = float(np.mean(a) * np.mean(b) + np.mean(~a) * np.mean(~b))
    if pe == 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def load_rows() -> dict[str, list[dict]]:
    by_dir: dict[str, list[dict]] = {}
    seen: set[tuple[str, str]] = set()
    for line in open(VERDICTS_PATH):
        r = json.loads(line)
        key = (r["direction"], r["prompt_id"])
        assert key not in seen, f"duplicate row {key}"
        seen.add(key)
        assert r["n"] == r["stored_n_baseline"] == 30, key
        by_dir.setdefault(r["direction"], []).append(r)
    return by_dir


def direction_metrics(rows: list[dict]) -> dict:
    self_rate = np.array([r["stored_k_baseline"] / r["stored_n_baseline"] for r in rows])
    cross_rate = np.array([r["k"] / r["n"] for r in rows])
    delta = cross_rate - self_rate
    self_k = np.array([r["stored_k_baseline"] for r in rows])
    cross_k = np.array([r["k"] for r in rows])

    m: dict = {
        "n_units": len(rows),
        "pearson_r": float(stats.pearsonr(self_rate, cross_rate)[0]),
        "spearman_rho": float(stats.spearmanr(self_rate, cross_rate)[0]),
        "mean_abs_delta_pp": float(np.mean(np.abs(delta)) * 100),
        "median_abs_delta_pp": float(np.median(np.abs(delta)) * 100),
        "frac_abs_delta_gt20pp": float(np.mean(np.abs(delta) > 0.20)),
        "frac_abs_delta_gt50pp": float(np.mean(np.abs(delta) > 0.50)),
        "truncated_frac_graded": float(sum(r["n_truncated"] for r in rows) / sum(r["n"] for r in rows)),
    }
    for label, thr in PRESENCE_THRESHOLDS.items():
        a, b = self_k >= thr, cross_k >= thr
        m[f"agree {label}"] = float(np.mean(a == b))
        m[f"kappa {label}"] = cohens_kappa(a, b)
    # Deterministic slices: where the bank's subject behaves (almost) deterministically,
    # how often does the other model match within 3/30?
    det0 = self_k == 0
    det30 = self_k == 30
    m["n_self_0of30"] = int(det0.sum())
    m["frac_match_le3_given_self_0"] = float(np.mean(cross_k[det0] <= 3)) if det0.any() else float("nan")
    m["n_self_30of30"] = int(det30.sum())
    m["frac_match_ge27_given_self_30"] = float(np.mean(cross_k[det30] >= 27)) if det30.any() else float("nan")
    m["_self_rate"] = self_rate
    m["_cross_rate"] = cross_rate
    return m


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%" if np.isfinite(x) else "n/a"


def main() -> None:
    by_dir = load_rows()
    for d, (_, _, expected) in CROSS_DIRECTIONS.items():
        got = len(by_dir.get(d, []))
        assert got == expected, f"{d}: {got}/{expected} units graded — finish grade_cross.py first"
    for d, (_, expected) in SELF_DIRECTIONS.items():
        got = len(by_dir.get(d, []))
        if got != expected:
            print(f"WARNING: {d}: {got}/{expected} self re-grade units present")

    metrics = {d: direction_metrics(by_dir[d]) for d in CROSS_DIRECTIONS}
    pooled_rows = [r for d in CROSS_DIRECTIONS for r in by_dir[d]]
    metrics["pooled"] = direction_metrics(pooled_rows)

    # Self re-grade reproducibility (stored pipeline rate vs this run's instrument)
    repro = {}
    for d in SELF_DIRECTIONS:
        rows = by_dir.get(d, [])
        if not rows:
            continue
        s = np.array([r["stored_k_baseline"] / 30 for r in rows])
        g = np.array([r["k"] / 30 for r in rows])
        repro[d] = {
            "n": len(rows),
            "pearson_r": float(stats.pearsonr(s, g)[0]),
            "mean_abs_delta_pp": float(np.mean(np.abs(g - s)) * 100),
            "max_abs_delta_pp": float(np.max(np.abs(g - s)) * 100),
        }

    # ---------------------------------------------------------------- plots
    PLOTS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (d, (bank_label, graded_label, _)) in zip(axes, CROSS_DIRECTIONS.items()):
        m = metrics[d]
        jitter = np.random.default_rng(0).uniform(-0.008, 0.008, (2, m["n_units"]))
        ax.scatter(m["_self_rate"] + jitter[0], m["_cross_rate"] + jitter[1], s=10, alpha=0.35)
        ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--")
        ax.set_xlabel(f"{bank_label.split(' bank')[0]} rate (stored ground truth)")
        ax.set_ylabel(f"{graded_label} rate (cross-graded)")
        ax.set_title(f"{bank_label} → {graded_label}\nr={m['pearson_r']:.2f}, mean|Δ|={m['mean_abs_delta_pp']:.1f}pp")
    fig.suptitle("Behavior rate: bank subject vs the other model (same prompts, same classifier question)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "rate_scatter.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for d, (bank_label, graded_label, _) in CROSS_DIRECTIONS.items():
        m = metrics[d]
        ax.hist(
            (m["_cross_rate"] - m["_self_rate"]) * 100,
            bins=np.arange(-102.5, 106, 5),
            alpha=0.55,
            label=f"{bank_label} → {graded_label}",
        )
    ax.set_xlabel("cross-model rate − bank subject rate (pp)")
    ax.set_ylabel("units")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "delta_hist.png", dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------- summary
    def table(keys: list[str], row_fmt) -> list[str]:
        header = "| metric | " + " | ".join(
            f"{CROSS_DIRECTIONS[d][0]} → {CROSS_DIRECTIONS[d][1]}" for d in CROSS_DIRECTIONS
        ) + " | pooled |"
        sep = "|---" * (len(CROSS_DIRECTIONS) + 2) + "|"
        lines = [header, sep]
        for k in keys:
            cells = [row_fmt(metrics[d][k]) for d in [*CROSS_DIRECTIONS, "pooled"]]
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        return lines

    lines = [
        "# Subject behavior similarity: Qwen3-8B vs Llama-3.1-8B on the eval banks",
        "",
        "For every (prompt, behavior_question) unit in each bank, the OTHER subject model's 30",
        "baseline completions (same prompt pool, identical conversations) were graded with the",
        "bank's classifier question (grader: Sonnet 4.6, thinking 30k — same model and prompt",
        "construction as the pipeline that produced the stored rates; the pipeline used thinking",
        "2048, drift measured by the self re-grade below). `rate_self` = the bank's stored",
        "`k_baseline/30`; `rate_cross` = the other model's graded `k/30`.",
        "",
        "> Regenerate: `run_all.sh` in this folder (grade → analyze → spot check).",
        "",
        "## Headline",
        "",
        "**The two subjects do NOT behave the same on the banks' probed behaviors** — the",
        "privileged-access cross-predictor parity is not explained by behavioral identity of the",
        "subjects. Pooled over",
        f"{metrics['pooled']['n_units']} units, the cross-model rate correlates with the stored rate at only",
        f"r = {metrics['pooled']['pearson_r']:.2f} (mean |Δ| = {metrics['pooled']['mean_abs_delta_pp']:.0f}pp; "
        f"{fmt_pct(metrics['pooled']['frac_abs_delta_gt20pp'])} of units differ by >20pp, "
        f"{fmt_pct(metrics['pooled']['frac_abs_delta_gt50pp'])} by >50pp), while the self re-grade",
        "reproducibility check shows the grading instrument itself is tight (r ≈ 0.96, mean |Δ| ≈ 2-3pp)",
        "— the disagreement is real behavioral divergence, not grader noise. The effect is symmetric",
        "and signed: each bank's behaviors are far more characteristic of its own subject (mean rate",
        f"{np.mean(metrics['qwen_bank_on_llama']['_self_rate']):.2f} / "
        f"{np.mean(metrics['llama_bank_on_qwen']['_self_rate']):.2f} on the Qwen / Llama banks) than of "
        f"the other model ({np.mean(metrics['qwen_bank_on_llama']['_cross_rate']):.2f} / "
        f"{np.mean(metrics['llama_bank_on_qwen']['_cross_rate']):.2f}),",
        "as expected since classifier questions are authored from observed subject behavior.",
        "Even where the bank subject is fully deterministic (30/30), the other model matches (≥27/30)",
        f"only {fmt_pct(metrics['pooled']['frac_match_ge27_given_self_30'])} of the time.",
        "",
        "Interpretation caveats: (1) self rates are >0 in ~97% of units by construction, so the",
        "'>0% present' agreement/κ rows are near-degenerate — the graded thresholds (≥10/30/50%) are",
        "the informative ones, and κ is ≈0-0.14 everywhere; (2) qualitative review",
        "(`results/spot_check.md`) confirms top disagreements are genuine behavior differences",
        "(tool-call target choice, empty-array vs function call, response language), not grader",
        "artifacts.",
        "",
        "## Rate agreement",
        "",
        *table(
            ["n_units", "pearson_r", "spearman_rho", "mean_abs_delta_pp", "median_abs_delta_pp"],
            lambda v: f"{v:.3f}" if isinstance(v, float) else str(v),
        ),
        "",
        "## Disagreement prevalence",
        "",
        *table(
            ["frac_abs_delta_gt20pp", "frac_abs_delta_gt50pp"],
            fmt_pct,
        ),
        "",
        "## Binarized 'behavior present' agreement (threshold sensitivity)",
        "",
    ]
    for label in PRESENCE_THRESHOLDS:
        lines += table([f"agree {label}", f"kappa {label}"], lambda v: f"{v:.3f}")
        lines.append("")
    lines += [
        "## Deterministic-behavior slices",
        "",
        "Units where the bank's subject is deterministic; does the other model match within 3/30?",
        "",
        *table(
            [
                "n_self_0of30",
                "frac_match_le3_given_self_0",
                "n_self_30of30",
                "frac_match_ge27_given_self_30",
            ],
            lambda v: fmt_pct(v) if isinstance(v, float) else str(v),
        ),
        "",
        "## Self re-grade reproducibility check",
        "",
        "Same instrument as the cross grading, applied to the bank subject's own completions;",
        "measures grader drift vs the stored pipeline rates (graded with thinking 2048).",
        "",
        "| direction | n | pearson r | mean |Δ| (pp) | max |Δ| (pp) |",
        "|---|---|---|---|---|",
    ]
    for d, r in repro.items():
        lines.append(
            f"| {SELF_DIRECTIONS[d][0]} | {r['n']} | {r['pearson_r']:.3f} | "
            f"{r['mean_abs_delta_pp']:.1f} | {r['max_abs_delta_pp']:.1f} |"
        )
    lines += [
        "",
        "## Notes / caveats",
        "",
        "- Truncation (pre-existing, both sides): fraction of graded completions with"
        " `finish_reason=='length'`: "
        + ", ".join(
            f"{CROSS_DIRECTIONS[d][1]} {fmt_pct(metrics[d]['truncated_frac_graded'])}"
            for d in CROSS_DIRECTIONS
        )
        + ". The stored rates were computed on equally-truncated self completions.",
        "- Baseline behavior only; counterfactual-side rates for the other model do not exist"
        " (see the caveats in README.md).",
        "- Qualitative review: `results/spot_check.md` (make_spot_check.py).",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {SUMMARY_PATH} and {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
