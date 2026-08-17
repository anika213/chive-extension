"""Paired analysis for the counterfactual-simulatability self-explanation experiment.

Reads ONLY results/<slug>/ next to this script. Per run x simulator, pairs each self-explanation
arm against the transcript-only `plain` baseline *for the same simulator*, over the claim set both
scored (selfexpl arms only cover prompt_ids with a cached explanation, so a subset of plain's), and reports:
  - AUROC for plain / selfexpl / selfexpl_base (+ bootstrap CI),
  - Delta(selfexpl - plain) and Delta(selfexpl_base - plain), paired claim-bootstrap CI,
  - parse-skip + truncation counts (from the per-file .meta.json sidecars).

The story is the capability gradient (weak simulators helped, strong simulators hurt), so the
headline figure orders simulators weak->strong, colored by family. Writes results/SUMMARY.md +
results/figure_selfexpl.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from chive import paths
from experiments.common.gold_slice import (
    RUNS as SLICE_RUNS,
    RUNS_DIR as SLICE_RUNS_DIR,
    quality_ids,
)


def _quality_keep(slug: str) -> set:
    """claim_ids passing the paper's QUALITY slice (concreteness ∩ cf-correct ∩ not-confounded)
    for this run — the same population the interpretability and direct-prediction figures use.
    Recomputed from source; fails loudly if the slug has no mapped run dir."""
    run_dir = str(Path(SLICE_RUNS_DIR) / SLICE_RUNS[slug][1])
    return quality_ids(run_dir)


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Fast tie-aware (average-rank, Mann-Whitney U) AUROC on numpy arrays — matches the
    reference run_binary_eval.auroc but vectorized so the 10k bootstrap is tractable."""
    y = np.asarray(y, dtype=bool)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p, method="average")
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


RNG = np.random.default_rng(0)
N_BOOT = 10000
RESULTS = paths.experiment_data_dir("self_explanation") / "results"
OUT = RESULTS / "SUMMARY.md"
FIG = RESULTS / "figure_selfexpl.png"

# (slug, display). One results subdir per run.
RUNS = [
    ("qwen8b", "Qwen3-8B"),
    ("q397_5000", "Qwen3.5-397B (5000)"),
    ("q397_petri", "Qwen3.5-397B (petri)"),
]

# simulator model id -> (rank weak->strong, display, family). The gradient IS the result.
SIMS = {
    "google/gemma-3-12b-it": (0, "Gemma-3-12B", "Gemma"),
    "google/gemma-3-27b-it": (1, "Gemma-3-27B", "Gemma"),
    "Qwen/Qwen3.6-27B": (2, "Qwen3.6-27B", "Qwen"),
    "claude-haiku-4-5-20251001": (3, "Haiku-4.5", "Claude"),
    "claude-opus-4-8": (4, "Opus-4.8", "Claude"),
}
FAMILY_COLOR = {"Gemma": "#2ca02c", "Qwen": "#9467bd", "Claude": "#1f77b4"}
ARM_DISPLAY = {"plain": "plain", "selfexpl": "selfexpl (trained)", "selfexpl_base": "selfexpl (base)"}


def load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    for l in open(path):
        d = json.loads(l)
        rows[d["claim_id"]] = d
    return rows


def sim_of(path: Path) -> str:
    return json.loads(open(path).readline())["answerer_model"]


def arm_files(rd: Path, arm: str) -> list[Path]:
    """Result files for an arm. 'selfexpl' must NOT capture 'selfexpl_base_*' (prefix overlap)."""
    if arm == "plain":
        return sorted(rd.glob("plain_*.jsonl"))
    if arm == "selfexpl_base":
        return sorted(rd.glob("selfexpl_base_*.jsonl"))
    return sorted(p for p in rd.glob("selfexpl_*.jsonl") if not p.name.startswith("selfexpl_base_"))


def index_by_sim(files: list[Path], arm: str) -> dict[str, Path]:
    """Map answerer_model -> result file, failing LOUDLY if two files map to the same simulator.
    The arm globs match any checkpoint, so a stale checkpoint's file sitting next to the current
    one (e.g. an old and a current checkpoint's file both present) would otherwise be silently
    deduped by dict-insertion order and the AUROC would depend on which filename sorted last.
    One result file per (arm, simulator) per dir — archive/remove stale checkpoints."""
    out: dict[str, Path] = {}
    for p in files:
        s = sim_of(p)
        if s in out:
            raise RuntimeError(
                f"duplicate {arm} result files for simulator {s!r} in {p.parent} — more than one "
                f"checkpoint present:\n  {out[s].name}\n  {p.name}\n"
                f"Archive/remove the stale one so the analysis is unambiguous.")
        out[s] = p
    return out


def meta_counts(path: Path) -> tuple[int, int]:
    """(n_skipped_parse, n_truncated) from the sidecar, or (0,0) if absent (older runs)."""
    m = path.with_suffix(".meta.json")
    if not m.exists():
        return 0, 0
    d = json.loads(m.read_text())
    return d.get("n_skipped_parse", 0), d.get("n_truncated", 0)


def auroc_ci(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    a = auroc(y, p)
    boots = []
    n = len(y)
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        yb = y[idx]
        s = yb.sum()
        if s == 0 or s == n:
            continue
        boots.append(auroc(yb, p[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return a, lo, hi


def paired_delta_ci(y: np.ndarray, p_arm: np.ndarray, p_plain: np.ndarray) -> tuple[float, float, float]:
    delta = auroc(y, p_arm) - auroc(y, p_plain)
    boots = []
    n = len(y)
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        yb = y[idx]
        s = yb.sum()
        if s == 0 or s == n:
            continue
        boots.append(auroc(yb, p_arm[idx]) - auroc(yb, p_plain[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return delta, lo, hi


def col(arm: dict[str, dict], ids: list[str], field: str) -> np.ndarray:
    return np.array([arm[c][field] for c in ids])


def analyze_run(slug: str, se_suffix: str = "") -> list[dict]:
    """One record per simulator present in this run. `plain` always comes from results/<slug>/;
    the selfexpl arms come from results/<slug><se_suffix>/ (se_suffix='' for the canonical follow
    variant; '_prioritize' for the pure-extrapolation variant, which reuses the canonical plain)."""
    rd_plain = RESULTS / slug
    rd_se = RESULTS / f"{slug}{se_suffix}"
    if not rd_se.exists():
        return []
    keep = _quality_keep(slug)  # paper QUALITY slice; intersect every arm's id set with this
    plain_by_sim = index_by_sim(arm_files(rd_plain, "plain"), "plain")
    se_by_sim = index_by_sim(arm_files(rd_se, "selfexpl"), "selfexpl")
    base_by_sim = index_by_sim(arm_files(rd_se, "selfexpl_base"), "selfexpl_base")
    records = []
    for sim in sorted(set(plain_by_sim) | set(se_by_sim) | set(base_by_sim),
                      key=lambda s: SIMS.get(s, (99, s, "?"))[0]):
        rec: dict = {"slug": slug, "sim": sim}
        if sim not in plain_by_sim:
            rec["error"] = "no plain baseline"
            records.append(rec)
            continue
        plain = load_rows(plain_by_sim[sim])
        rec["plain_skip"], rec["plain_trunc"] = meta_counts(plain_by_sim[sim])
        # plain AUROC over the quality slice (intersect the scored ids with `keep`).
        ids_p = sorted(set(plain) & keep)
        yp = col(plain, ids_p, "is_true").astype(int)
        rec["n_plain"] = len(ids_p)
        rec["plain_auroc"] = auroc_ci(yp, col(plain, ids_p, "prob_true"))
        rec["plain_acc"] = float(np.mean(col(plain, ids_p, "pred_true").astype(bool) == yp.astype(bool)))
        for arm, by_sim in [("selfexpl", se_by_sim), ("selfexpl_base", base_by_sim)]:
            if sim not in by_sim:
                continue
            sed = load_rows(by_sim[sim])
            rec[f"{arm}_skip"], rec[f"{arm}_trunc"] = meta_counts(by_sim[sim])
            ids = sorted(set(plain) & set(sed) & keep)
            y = col(sed, ids, "is_true").astype(int)
            ps, pp = col(sed, ids, "prob_true"), col(plain, ids, "prob_true")
            rec[f"n_{arm}"] = len(ids)
            rec[f"{arm}_auroc"] = auroc_ci(y, ps)
            rec[f"{arm}_delta"] = paired_delta_ci(y, ps, pp)
            rec[f"{arm}_acc"] = float(np.mean(col(sed, ids, "pred_true").astype(bool) == y.astype(bool)))
        records.append(rec)
    return records


def fmt_ci(t: tuple[float, float, float] | None) -> str:
    if t is None:
        return "—"
    a, lo, hi = t
    return f"{a:.3f} [{lo:.3f}, {hi:.3f}]"


def fmt_delta(t: tuple[float, float, float] | None) -> str:
    if t is None:
        return "—"
    d, lo, hi = t
    return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def make_figure(all_records: list[dict], fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Aggregate Δ across runs per simulator (mean of per-run deltas; points show per-run spread).
    by_sim: dict[str, list[dict]] = {}
    for r in all_records:
        if "error" in r:
            continue
        by_sim.setdefault(r["sim"], []).append(r)
    # Order by MEASURED reader strength (mean plain AUROC across runs), not the a-priori tier —
    # empirically Qwen3.6 (~0.75) is a stronger reader than Haiku (~0.70), and ordering by the
    # measured axis makes the Δ-vs-capability gradient read monotonically (the headline).
    def mean_plain(s: str) -> float:
        return float(np.mean([r["plain_auroc"][0] for r in by_sim[s]]))
    sims = sorted(by_sim, key=mean_plain)
    if not sims:
        print("[analyze] no records to plot")
        return
    labels = [SIMS.get(s, (99, s, "?"))[1] for s in sims]
    colors = [FAMILY_COLOR.get(SIMS.get(s, (99, s, "?"))[2], "#888") for s in sims]
    x = np.arange(len(sims))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: absolute AUROC per arm (mean over runs).
    for arm, marker, off in [("plain", "o", -0.18), ("selfexpl", "s", 0.0), ("selfexpl_base", "^", 0.18)]:
        means, errs = [], []
        for s in sims:
            vals = [r[f"{arm}_auroc"][0] for r in by_sim[s] if f"{arm}_auroc" in r]
            means.append(np.mean(vals) if vals else np.nan)
            errs.append(np.std(vals) if len(vals) > 1 else 0.0)
        ax0.errorbar(x + off, means, yerr=errs, marker=marker, ls="-", capsize=3,
                     label=ARM_DISPLAY[arm])
    ax0.axhline(0.5, color="gray", ls=":", lw=1)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=20, ha="right")
    for tick, c in zip(ax0.get_xticklabels(), colors):
        tick.set_color(c)
    ax0.set_ylabel("AUROC (mean over 3 runs)")
    ax0.set_title("Simulator AUROC by arm (weak → strong)")
    ax0.legend(fontsize=9)

    # Right: ΔAUROC vs plain (the headline). Points per run + mean bar.
    for arm, marker, off in [("selfexpl", "s", -0.1), ("selfexpl_base", "^", 0.1)]:
        for i, s in enumerate(sims):
            vals = [r[f"{arm}_delta"][0] for r in by_sim[s] if f"{arm}_delta" in r]
            if not vals:
                continue
            ax1.scatter([x[i] + off] * len(vals), vals, color=colors[i],
                        marker=marker, alpha=0.5, s=30, zorder=3)
            ax1.hlines(np.mean(vals), x[i] + off - 0.07, x[i] + off + 0.07,
                       color=colors[i], lw=2.5, zorder=4)
    ax1.axhline(0.0, color="black", ls="-", lw=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    for tick, c in zip(ax1.get_xticklabels(), colors):
        tick.set_color(c)
    ax1.set_ylabel("ΔAUROC vs plain")
    ax1.set_title("Does the self-explanation help? (■ trained, ▲ base)")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"[analyze] figure -> {fig_path}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="follow", choices=["follow", "prioritize"],
                    help="follow: canonical results/<slug>/. prioritize: selfexpl arms from "
                         "results/<slug>_prioritize/, plain reused from results/<slug>/.")
    args = ap.parse_args()
    se_suffix = "" if args.variant == "follow" else f"_{args.variant}"
    out = OUT if args.variant == "follow" else RESULTS / f"SUMMARY_{args.variant}.md"
    fig_path = FIG if args.variant == "follow" else RESULTS / f"figure_selfexpl_{args.variant}.png"

    all_records: list[dict] = []
    lines = [
        "# Counterfactual-simulatability self-explanation — results", "",
        "Per run x simulator, each self-explanation arm vs the transcript-only `plain` baseline for",
        "the **same simulator**, paired over the claim set both scored. AUROC has a bootstrap 95% CI",
        "(10k); ΔAUROC is a paired claim-bootstrap. ΔAUROC<0 ⇒ the explanation HURT that simulator.",
        "Self-contained: reads results/ only. `skip` = PROBABILITY parse failures, `trunc` = hit max",
        "tokens (both should be ~0).", "",
        "**Population: the QUALITY slice** (concreteness≥3 ∩ cf-correct ∩ not-confounded, via "
        "`gold_slice.quality_ids`) — the paper's primary population, consistent with the "
        "interpretability and direct-prediction figures. The comparator-dependent baseline-unsure "
        "gate is not applied.", "",
    ]
    if args.variant == "prioritize":
        lines[1:1] = [
            "**Variant: `prioritize` (pure extrapolation).** The simulator was instructed to predict "
            "SOLELY from the self-explanation (deferring even against the transcript / its world "
            "knowledge). `plain` is the SAME canonical baseline as the follow run; only the selfexpl "
            "arms differ. Expect the cross-simulator spread to collapse if this isolates explanation "
            "accuracy.", "",
        ]
    for slug, disp in RUNS:
        recs = analyze_run(slug, se_suffix)
        all_records += recs
        lines += [f"## {disp} (`{slug}`)", ""]
        if not recs:
            lines += ["_(no results yet)_", ""]
            continue
        lines += [
            "| simulator | family | n (se/base) | plain AUROC | selfexpl AUROC | selfexpl_base AUROC "
            "| Δ(se−plain) | Δ(base−plain) | skip se/base | trunc se/base |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in recs:
            _, sdisp, fam = SIMS.get(r["sim"], (99, r["sim"], "?"))
            if "error" in r:
                lines.append(f"| {sdisp} | {fam} | — | **{r['error']}** | | | | | | |")
                continue
            n_se = r.get("n_selfexpl", "—")
            n_base = r.get("n_selfexpl_base", "—")
            lines.append(
                f"| {sdisp} | {fam} | {n_se}/{n_base} | {fmt_ci(r.get('plain_auroc'))} "
                f"| {fmt_ci(r.get('selfexpl_auroc'))} | {fmt_ci(r.get('selfexpl_base_auroc'))} "
                f"| {fmt_delta(r.get('selfexpl_delta'))} | {fmt_delta(r.get('selfexpl_base_delta'))} "
                f"| {r.get('selfexpl_skip', 0)}/{r.get('selfexpl_base_skip', 0)} "
                f"| {r.get('selfexpl_trunc', 0)}/{r.get('selfexpl_base_trunc', 0)} |")
        lines.append("")

    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {out}")
    make_figure(all_records, fig_path)


if __name__ == "__main__":
    main()
