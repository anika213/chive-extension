"""Paper analysis — arm accuracy on the "solvable AND hard" slice.

The headline binary-eval result (tools give ~no uplift over a transcript-only baseline)
is computed on the full balanced set. A natural worry: the full set is diluted by claims
that are trivial (plain already nails them) or unsolvable (even the counterfactual ceiling
fails), where no tool *could* help. This script isolates the subset where a tool has the
most room to help and re-scores every arm there:

  SLICE = { claims where the CEILING arm (cf — the agent runs the real intervention) is
            CORRECT, AND the BASELINE arm (plain) is genuinely UNCERTAIN
            (prob_true in [--conf-lo, --conf-hi], default 0.25-0.75) }

  - cf-correct  => the claim IS solvable with real counterfactual access (not noise).
  - plain uncertain => the baseline has NOT already settled it; there is headroom.

If the activation tools carried real signal, this is exactly where it should show up.
We report, on the FULL common set and on the SLICE, for every arm:
  n, accuracy, AUROC (calibration-invariant discrimination), and stays-lean
  (fraction of predictions that the behavior is STABLE — the systematic-bias metric).

This is analysis-only: it reads the arm result files that the binary eval already wrote.

Usage (from repo root):
  .venv/bin/python experiments/common/analyze_solvable_slice.py \
      --binary-evals-dir data/runs/opus_investigates_gemma3_27b_it_5000_thorough_n30/cf5/binary_evals \
      --label cf5_n2000
"""
import argparse
import glob
import json
from pathlib import Path

# file-name prefix per arm; cf = counterfactual ceiling, the rest are activation tools / baseline
ARM_PREFIX = {"plain": "plain", "nla": "nla_", "sae": "sae_", "ao": "ao_", "cf": "cf_"}
# analysis artifacts that share an arm prefix but are NOT result files
_SKIP = ("_toolcalls", "read_labels", "_evidence", "_roles", "paired", "error_classification",
         "_simple", "_debias")


def arm_file(be_dir: Path, arm: str, label: str) -> Path:
    pat = str(be_dir / f"{ARM_PREFIX[arm]}*_{label}.jsonl")
    hits = [h for h in glob.glob(pat) if not any(s in h for s in _SKIP)]
    assert len(hits) == 1, f"expected 1 {arm} file for label {label}, found {hits} ({pat})"
    return Path(hits[0])


def load_arm(path: Path) -> dict[str, dict]:
    return {json.loads(l)["claim_id"]: json.loads(l) for l in open(path)}


def auroc(scores: list[float], labels: list[bool]) -> float:
    """Rank-based (Mann-Whitney) AUROC; nan if only one class present."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # average rank (1-based), ties shared
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(r for r, y in zip(ranks, labels) if y)
    n_pos, n_neg = len(pos), len(neg)
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def predicts_stays(row: dict) -> bool:
    """The agent effectively predicts the behavior is STABLE. A claim's suffix (claim_type
    ...'_stays'/'_changes') is what it ASSERTS; the agent asserts 'stays' iff
    (claim asserts stays) == (agent says the claim is true)."""
    asserts_stays = str(row.get("claim_type", "")).endswith("stays")
    return asserts_stays == bool(row["pred_true"])


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "acc": float("nan"), "auroc": float("nan"), "stays": float("nan")}
    acc = sum(bool(r["correct"]) for r in rows) / n
    au = auroc([float(r["prob_true"]) for r in rows], [bool(r["is_true"]) for r in rows])
    stays = sum(predicts_stays(r) for r in rows) / n
    return {"n": n, "acc": acc, "auroc": au, "stays": stays}


def table(title: str, arms: list[str], rows_by_arm: dict[str, list[dict]]) -> str:
    lines = [f"### {title}", "", "| arm | n | accuracy | AUROC | stays-lean |", "|---|---|---|---|---|"]
    for a in arms:
        m = metrics(rows_by_arm[a])
        au = "n/a" if m["auroc"] != m["auroc"] else f"{m['auroc']:.3f}"
        lines.append(f"| {a} | {m['n']} | {m['acc']:.1%} | {au} | {m['stays']:.1%} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary-evals-dir", type=Path, required=True)
    ap.add_argument("--label", default="cf5_n2000")
    ap.add_argument("--arms", default="plain,nla,sae,ao,cf",
                    help="comma list of arms to score (file prefixes); must include the ceiling+conf arms")
    ap.add_argument("--ceiling-arm", default="cf", help="arm that must be CORRECT for a claim to be solvable")
    ap.add_argument("--conf-arm", default="plain", help="arm whose prob_true defines 'uncertain'")
    ap.add_argument("--conf-lo", type=float, default=0.25)
    ap.add_argument("--conf-hi", type=float, default=0.75)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for need in (args.ceiling_arm, args.conf_arm):
        assert need in arms, f"--arms must include the {need!r} arm"
    loaded = {a: load_arm(arm_file(args.binary_evals_dir, a, args.label)) for a in arms}

    # common claim_ids across every arm so each row set is comparable
    common = sorted(set.intersection(*[set(d) for d in loaded.values()]))

    ceil, conf = loaded[args.ceiling_arm], loaded[args.conf_arm]
    def in_slice(cid: str) -> bool:
        if not ceil[cid]["correct"]:
            return False
        p = float(conf[cid]["prob_true"])
        return args.conf_lo <= p <= args.conf_hi

    slice_ids = [c for c in common if in_slice(c)]
    full_rows = {a: [loaded[a][c] for c in common] for a in arms}
    slice_rows = {a: [loaded[a][c] for c in slice_ids] for a in arms}

    # how the slice whittles down, for transparency
    n_ceil = sum(ceil[c]["correct"] for c in common)
    n_conf = sum(args.conf_lo <= float(conf[c]["prob_true"]) <= args.conf_hi for c in common)

    report = []
    report.append(f"# Arm accuracy on the solvable-and-hard slice ({args.label})\n")
    report.append(
        f"**Slice definition:** `{args.ceiling_arm}` correct (solvable) AND `{args.conf_arm}` "
        f"prob_true in [{args.conf_lo}, {args.conf_hi}] (uncertain).\n")
    report.append(
        f"Common claims across {arms}: **{len(common)}**. "
        f"{args.ceiling_arm}-correct: {n_ceil}. {args.conf_arm}-uncertain: {n_conf}. "
        f"Both (the slice): **{len(slice_ids)}**.\n")
    report.append(table("Full common set", arms, full_rows))
    report.append("")
    report.append(table(f"Slice ({len(slice_ids)} claims)", arms, slice_rows))
    report.append("")
    report.append(f"*(`{args.ceiling_arm}` is 100% / AUROC 1.000 on the slice by construction — "
                  f"the slice is defined as {args.ceiling_arm}-correct.)*")
    report.append("")
    # the headline comparison: do tools beat plain ON THE SLICE?
    base = metrics(slice_rows[args.conf_arm])["acc"]
    deltas = []
    for a in arms:
        if a in (args.conf_arm, args.ceiling_arm):
            continue
        deltas.append(f"- **{a}**: {metrics(slice_rows[a])['acc']:.1%} "
                      f"({metrics(slice_rows[a])['acc'] - base:+.1%} vs {args.conf_arm})")
    report.append(f"**Tool uplift on the slice (vs {args.conf_arm} {base:.1%}):**")
    report.extend(deltas)
    text = "\n".join(report) + "\n"

    out_path = args.out or (args.binary_evals_dir / f"solvable_slice_{args.label}.md")
    out_path.write_text(text)
    print(text)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
