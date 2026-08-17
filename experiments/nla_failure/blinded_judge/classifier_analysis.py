"""Post hoc: can the blinded judge's per-claim read counts classify factor_status?

The blinded judge (blinded_judge.py) flagged, per claim, which selected NLA reads
support the hypothesized X->Y link and which weigh against it. Question: used as a
bare classifier of the ground truth (factor_status == supported, i.e. "editing X
really moves Y"), how good are those counts — and how do they compare to the plain
(transcript-only) and NLA-arm agents' own implied P(change) on the same claims?

Every agent prob_true is converted to P(change): prob_true if the claim asserts
"changes", else 1 - prob_true, so all classifiers target the same label.

AUROCs are reported with 95% bootstrap CIs (2000 resamples, seed 0, claim-level).
The feature-combination rows use 5-fold cross-validated logistic regression so no
number is in-sample.

Run (repo root, .venv):
  python investigations/.../blinded_judge/classifier_analysis.py \
      --blinded results/blinded_labels.jsonl --labels <spectrum.jsonl> \
      --eval-set <eval_set.jsonl> --plain <plain.jsonl> --nla <nla.jsonl>
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def roc_auc_score(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC with tie correction (midranks)."""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fit_logistic(X: np.ndarray, y: np.ndarray, n_iter: int = 50) -> np.ndarray:
    """IRLS logistic regression on standardized features + intercept."""
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-Xb @ w))
        W = p * (1 - p) + 1e-9
        H = Xb.T @ (Xb * W[:, None]) + 1e-6 * np.eye(Xb.shape[1])
        g = Xb.T @ (y.astype(float) - p)
        step = np.linalg.solve(H, g)
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def cv_logistic_scores(X: np.ndarray, y: np.ndarray, seed: int = 0, k: int = 5) -> np.ndarray:
    """5-fold stratified CV out-of-fold probabilities."""
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
    Xs = (X - mu) / sd
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), dtype=int)
    for cls in (True, False):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % k
    scores = np.empty(len(y))
    for f in range(k):
        tr, te = folds != f, folds == f
        w = fit_logistic(Xs[tr], y[tr])
        scores[te] = 1 / (1 + np.exp(-(np.column_stack([np.ones(te.sum()), Xs[te]]) @ w)))
    return scores


@dataclass
class Row:
    claim_id: str
    supported: bool          # factor_status == "supported" (the label)
    n_reads: int
    n_supports: int
    n_against: int
    leaked_true_cause: int   # leaked-judge true-cause read count (contaminated; reference)
    plain_p_change: float
    nla_p_change: float


LEAKED_TRUE = ("explicit_causal", "interpretation", "implied_connection")


def p_change(rec: dict, ev: dict) -> float:
    if "prediction" in ev:  # cf5: claims assert stays or changes
        return rec["prob_true"] if ev["prediction"] == "changes" else 1.0 - rec["prob_true"]
    # cf7: every claim asserts "changes" (is_true == (factor_status == supported))
    assert ev["claim_type"] in ("supported", "inert"), ev["claim_type"]
    return rec["prob_true"]


def load_rows(args) -> list[Row]:
    blind = {json.loads(l)["claim_id"]: json.loads(l) for l in open(args.blinded)}
    lab = {json.loads(l)["claim_id"]: json.loads(l) for l in open(args.labels)}
    ev = {json.loads(l)["claim_id"]: json.loads(l) for l in open(args.eval_set)}
    plain = {json.loads(l)["claim_id"]: json.loads(l) for l in open(args.plain)}
    nla = {json.loads(l)["claim_id"]: json.loads(l) for l in open(args.nla)}
    dropped = [cid for cid in blind if cid not in ev]
    if dropped:
        print(f"NOTE: {len(dropped)} judged claims are not in this eval set "
              f"(release filtering) and are excluded from every number below.")
    rows = []
    for cid, b in blind.items():
        if cid not in ev:
            continue
        e = ev[cid]
        rows.append(Row(
            claim_id=cid,
            supported=e["factor_status"] == "supported",
            n_reads=b["n_reads"],
            n_supports=len(b["supports"]),
            n_against=len(b["against"]),
            leaked_true_cause=(sum(lab[cid]["counts"][k] for k in LEAKED_TRUE)
                               if "counts" in lab[cid] else -1),
            plain_p_change=p_change(plain[cid], e),
            nla_p_change=p_change(nla[cid], e),
        ))
    return rows


def auroc_ci(y: np.ndarray, s: np.ndarray, n_boot: int = 2000, seed: int = 0):
    point = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blinded", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--eval-set", type=Path, required=True)
    ap.add_argument("--plain", type=Path, required=True)
    ap.add_argument("--nla", type=Path, required=True)
    args = ap.parse_args()

    rows = load_rows(args)
    y = np.array([r.supported for r in rows])
    n_reads = np.array([r.n_reads for r in rows], dtype=float)
    sup = np.array([r.n_supports for r in rows], dtype=float)
    ag = np.array([r.n_against for r in rows], dtype=float)
    sup_f, ag_f = sup / n_reads, ag / n_reads
    plain_p = np.array([r.plain_p_change for r in rows])
    nla_p = np.array([r.nla_p_change for r in rows])
    leaked = np.array([r.leaked_true_cause for r in rows], dtype=float)

    print(f"n={len(rows)} claims, {y.sum()} supported / {(~y).sum()} inert\n")
    print("=== AUROC for predicting factor_status == supported ===")
    singles = [
        ("n_supports (count of link-supporting reads)", sup),
        ("supports fraction (n_supports / n_reads)", sup_f),
        ("n_against (inverted)", -ag),
        ("against fraction (inverted)", -ag_f),
        ("supports - against (count diff)", sup - ag),
        ("supports - against (fraction diff)", sup_f - ag_f),
        ("[ref, leaked judge] true-cause read count", leaked),
        ("plain agent P(change)  (transcript only)", plain_p),
        ("NLA agent P(change)", nla_p),
    ]
    if leaked[0] < 0:  # no spectrum counts for this run
        singles = [(n, s) for n, s in singles if not n.startswith("[ref, leaked")]
    for name, s in singles:
        pt, lo, hi = auroc_ci(y, s)
        print(f"  {name:45s} {pt:.3f} [{lo:.3f}, {hi:.3f}]")

    print("\n=== 5-fold CV logistic combinations ===")
    combos = [
        ("supports_f + against_f", np.column_stack([sup_f, ag_f])),
        ("supports_f + against_f + n_reads", np.column_stack([sup_f, ag_f, n_reads])),
        ("plain P(change) alone (ref)", plain_p.reshape(-1, 1)),
        ("plain + supports_f + against_f", np.column_stack([plain_p, sup_f, ag_f])),
        ("NLA + supports_f + against_f", np.column_stack([nla_p, sup_f, ag_f])),
        ("plain + NLA (ref)", np.column_stack([plain_p, nla_p])),
        ("plain + NLA + supports_f + against_f",
         np.column_stack([plain_p, nla_p, sup_f, ag_f])),
    ]
    for name, X in combos:
        s = cv_logistic_scores(X, y)
        pt, lo, hi = auroc_ci(y, s)
        print(f"  {name:45s} {pt:.3f} [{lo:.3f}, {hi:.3f}]")

    print("\n=== paired bootstrap: (plain + supports_f + against_f) - plain alone ===")
    s_combo = cv_logistic_scores(np.column_stack([plain_p, sup_f, ag_f]), y)
    rng = np.random.default_rng(0)
    n = len(y)
    deltas = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], s_combo[idx]) - roc_auc_score(y[idx], plain_p[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    pt = roc_auc_score(y, s_combo) - roc_auc_score(y, plain_p)
    print(f"  delta AUROC = {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]"
          f"  ({'CI excludes 0' if lo > 0 or hi < 0 else 'CI includes 0'})")


if __name__ == "__main__":
    main()
