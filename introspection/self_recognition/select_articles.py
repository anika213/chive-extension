"""Pick the articles where fine-tuned models recognise their own summary better than the base model.

Reads position-test results (position_test.py) for a base model and its fine-tunes, in one
answer format. For each article it computes every model's own-summary rate with the model's
summary listed first and listed second. An article qualifies if every fine-tune

  - picks its own summary in at least --min-rate of answers in BOTH orders, so the result
    is not position bias, and
  - beats the base model's rate (both orders pooled) by at least --min-gain.

Qualifying articles are ranked by the smallest fine-tune gain over the base model, and the
top --n are written as prompt records (build_prompts.make_record) with --format and
--investigation. The original slot of the model's own summary alternates 2, 1, 2, ... so
the investigations start from both orders.

Usage:
    .venv/bin/python introspection/self_recognition/select_articles.py \
        --prompts introspection/self_recognition/cnn_candidates.jsonl \
        --results data/experiments/selfrec_cnn_screen/*.jsonl --base base \
        --n 2 --out introspection/self_recognition/cnn_cues.jsonl
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from chive.pipeline.pool_common import read_jsonl, write_jsonl
from introspection.self_recognition.build_prompts import FORMATS, INVESTIGATIONS, make_record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, required=True, help="candidate prompt records (build_prompts.py)")
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="position_test.py output files")
    parser.add_argument("--base", default="base", help="name of the base model in the results")
    parser.add_argument("--format", choices=list(FORMATS), default="syspopped",
                        help="answer format to select on, and to write the chosen prompts in")
    parser.add_argument("--investigation", choices=list(INVESTIGATIONS), default="cues")
    parser.add_argument("--n", type=int, default=2, help="articles to pick")
    parser.add_argument("--min-rate", type=float, default=0.8)
    parser.add_argument("--min-gain", type=float, default=0.2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # (article_id, model) -> {own_position: (own, n)}
    cells = defaultdict(dict)
    for path in args.results:
        for r in read_jsonl(path):
            if r["format"] == args.format:
                cells[(r["article_id"], r["model"])][r["own_position"]] = (r["own"], r["n"])
    models = sorted({m for _, m in cells})
    tuned = [m for m in models if m != args.base]
    assert args.base in models and tuned, f"need {args.base!r} and at least one fine-tune; found {models}"

    def rate(a, m, pos=None):
        c = cells[(a, m)]
        own, n = (sum(x) for x in zip(*(c.values() if pos is None else [c[pos]])))
        return own / n

    prompts = {r["article_id"]: r for r in read_jsonl(args.prompts)}
    rows = []
    for a in prompts:
        if not all(len(cells.get((a, m), {})) == 2 for m in models):
            continue  # not tested in both orders for every model
        base = rate(a, args.base)
        gain = min(rate(a, m) for m in tuned) - base
        floor = min(rate(a, m, pos) for m in tuned for pos in (1, 2))
        rows.append((a, base, gain, floor, gain >= args.min_gain and floor >= args.min_rate))
    rows.sort(key=lambda r: (not r[4], -r[2]))

    print(f"{'article':10s} {'base':>5s} " + " ".join(f"{m + ' 1st|2nd':>13s}" for m in tuned) + "   gain  pick")
    picked = [r[0] for r in rows if r[4]][: args.n]
    for a, base, gain, _, ok in rows:
        per = " ".join(f"{rate(a, m, 1):6.0%}|{rate(a, m, 2):<6.0%}" for m in tuned)
        print(f"{a[:8]:10s} {base:5.0%} {per}  {gain:+5.0%}  {'*' if a in picked else ''}")

    records = []
    for i, a in enumerate(picked):
        p = prompts[a]
        records.append(make_record(p["dataset"], a, p["self_summary"], 2 if i % 2 == 0 else 1, p["self_model"],
                                   args.format, args.investigation))
    write_jsonl(records, args.out)
    print(f"\n{len(records)} of {sum(r[4] for r in rows)} qualifying articles -> {args.out}")


if __name__ == "__main__":
    main()
