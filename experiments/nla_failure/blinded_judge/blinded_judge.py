"""Blinded re-run of the NLA relationship-evidence judging.

The retired six-bucket spectrum judge (`../label_nla_reads.py`) was shown the ground
truth (baseline/counterfactual rates, factor_status, "THE CLAIM IS TRUE") before
labeling reads, so its ~45-50% "true-cause signal" rate may be leakage-driven
confabulation. This script re-judges the SAME claims and the SAME selected reads with
the ground truth withheld, using a deliberately coarse three-way rubric per claim:

  supports — >=1 read that on its own asserts/strongly implies that editing factor X
             would move behavior Y (the claim's hypothesized causal link)
  against  — >=1 read that rules against the link OR attributes Y to a different,
             non-edited cause (combined per Adam's request)
  none     — neither (derived: both lists empty)

The judge sees exactly what the predictor agent saw: the hypothesis (X and Y), the
question, the transcript, and the selected reads. It does NOT see rates, factor_status,
is_true, the claim's stays/changes prediction, or the grounding text.

Run (repo root, .venv, .env sourced) — see README.md for the full commands:
  python experiments/nla_failure/blinded_judge/blinded_judge.py \
      --labels <reads.jsonl> --eval-set <eval_set.jsonl> --out <labels.jsonl>

Post hoc (no API):
  ... --summarize --plain <plain.jsonl> --nla <nla.jsonl> [--dump-examples 3]
"""

import argparse
import asyncio
import collections
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from chive.pipeline.llm_client import (
    call_for_tool,
    make_llm_client,
    set_max_concurrency,
)

JUDGE_SYSTEM = (
    "You are auditing the outputs of an interpretability tool ('NLA', a natural-language "
    "autoencoder). The tool reads a target language model's internal activation at a single "
    "token of its own transcript and emits a short natural-language description of what that "
    "token's hidden state encodes. The descriptions are noisy and per-token. FORMAT NOTE: each "
    "description typically ends with a clause guessing likely next-token continuations; a read "
    "that merely predicts the literal next words of the visible transcript is not evidence of "
    "anything beyond next-token prediction.\n\n"
    "You are given a HYPOTHESIS about the target model: that a specific FACTOR X of the input "
    "drives a specific BEHAVIOR Y of its response (i.e., editing X would substantially change "
    "whether Y occurs). You are also given the full transcript and the NLA reads an "
    "investigator collected. You do NOT know whether the hypothesis is true. Judge only what "
    "the reads themselves say.\n\n"
    "Identify reads in exactly two categories:\n"
    "  SUPPORTS: the read, on its own content, asserts or strongly implies that X drives Y — "
    "a stated or clearly implied link between the two, not merely X or Y being mentioned or "
    "both appearing without connection.\n"
    "  AGAINST: the read, on its own content, weighs against the hypothesis — it indicates Y "
    "is driven by some OTHER, non-edited factor, or that X is incidental/not load-bearing for "
    "Y. Merely failing to mention X or Y is NOT against; the read must affirmatively point "
    "away from the link.\n\n"
    "Every other read (endpoint-only mentions, off-topic, boilerplate, pure next-token echo of "
    "visible text) is irrelevant — omit it. A read goes in at most one category. For each "
    "flagged read include a <=20-word quote from the read justifying the label.\n\n"
    "Return your labels via the rate_reads tool, referencing reads by their [tok N] index. "
    "If no read qualifies for a category, return an empty list for it."
)

TOOL = {
    "name": "rate_reads",
    "description": "Report which NLA reads support or weigh against the hypothesized X->Y link.",
    "input_schema": {
        "type": "object",
        "properties": {
            "supports": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "token_index": {"type": "integer"},
                        "quote": {"type": "string"},
                    },
                    "required": ["token_index", "quote"],
                },
            },
            "against": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "token_index": {"type": "integer"},
                        "quote": {"type": "string"},
                    },
                    "required": ["token_index", "quote"],
                },
            },
        },
        "required": ["supports", "against"],
    },
}


def behavior_of(c: dict) -> str:
    # cf5 eval sets call the field behavior_predicate; cf7 calls it behavior_question
    return c["behavior_predicate"] if "behavior_predicate" in c else c["behavior_question"]


def hypothesis_block(c: dict) -> str:
    # Blinded: no rates, no factor_status, no is_true, no stays/changes prediction,
    # no grounding. Only what the predictor agent itself was given.
    return (
        f"HYPOTHESIS under investigation: editing FACTOR X would substantially change BEHAVIOR Y.\n"
        f"FACTOR X (the input factor an intervention would edit): {c['intervention']}\n"
        f"BEHAVIOR Y (the response behavior being measured): {behavior_of(c)}\n"
        f"QUESTION the investigator is answering: {c['question']}"
    )


def transcript_block(c: dict) -> str:
    parts = [f"===== {str(m['role']).upper()} =====\n{m['content']}" for m in c["messages"]]
    parts.append(f"===== ASSISTANT (the completion under study) =====\n{c['chosen_completion']}")
    return "\n\n".join(parts)


def reads_block(reads: list[dict]) -> str:
    return "\n".join(f"[tok {r['token_index']}] {r['token']!r}\n  NLA: {r['verbalization']}"
                     for r in reads)


def build_user_content(claim: dict, reads: list[dict]) -> str:
    return (
        hypothesis_block(claim)
        + "\n\n================ FULL TRANSCRIPT ================\n"
        + transcript_block(claim)
        + f"\n\n================ NLA READS ({len(reads)}) ================\n"
        + reads_block(reads)
        + "\n\nIdentify the supporting and against reads (if any) via the rate_reads tool."
    )


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)]


N_MALFORMED_ITEMS = 0


def norm_hits(items: list, valid_indices: set[int]) -> list[dict]:
    """Tolerate flaky tool-output items (bare ints/strings instead of objects).

    Malformed or unknown-index items are dropped and counted (AGENTS.md flaky
    per-entry convention); the caller warns loudly at the end of the run.
    """
    global N_MALFORMED_ITEMS
    out = []
    for h in items:
        if isinstance(h, dict) and isinstance(h.get("token_index"), int):
            ti, quote = h["token_index"], str(h.get("quote", ""))
        elif isinstance(h, int) or (isinstance(h, str) and h.strip().isdigit()):
            ti, quote = int(h), ""
        else:
            N_MALFORMED_ITEMS += 1
            continue
        if ti not in valid_indices:
            N_MALFORMED_ITEMS += 1
            continue
        out.append({"token_index": ti, "quote": quote})
    return out


async def run(args) -> None:
    labels = load_jsonl(args.labels)
    evalset = {c["claim_id"]: c for c in load_jsonl(args.eval_set)}
    if args.max_claims:
        labels = labels[: args.max_claims]

    done = set()
    if args.out.exists() and args.resume:
        done = {r["claim_id"] for r in load_jsonl(args.out)}
        print(f"resume: {len(done)} claims already done")
    jobs = [(d["claim_id"], evalset[d["claim_id"]], d["reads"]) for d in labels
            if d["claim_id"] not in done]
    print(f"{len(jobs)} claims to judge")

    set_max_concurrency(args.concurrency)
    client = make_llm_client(args.model)
    out_f = open(args.out, "a" if (args.resume and done) else "w")
    lock = asyncio.Lock()
    n_skip = 0

    async def one(claim_id: str, claim: dict, reads: list[dict]):
        nonlocal n_skip
        res = await call_for_tool(
            client,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": build_user_content(claim, reads)}],
            tool=TOOL,
            required_keys=["supports", "against"],
            max_tokens=args.max_tokens,
            effort=args.effort,
        )
        if res is None:
            n_skip += 1
            print(f"  [skip] {claim_id}: no valid tool call", flush=True)
            return
        tool_input, _ = res
        by_index = {r["token_index"] for r in reads}
        rec = {
            "claim_id": claim_id,
            "n_reads": len(reads),
            "supports": norm_hits(tool_input["supports"], by_index),
            "against": norm_hits(tool_input["against"], by_index),
        }
        async with lock:
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
        print(f"  [ok] {claim_id}: {len(reads)} reads -> "
              f"supports={len(rec['supports'])} against={len(rec['against'])}", flush=True)

    gathered = await asyncio.gather(*[one(*j) for j in jobs], return_exceptions=True)
    for job, r in zip(jobs, gathered):
        if isinstance(r, Exception):
            n_skip += 1
            print(f"  [error] {job[0]}: {r!r}", flush=True)
    out_f.close()
    if n_skip:
        rate = n_skip / len(jobs)
        print(f"WARNING: {n_skip}/{len(jobs)} claims skipped ({rate:.1%})"
              + (" — EXCEEDS 0.1%, investigate" if rate > 0.001 else ""))
    if N_MALFORMED_ITEMS:
        print(f"WARNING: {N_MALFORMED_ITEMS} malformed/unknown-index tool-output items dropped")


# ---- post hoc ----------------------------------------------------------------------

@dataclass
class ClaimRow:
    claim_id: str
    factor_status: str
    is_true: bool
    group: str            # supports_only | against_only | both | none
    leaked_group: str     # grouping from the ground-truth-leaked spectrum judge
    plain_prob: float
    plain_correct: bool
    nla_prob: float
    nla_correct: bool

    @property
    def toward(self) -> float:
        sign = 1.0 if self.is_true else -1.0
        return (self.nla_prob - self.plain_prob) * sign


LEAKED_TRUE = {"explicit_causal", "interpretation", "implied_connection"}


def leaked_group_of(counts: dict) -> str:
    t = any(counts[k] for k in LEAKED_TRUE)
    a = bool(counts["implies_alternative"])
    return "both" if (t and a) else "true_cause" if t else "alt_cause" if a else "none"


def group_of(rec: dict) -> str:
    s, a = bool(rec["supports"]), bool(rec["against"])
    return "both" if (s and a) else "supports_only" if s else "against_only" if a else "none"


def summarize(args) -> None:
    blinded = {r["claim_id"]: r for r in load_jsonl(args.out)}
    labels = {d["claim_id"]: d for d in load_jsonl(args.labels)}
    evalset = {c["claim_id"]: c for c in load_jsonl(args.eval_set)}
    probs = {}
    for name, path in (("plain", args.plain), ("nla", args.nla)):
        probs[name] = {d["claim_id"]: d for d in load_jsonl(path)}

    dropped = [cid for cid in blinded if cid not in evalset]
    if dropped:
        print(f"NOTE: {len(dropped)} judged claims are not in this eval set "
              f"(release filtering) and are excluded from every number below.")
    rows = []
    for cid, rec in blinded.items():
        if cid not in evalset:
            continue
        lab, ev = labels[cid], evalset[cid]
        rows.append(ClaimRow(
            claim_id=cid,
            factor_status=ev["factor_status"],
            is_true=ev["is_true"],
            group=group_of(rec),
            leaked_group=leaked_group_of(lab["counts"]) if "counts" in lab else "n/a",
            plain_prob=probs["plain"][cid]["prob_true"],
            plain_correct=probs["plain"][cid]["correct"],
            nla_prob=probs["nla"][cid]["prob_true"],
            nla_correct=probs["nla"][cid]["correct"],
        ))
    n = len(rows)
    print(f"n={n} claims judged (blinded)\n")

    print("=== 1. Blinded category x factor_status ===")
    print(f"{'group':14s} {'supported':>18s} {'inert':>18s}")
    for g in ("supports_only", "against_only", "both", "none"):
        cells = []
        for fs in ("supported", "inert"):
            sub = [r for r in rows if r.factor_status == fs]
            k = sum(1 for r in sub if r.group == g)
            cells.append(f"{k:4d} ({k / len(sub):6.1%})")
        print(f"{g:14s} {cells[0]:>18s} {cells[1]:>18s}")

    if all(r.leaked_group != "n/a" for r in rows):
        print("\n=== 2. Blinded vs ground-truth-leaked judge (claim-level crosstab) ===")
        print(f"{'':14s}"
              + "".join(f"{lg:>12s}" for lg in ("true_cause", "alt_cause", "both", "none"))
              + "   (leaked ->)")
        for g in ("supports_only", "against_only", "both", "none"):
            cells = [sum(1 for r in rows if r.group == g and r.leaked_group == lg)
                     for lg in ("true_cause", "alt_cause", "both", "none")]
            print(f"{g:14s}" + "".join(f"{c:12d}" for c in cells))
    else:
        print("\n=== 2. (skipped: no leaked-judge spectrum counts for this run) ===")

    print("\n=== 3. Does the blinded label predict NLA outcomes? ===")
    for fs in ("supported", "inert"):
        print(f"  --- factor_status={fs} ---")
        for g in ("supports_only", "against_only", "both", "none"):
            sub = [r for r in rows if r.factor_status == fs and r.group == g]
            if not sub:
                print(f"    {g:14s} n=   0")
                continue
            pacc = sum(r.plain_correct for r in sub) / len(sub)
            nacc = sum(r.nla_correct for r in sub) / len(sub)
            tow = statistics.mean(r.toward for r in sub)
            print(f"    {g:14s} n={len(sub):4d}  plain {pacc:5.1%}  NLA {nacc:5.1%}  "
                  f"delta {nacc - pacc:+6.1%}  toward {tow:+.4f}")

    print("\n=== 4. Read-level flag rates ===")
    kept = [r for cid, r in blinded.items() if cid in evalset]
    n_reads = sum(r["n_reads"] for r in kept)
    n_sup = sum(len(r["supports"]) for r in kept)
    n_ag = sum(len(r["against"]) for r in kept)
    print(f"  reads={n_reads}  supports={n_sup} ({n_sup / n_reads:.2%})  "
          f"against={n_ag} ({n_ag / n_reads:.2%})")

    if args.dump_examples:
        dump_examples(args, blinded, labels, evalset, rows)


def dump_examples(args, blinded, labels, evalset, rows) -> None:
    out = args.out.parent / "blinded_examples.md"
    by_id = {r.claim_id: r for r in rows}
    lines = ["# Blinded-judge flagged examples (full reads, for manual review)\n"]
    for kind in ("supports", "against"):
        picked = [cid for cid, rec in blinded.items() if rec[kind]][: args.dump_examples]
        lines.append(f"\n# {kind.upper()} examples\n")
        for cid in picked:
            rec, ev, row = blinded[cid], evalset[cid], by_id[cid]
            reads = {r["token_index"]: r for r in labels[cid]["reads"]}
            lines += [
                f"\n---\n\n## {cid} ({ev['claim_type']}, factor_status={ev['factor_status']})\n",
                f"**X:** {ev['intervention']}\n",
                f"**Y:** {behavior_of(ev)}\n",
                f"**Ground truth (NOT shown to judge):** {ev['baseline_rate']} -> "
                f"{ev['counterfactual_rate']} ({ev['effect_pp']}pp); claim is "
                f"{'TRUE' if ev['is_true'] else 'FALSE'}\n",
                f"**Outcome:** plain p={row.plain_prob:.2f} "
                f"({'correct' if row.plain_correct else 'WRONG'}) -> NLA p={row.nla_prob:.2f} "
                f"({'correct' if row.nla_correct else 'WRONG'})\n",
            ]
            for h in rec[kind]:
                r = reads[h["token_index"]]
                lines += [
                    f"\n**[{kind}] token {h['token_index']} = `{r['token']}`** — judge quote: "
                    f"_{h['quote']}_\n",
                    "\n```\n" + r["verbalization"].strip() + "\n```\n",
                ]
    out.write_text("\n".join(lines))
    print(f"\nwrote examples -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True,
                    help="spectrum label jsonl (source of the selected reads per claim)")
    ap.add_argument("--eval-set", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="anthropic:claude-opus-4-8")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--concurrency", type=int, default=80)
    ap.add_argument("--max-claims", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--plain", type=Path, default=None, help="summarize: plain arm jsonl")
    ap.add_argument("--nla", type=Path, default=None, help="summarize: NLA arm jsonl")
    ap.add_argument("--dump-examples", type=int, default=0,
                    help="summarize: dump N full examples per category to blinded_examples.md")
    args = ap.parse_args()

    if args.summarize:
        assert args.plain and args.nla, "--summarize needs --plain and --nla"
        summarize(args)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
