"""Build the blinded-judge reads file for the cf7 final-eval run.

Takes the pinned 1000-claim id list from the tier-1 stage-A output
(nla_read_labels_explicit_selected_cf7_short1000.jsonl — the exact claims behind the
paper's 0.6% explicit-link number) and re-extracts each claim's agent-selected NLA
reads from the cf7 NLA arm's agent transcripts, using the same parser as
label_nla_reads.py. Output rows match the schema blinded_judge.py expects
({claim_id, factor_status, ..., n_reads, reads}); no spectrum "counts" field exists
for cf7 (that judge was never run here), so downstream leaked-judge comparisons are
skipped.

Sanity check: per-claim read counts must exactly match stage-A's n_reads — a mismatch
means we parsed a different arm variant than the paper did, and the script fails.
"""

import argparse
import json
import re
from pathlib import Path

READ_BLOCK_RE = re.compile(
    r"\[(\d+)\] token=(?:'(.*?)'|\"(.*?)\")\n(.*?)(?=\n\[\d+\] token=|\Z)", re.DOTALL)


def parse_selected_reads(row: dict) -> list[dict]:
    seen: dict[int, dict] = {}
    for msg in row["agent_transcript"]:
        for tr in msg.get("tool_results", []):
            if tr.get("name") != "query_nla":
                continue
            for m in READ_BLOCK_RE.finditer(tr.get("output", "")):
                ti = int(m.group(1))
                if ti in seen:
                    continue
                tok = m.group(2) if m.group(2) is not None else (m.group(3) or "")
                seen[ti] = {"token_index": ti, "token": tok,
                            "verbalization": m.group(4).strip()}
    return [seen[k] for k in sorted(seen)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a", type=Path, required=True,
                    help="nla_read_labels_explicit_selected_cf7_short1000.jsonl (pins ids + n_reads)")
    ap.add_argument("--nla-arm", type=Path, required=True,
                    help="cf7 NLA binary-evals jsonl (agent transcripts)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    stage_a = [json.loads(l) for l in open(args.stage_a)]
    arm = {json.loads(l)["claim_id"]: json.loads(l) for l in open(args.nla_arm)}

    mismatches = []
    with open(args.out, "w") as f:
        for rec in stage_a:
            cid = rec["claim_id"]
            reads = parse_selected_reads(arm[cid])
            if len(reads) != rec["n_reads"]:
                mismatches.append((cid, len(reads), rec["n_reads"]))
                continue
            f.write(json.dumps({
                "claim_id": cid,
                "is_true": rec["is_true"],
                "claim_type": rec["claim_type"],
                "factor_status": rec["factor_status"],
                "source": "selected",
                "n_reads": len(reads),
                "reads": reads,
            }) + "\n")

    if mismatches:
        for cid, got, want in mismatches[:10]:
            print(f"MISMATCH {cid}: parsed {got} reads, stage-A says {want}")
        raise SystemExit(
            f"{len(mismatches)}/{len(stage_a)} claims mismatch stage-A read counts — "
            "wrong arm file?")
    print(f"wrote {len(stage_a)} claims -> {args.out}")


if __name__ == "__main__":
    main()
