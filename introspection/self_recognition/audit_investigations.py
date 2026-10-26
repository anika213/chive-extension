"""Audit a self-recognition CHIVE run against its logs.

The investigator's write-up is not reliable: it has described order swaps that never
happened, and run edits that give the answer away or drop the model's own summary. This
script ignores the write-up and rebuilds every experiment from the exact prompt that was
sent to the target:

  layout   each labelled summary in the prompt, matched to the model's own summary (OWN)
           or the other model's (OTHER) by text similarity; "~0.83" marks an edited
           summary, "?" one that matches neither (e.g. a new summary the investigator wrote)
  diff     what the edit changed relative to the original prompt
  rate     own-summary picks after the edit, with a Fisher exact p-value against the
           original prompt's rate
  claim    the investigator's own description, to compare with the above

Leaks (edits that tell the model which summary it wrote, or describe its style) are not
detected automatically: read the diff line.

Usage:
    .venv/bin/python introspection/self_recognition/audit_investigations.py \
        --run data/runs/selfrec_batch_5 --prompts introspection/self_recognition/batch_5.jsonl
"""

import argparse
import difflib
import json
import re
from pathlib import Path

from scipy.stats import fisher_exact

# A labelled summary in the prompt: a paragraph whose first line is a short label ending
# in ":" (e.g. "Summary1:"), followed by the summary, one or more lines (CNN highlights).
LABELLED = re.compile(r"^([^\n]{1,40}):\n(.+)", re.S)
SAME = 0.97   # similarity above which a summary counts as unedited
MATCH = 0.5   # similarity below which it matches neither summary


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def layout(text: str, own: str, other: str) -> list[str]:
    out = []
    for para in text.split("\n\n"):
        m = LABELLED.match(para.strip())
        if not m:
            continue
        label, body = m.group(1).strip(), m.group(2).strip()
        if label.lower().startswith("article"):
            continue
        r_own = difflib.SequenceMatcher(None, body, own).ratio()
        r_other = difflib.SequenceMatcher(None, body, other).ratio()
        who = "OWN" if r_own > max(r_other, MATCH) else "OTHER" if r_other > MATCH else "?"
        best = max(r_own, r_other)
        out.append(f"{label}={who}" + ("" if best > SAME else f"~{best:.2f}"))
    return out


def diff(orig: list[dict], new: list[dict]) -> list[str]:
    parts = []
    for o, n in zip(orig, new):
        if o["content"] == n["content"]:
            continue
        sm = difflib.SequenceMatcher(None, o["content"], n["content"], autojunk=False)
        for op, a1, a2, b1, b2 in sm.get_opcodes():
            if op != "equal":
                parts.append(f"{n['role']}:{op} {o['content'][a1:a2][:80]!r} -> {n['content'][b1:b2][:160]!r}")
    if len(orig) != len(new):
        parts.append(f"turns {len(orig)} -> {len(new)}")
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="CHIVE run dir (has investigations.jsonl)")
    parser.add_argument("--prompts", type=Path, required=True, help="the build_prompts.py file the run used")
    args = parser.parse_args()

    meta = {r["id"]: r for r in read_jsonl(args.prompts)}
    prompts = {r["id"]: r for r in read_jsonl(args.run / "prompts.jsonl")}
    ver_path = args.run / "verification.jsonl"
    ver = {r["prompt_id"]: r for r in read_jsonl(ver_path)} if ver_path.exists() else {}

    for inv in read_jsonl(args.run / "investigations.jsonl"):
        pid = inv["prompt_id"]
        m, f = meta[pid], inv.get("structured_findings") or {}
        own, other = m["self_summary"], m["other_summary"]
        base, nb = inv["baseline_grader_yes"], len(inv["baseline_grader_verdicts"])
        score = ver.get(pid, {}).get("score", "-")
        print("=" * 100)
        print(f"{pid} | own slot {m['self_slot']} | original {base}/{nb} | verifier {score} | confidence {f.get('confidence')}")
        print("  original layout", layout(prompts[pid]["messages"][-1]["content"], own, other))
        claims = {s["experiment_number"]: s for s in f.get("experiment_summaries", [])}
        for e in inv["experiment_log"]:
            y, n = e["grader_yes"], len(e["grader_verdicts"])
            p = fisher_exact([[base, nb - base], [y, n - y]])[1]
            print(f"  exp {e['exp_num']}: {y}/{n} p={p:.3f} layout={layout(e['messages'][-1]['content'], own, other)}")
            print(f"      claim: {claims.get(e['exp_num'], {}).get('summary', '(none)')[:260]}")
            print(f"      diff:  {' | '.join(diff(prompts[pid]['messages'], e['messages']))[:600]}")
        if pid in ver:
            print("  verifier issues:", ver[pid].get("issues"))


if __name__ == "__main__":
    main()
