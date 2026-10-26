"""Position test: does each model pick its own summary with it listed first AND second?

For every prompt record (from build_prompts.py), model and answer format, the two
summaries are shown in both orders and each cell is sampled N times with the
pipeline's settings (temperature 1.0, no thinking). A model that recognises its own
summary is accurate in both positions; a position-biased one is accurate in one.

Grading:
  binary / syspopped  the answer must be "1" or "2"; parsed in code (no grader call)
  explain             the record's prewritten classifier question, applied by the
                      CHIVE grader (same instrument as the investigations)

"Own" is the record's self_summary, i.e. what the summary-generating model wrote
(for the sgtr-em fine-tunes: the base model's summary, as in their training data).

Usage:
    .venv/bin/python introspection/self_recognition/position_test.py \
        --input introspection/self_recognition/cnn_6.jsonl \
        --models base=vllm:qwen35-27b@http://localhost:8000 sgtr0=vllm:qwen35-27b-sgtr0@http://localhost:8000 \
        --formats syspopped explain --n 30 --out data/experiments/selfrec_cnn_position/results.jsonl
"""

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from chive.pipeline.config import DEFAULTS
from chive.pipeline.investigate import grade_completions, run_endpoint_prompt
from chive.pipeline.llm_client import make_llm_client, set_max_concurrency
from chive.pipeline.pool_common import read_jsonl, write_jsonl
from introspection.self_recognition.build_prompts import FORMATS, load_source, prompt_messages, screening_fields

CHOICE = re.compile(r"^\W*(?:summary\s*)?([12])\b", re.I)


def parse_choice(text: str) -> int | None:
    m = CHOICE.match(text.strip())
    return int(m.group(1)) if m else None


async def run_cell(rec, article, fmt, own_position, n, target, grader):
    pair = (rec["self_summary"], rec["other_summary"]) if own_position == 1 else (rec["other_summary"], rec["self_summary"])
    messages = prompt_messages(fmt, article, *pair)
    system = messages[0]["content"] if messages[0]["role"] == "system" else None
    completions = await run_endpoint_prompt(
        target, system_prompt=system, messages=[m for m in messages if m["role"] != "system"], n=n,
        temperature=1.0, max_tokens=DEFAULTS["target_max_tokens"], thinking_budget=None,
    )
    unparsed = 0
    if fmt == "explain":
        question = (rec.get("screening") or screening_fields(rec["self_summary"], own_position))["classifier_question"]
        verdicts = await grade_completions(grader, question, messages, completions)
    else:
        picks = [parse_choice(c) for c in completions]
        unparsed = sum(p is None for p in picks)
        verdicts = [p == own_position for p in picks]
    return {"article_id": rec["article_id"], "format": fmt, "own_position": own_position,
            "own": sum(verdicts), "n": len(verdicts), "unparsed": unparsed, "verdicts": verdicts,
            "messages": messages, "completions": completions}


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="prompt records from build_prompts.py")
    parser.add_argument("--models", nargs="+", required=True, metavar="NAME=SPEC")
    parser.add_argument("--formats", nargs="+", choices=list(FORMATS), default=["syspopped", "explain"])
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--grader", default=os.environ.get("CHIVE_GRADER_LLM") or DEFAULTS["grader_llm"])
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    sources = {d: load_source(d)[0] for d in {r.get("dataset", "xsum") for r in records}}
    set_max_concurrency(args.concurrency)
    grader = make_llm_client(args.grader) if "explain" in args.formats else None

    rows = []
    for name, spec in (m.split("=", 1) for m in args.models):
        target = make_llm_client(spec)
        cells = await asyncio.gather(*[
            run_cell(rec, sources[rec.get("dataset", "xsum")][rec["article_id"]], fmt, pos, args.n, target, grader)
            for rec in records for fmt in args.formats for pos in (1, 2)
        ])
        rows += [{"model": name, "spec": spec, **c} for c in cells]
        write_jsonl(rows, args.out)  # rewrite after each model so a disconnect keeps finished models

        print(f"\n{name} ({spec})")
        for fmt in args.formats:
            tot = {1: [0, 0], 2: [0, 0]}
            for c in cells:
                if c["format"] == fmt:
                    tot[c["own_position"]][0] += c["own"]; tot[c["own_position"]][1] += c["n"]
            (a, na), (b, nb) = tot[1], tot[2]
            unparsed = sum(c["unparsed"] for c in cells if c["format"] == fmt)
            print(f"  {fmt:>9}: own listed first {a}/{na} ({a/na:.0%}) | listed second {b}/{nb} ({b/nb:.0%}) | "
                  f"average {(a/na + b/nb)/2:.0%}" + (f" | unparsed {unparsed}" if unparsed else ""))


if __name__ == "__main__":
    asyncio.run(main())
