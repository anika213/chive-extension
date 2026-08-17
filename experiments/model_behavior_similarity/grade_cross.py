"""Cross-model behavior grading for the behavior-similarity check (see README.md).

For every (prompt, behavior_question) unit in each bank, grade the OTHER subject
model's 30 baseline completions with the bank's classifier question, using the
same grader model + prompt construction as the pipeline that produced the stored
ground-truth rates. Also re-grades a 50-unit-per-bank SELF sample as a
reproducibility check (stored rate vs regraded rate under this run's settings).

CPU/API only — run with .env sourced, repo root cwd.
Resumable: rows already in results/verdicts.jsonl are skipped on re-run.
"""

import argparse
import asyncio
import functools
import json
import random
from pathlib import Path

print = functools.partial(print, flush=True)

from chive.pipeline.investigate import (
    GRADER_SYSTEM_PROMPT,
    GRADER_TOOL,
    GradeCompletionsInput,
    build_grader_user_content,
)
from chive.pipeline.llm_client import make_llm_client
from chive.pipeline.tool_validate import call_for_validated_tool
from chive import paths

RUNS = paths.RUNS_DIR
QWEN_RUN = "opus_investigates_qwen3_8b_5000_thorough_n30"
LLAMA_RUN = "opus_investigates_llama3_1_8b_5000_thorough_n30"
BANK_NAME = "eval_set_conc3_n2000.jsonl"
OUT_PATH = paths.experiment_data_dir("model_behavior_similarity") / "results" / "verdicts.jsonl"

GRADER_MODEL = "anthropic:claude-sonnet-4-6"  # same model as the pipeline grader
# User-approved bump over the pipeline grader's 2048 thinking budget (~1k per
# completion for 30 completions). The self_regrade directions use these SAME
# settings, so stored-vs-regraded drift directly measures the instrument shift.
THINKING_BUDGET = 30_000
MAX_TOKENS = 36_000
PER_ATTEMPT_TIMEOUT_S = 600.0  # 30k thinking can exceed the 240s default
CONCURRENCY = 150
N_SELF_REGRADE_PER_BANK = 50
SEED = 0

# direction -> (bank run, graded run)
DIRECTIONS = {
    "qwen_bank_on_llama": (QWEN_RUN, LLAMA_RUN),
    "llama_bank_on_qwen": (LLAMA_RUN, QWEN_RUN),
    "self_regrade_qwen_bank": (QWEN_RUN, QWEN_RUN),
    "self_regrade_llama_bank": (LLAMA_RUN, LLAMA_RUN),
}


def load_bank_units(run: str) -> dict[str, dict]:
    """Dedupe a bank's claims to one unit per prompt; assert the within-prompt
    invariants (single behavior_question / baseline rate / conversation)."""
    units: dict[str, dict] = {}
    for line in open(RUNS / run / BANK_NAME):
        c = json.loads(line)
        assert c["behavior_question"].strip(), c["claim_id"]
        u = units.setdefault(
            c["prompt_id"],
            {
                "behavior_question": c["behavior_question"],
                "k_baseline": c["k_baseline"],
                "n_baseline": c["n_baseline"],
                "messages": c["messages"],
            },
        )
        assert u["behavior_question"] == c["behavior_question"], c["claim_id"]
        assert (u["k_baseline"], u["n_baseline"]) == (c["k_baseline"], c["n_baseline"]), c["claim_id"]
        assert u["messages"] == c["messages"], c["claim_id"]
    return units


def load_completions(run: str, pids: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in open(RUNS / run / "completions.jsonl"):
        d = json.loads(line)
        if d["id"] in pids:
            out[d["id"]] = d
    missing = pids - set(out)
    assert not missing, f"{run}: {len(missing)} bank prompts missing completions: {sorted(missing)[:5]}"
    return out


def build_tasks() -> list[dict]:
    bank_units = {QWEN_RUN: load_bank_units(QWEN_RUN), LLAMA_RUN: load_bank_units(LLAMA_RUN)}
    all_pids = set(bank_units[QWEN_RUN]) | set(bank_units[LLAMA_RUN])
    comps = {run: load_completions(run, all_pids) for run in (QWEN_RUN, LLAMA_RUN)}

    tasks = []
    for direction, (bank_run, graded_run) in DIRECTIONS.items():
        units = bank_units[bank_run]
        pids = sorted(units)
        if direction.startswith("self_regrade"):
            pids = random.Random(SEED).sample(pids, N_SELF_REGRADE_PER_BANK)
        for pid in pids:
            u = units[pid]
            comp = comps[graded_run][pid]
            # The two runs share the prompt pool; the grading conversation must be
            # the exact conversation the graded completions respond to.
            assert comp["messages"] == u["messages"], (direction, pid)
            assert len(comp["completions"]) == u["n_baseline"] == 30, (direction, pid)
            tasks.append(
                {
                    "direction": direction,
                    "bank_run": bank_run,
                    "graded_run": graded_run,
                    "prompt_id": pid,
                    "behavior_question": u["behavior_question"],
                    "messages": u["messages"],
                    "completions": comp["completions"],
                    "stored_k_baseline": u["k_baseline"],
                    "stored_n_baseline": u["n_baseline"],
                    "n_truncated": sum(fr == "length" for fr in comp["finish_reasons"]),
                }
            )
    return tasks


async def grade_one(client, sem: asyncio.Semaphore, task: dict) -> dict | None:
    """grade_completions() with this run's token budgets; same prompt, tool,
    system prompt, and coverage check as the pipeline grader."""
    completions = task["completions"]
    user_content = build_grader_user_content(
        task["behavior_question"], task["messages"], completions
    )
    expected = set(range(1, len(completions) + 1))
    async with sem:
        for attempt in range(2):
            outcome = await call_for_validated_tool(
                client,
                system=GRADER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
                tool=GRADER_TOOL,
                model_cls=GradeCompletionsInput,
                max_tokens=MAX_TOKENS,
                thinking_budget=THINKING_BUDGET,
                per_attempt_timeout_seconds=PER_ATTEMPT_TIMEOUT_S,
            )
            if outcome is None:
                continue
            tool_input, _resp = outcome
            got = [v.completion for v in tool_input.verdicts]
            if set(got) != expected or len(got) != len(expected):
                print(
                    f"  [grader] {task['direction']}/{task['prompt_id']} coverage mismatch "
                    f"(attempt {attempt + 1}/2): got {sorted(got)}"
                )
                continue
            by_num = {v.completion: v.answer == "yes" for v in tool_input.verdicts}
            verdicts = [by_num[i] for i in range(1, len(completions) + 1)]
            return {
                "direction": task["direction"],
                "bank_run": task["bank_run"],
                "graded_run": task["graded_run"],
                "prompt_id": task["prompt_id"],
                "behavior_question": task["behavior_question"],
                "verdicts": verdicts,
                "k": sum(verdicts),
                "n": len(verdicts),
                "stored_k_baseline": task["stored_k_baseline"],
                "stored_n_baseline": task["stored_n_baseline"],
                "n_truncated": task["n_truncated"],
                "grader_model": GRADER_MODEL,
                "thinking_budget": THINKING_BUDGET,
            }
    return None


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="grade only the first N pending units (smoke test)")
    args = parser.parse_args()

    tasks = build_tasks()
    done: set[tuple[str, str]] = set()
    if OUT_PATH.exists():
        for line in open(OUT_PATH):
            r = json.loads(line)
            done.add((r["direction"], r["prompt_id"]))
    todo = [t for t in tasks if (t["direction"], t["prompt_id"]) not in done]
    print(f"{len(tasks)} units total, {len(done)} already graded, {len(todo)} to grade")
    if args.limit is not None:
        todo = todo[: args.limit]
        print(f"--limit {args.limit}: grading {len(todo)} units")
    if not todo:
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = make_llm_client(GRADER_MODEL)
    sem = asyncio.Semaphore(CONCURRENCY)
    n_ok = n_fail = 0
    with open(OUT_PATH, "a") as out_f:
        pending = [asyncio.create_task(grade_one(client, sem, t)) for t in todo]
        for fut in asyncio.as_completed(pending):
            row = await fut
            if row is None:
                n_fail += 1
                continue
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            n_ok += 1
            if n_ok % 50 == 0:
                print(f"  graded {n_ok}/{len(todo)} (failures so far: {n_fail})")

    print(f"done: {n_ok} graded, {n_fail} failed")
    if n_fail:
        rate = n_fail / len(todo)
        print(
            f"WARNING: {n_fail}/{len(todo)} units ({rate:.1%}) failed grading — "
            f"re-run this script to retry them (resume skips completed units)."
        )


if __name__ == "__main__":
    asyncio.run(main())
