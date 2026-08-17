"""Sanity-check that target generations have thinking OFF.

For the held-out eval dataset builds, the TARGET model (Qwen/Gemma) must generate
with thinking disabled in Stage 1 (completions) AND Stage 3 (run_prompt
counterfactual targets), and in PETRI audits. The Opus/Sonnet INVESTIGATOR/AUDITOR
reasoning is separate and expected — this script only inspects target output.

It scans for any leaked `<think>`/`</think>` in target completions and (when
present) any nonzero reasoning-token counts. Empirical ground truth, not a
re-read of the config.

Usage:
    python check_thinking_off.py --run-dir <mu_run_dir> [--n 300]
    python check_thinking_off.py --petri-transcripts <petri_transcripts.json> [--n 300]

--n samples the first N records (these files are 100s of MB); pass --n 0 for all.
"""
import argparse
import json
from pathlib import Path

THINK_MARKERS = ("<think>", "</think>")


def _scan(strings):
    """Return (n_total, n_with_think, first_example)."""
    n_total = 0
    n_think = 0
    example = None
    for s in strings:
        n_total += 1
        if any(m in s for m in THINK_MARKERS):
            n_think += 1
            if example is None:
                example = s[:500]
    return n_total, n_think, example


def check_stage1(run_dir: Path, n: int):
    path = run_dir / "completions.jsonl"
    if not path.exists():
        print(f"[stage1] no completions.jsonl in {run_dir}")
        return
    comps = []
    reasoning_tokens = 0
    n_records = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            rec = json.loads(line)
            n_records += 1
            comps.extend(rec.get("completions", []))
            gu = rec.get("generation_usage") or {}
            # generation_usage may be a dict or list-of-dicts depending on backend
            for u in (gu if isinstance(gu, list) else [gu]):
                if isinstance(u, dict):
                    reasoning_tokens += u.get("reasoning_tokens", 0) or 0
    tot, think, ex = _scan(comps)
    print(f"[stage1] {n_records} prompts, {tot} completions, "
          f"{think} with <think> markers, reasoning_tokens={reasoning_tokens}")
    if think:
        print(f"  !!! LEAK example: {ex!r}")


def check_stage3(run_dir: Path, n: int):
    path = run_dir / "investigations.jsonl"
    if not path.exists():
        print(f"[stage3] no investigations.jsonl in {run_dir}")
        return
    target_comps = []
    n_records = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            rec = json.loads(line)
            n_records += 1
            # baseline completions (Stage-1 samples re-graded) + every experiment's
            # run_prompt completions are ALL target output.
            target_comps.extend(rec.get("completions", []))
            for exp in rec.get("experiment_log", []) or []:
                target_comps.extend(exp.get("completions", []))
    tot, think, ex = _scan(target_comps)
    print(f"[stage3] {n_records} investigations, {tot} target completions "
          f"(baseline + all experiments), {think} with <think> markers")
    if think:
        print(f"  !!! LEAK example: {ex!r}")


def check_petri(transcripts_path: Path, n: int):
    d = json.load(open(transcripts_path))
    ts = d["transcripts"]
    if n:
        ts = ts[:n]
    contents = []
    for t in ts:
        for turn in t["turns"]:
            if turn["kind"] == "assistant":
                contents.append(turn.get("content") or "")
    tot, think, ex = _scan(contents)
    meta = d.get("metadata", {})
    print(f"[petri] {len(ts)} transcripts (target={meta.get('target_model')}, "
          f"auditor={meta.get('auditor_model')}), {tot} target assistant turns, "
          f"{think} with <think> markers")
    if think:
        print(f"  !!! LEAK example: {ex!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--petri-transcripts", type=Path, default=None)
    ap.add_argument("--n", type=int, default=300, help="sample first N records (0 = all)")
    args = ap.parse_args()
    if args.run_dir:
        print(f"=== {args.run_dir} (first {args.n or 'ALL'} records) ===")
        check_stage1(args.run_dir, args.n)
        check_stage3(args.run_dir, args.n)
    if args.petri_transcripts:
        print(f"=== {args.petri_transcripts} (first {args.n or 'ALL'} records) ===")
        check_petri(args.petri_transcripts, args.n)
    if not args.run_dir and not args.petri_transcripts:
        ap.error("pass --run-dir and/or --petri-transcripts")


if __name__ == "__main__":
    main()
