"""
Pipeline orchestrator for model understanding data generation.

Runs the full pipeline end-to-end with resume support. Every model-facing stage
selects its backend with the same llm_client spec (vllm:/openrouter:/anthropic:):
- Stage 0: Slice the first N rows of the canonical prompt pool into the run dir
- Stage 1: Generate completions via stage1_llm (vLLM server / OpenRouter)
- Stage 2: Screen completions for interesting behaviors (judge_llm)
- Stage 3: Deep investigation with counterfactual experiments (stage3_llm + target_llm)
- Stage 4: Verification judge scoring (judge_llm)
- Stage 5: Synthetic data generation

Usage:
    source .env
    .venv/bin/python chive/pipeline/run_pipeline.py \
        --config data/runs/test_mixed/config.json
"""

import argparse
import asyncio
import functools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

print = functools.partial(print, flush=True)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

from chive.pipeline.utils import load_dotenv

load_dotenv()

import anthropic

# Import stage modules
import chive.pipeline.screen_completions as screen_completions
import chive.pipeline.investigate as investigate_mod
import chive.pipeline.verify_investigations as verify_investigations
import chive.pipeline.generate_synthetic_data as generate_synthetic_data
import chive.pipeline.generate_completions as generate_completions_mod
from chive.pipeline.config import load_config
from chive.pipeline.llm_client import make_llm_client, set_max_concurrency
from chive.pipeline.run_files import load_stage_records
from chive.pipeline.build_prompt_pool import (
    content_hash,
    load_exclude_hashes,
)

# All pipeline defaults live in chive/pipeline/config.py (DEFAULTS) — the single
# source of truth shared with the standalone stage CLIs. load_config merges a
# run's config.json over them and fails loudly on unknown keys.


def report_stage_time(name: str, elapsed: float, status: dict, stage: str, count_key: str):
    """Print a stage's wall-clock plus per-item throughput.

    Reads the item count the stage just recorded in status so the line shows
    "<name> took <s>s (<n> items, <rate>/min, <s/item>s/item)" — enough to gauge
    rate during a run and to extrapolate total time from a smaller test.
    """
    n = status.get(stage, {}).get(count_key)
    line = f"  {name} took {elapsed:.0f}s"
    if n:
        line += f" ({n} items, {n / elapsed * 60:.1f}/min, {elapsed / n:.1f}s/item)"
    print(line)


def update_status(run_dir: Path, status: dict, stage: str, data: dict):
    """Update a stage's status and write to disk."""
    status[stage] = data
    status["last_updated"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))
    print(f"  Status: {stage} = {data.get('status', '?')}")


# ---------------------------------------------------------------------------
# Stage 1: Generate completions (single + multi-turn, via vLLM server)
# ---------------------------------------------------------------------------

async def run_stage0(config: dict, run_dir: Path, status: dict) -> Path:
    """Slice the first N rows of the canonical pool into run_dir/prompts.jsonl.

    Model-agnostic: copies prompt records (messages only) from the pool. Optional
    exclude_files drop prompts (by conversation_hash / content hash) seen in prior
    runs. This is all Stage 0 does now — no WildChat streaming, no tokenization.
    """
    out = run_dir / "prompts.jsonl"
    if status.get("stage0", {}).get("status") == "done" and out.exists():
        print("Stage 0 already done, skipping")
        return out

    pool_path = Path(config["prompt_pool"])
    if not pool_path.is_absolute():
        pool_path = REPO_ROOT / pool_path
    assert pool_path.exists(), (
        f"Prompt pool not found: {pool_path}. Build it once with build_prompt_pool.py."
    )

    exclude_conv = exclude_content = None
    if config["exclude_files"]:
        exclude_conv, exclude_content = load_exclude_hashes(config["exclude_files"])

    offset = config["stage1_offset"]
    n = config["stage1_n_prompts"]
    selected: list[dict] = []
    with open(pool_path) as f:
        for i, line in enumerate(f):
            if i < offset or not line.strip():
                continue
            rec = json.loads(line)
            if exclude_conv and rec.get("conversation_hash", "") in exclude_conv:
                continue
            if exclude_content and content_hash(rec["messages"]) in exclude_content:
                continue
            selected.append(rec)
            if len(selected) >= n:
                break

    assert selected, f"No prompts selected from pool {pool_path} (offset={offset}, n={n})"
    if len(selected) < n:
        print(f"  WARNING: only {len(selected)} prompts available from pool (requested {n})")
    with open(out, "w") as f:
        for rec in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Stage 0: wrote {len(selected)} prompts -> {out}")
    update_status(run_dir, status, "stage0", {"status": "done", "n_prompts": len(selected)})
    return out


def ensure_vllm_server(job_name: str) -> str:
    """Find a running vLLM server by Slurm job name. Fails if not found."""
    return investigate_mod.discover_vllm_url(job_name=job_name)


def _resolve_stage1_spec(config: dict) -> str:
    """Resolve the Stage-1 backend spec, discovering the vLLM URL if needed.

    A bare "vllm:MODEL" (no @url) is resolved against the running vLLM server
    found by vllm_job_name. Full specs (with @url, or openrouter:...) pass through.
    """
    spec = config["stage1_llm"]
    assert spec, (
        "config['stage1_llm'] is required "
        "(e.g. 'openrouter:Qwen/Qwen3.5-397B-A17B' or 'vllm:Qwen/Qwen3-32B-FP8@http://node:8000')"
    )
    if spec.startswith("vllm:") and "@" not in spec:
        url = ensure_vllm_server(job_name=config["vllm_job_name"])
        spec = f"{spec}@{url}"
    return spec


async def run_stage1(config: dict, run_dir: Path, status: dict, prompts_path: Path) -> Path:
    """Generate completions from the Stage-0 slice via an llm_client backend."""
    output_file = run_dir / "completions.jsonl"
    if status.get("stage1", {}).get("status") == "done" and output_file.exists():
        print("Stage 1 already done, skipping")
        return output_file

    update_status(run_dir, status, "stage1", {"status": "running"})
    spec = _resolve_stage1_spec(config)
    client = make_llm_client(spec)
    candidates = generate_completions_mod.load_prompts(str(prompts_path), n_prompts=None, offset=0)
    n_completions = config["n_completions"]
    max_tokens = config["target_max_tokens"]
    thinking_budget = config["target_thinking_budget"]
    print(f"  Stage 1: {len(candidates)} prompts via {spec} "
          f"x {n_completions} completions x {max_tokens} tok "
          f"(thinking_budget={thinking_budget})")

    prompt_results = await generate_completions_mod.generate_completions_api(
        client=client,
        candidates=candidates,
        n_completions=n_completions,
        max_tokens=max_tokens,
        concurrency=config["concurrency"],
        jsonl_path=output_file,
        thinking_budget=thinking_budget,
    )

    total_completions = sum(len(p.completions) for p in prompt_results)
    n_length = sum(1 for p in prompt_results for fr in p.finish_reasons if fr == "length")
    trunc_rate = n_length / total_completions if total_completions else 0.0
    from collections import Counter
    turn_dist = Counter(p.n_turns for p in prompt_results)
    # The .jsonl written incrementally above IS the stage output; run metadata
    # (model spec, n_completions, max tokens, thinking budget) lives in
    # config.json, counts/rates in status.json below.
    print(f"  Saved {len(prompt_results)} prompts to {output_file} (truncated {trunc_rate:.1%})")

    update_status(run_dir, status, "stage1", {
        "status": "done",
        "n_prompts": len(prompt_results),
        "total_completions": total_completions,
        "truncated_rate": trunc_rate,
        "turn_distribution": dict(turn_dist),
    })
    return output_file


# ---------------------------------------------------------------------------
# Stage 2: Screen completions
# ---------------------------------------------------------------------------

async def run_stage2(
    config: dict, run_dir: Path, status: dict, completions_path: Path,
):
    """Screen completions for interesting behaviors."""
    if status.get("stage2", {}).get("status") == "done":
        print("Stage 2 already done, skipping")
        return

    update_status(run_dir, status, "stage2", {"status": "running"})

    prompts = screen_completions.load_prompts(
        completions_path, n=config["stage2_n"],
    )

    screening_path = run_dir / "screening.jsonl"
    mode = config["stage2_mode"]

    judge_spec = config["judge_llm"]
    assert judge_spec, (
        "config['judge_llm'] is required "
        "(e.g. 'anthropic:claude-opus-4-6' or 'vllm:Qwen/Qwen3-32B-FP8@http://node:8000')"
    )
    judge_args = dict(
        llm_spec=judge_spec,
        max_tokens=config["judge_max_tokens"],
        thinking_budget=config["judge_thinking_budget"] if config["judge_thinking_budget"] > 0 else None,
    )

    if config["screening_file"]:
        raw = config["screening_file"]
        prewritten = Path(raw) if Path(raw).is_absolute() else REPO_ROOT / raw
        print(f"  Using prewritten screening from {prewritten} (no judge call)")
        results = screen_completions.screen_prewritten(prompts, prewritten, screening_path)
    elif mode == "batch":
        assert judge_spec.startswith("anthropic:"), (
            "Stage 2 batch mode is Anthropic-only"
        )
        results = screen_completions.screen_batch(
            prompts, screening_path,
            model=judge_spec.split(":", 1)[1],
            batch_chunk_size=config["batch_chunk_size"],
            max_tokens=config["judge_max_tokens"],
            thinking_budget=config["judge_thinking_budget"] if config["judge_thinking_budget"] > 0 else None,
        )
    elif mode == "async":
        results = await screen_completions.screen_async(
            prompts, screening_path,
            concurrency=config["concurrency"],
            **judge_args,
        )
    else:
        raise ValueError(f"Unknown stage2_mode: {mode!r}")

    screen_completions.summarize_results(results, screening_path)

    score_dist = {}
    for s in range(1, 6):
        count = sum(1 for r in results if r.interest_score == s)
        if count > 0:
            score_dist[str(s)] = count

    # "partial" (not "done") when items were skipped, so a re-run resumes the missing ones.
    update_status(run_dir, status, "stage2", {
        "status": "done" if len(results) == len(prompts) else "partial",
        "total": len(prompts),
        "completed": len(results),
        "score_distribution": score_dist,
    })


# ---------------------------------------------------------------------------
# Stage 3: Investigation
# ---------------------------------------------------------------------------

async def run_stage3(config: dict, run_dir: Path, status: dict):
    """Deep investigation of interesting behaviors."""
    if status.get("stage3", {}).get("status") == "done":
        print("Stage 3 already done, skipping")
        return

    # Load screening results
    min_score = config["stage3_min_score"]
    candidates = [
        r for r in load_stage_records(run_dir, "screening")
        if r["interest_score"] >= min_score
    ]
    candidates.sort(key=lambda r: r["interest_score"], reverse=True)
    if config["stage3_n"] is not None:
        candidates = candidates[:config["stage3_n"]]

    print(f"  {len(candidates)} prompts with score >= {min_score}")

    if not candidates:
        print("  No prompts to investigate!")
        update_status(run_dir, status, "stage3", {
            "status": "done", "total": 0, "completed": 0,
        })
        return

    target_llm_spec = config["target_llm"]
    assert target_llm_spec, (
        "config['target_llm'] is required — the investigated model for run_prompt "
        "counterfactuals (e.g. 'vllm:Qwen/Qwen3-32B-FP8@http://node:8000')"
    )
    target_client = make_llm_client(target_llm_spec)
    test = await investigate_mod.run_endpoint_prompt(
        target_client, user_message="Say hello.", n=1, max_tokens=20,
    )
    print(f"  Target backend OK: {test[0][:80]}")

    agent_llm_spec = config["stage3_llm"]
    assert agent_llm_spec, (
        "config['stage3_llm'] is required "
        "(e.g. 'anthropic:claude-opus-4-6' or 'vllm:Qwen/Qwen3-32B-FP8@http://node:8000')"
    )
    agent_client = make_llm_client(agent_llm_spec)
    grader_client = make_llm_client(config["grader_llm"])

    update_status(run_dir, status, "stage3", {
        "status": "running",
        "total": len(candidates),
        "target_llm": target_llm_spec,
    })

    jsonl_path = run_dir / "investigations.jsonl"
    outcome = await investigate_mod.run_investigations(
        candidates,
        agent_client=agent_client,
        target_client=target_client,
        grader_client=grader_client,
        jsonl_path=jsonl_path,
        concurrency=config["concurrency"],
        max_tokens=config["stage3_max_tokens"],
        thinking_budget=config["stage3_thinking_budget"] if config["stage3_thinking_budget"] > 0 else None,
        max_agent_turns=config["stage3_max_agent_turns"],
        vllm_n_completions=config["n_completions"],
        thorough=config["stage3_thorough"],
        target_thinking_budget=config["target_thinking_budget"],
        target_max_tokens=config["target_max_tokens"],
        n_completions_shown=config["stage3_n_completions_shown"],
        max_skip_rate=config["stage3_max_skip_rate"],
    )

    print(f"\n  {len(outcome.records)} investigations in {jsonl_path}")

    update_status(run_dir, status, "stage3", {
        "status": "done" if outcome.n_skipped == 0 else "partial",
        "total": len(candidates),
        "completed": len(outcome.records),
        "skipped": outcome.n_skipped,
    })


# ---------------------------------------------------------------------------
# Stage 4: Verification
# ---------------------------------------------------------------------------

async def run_stage4(config: dict, run_dir: Path, status: dict):
    """Verification judge scoring."""
    if status.get("stage4", {}).get("status") == "done":
        print("Stage 4 already done, skipping")
        return

    investigations = verify_investigations.load_investigations(
        run_dir / "investigations.jsonl", n=None,
    )

    if not investigations:
        print("  No investigations to verify!")
        update_status(run_dir, status, "stage4", {
            "status": "done", "total": 0, "completed": 0,
        })
        return

    update_status(run_dir, status, "stage4", {
        "status": "running",
        "total": len(investigations),
    })

    verification_path = run_dir / "verification.jsonl"
    mode = config["stage4_mode"]

    judge_spec = config["judge_llm"]
    assert judge_spec, (
        "config['judge_llm'] is required "
        "(e.g. 'anthropic:claude-opus-4-6' or 'vllm:Qwen/Qwen3-32B-FP8@http://node:8000')"
    )
    judge_args = dict(
        llm_spec=judge_spec,
        max_tokens=config["judge_max_tokens"],
    )

    if mode == "batch":
        assert judge_spec.startswith("anthropic:"), (
            "Stage 4 batch mode is Anthropic-only"
        )
        results = verify_investigations.verify_batch(
            investigations, verification_path,
            model=judge_spec.split(":", 1)[1],
            batch_chunk_size=config["stage4_batch_chunk_size"],
            max_tokens=config["judge_max_tokens"],
            thinking_budget=config["judge_thinking_budget"] if config["judge_thinking_budget"] > 0 else None,
        )
    elif mode == "async":
        results = await verify_investigations.verify_async(
            investigations, verification_path,
            thinking_budget=config["judge_thinking_budget"] if config["judge_thinking_budget"] > 0 else None,
            concurrency=config["concurrency"],
            n_verifications=config["stage4_n_verifications"],
            **judge_args,
        )
    else:
        raise ValueError(f"Unknown stage4_mode: {mode!r}")

    verify_investigations.summarize_results(results)

    # Score is a float (ensemble mean) — bucket by round() for the distribution.
    # Display only: round() admits 7.5-7.99, so summing buckets 8-10 overcounts
    # the kept set by ~10%. The keep gate is a strict float compare (>=8 eval,
    # >=7 training); report n_score_ge_8 below, never the buckets.
    score_dist = {}
    for s in range(1, 11):
        count = sum(1 for r in results if round(r.score) == s)
        if count > 0:
            score_dist[str(s)] = count

    update_status(run_dir, status, "stage4", {
        "status": "done" if len(results) == len(investigations) else "partial",
        "total": len(investigations),
        "completed": len(results),
        "score_distribution": score_dist,
        "n_score_ge_8": sum(1 for r in results if r.score >= 8),
        "n_score_ge_7": sum(1 for r in results if r.score >= 7),
        "mean_score": round(
            sum(r.score for r in results) / len(results), 2,
        ) if results else 0,
    })


# ---------------------------------------------------------------------------
# Stage 5: Synthetic data generation
# ---------------------------------------------------------------------------

async def run_stage5(config: dict, run_dir: Path, status: dict):
    """Generate synthetic training data from pipeline outputs."""
    if status.get("stage5", {}).get("status") == "done":
        print("Stage 5 already done, skipping")
        return

    screening_path = run_dir / "screening.jsonl"
    investigations_path = run_dir / "investigations.jsonl"
    verification_path = run_dir / "verification.jsonl"
    output_path = run_dir / "synthetic_data.json"

    for p in [screening_path, investigations_path, verification_path]:
        assert p.exists(), f"Required file missing: {p}"

    update_status(run_dir, status, "stage5", {"status": "running"})

    stage5_spec = config["stage5_llm"]
    assert stage5_spec, (
        "config['stage5_llm'] is required "
        "(e.g. 'anthropic:claude-sonnet-4-5-20250929' or 'vllm:Qwen/...@http://node:8000')"
    )
    # Model name embedded in each request dict; only the Anthropic batch path
    # sends it. The async path routes by stage5_spec and ignores it.
    model_name = make_llm_client(stage5_spec).model
    min_interest = config["stage5_min_interest_score"]
    min_verification = config["stage5_min_verification_score"]
    n_cf_experiments = config["stage5_n_counterfactual_experiments"]
    seed = config["stage5_seed"]

    examples = generate_synthetic_data.load_joined_data(
        screening_path, investigations_path, verification_path,
        min_interest_score=min_interest,
        min_verification_score=min_verification,
    )

    # Stage 5 thinking is part of the data-generating process (see the
    # stage5_thinking_budget comment in config.DEFAULTS). ONE value for both
    # modes: prepare_requests bakes it into the request params (what batch mode
    # sends) and generate_async receives the same value.
    s5_think = config["stage5_thinking_budget"]
    s5_thinking_budget = s5_think if s5_think > 0 else None

    requests = generate_synthetic_data.prepare_requests(
        examples,
        n_counterfactual_experiments=n_cf_experiments,
        seed=seed,
        model=model_name,
        thinking_budget=s5_thinking_budget,
    )

    mode = config["stage5_mode"]
    if mode == "async":
        results = await generate_synthetic_data.generate_async(
            requests, output_path,
            llm_spec=stage5_spec,
            concurrency=config["concurrency"],
            thinking_budget=s5_thinking_budget,
        )
    elif mode == "batch":
        assert stage5_spec.startswith("anthropic:"), "Stage 5 batch mode is Anthropic-only"
        results = generate_synthetic_data.generate_batch(
            requests, output_path,
            batch_chunk_size=config["batch_chunk_size"],
        )
    else:
        raise ValueError(f"Unknown stage5_mode: {mode!r}")

    # Build args namespace for save_results (records the spec as provenance)
    args = argparse.Namespace(
        model=stage5_spec,
        investigations=investigations_path,
        screening=screening_path,
        verification=verification_path,
        min_interest_score=min_interest,
        min_verification_score=min_verification,
        n_counterfactual_experiments=n_cf_experiments,
        seed=seed,
    )
    generate_synthetic_data.save_results(results, output_path, args)

    bp_count = sum(1 for r in results
                   if r.example_type == "behavior_prediction")
    cf_count = sum(1 for r in results
                   if r.example_type == "counterfactual_prediction")

    update_status(run_dir, status, "stage5", {
        "status": "done",
        "n_examples": len(results),
        "n_behavior_predictions": bp_count,
        "n_counterfactual_predictions": cf_count,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Run model understanding pipeline (stages 1-5)",
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to config JSON (run directory = config file's parent dir)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = args.config.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load or create status
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    print(f"Pipeline: {config['run_name']}")
    print(f"Run dir:  {run_dir}")
    print(f"Config:\n{json.dumps(config, indent=2)}")

    # Single global in-flight cap per endpoint, enforced at LLMClient.call().
    # Every stage (and all nested fan-out) shares this budget.
    set_max_concurrency(config["concurrency"])

    t0 = time.time()

    # Stage 0 (slice pool) + Stage 1 (generate), unless a pre-built completions
    # .jsonl (one PromptCompletions record per line) is supplied via config.
    if config["completions_file"]:
        raw = config["completions_file"]
        completions_path = Path(raw) if Path(raw).is_absolute() else REPO_ROOT / raw
        assert completions_path.exists(), (
            f"Completions file not found: {completions_path}"
        )
        print(f"\nUsing existing completions: {completions_path}")
    else:
        print(f"\n{'='*60}")
        print("STAGE 0: Prompt slice")
        print(f"{'='*60}")
        prompts_path = await run_stage0(config, run_dir, status)

        print(f"\n{'='*60}")
        print("STAGE 1: Generate completions")
        print(f"{'='*60}")
        t1 = time.time()
        completions_path = await run_stage1(config, run_dir, status, prompts_path)
        report_stage_time("Stage 1", time.time() - t1, status, "stage1", "n_prompts")

    # Stage 2: Screening
    print(f"\n{'='*60}")
    print("STAGE 2: Screening")
    print(f"{'='*60}")
    t2 = time.time()
    await run_stage2(config, run_dir, status, completions_path)
    report_stage_time("Stage 2", time.time() - t2, status, "stage2", "completed")

    # Stage 3: Investigation
    print(f"\n{'='*60}")
    print("STAGE 3: Investigation")
    print(f"{'='*60}")
    t3 = time.time()
    await run_stage3(config, run_dir, status)
    report_stage_time("Stage 3", time.time() - t3, status, "stage3", "completed")

    # Stage 4: Verification
    print(f"\n{'='*60}")
    print("STAGE 4: Verification")
    print(f"{'='*60}")
    t4 = time.time()
    await run_stage4(config, run_dir, status)
    report_stage_time("Stage 4", time.time() - t4, status, "stage4", "completed")

    # Stage 5: Synthetic data generation
    if config["skip_stage5_synthetic"]:
        print(f"\nStage 5 skipped (skip_stage5_synthetic=true)")
        update_status(run_dir, status, "stage5", {
            "status": "skipped",
            "reason": "skip_stage5_synthetic=true in config",
        })
    else:
        print(f"\n{'='*60}")
        print("STAGE 5: Synthetic data generation")
        print(f"{'='*60}")
        t5 = time.time()
        await run_stage5(config, run_dir, status)
        report_stage_time("Stage 5", time.time() - t5, status, "stage5", "n_examples")

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
