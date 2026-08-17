"""Stage 1 — generate the freeform self-explanation cache (method B) for any unified Claim bank.

For each bank row (a cued prompt + one sampled behavior completion), the self-explanation model is
asked the row's `freeform_question` ("Why did you answer (C)?" / "Why did you conclude that the
poster is the asshole?") and produces a freeform explanation. We cache the full explanation AND the
full rendered prompt sent to the model, so the downstream LLM-judge extraction (Stage 2) can be
re-iterated without re-running generation.

Conversation shape = the EXACT training prefix (cf. experiments/self_explanation/
precompute_selfexpl_cache.build_conversation): transcript → behavior completion (assistant) → the
self-explanation question (user), rendered no-think with add_generation_prompt, NO system prompt.

Self-contained to this folder. One arm per invocation:
  --arm trained  + (--lora-path <dir>  [vllm 8B]  |  --tinker-weights tinker://… [tinker 397B])
  --arm base     + --backend {vllm,tinker}        (un-fine-tuned base, same question, zero-shot)

Generation matches the in-distribution self-expl cache: temperature 0.7, no-think, n=1.
"""

from __future__ import annotations

import os

# Must precede any transitive torch/CUDA import so vLLM re-inits CUDA in its worker
# (AGENTS.md vLLM-offline footgun). Harmless for the tinker path.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import asyncio
import json
from pathlib import Path

from chive.pipeline.utils import report_skip_rate

TEMPERATURE = 0.7
ENABLE_THINKING = False


def build_conversation(row: dict, suffix: str = "") -> list[dict]:
    """transcript (prompt+cue) → behavior completion (assistant) → self-explanation question (user).

    The question comes straight from the unified bank's `freeform_question` (e.g. "Why did you
    answer (C)?" / "Why did you conclude that the poster is the asshole?"), so this runs unchanged
    on ANY unified Claim bank (MCQ settings + sycophancy). `suffix` is an optional elicitation string
    appended to the question (used for base-model controls, e.g. "…explain the specific factors…")."""
    assert len(row["messages"]) == 1 and row["messages"][0]["role"] == "user", \
        f"expected a single user turn, got {row['messages']}"
    return [
        row["messages"][0],
        {"role": "assistant", "content": row["chosen_completion"]},
        {"role": "user", "content": row["freeform_question"] + suffix},
    ]


def load_eval_set(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in open(path)]
    # One claim per prompt_id in the sycophancy builder, but guard anyway.
    seen: dict[str, dict] = {}
    for r in rows:
        seen.setdefault(r["prompt_id"], r)
    assert len(seen) == len(rows), f"duplicate prompt_ids in {path} ({len(rows)} rows, {len(seen)} ids)"
    return rows


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def generate_vllm(rows: list[dict], base_model: str, lora_path: str | None,
                  max_new_tokens: int, suffix: str = "") -> list[dict]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # Trained: load tokenizer from the checkpoint (its saved chat_template) so the prefix is
    # byte-identical to training. Base: load from the base model. Vocab identical either way.
    tokenizer = AutoTokenizer.from_pretrained(lora_path if lora_path else base_model)
    convs = [build_conversation(r, suffix) for r in rows]
    rendered = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=ENABLE_THINKING) for c in convs]
    prompt_ids = [tokenizer.apply_chat_template(c, tokenize=True, add_generation_prompt=True,
                                                enable_thinking=ENABLE_THINKING) for c in convs]
    llm = LLM(model=base_model, enable_lora=bool(lora_path), max_lora_rank=64,
              gpu_memory_utilization=0.90, max_model_len=8192)
    sp = SamplingParams(temperature=TEMPERATURE, max_tokens=max_new_tokens, n=1)
    lora_request = LoRARequest("selfexpl", 1, lora_path) if lora_path else None
    outs = llm.generate(prompts=[{"prompt_token_ids": ids} for ids in prompt_ids],
                        sampling_params=sp, lora_request=lora_request)
    results = []
    for r, rp, out in zip(rows, rendered, outs):
        o = out.outputs[0]
        results.append({"prompt_id": r["prompt_id"], "rendered_prompt": rp,
                        "self_explanation": o.text, "finish_reason": o.finish_reason,
                        "n_tokens": len(o.token_ids)})
    return results


def generate_tinker(rows: list[dict], base_model: str, tinker_weights: str | None,
                    concurrency: int, max_new_tokens: int, suffix: str = "") -> list[dict]:
    from transformers import AutoTokenizer

    from chive.pipeline.investigate import run_endpoint_prompt
    from chive.pipeline.llm_client import make_llm_client

    spec = f"tinker:{base_model}@{tinker_weights}" if tinker_weights else f"tinker:{base_model}"
    client = make_llm_client(spec)
    # Reconstruct the rendered prompt for the record (Tinker applies the same base-tokenizer
    # template no-think internally; we render it here only to persist what was sent).
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    sem = asyncio.Semaphore(concurrency)

    async def one(r: dict) -> dict | None:
        conv = build_conversation(r, suffix)
        rp = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=ENABLE_THINKING)
        async with sem:
            for _ in range(4):  # tinker occasionally returns an empty sample; retry then drop
                try:
                    completions = await run_endpoint_prompt(
                        target_client=client, messages=conv, n=1,
                        temperature=TEMPERATURE, max_tokens=max_new_tokens)
                except AssertionError:
                    continue
                return {"prompt_id": r["prompt_id"], "rendered_prompt": rp,
                        "self_explanation": completions[0], "finish_reason": "stop",
                        "n_tokens": None}
        print(f"[gen] DROP {r['prompt_id']}: empty completion after retries")
        return None

    async def run() -> list[dict]:
        return [x for x in await asyncio.gather(*[one(r) for r in rows]) if x is not None]

    return asyncio.run(run())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-set", required=True, help="unified Claim bank jsonl")
    p.add_argument("--base-model", required=True)
    p.add_argument("--arm", required=True, choices=["trained", "base"])
    p.add_argument("--checkpoint-name", required=True, help="identity recorded in rows / cache name")
    p.add_argument("--backend", required=True, choices=["vllm", "tinker"])
    p.add_argument("--lora-path", default=None, help="trained vllm: local LoRA adapter dir")
    p.add_argument("--tinker-weights", default=None, help="trained tinker: tinker:// sampler_weights path")
    p.add_argument("--out", required=True, help="output cache jsonl")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--concurrency", type=int, default=30, help="tinker only")
    p.add_argument("--max-prompts", type=int, default=None, help="smoke test cap")
    p.add_argument("--question-suffix", default="", help="optional elicitation string appended to the "
                   "freeform question (base-model control, e.g. '…explain the specific factors…')")
    args = p.parse_args()

    if args.arm == "trained":
        assert bool(args.lora_path) ^ bool(args.tinker_weights), \
            "trained: pass exactly one of --lora-path (vllm) / --tinker-weights (tinker)"
    else:
        assert not (args.lora_path or args.tinker_weights), "base arm takes no adapter"

    rows = load_eval_set(Path(args.eval_set))
    if args.max_prompts is not None:
        rows = rows[:args.max_prompts]
    print(f"[gen] {len(rows)} prompts | arm={args.arm} backend={args.backend} ckpt={args.checkpoint_name}")

    if args.backend == "tinker":
        gen = generate_tinker(rows, args.base_model, args.tinker_weights, args.concurrency,
                              args.max_new_tokens, args.question_suffix)
    else:
        gen = generate_vllm(rows, args.base_model, args.lora_path, args.max_new_tokens,
                            args.question_suffix)

    by_pid = {r["prompt_id"]: r for r in rows}
    gen = [g for g in gen if g["self_explanation"].strip()]
    report_skip_rate(len(rows) - len(gen), len(rows), "empty/failed generation")

    truncated = sum(1 for g in gen if g["finish_reason"] == "length")
    if truncated:
        rate = truncated / len(gen)
        print(f"[gen] *** {truncated}/{len(gen)} ({rate:.1%}) explanations hit max_tokens ***")
        if args.arm == "trained":
            assert rate <= 0.01, "trained truncation rate too high — raise --max-new-tokens"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for g in gen:
            src = by_pid[g["prompt_id"]]
            f.write(json.dumps({
                "prompt_id": g["prompt_id"],
                "claim_id": src["claim_id"],
                "dataset": src["dataset"],
                "cue": src["cue"],
                "is_true": src["is_true"],
                "effect_pp": src["effect_pp"],
                "hint_text": src["hint_text"],          # the verbatim injected cue (generic)
                "question": src["question"],
                "self_explanation_question": src["freeform_question"],
                "rendered_prompt": g["rendered_prompt"],
                "self_explanation": g["self_explanation"],
                "finish_reason": g["finish_reason"],
                "n_tokens": g["n_tokens"],
                "base_model": args.base_model,
                "arm": args.arm,
                "checkpoint": args.checkpoint_name,
            }) + "\n")
    meta = {
        "eval_set": args.eval_set, "base_model": args.base_model, "arm": args.arm,
        "checkpoint": args.checkpoint_name, "backend": args.backend,
        "lora_path": args.lora_path, "tinker_weights": args.tinker_weights,
        "n_prompts": len(rows), "n_generated": len(gen), "n_truncated": truncated,
        "temperature": TEMPERATURE, "max_new_tokens": args.max_new_tokens,
        "enable_thinking": ENABLE_THINKING, "question_suffix": args.question_suffix,
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[gen] wrote {len(gen)} -> {out_path}")


if __name__ == "__main__":
    main()
