"""Score the binary-eval claims with a trained model in the STRUCTURED-CLAIM format (matches
the balance_and_export.py training format): the model reads the claim STATEMENT as a chat turn
and answers Yes/No, and we read the first-token Yes/No distribution to get a graded p_yes.

Prompt (identical shape to the training examples):
  [messages] + assistant: chosen_completion + user: statement + QUESTION_INSTRUCTION
rendered with the Qwen3 chat template at enable_thinking=False (first generated token = the answer).
The statement already encodes the prediction, so Yes = "the claim is true" directly
(prob_true = p_yes) — no question rephrasing and no prediction/polarity remap.

Two backends, one code path (--backend):
  vllm   — local LoRA on a single H200 (default, e.g. the Qwen3-8B ceiling). Reads the top-20
           logprobs at the first generated position directly.
  tinker — a Tinker-trained LoRA (e.g. the Qwen3.5-397B ceiling), served via the Tinker SDK.
           Tinker's prompt-logprobs are teacher-forced (logprob[i] = P(token_i | tokens<i),
           index 0 = None), so the first-token distribution lives at index len(prompt): we append
           ONE token to create that slot and read topk_prompt_logprobs[len(prompt)]. The appended
           token's value is irrelevant — that position is conditioned only on the prompt. This is
           the same mechanism as adamkarvonen/token-difr (append continuation, slice
           topk_prompt_logprobs[prompt_len:]). CPU-only orchestrator → run in tmux, no GPU/Slurm.

Both backends emit the same Yes/No top-k dict -> first_token_p_yes, and write rows that are
schema-compatible with run_binary_eval.BinaryResult (so a ceiling file drops straight next to the
opus binary_evals/ files and is comparable by claim_id).
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import asyncio
import collections
import json
import math
from pathlib import Path

from transformers import AutoTokenizer

from chive import paths
from chive.training.balance_and_export import QUESTION_INSTRUCTION
from chive.binary_eval.run_binary_eval import (
    BinaryResult,
    auroc,
    brier,
    print_report,
)

DEFAULT_EVAL_SET = str(
    paths.RUNS_DIR / "opus_investigates_qwen3_8b_5000_thorough_n30" / "eval_set_conc3_n2000.jsonl"
)
# Default base model (the tokenizer + chat template) per backend. The trained LoRA is a separate
# flag (--lora-path / --model-path); the base must match the model the bank's transcripts came from.
DEFAULT_BASE = {"vllm": "Qwen/Qwen3-8B", "tinker": "Qwen/Qwen3.5-397B-A17B"}


def yes_no_token_ids(tokenizer) -> dict[str, list[int]]:
    yes = ["yes", " yes", "Yes", " Yes", "YES", " YES"]
    no = ["no", " no", "No", " No", "NO", " NO"]
    # Each variant must be a single token, else taking pieces[0] silently corrupts p_yes
    # (tokenizer-dependent — matters for cross-family runs, e.g. Gemma/Llama).
    for v in yes + no:
        pieces = tokenizer.tokenize(v)
        assert len(pieces) == 1, f"{v!r} tokenizes to {len(pieces)} tokens: {pieces}"
    return {
        "yes": [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(v)[0]) for v in yes],
        "no": [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(v)[0]) for v in no],
    }


def first_token_p_yes(
    logprobs_dict: dict[int, float], ids: dict[str, list[int]]
) -> tuple[float, float | None, float | None]:
    """p_yes = softmax over (best yes-variant logprob, best no-variant logprob), reading a plain
    {token_id: logprob} top-k dict. Returns (p_yes, yes_logprob, no_logprob); 0.5 if neither is
    in the top-k."""
    yes_lp = max((logprobs_dict[t] for t in ids["yes"] if t in logprobs_dict), default=None)
    no_lp = max((logprobs_dict[t] for t in ids["no"] if t in logprobs_dict), default=None)
    if yes_lp is None and no_lp is None:
        return 0.5, None, None
    if yes_lp is None:
        return 0.0, None, no_lp
    if no_lp is None:
        return 1.0, yes_lp, None
    m = max(yes_lp, no_lp)
    p_yes = math.exp(yes_lp - m) / (math.exp(yes_lp - m) + math.exp(no_lp - m))
    return p_yes, yes_lp, no_lp


def template_requires_alternation(tokenizer) -> bool:
    """Capability probe: does the chat template reject consecutive same-role turns (Gemma does;
    Qwen/Llama don't)? Two-message probe render — the try/except is the introspection mechanism,
    not an error guard."""
    probe = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    try:
        tokenizer.apply_chat_template(probe, tokenize=True, add_generation_prompt=True)
        return False
    except Exception:
        return True


def roles_alternate(r: dict) -> bool:
    """True if messages (after an optional leading system turn) strictly alternate starting with
    user — the only structure a strict-alternation template (Gemma) can render."""
    roles = [m["role"] for m in r["messages"]]
    body = roles[1:] if roles[0] == "system" else roles
    return bool(body) and body[0] == "user" and all(a != b for a, b in zip(body, body[1:]))


def merge_consecutive_roles(r: dict) -> dict:
    """Concatenate consecutive same-role messages with a blank line, for strict-alternation
    templates (Gemma). Same convention as Gemma's own template, which folds the system turn into
    the first user turn with a blank-line join. Content is preserved; only the turn boundary is
    lost. Returns a shallow-copied row with merged messages."""
    merged: list[dict] = []
    for m in r["messages"]:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1] = {"role": m["role"], "content": merged[-1]["content"] + "\n\n" + m["content"]}
        else:
            merged.append(dict(m))
    return {**r, "messages": merged}


def build_conv(r: dict, answer_instruction: str) -> list[dict]:
    return (
        list(r["messages"])
        + [{"role": "assistant", "content": r["chosen_completion"]}]
        + [{"role": "user", "content": r["statement"] + QUESTION_INSTRUCTION + answer_instruction}]
    )


def build_prompt(tokenizer, r: dict, answer_instruction: str) -> str:
    return tokenizer.apply_chat_template(
        build_conv(r, answer_instruction), tokenize=False,
        add_generation_prompt=True, enable_thinking=False)


def build_prompt_ids(tokenizer, r: dict, answer_instruction: str) -> list[int]:
    return tokenizer.apply_chat_template(
        build_conv(r, answer_instruction), tokenize=True,
        add_generation_prompt=True, enable_thinking=False)


# --- per-claim score record: aligned to rows, fed into the shared BinaryResult builder -------
def _score(p_yes, yes_lp, no_lp, argmax_text, argmax_is_yesno, prompt):
    return {"p_yes": p_yes, "yes_lp": yes_lp, "no_lp": no_lp, "argmax_text": argmax_text,
            "argmax_is_yesno": argmax_is_yesno, "prompt": prompt}


def score_vllm(rows, base_model, lora_path, answer_instruction, max_model_len) -> list[dict]:
    import vllm

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    ids = yes_no_token_ids(tokenizer)
    yes_set, no_set = set(ids["yes"]), set(ids["no"])
    prompts = [build_prompt(tokenizer, r, answer_instruction) for r in rows]
    # Feed vLLM the chat-template token ids directly (exactly what training rendered) instead of
    # letting it re-tokenize the string — re-encoding a rendered string prepends a second BOS for
    # Gemma/Llama tokenizers. Token-identical for Qwen (its encode adds no special tokens).
    prompt_ids = [build_prompt_ids(tokenizer, r, answer_instruction) for r in rows]
    print("[vllm] sample rendered tail:\n" + repr(prompts[0][-300:]), flush=True)

    lora_request = None
    if lora_path is not None:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("structured_sft_lora", 1, lora_path=lora_path)
    llm = vllm.LLM(
        model=base_model, max_model_len=max_model_len, gpu_memory_utilization=0.90,
        enable_lora=lora_path is not None, max_lora_rank=64 if lora_path is not None else None,
    )
    sp = vllm.SamplingParams(temperature=0.0, max_tokens=10, logprobs=20)
    outputs = llm.generate([{"prompt_token_ids": pi} for pi in prompt_ids], sp,
                           lora_request=lora_request)
    del llm

    scored = []
    for prompt, output in zip(prompts, outputs):
        lp0 = {int(t): lp.logprob for t, lp in output.outputs[0].logprobs[0].items()}
        first_tok_id = output.outputs[0].token_ids[0]
        p_yes, yes_lp, no_lp = first_token_p_yes(lp0, ids)
        scored.append(_score(p_yes, yes_lp, no_lp, output.outputs[0].text.strip(),
                             first_tok_id in yes_set or first_tok_id in no_set, prompt))
    return scored


async def score_tinker(rows, base_model, model_path, answer_instruction, concurrency, topk) -> list[dict]:
    import tinker

    from chive.pipeline.utils import load_dotenv
    load_dotenv()
    project_id = os.environ.get("TINKER_PROJECT_ID")
    owner = os.environ.get("TINKER_OWNER")
    assert project_id and owner, "TINKER_PROJECT_ID / TINKER_OWNER required (see .env)"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    ids = yes_no_token_ids(tokenizer)
    yes_no_set = set(ids["yes"]) | set(ids["no"])
    pad_id = ids["yes"][0]  # arbitrary; only creates the slot at index len(prompt)
    print(f"[tinker] base={base_model} model_path={model_path} concurrency={concurrency}", flush=True)
    print("[tinker] sample rendered tail:\n"
          + repr(build_prompt(tokenizer, rows[0], answer_instruction)[-300:]), flush=True)

    service = tinker.ServiceClient(project_id=project_id, user_metadata={"owner": owner})
    # model_path=None -> sample the BASE model (no LoRA), matching the vllm base arm.
    sc = service.create_sampling_client(
        base_model=base_model if model_path is None else None, model_path=model_path)
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(r) -> dict:
        nonlocal done
        prompt_ids = build_prompt_ids(tokenizer, r, answer_instruction)
        prompt_text = build_prompt(tokenizer, r, answer_instruction)
        answer_pos = len(prompt_ids)
        full = prompt_ids + [pad_id]  # append one token -> topk slot at answer_pos
        async with sem:
            resp = await sc.sample_async(
                prompt=tinker.ModelInput.from_ints(full),
                num_samples=1,
                sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0.0),
                include_prompt_logprobs=True,
                topk_prompt_logprobs=topk,
            )
        row = resp.topk_prompt_logprobs[answer_pos]  # list[(token_id, logprob)] | None
        lp0 = {int(t): float(lp) for t, lp in row} if row else {}
        p_yes, yes_lp, no_lp = first_token_p_yes(lp0, ids)
        if lp0:
            best_id = max(lp0, key=lambda k: lp0[k])
            argmax_text, argmax_is_yesno = tokenizer.decode([best_id]).strip(), best_id in yes_no_set
        else:
            argmax_text, argmax_is_yesno = "", False
        done += 1
        if done % 100 == 0:
            print(f"[tinker] {done}/{len(rows)} scored", flush=True)
        return _score(p_yes, yes_lp, no_lp, argmax_text, argmax_is_yesno, prompt_text)

    return await asyncio.gather(*[one(r) for r in rows])  # order-preserving


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["vllm", "tinker"], default="vllm")
    p.add_argument("--eval-set", default=DEFAULT_EVAL_SET)
    p.add_argument("--base-model", default=None, help="tokenizer/chat-template base (default per backend)")
    p.add_argument("--lora-path", default=None, help="vllm: local LoRA adapter dir (omit for base model)")
    p.add_argument("--model-path", default=None, help="tinker: tinker:// sampler-weights path of the LoRA")
    p.add_argument("--out", required=True)
    p.add_argument("--max-claims", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=16384, help="vllm only")
    p.add_argument("--concurrency", type=int, default=32, help="tinker only")
    p.add_argument("--topk", type=int, default=20, help="tinker only: top-k prompt logprobs to request")
    p.add_argument("--answer-instruction", default="",
                   help="appended to each statement turn. The LoRA answers Yes/No directly (none "
                        "needed); a BASE model otherwise replies in prose, so set e.g. ' Answer Yes or No.'")
    args = p.parse_args()
    base_model = args.base_model or DEFAULT_BASE[args.backend]
    # Both backends support a base arm: vllm omits --lora-path, tinker omits --model-path.
    ckpt = (args.model_path if args.backend == "tinker" else args.lora_path) or "base"

    rows = [json.loads(line) for line in open(args.eval_set)]
    if args.max_claims is not None:
        rows = rows[: args.max_claims]
    # Strict-alternation templates (Gemma) cannot render claims whose transcript has consecutive
    # same-role turns (100/1442 in the qwen8b petri bank). For such families only, MERGE
    # consecutive same-role messages with a blank-line join (Gemma's own system-turn convention).
    # Content preserved, turn boundary lost; affected claim_ids are logged so the analysis can
    # run a strict-alternating sensitivity slice.
    if template_requires_alternation(AutoTokenizer.from_pretrained(base_model)):
        affected = [r["claim_id"] for r in rows if not roles_alternate(r)]
        if affected:
            print(f"\n*** NOTE: {base_model} chat template enforces strict role alternation; "
                  f"MERGING consecutive same-role turns on {len(affected)}/{len(rows)} claims "
                  f"(first 5: {affected[:5]}) ***\n", flush=True)
            rows = [r if roles_alternate(r) else merge_consecutive_roles(r) for r in rows]
    arm = f"structured_claim_{args.backend}"
    print(f"[structured] arm={arm}  backend={args.backend}  {len(rows)} claims  base={base_model}  "
          f"ckpt={ckpt}  answer_instruction={args.answer_instruction!r}", flush=True)

    if args.backend == "vllm":
        scored = score_vllm(rows, base_model, args.lora_path, args.answer_instruction, args.max_model_len)
    else:
        scored = asyncio.run(score_tinker(
            rows, base_model, args.model_path, args.answer_instruction, args.concurrency, args.topk))

    results: list[BinaryResult] = []
    n_missing = n_top1_yesno = 0
    argmax_tokens: collections.Counter = collections.Counter()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r, s in zip(rows, scored):
            if s["argmax_is_yesno"]:
                n_top1_yesno += 1
            argmax_tokens[s["argmax_text"][:12]] += 1
            if s["yes_lp"] is None and s["no_lp"] is None:
                n_missing += 1
            prob_true = s["p_yes"]  # statement encodes the prediction -> Yes = claim is true
            pred = prob_true >= 0.5
            res = BinaryResult(
                claim_id=r["claim_id"], prompt_id=r["prompt_id"], claim_type=r["claim_type"],
                is_true=r["is_true"], prob_true=prob_true, pred_true=pred,
                correct=(pred == r["is_true"]),
                reasoning=f"[{arm}] p_yes={prob_true:.3f} (yes_lp={s['yes_lp']}, no_lp={s['no_lp']}); "
                          f"argmax={s['argmax_text']!r}",
                answerer=arm, answerer_model=base_model, effort=ckpt, prompt=s["prompt"],
            )
            results.append(res)
            f.write(json.dumps({**res.__dict__, "p_yes": s["p_yes"], "yes_logprob": s["yes_lp"],
                                "no_logprob": s["no_lp"], "argmax_token": s["argmax_text"]}) + "\n")
    print(f"[structured] wrote {len(results)} -> {out_path}")

    n = len(results)
    print(f"\n[diag] arm={arm}")
    print(f"[diag] argmax first token IS yes/no: {n_top1_yesno}/{n} ({n_top1_yesno/n:.1%})")
    print(f"[diag] yes/no present in top-{args.topk if args.backend=='tinker' else 20}: "
          f"{n - n_missing}/{n} ({(n - n_missing)/n:.1%})")
    print(f"[diag] top first-token strings: {argmax_tokens.most_common(12)}")
    if n_missing:
        print(f"\n*** WARNING: {n_missing}/{n} claims had NO yes/no token in top-k (scored p_yes=0.5). ***")
    if n_top1_yesno / n < 0.9:
        print(f"\n*** WARNING: only {n_top1_yesno/n:.0%} of argmax first tokens are yes/no. ***")

    print_report(results, Path(args.eval_set).name)
    labels = [r.is_true for r in results]
    scores = [r.prob_true for r in results]
    auc = auroc(labels, scores)
    print(f"\n[structured] arm={arm}  acc={sum(r.correct for r in results)/len(results):.1%}  "
          f"AUROC={auc:.3f}  Brier={brier(labels, scores):.3f}"
          if auc is not None else f"[structured] arm={arm}  AUROC n/a")


if __name__ == "__main__":
    main()
