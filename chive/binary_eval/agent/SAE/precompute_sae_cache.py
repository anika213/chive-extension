"""Precompute the SAE cache for a question bank (Gemma Scope 2 or batch-top-k SAEs).

SAE feature activations are (like NLA) claim-independent and per-token, so we compute
them once per investigation and cache them; the agent's SAE tool is then a pure dict
lookup joined with the Neuronpedia autointerp labels — no GPU / server at eval time
(see ../sae_tool.py).

Per question:
  - CAPTURE: HTTP to the existing AO gemma vllm-lens server — POST the chat-templated
    token sequence, request output_residual_stream=[40] (the SAE's hook layer,
    resid_post of block 40). Same capture path as ao_tool / NLA precompute, different
    layer. One cheap max_tokens=1 call per question.
  - ENCODE: one JumpReLU encode of all positions on the local GPU
    ([seq, 5376] @ [5376, 65536] in f32 — instant on GPU, minutes on CPU, hence the
    1-GPU job). Keep the top --top-k (default 60 = the SAE's trained L0) features per
    position, store (feature, activation) pairs; entries with activation 0 are dropped.

This cache covers EVERY position uniformly, BOS included — the SAE encode is free, and
chat-template tokens (<start_of_turn>, <end_of_turn>, role markers) often carry the most
legible summary features. SAEs produce garbage on the BOS token (its L0 blows up; it
encodes nothing useful), but we cache it like any other position rather than special-
casing the data; the query_sae tool tells the agent to ignore position 0.

Layer convention: the vllm-lens output_residual_stream index is asserted against the
SAE config's hf_hook_point_in layer, and was verified empirically with --verify-layers
(see SAE_ITERATION_LOG.md): the SAE's expected L0 / low FVU single out exactly one
capture index, so an off-by-one cannot pass the built-in sanity asserts below.

Integrity asserts per question: median L0 within [0.33x, 3x] of the SAE's trained L0,
FVU < 0.5 (both QC stats computed excluding BOS — it is a calibrated outlier the gate
would otherwise be dominated by; the cache still covers BOS — saved per row as median_l0 / fvu).
These catch GROSS miswiring (wrong model, wrong scale, far-wrong layer) — they can NOT
catch an off-by-one layer (adjacent-layer FVU differs by only ~0.01); that is prevented
by the capture_layer == SAE-config hook layer assert plus the one-time --verify-layers
empirical check. Measured on the canonical bank (n=1254): median_l0 mean 58.7 (trained
60), FVU mean 0.121 / p10 0.077 / p90 0.211 / max 0.391. For calibration: ~0.10 FVU is
the IN-distribution level for this -it SAE — the same pipeline on plain encyclopedic
prose (chat-formatted or not) measures FVU ~0.51, and per-feature firing frequencies on
our data correlate r=0.76 with the feature_frequencies Google ships in
examples.safetensors (their corpus L0: 50). So a question up near the 0.5 bound means
unusual content, not a pipeline bug.

Run as a 1-GPU Slurm job:
    sbatch chive/binary_eval/agent/SAE/jobs/precompute_sae_cache.sh \
        <run_dir>/binary_counterfactual_questions.jsonl http://node-XX:8414
    # -> <run_dir>/sae_cache/<sae_slug>.jsonl   (next to the bank; checkpoint-named)

--bank accepts ANY jsonl whose rows carry prompt_id/messages/chosen_completion (a full
claim bank or an eval set); rows are deduped by prompt_id (claims share their
investigation's transcript). Precompute on the FULL bank so one cache serves every
eval set derived from it.

Verification mode (one short GPU job, prints a table and exits — no cache written):
    ... precompute_sae_cache.py --bank ... --capture-url ... --verify-layers 39,40,41 --max-questions 3

batch-top-k mode (--sae-arch batch_topk; e.g. adamkarvonen/qwen3-8b-saes for Qwen3-8B):
  - loads via batch_topk_sae.py (ae.pt checkpoint dir), threshold-mode inference — per-
    position L0 varies around the trained k (not capped at it).
  - ATTENTION-SINK FILTER (experiment-semantic, matches the SAE's training): Qwen has
    random attention-sink positions hundreds of tokens in (~0.1% of positions, residual
    norms 100-1000x the median); the SAE was trained with them EXCLUDED, so encoded acts
    at positions with norm > 10x the per-question median are zeroed. Those rows are
    stored with an empty feature list plus their indices in "sink_positions", so the
    eval-time tool can render an explicit "attention sink — unreadable" marker instead
    of a silently-empty list. QC stats exclude sink positions (and position 0).
  - "neuronpedia_source" ("<np_model>/<np_source>") is recorded per row: the eval-time
    labels file is derived from it (the Gemma regex derivation doesn't apply here).
  - Qwen3 has NO BOS token — position 0 is the chat template's first token
    (<|im_start|>), itself an attention sink; the BOS asserts are skipped when the
    tokenizer has no BOS, and position 0 stays excluded from QC either way.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import torch
from transformers import AutoTokenizer

from chive.binary_eval.agent.SAE.batch_topk_sae import (
    BatchTopKSAE, load_batch_topk_sae)
from chive.binary_eval.agent.SAE.gemma_scope_sae import (
    GemmaScopeSAE, load_gemma_scope_sae)
from chive.binary_eval.agent.transcript_segments import segment_transcript
from chive.binary_eval.agent_tools import sae_slug


async def post_json(client, url, payload, retries=4):
    """POST with retries on transient connection resets (same as NLA precompute)."""
    for attempt in range(retries):
        try:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            return r.json()
        except httpx.TransportError:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(0.5 * (attempt + 1))
    raise RuntimeError("unreachable")


async def capture_residual(client, capture_url, base_model, token_ids, layers, d_model):
    """Capture residual stream at `layers` over token_ids -> [len(layers), seq, d]."""
    from vllm_lens._helpers._serialize import decode_activations
    body = await post_json(client, f"{capture_url}/v1/completions", {
        "model": base_model,
        "prompt": token_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "vllm_xargs": {"output_residual_stream": json.dumps(layers)},
    })
    residual = decode_activations(body)["residual_stream"]
    assert residual.shape[0] == len(layers), (
        f"expected {len(layers)} captured layer(s), got {tuple(residual.shape)} — is the "
        f"vllm-lens output_residual_stream JSON-decode patch applied + server restarted?"
    )
    assert residual.shape[1] == len(token_ids), (residual.shape, len(token_ids))
    assert residual.shape[2] == d_model, (residual.shape, d_model)
    return residual


def load_questions(bank_path: Path, max_questions: int | None) -> list[dict]:
    """One row per investigation: dedupe bank rows (many claims share a prompt_id)."""
    by_id: dict[str, dict] = {}
    for line in open(bank_path):
        d = json.loads(line)
        if d["prompt_id"] not in by_id:
            by_id[d["prompt_id"]] = {k: d[k] for k in ("prompt_id", "messages", "chosen_completion")}
    qs = list(by_id.values())
    return qs[:max_questions] if max_questions is not None else qs


def question_stats(sae: GemmaScopeSAE | BatchTopKSAE, acts: torch.Tensor,
                   resid_f32: torch.Tensor) -> tuple[float, float]:
    """(median L0, FVU) over the given (BOS-excluded) positions."""
    l0 = (acts > 0).sum(dim=-1)
    return float(l0.float().median()), sae.fvu(resid_f32)


async def verify_layers(args, sae, tokenizer, questions, device):
    """Encode several candidate capture layers per question; the right layer is the one
    whose median L0 matches the SAE's trained L0 with clearly-lowest FVU."""
    layers = [int(x) for x in args.verify_layers.split(",")]
    print(f"[verify] candidate capture layers {layers} on {len(questions)} question(s); "
          f"SAE trained l0={sae.expected_l0} at hook_layer={sae.hook_layer}")
    async with httpx.AsyncClient(timeout=600.0) as client:
        for q in questions:
            seg = segment_transcript(tokenizer, q["messages"], q["chosen_completion"])
            residual = await capture_residual(
                client, args.capture_url, args.base_model, seg.token_ids, layers, sae.d_in)
            print(f"[verify] {q['prompt_id']} ({len(seg.token_ids)} tokens, BOS excluded):")
            for i, layer in enumerate(layers):
                x = residual[i, 1:].to(device).float()
                med_l0, fvu = question_stats(sae, sae.encode(x), x)
                print(f"[verify]   layer {layer:3d}: median_l0={med_l0:7.1f}  fvu={fvu:.4f}")
    print("[verify] done — pick the layer with l0 ~= trained l0 and lowest fvu; "
          "no cache written.")


async def run(args):
    device = "cuda"
    assert torch.cuda.is_available(), "precompute needs a GPU for the encode matmul"
    if args.sae_arch == "batch_topk":
        sae = load_batch_topk_sae(args.sae_repo, args.sae_path, device)
        assert args.base_model == sae.lm_name, (
            f"--base-model {args.base_model} != SAE's training model {sae.lm_name}")
        assert args.neuronpedia_source, (
            "--neuronpedia-source is required for batch_topk (labels file derivation; "
            "e.g. qwen3-8b/18-resid-batchtopk-65k__l0-80)")
    else:
        sae = load_gemma_scope_sae(args.sae_repo, args.sae_path, device)
    top_k = args.top_k if args.top_k is not None else sae.expected_l0
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    questions = load_questions(Path(args.bank), args.max_questions)
    print(f"[precompute] {len(questions)} investigations from {args.bank}  "
          f"capture={args.capture_url} layer={sae.hook_layer}  top_k={top_k}", flush=True)

    if args.verify_layers:
        await verify_layers(args, sae, tokenizer, questions, device)
        return

    assert args.capture_layer == sae.hook_layer, (
        f"--capture-layer {args.capture_layer} != SAE hook layer {sae.hook_layer} "
        f"({args.sae_path}). The capture layer is pinned to the SAE; pass the matching one."
    )
    # Cache lands next to the bank, named after the SAE checkpoint (run identity = the
    # directory, checkpoint identity = the filename — see agent_tools.default_cache_path).
    sae_id = f"{args.sae_repo}/{args.sae_path}"
    out_path = (Path(args.out) if args.out
                else Path(args.bank).parent / "sae_cache" / f"{sae_slug(sae_id)}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if args.resume and out_path.exists():
        for line in open(out_path):
            d = json.loads(line)
            # A resume must continue the SAME cache: identical base model / SAE / top_k.
            existing = (d["base_model"], d["sae"], d["top_k"])
            current = (args.base_model, sae_id, top_k)
            assert existing == current, (
                f"--resume into {out_path} would mix settings: existing rows have "
                f"{existing}, this invocation has {current}."
            )
            done.add(d["prompt_id"])
        print(f"[precompute] --resume: {len(done)} already in {out_path}", flush=True)
    todo = [q for q in questions if q["prompt_id"] not in done]

    t0 = time.time()
    n_written = n_skipped = 0
    n_sink_total = n_pos_total = 0        # both exclude position 0 (always an outlier)
    l0s, fvus = [], []
    fout = open(out_path, "a" if args.resume else "w")
    async with httpx.AsyncClient(timeout=600.0) as client:
        for q in todo:
            try:
                seg = segment_transcript(tokenizer, q["messages"], q["chosen_completion"])
            except AssertionError as e:
                print(f"[precompute] {q['prompt_id']}: SKIPPED — segmentation failed: {e}", flush=True)
                n_skipped += 1
                continue
            if tokenizer.bos_token_id is not None:         # Qwen3 has no BOS token
                assert seg.token_ids[0] == tokenizer.bos_token_id, (
                    f"{q['prompt_id']}: token 0 is {seg.token_ids[0]}, not BOS")
                assert tokenizer.bos_token_id not in seg.token_ids[1:], (
                    f"{q['prompt_id']}: stray BOS inside sequence (double-BOS bug?)")

            residual = await capture_residual(
                client, args.capture_url, args.base_model, seg.token_ids, [sae.hook_layer], sae.d_in)
            x = residual[0].to(device).float()             # ALL positions, position 0 included
            acts = sae.encode(x)                           # [seq, d_sae]
            sink_positions: list[int] = []
            if args.sae_arch == "batch_topk":
                # Attention-sink filter (matches SAE training; see module docstring):
                # zero encoded acts where residual norm > 10x the per-question median.
                norm_mask = x.norm(dim=-1) > x.norm(dim=-1).median() * 10
                acts = acts * (~norm_mask)[:, None]
                sink_positions = norm_mask.nonzero().flatten().tolist()
            # QC stats exclude position 0 (BOS / first template token — a calibrated
            # L0/reconstruction outlier that would dominate the gate) and sink-masked
            # positions (their acts are zeroed by construction). The cache below still
            # covers every position — no token is special-cased out of the data.
            qc_idx = [i for i in range(1, len(seg.token_ids)) if i not in set(sink_positions)]
            med_l0, fvu = question_stats(sae, acts[qc_idx], x[qc_idx])
            assert sae.expected_l0 / 3 <= med_l0 <= sae.expected_l0 * 3, (
                f"{q['prompt_id']}: median L0 {med_l0:.0f} far from trained {sae.expected_l0} "
                f"— wrong capture layer / model / scale?")
            assert fvu < 0.5, f"{q['prompt_id']}: FVU {fvu:.3f} — SAE is not reconstructing"
            n_sink_total += len([p for p in sink_positions if p != 0])
            n_pos_total += len(seg.token_ids) - 1

            vals, idxs = acts.topk(top_k, dim=-1)          # [seq, top_k]
            vals, idxs = vals.cpu(), idxs.cpu()
            topk: dict[str, list] = {}
            for pos in range(len(seg.token_ids)):
                keep = int((vals[pos] > 0).sum())
                topk[str(pos)] = [[int(i), round(float(v), 2)]
                                  for i, v in zip(idxs[pos, :keep], vals[pos, :keep])]

            row = {
                "prompt_id": q["prompt_id"], "base_model": args.base_model,
                "capture_layer": sae.hook_layer,
                "sae": sae_id, "top_k": top_k,
                "n_tokens": len(seg.token_ids), "token_ids": seg.token_ids,
                "topk": topk, "median_l0": med_l0, "fvu": round(fvu, 4),
            }
            if args.sae_arch == "batch_topk":
                row["sink_positions"] = sink_positions
                row["neuronpedia_source"] = args.neuronpedia_source
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            n_written += 1
            l0s.append(med_l0); fvus.append(fvu)
            if n_written % 25 == 0 or n_written == len(todo):
                dt = max(time.time() - t0, 1e-9)
                print(f"[precompute] {n_written}/{len(todo)} q @ {n_written/dt:.2f} q/s | "
                      f"last={q['prompt_id']} l0={med_l0:.0f} fvu={fvu:.3f}", flush=True)
    fout.close()

    print("\n" + "=" * 80)
    print(f"[precompute] wrote {n_written} questions -> {out_path}  (skipped {n_skipped}, "
          f"resumed past {len(done)})")
    if l0s:
        l0s.sort(); fvus.sort()
        print(f"[precompute] median-of-medians L0={l0s[len(l0s)//2]:.0f}  "
              f"median FVU={fvus[len(fvus)//2]:.4f}  worst FVU={fvus[-1]:.4f}")
    if args.sae_arch == "batch_topk" and n_pos_total:
        rate = n_sink_total / n_pos_total
        print(f"[precompute] attention-sink positions (excl. position 0): "
              f"{n_sink_total}/{n_pos_total} = {rate:.4%}  (expected ~0.1%)")
        if not 0.0001 <= rate <= 0.01:
            print(f"*** WARNING: sink rate {rate:.4%} is far from the expected ~0.1% — "
                  f"investigate before using this cache ***")
    if n_skipped and n_skipped / max(n_written + n_skipped, 1) > 0.001:
        print(f"*** WARNING: {n_skipped} questions skipped (segmentation) — "
              f"{n_skipped/(n_written+n_skipped):.1%} ***")
    print("=" * 80)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", required=True,
                   help="jsonl with prompt_id/messages/chosen_completion rows "
                        "(full claim bank or eval set); deduped by prompt_id")
    p.add_argument("--capture-url", required=True,
                   help="AO gemma vllm-lens server, e.g. http://node-12:8414")
    p.add_argument("--base-model", required=True,
                   help="capture/base model — MUST match the model that produced the bank's "
                        "transcripts (recorded per row; asserted at eval time)")
    p.add_argument("--sae-repo", default="google/gemma-scope-2-27b-it")
    p.add_argument("--sae-path", default="resid_post/layer_40_width_65k_l0_medium",
                   help="SAE folder inside the repo; defines hook layer + trained L0")
    p.add_argument("--sae-arch", choices=["jump_relu", "batch_topk"], default="jump_relu",
                   help="jump_relu = Gemma Scope safetensors; batch_topk = "
                        "dictionary_learning ae.pt checkpoints (adds the attention-sink "
                        "filter — see module docstring)")
    p.add_argument("--neuronpedia-source", default=None,
                   help="'<np_model>/<np_source>' recorded per cache row; REQUIRED for "
                        "batch_topk (the eval derives the labels file from it), e.g. "
                        "qwen3-8b/18-resid-batchtopk-65k__l0-80")
    p.add_argument("--capture-layer", type=int, default=None,
                   help="REQUIRED for cache builds: must equal the SAE's hook layer "
                        "(explicit, never inferred — see AGENTS.md layer-selection rule)")
    p.add_argument("--top-k", type=int, default=None,
                   help="features kept per position (default: the SAE's trained L0/k)")
    p.add_argument("--out", default=None,
                   help="cache path (default <bank_dir>/sae_cache/<sae_slug>.jsonl)")
    p.add_argument("--resume", action="store_true",
                   help="append, skipping prompt_ids already in the output cache")
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--verify-layers", default=None,
                   help="comma-separated candidate capture layers; print L0/FVU per "
                        "layer and exit (no cache written)")
    args = p.parse_args()
    assert args.verify_layers or args.capture_layer is not None, (
        "--capture-layer is required for cache builds (or use --verify-layers)")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
