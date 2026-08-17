"""Precompute the NLA cache for a claim bank — ONLINE capture + OFFLINE AV decode.

NLA is UNCONDITIONAL and PER-TOKEN: a token's verbalization doesn't depend on the
claim, so we compute it once for every addressable token in every transcript and
cache the text. The agent's NLA tool is then a pure dict lookup — no GPU/server in
the hot path (see ../nla_tool.py).

This is the production consumer of the shared inference module (nla_inference.py):
  - CAPTURE is ONLINE: HTTP to one shared vllm-lens base server (capture_residual).
    One cheap forward pass per transcript yields every position's activation, so a
    single capture server feeds many decode workers.
  - AV DECODE is OFFLINE: each worker owns an AVDecoder (vllm.LLM, prompt_embeds,
    async_scheduling=False) and decodes its shard. This is the fan-out — run several
    decode workers (Slurm array) against the one capture server.

--spec names a checkpoint in the nla_inference NLASpec registry (base model, AV repo,
subject extraction layer — the single source of truth). --bank is ANY jsonl whose rows
carry prompt_id/messages/chosen_completion (a full claim bank or an eval set); rows are
deduped by prompt_id. The cache lands NEXT TO the bank, named after the spec (run
identity = the directory, checkpoint identity = the filename):

    <bank_dir>/nla_cache/<spec.name>.jsonl

Parallelism: --num-shards N --shard-idx I (one Slurm array task per shard; see
jobs/precompute_nla_array.sh). Shard outputs land in <bank_dir>/nla_cache/shards/;
cat them into <bank_dir>/nla_cache/<spec.name>.jsonl afterwards.

EVERY token position is verbalized (content + chat-template scaffolding) — full
coverage is the cache invariant nla_tool.load_nla_cache enforces. Decoding is SAMPLED
at temperature 1.0 (the in-distribution training setting); decode params are recorded
per row.

    sbatch chive/binary_eval/agent/NLA/jobs/precompute_nla_array.sh \
        <run>/binary_counterfactual_questions.jsonl http://node-XX:PORT qwen3-8b-baseline-nosalvage
"""

from __future__ import annotations

import os
# Required before any torch/vllm import — see AGENTS.md "vLLM offline" note.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer

from chive.binary_eval.agent.NLA.nla_inference import (
    AV_MAX_NEW_TOKENS,
    SPECS,
    AVDecoder,
    capture_residual,
)
from chive.binary_eval.agent.transcript_segments import segment_transcript


def load_questions(bank_path: Path, num_shards: int, shard_idx: int,
                   max_questions: int | None) -> list[dict]:
    """One row per investigation: dedupe bank rows by prompt_id (claims share their
    investigation's transcript), then take this shard (index mod num_shards over the
    deduped order — interleaves long and short transcripts across shards)."""
    by_id: dict[str, dict] = {}
    for line in open(bank_path):
        d = json.loads(line)
        if d["prompt_id"] not in by_id:
            by_id[d["prompt_id"]] = {k: d[k] for k in ("prompt_id", "messages", "chosen_completion")}
    qs = [q for i, q in enumerate(by_id.values()) if i % num_shards == shard_idx]
    return qs[:max_questions] if max_questions is not None else qs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, choices=sorted(SPECS),
                   help="checkpoint name from the NLASpec registry in nla_inference "
                        "(provides base model, AV repo, subject extraction layer)")
    p.add_argument("--bank", required=True,
                   help="jsonl with prompt_id/messages/chosen_completion rows (full claim bank "
                        "or eval set); deduped by prompt_id.")
    p.add_argument("--capture-url", required=True,
                   help="shared vllm-lens base server for capture, e.g. http://node-XX:PORT")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="AV decode temperature; 1.0 (sampled) is the in-distribution setting.")
    p.add_argument("--max-tokens", type=int, default=AV_MAX_NEW_TOKENS,
                   help="AV completion budget (NLA-trained length, 256)")
    p.add_argument("--max-model-len", type=int, default=2048,
                   help="offline AV engine window (AV prompt ~104 tok + budget — NOT transcript len)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--capture-max-len", type=int, default=8192,
                   help="must match the capture server's --max-model-len; transcripts longer "
                        "than this can't be captured and are skipped + counted")
    p.add_argument("--num-shards", type=int, default=1,
                   help="split the deduped investigations across N parallel decode workers")
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="single-job only (num_shards=1): skip prompt_ids already in the output "
                        "cache and APPEND the rest, so a top-up over a new eval set computes only "
                        "the delta. Refused if the existing cache's base_model/av_repo/"
                        "capture_layer don't match this invocation.")
    args = p.parse_args()
    assert 0 <= args.shard_idx < args.num_shards, (args.shard_idx, args.num_shards)
    spec = SPECS[args.spec]

    bank = load_questions(Path(args.bank), args.num_shards, args.shard_idx, args.max_questions)

    # Cache lands next to the bank, named after the spec (run identity = directory,
    # checkpoint identity = filename). Shards go to shards/ so the default cache lookup
    # never sees a partial cache.
    cache_dir = Path(args.bank).parent / "nla_cache"
    if args.num_shards == 1:
        out_path = cache_dir / f"{spec.name}.jsonl"
    else:
        out_path = cache_dir / "shards" / f"{spec.name}_shard{args.shard_idx}of{args.num_shards}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --resume: single-job append. Skip prompt_ids already present (continuing the SAME cache —
    # settings must match), so a top-up over a new eval set computes only the delta. Sharded
    # resume is intentionally unsupported: shard files partition a different bank and the
    # cat-merge would clobber/duplicate — for cold full builds, shard then merge instead.
    done: set[str] = set()
    if args.resume:
        assert args.num_shards == 1, (
            "--resume is single-job only (num_shards=1); for cold full builds, shard then cat-merge")
        if out_path.exists():
            for line in open(out_path):
                d = json.loads(line)
                existing = (d["base_model"], d["av_repo"], d["capture_layer"])
                current = (spec.base_model, spec.av_repo, spec.extraction_layer)
                assert existing == current, (
                    f"--resume into {out_path} would mix settings: existing rows have "
                    f"{existing}, this invocation has {current}.")
                done.add(d["prompt_id"])
            print(f"[precompute] --resume: {len(done)} already in {out_path}", flush=True)
    todo = [q for q in bank if q["prompt_id"] not in done]

    print(f"[precompute] spec={spec.name} base={spec.base_model} layer={spec.extraction_layer} "
          f"{len(todo)} to compute ({len(bank) - len(todo)} already cached) "
          f"(shard {args.shard_idx}/{args.num_shards}) from {args.bank} "
          f"capture={args.capture_url} temp={args.temperature} max_tokens={args.max_tokens}", flush=True)
    if not todo:
        print("[precompute] nothing to compute — exiting", flush=True)
        return

    decoder = AVDecoder(spec.av_repo, extraction_layer=spec.extraction_layer,
                        temperature=args.temperature, max_tokens=args.max_tokens,
                        max_model_len=args.max_model_len,
                        gpu_memory_utilization=args.gpu_memory_utilization)
    tokenizer = AutoTokenizer.from_pretrained(spec.base_model)

    t0 = time.time()
    n_written = n_skip_seg = n_skip_long = tot_decodes = tot_trunc = tot_missing = 0
    fout = open(out_path, "a" if args.resume else "w")
    for q in todo:
        try:
            seg = segment_transcript(tokenizer, q["messages"], q["chosen_completion"])
        except AssertionError as e:
            print(f"[precompute] {q['prompt_id']}: SKIPPED — segmentation failed: {e}", flush=True)
            n_skip_seg += 1
            continue
        if len(seg.token_ids) > args.capture_max_len:
            print(f"[precompute] {q['prompt_id']}: SKIPPED — {len(seg.token_ids)} tokens > "
                  f"capture-max-len {args.capture_max_len}", flush=True)
            n_skip_long += 1
            continue

        captured = capture_residual(args.capture_url, spec.base_model, seg.token_ids,
                                    decoder.extraction_layer, decoder.d_model)
        results = decoder.decode([captured[p] for p in range(len(seg.token_ids))])
        assert len(results) == len(seg.token_ids), (len(results), len(seg.token_ids))

        explanations = {str(p): r.explanation for p, r in enumerate(results)}
        n_trunc = sum(r.finish_reason == "length" for r in results)
        n_missing = sum(r.missing_tag for r in results)
        fout.write(json.dumps({
            "prompt_id": q["prompt_id"], "base_model": spec.base_model, "av_repo": spec.av_repo,
            "capture_layer": decoder.extraction_layer,
            "temperature": args.temperature, "max_tokens": args.max_tokens,
            "n_tokens": len(seg.token_ids), "token_ids": seg.token_ids,
            "explanations": explanations,
            "n_truncated": n_trunc, "n_missing_tags": n_missing,
        }) + "\n")
        fout.flush()   # incremental: a long run survives a crash
        n_written += 1
        tot_decodes += len(results); tot_trunc += n_trunc; tot_missing += n_missing
        dt = max(time.time() - t0, 1e-9)
        print(f"[precompute] {q['prompt_id']}: {len(results)} pos | cum {n_written} q, "
              f"{tot_decodes} decodes @ {tot_decodes/dt:.1f}/s | trunc {tot_trunc} miss {tot_missing}",
              flush=True)
    fout.close()

    n_skipped = n_skip_seg + n_skip_long
    print("\n" + "=" * 80)
    print(f"[precompute] wrote {n_written} questions -> {out_path}  "
          f"(skipped {n_skipped}: {n_skip_seg} segmentation, {n_skip_long} too-long)")
    print(f"[precompute] decodes={tot_decodes}  wall={time.time()-t0:.0f}s  -> "
          f"{tot_decodes/max(time.time()-t0,1e-9):.1f} decodes/s")
    sr = n_skipped / max(n_written + n_skipped, 1)
    if sr > 0.001:
        print(f"*** WARNING: {n_skipped}/{n_written + n_skipped} investigations skipped "
              f"({sr:.2%}) — these will be missing from the cache. ***")
    tr = tot_trunc / tot_decodes if tot_decodes else 0.0
    mr = tot_missing / tot_decodes if tot_decodes else 0.0
    for name, k, rate in (("truncated (hit max_tokens)", tot_trunc, tr),
                          ("missing <explanation> tags", tot_missing, mr)):
        if rate > 0.001:
            print(f"*** WARNING: {name}: {k}/{tot_decodes} ({rate:.2%}) — degenerate decodes "
                  f"are cached unflagged; check async_scheduling=False / injection scale / budget. ***")
        else:
            print(f"[precompute] {name}: {k}/{tot_decodes} ({rate:.2%})")
    if args.num_shards > 1:
        print(f"[precompute] shard {args.shard_idx}/{args.num_shards} done. When ALL shards "
              f"finish:\n  cat {out_path.parent}/{spec.name}_shard*of{args.num_shards}.jsonl "
              f"> {cache_dir}/{spec.name}.jsonl")
    print("=" * 80)


if __name__ == "__main__":
    main()
