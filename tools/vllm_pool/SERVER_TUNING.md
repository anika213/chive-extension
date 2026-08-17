# Per-model vLLM tuning notes

What we've learned running each model in the pool. Update as new lessons land.

For the proxy/registry mechanics, see `README.md`.

---

## Qwen models — always smoke-test reasoning before adding to pool

Qwen-3.5 / 3.6 reasoning paths can fail SILENTLY. The vllm server starts fine, `/v1/models` returns 200, but a `thinking_token_budget` request 400s with:

```
thinking_token_budget is set but reasoning_config is not configured.
Please set --reasoning-config to use thinking_token_budget.
```

The cure is `--reasoning-config '{}'` in the launcher. We've shipped servers without it more than once. Mitigation: **always** end the spinup with the reasoning smoke test (`README.md` § Step 6 "Smoke test reasoning ALWAYS") and only add to the registry when `choices[0].message.reasoning` comes back non-empty.

---

## Qwen3.5-397B-A17B-FP8

- **Tensor parallel:** 4 (TP=4 on a 4× H200 slice).
- **Per-backend cap:** **110**. Sweep at cap 100/120/140 on 3 backends showed identical inv/min throughput, so the cap is not the bottleneck once it's roughly right. Above ~140 with concurrent agentic load you start preempting; below ~80 you start under-utilizing.
- **Driver concurrency:** at `n_backends ≥ 6`, bump `concurrency` in the run's `canonical_config.json` from 800 → 1500. This leaves enough proxy-queue depth that preempt dips don't 503 the driver.
- **Per-backend gen tok/s observed:** 1,500–2,200 (varies by RunPod vs Slurm; cross-cluster RunPod is lower). 7-backend pool sustains ~12,000–13,000 gen tok/s total, ~25 inv/min at Stage 3 (10-turn agentic, ~33k gen tokens/inv).

**Semantic flags (must match across all members or `config_hash` warns):**
```
--served-model-name Qwen/Qwen3.5-397B-A17B-FP8
--tensor-parallel-size 4
--language-model-only
--reasoning-parser qwen3
--reasoning-config '{}'              # <-- without this, thinking_token_budget 400s
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
--max-model-len 81920
--max-num-seqs 1024
--max-num-batched-tokens 8192
```

**Reliability flags (perf/correctness workarounds — not in config_hash):**
```
--gpu-memory-utilization 0.90
--enable-prefix-caching
--gdn-prefill-backend triton
--no-enable-flashinfer-autotune
--disable-custom-all-reduce
--max-num-partial-prefills 1
--max-long-partial-prefills 1
--long-prefill-token-threshold 8192
--compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'   # required on RunPod stack; suppresses broken TRTLLM fused-allreduce compile path
```

**Env vars (must export before `vllm serve`):**
```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_DEEP_GEMM=0          # DeepGEMM JIT fails on cu129 torch + CUDA 12.8 — kills EngineCore
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP8=0 # FlashInfer FP8 MoE produced garbage output for Qwen3.5 FP8 on H200
# Slurm-only:
export NCCL_SOCKET_IFNAME=vxlan0
```

**Load time:** 397B weights ~378GB. Off Slurm NFS, expect ~3-15 min checkpoint load + ~5-10 min vLLM compile. RunPod from local `/workspace`-cached weights is faster (~10-15 min total cold, ~5 min with vLLM compile cache warm).

---

## Qwen3.6-35B-A3B-FP8

- **Tensor parallel:** 1 (single H200).
- **Per-backend cap:** **160** (with `--adaptive-cap --adaptive-cap-max 320`). The pilot ran cleanly at cap 256–384 on a 2-backend pool; for high-fanout 8+-backend pools, starting at 160 with adaptive headroom to 320 is the safe default.
- **Driver concurrency:** ~`n_backends × cap × 2`. For an 8-backend pool, 3000 is sized to keep the adaptive cap as the binding constraint, not the driver.
- **Per-backend gen tok/s observed (pilot):** ~10,800 at cap 256 short-output. ~21,500 cluster total at proxy concurrency 512 across 2 backends.

**Semantic flags:** same family as 397B (uses the same reasoning parser + reasoning config; you should also run the reasoning smoke test):
```
--served-model-name Qwen/Qwen3.6-35B-A3B-FP8
--language-model-only
--reasoning-parser qwen3
--reasoning-config '{}'              # same silent-fail trap as 397B if missing
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
--max-model-len 81920
```

**Reliability flags:** same family as 397B:
```
--gpu-memory-utilization 0.90
--enable-prefix-caching
--gdn-prefill-backend triton
--no-enable-flashinfer-autotune
--disable-custom-all-reduce
```

**Env vars:** same as 397B — `VLLM_USE_DEEP_GEMM=0`, `VLLM_USE_FLASHINFER_MOE_FP8=0`, etc. The Qwen3.x FP8 stack has the same FlashInfer/DeepGEMM landmines regardless of model size.

---

## Gemma-3-27B-IT (`google/gemma-3-27b-it`)

- **Tensor parallel:** 1 (single H200).
- **Per-backend cap:** **160** with `--adaptive-cap --adaptive-cap-max 320`.
- **Driver concurrency:** ~`n_backends × cap × 2`.

**Differences from Qwen:**
- Gemma 3 is multimodal — `--language-model-only` strips the vision tower so the server starts as a pure text LM (smaller mem, faster startup).
- **No** `--tool-call-parser`, `--reasoning-parser`, `--reasoning-config` — Gemma uses neither Qwen's tool nor reasoning protocols. The reasoning smoke test does not apply.
- `--max-model-len 4000` for the opus-investigates workload (prompts ≤ ~1.5k tokens, max_tokens=500). Bump if context grows.

**Semantic flags:**
```
--served-model-name google/gemma-3-27b-it
--language-model-only
--max-model-len 4000
```

**Reliability flags:**
```
--gpu-memory-utilization 0.90
--enable-prefix-caching
--disable-custom-all-reduce
```

---

## Generic operating envelope

| Signal | Meaning | Action |
|---|---|---|
| `kv_cache_usage_avg > 0.95` sustained | Pool oversubscribed | Lower `--max-seqs-per-upstream` 10-20% |
| `preemptions_per_sec > 1` sustained | KV thrash | Lower cap; check for orphan requests from a prior driver |
| `n_routable < n_upstreams` for >10 min | Real degradation | Check `/pool` per-backend `last_error`; clear blacklist (`README.md` § Blacklist) or restart the wedged backend |
| `n_routable` flapping 6-7/7 every few min | Health-check timeout on a slow/nginx-fronted backend | Cosmetic. Throughput unaffected. Bump proxy `--health-timeout 30` if it bothers you (requires proxy restart) |
| Driver `waiting_acquires` >> 0 sustained | Driver is feeding past pool cap | Working as intended (queue absorbs preempt dips). Worry if `n_routable` is also low |
| `fallback_route_rate > 0.7` | Affinity broken | Usually means pool is at cap; if pool is NOT at cap, look for one backend hogging affinity-hashed prompts |
| Slow startup on 397B (>20 min) | Cold checkpoint load | Normal for first cold load off NFS. Subsequent restarts hit the vLLM compile cache and are 5-10 min |
