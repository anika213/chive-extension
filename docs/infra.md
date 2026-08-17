# Infrastructure guide

How this repo's compute is organized, what assumes our cluster, and what you need to
change to run it on yours.

## Execution model

Everything follows one pattern:

1. **GPU work runs as Slurm batch jobs.** vLLM servers, AO activation capture, NLA/SAE
   precomputes, and LoRA training are all submitted with `sbatch`. There is no
   assumption that the machine you type on has a GPU.
2. **Drivers are CPU-side HTTP/API clients.** The pipeline (`chive.pipeline.run_pipeline`),
   the binary eval (`chive.binary_eval.run_binary_eval`), and the experiment scripts run on
   a login node or laptop and talk to (a) vLLM servers over their OpenAI-compatible HTTP
   API and (b) the Anthropic / OpenRouter APIs. The analysis scripts under
   `experiments/*/analyze.py` are pure CPU over downloaded data — no servers at all.
3. **Servers advertise themselves through small JSON files.** Slurm serve jobs write a
   server-info JSON when the server is healthy: the AO/CF serve jobs write
   `data/scratch/{ao_server*,cf_server}.json` (read by `chive/binary_eval/agent_tools.py`),
   and the pool backends write per-job registry JSONs (read by
   `tools/vllm_pool/pool_proxy.py`). Drivers can also be pointed at a URL directly
   (`--url http://node:8000`).

So to reproduce anything model-facing you need: any OpenAI-compatible vLLM endpoint for
the target/base models, plus API keys for the investigator/judge models. Nothing in the
driver code cares whether the endpoint is a Slurm job, a RunPod pod, or localhost.

## LLM backend specs

Every model-facing stage takes an `llm_spec` string, resolved by
`chive/pipeline/llm_client.py::make_llm_client`:

| spec | backend |
|---|---|
| `anthropic:claude-opus-4-6` | Anthropic API (thinking enabled; used for investigator/judge/verifier) |
| `openrouter:qwen/qwen3.5-397b-a17b` | OpenRouter (fp8 provider routing auto-applied for Qwen; fine for iteration, use vLLM for canonical runs) |
| `vllm:Qwen/Qwen3-8B@http://node:8000` | any OpenAI-compatible server (vLLM, or the pool proxy below) |
| `openai:MODEL@https://host/v1` | generic OpenAI-compatible endpoint with `OPENAI_API_KEY` |
| `tinker:BASE_MODEL[@tinker://LORA_PATH]` | Tinker sampling API (397B LoRA inference) |

Pipeline configs (`config.json` in a run dir) set one spec per stage
(`stage1_llm`, `judge_llm`, `stage3_llm`, `target_llm`, `stage5_llm`), so any stage can
be re-pointed at any backend independently.

Credentials come from `.env` at the repo root (loaded by `load_dotenv()`):
`ANTHROPIC_API_KEY` (optionally `ANTHROPIC_API_KEY_HIGH_PRIORITY` /
`ANTHROPIC_API_KEY_BATCH_API`), `OPENROUTER_API_KEY`, `OPENAI_API_KEY` (only for
`openai:` specs), `HF_TOKEN`, and `TINKER_*` / `WANDB_API_KEY` for training.

## Serving models: single server vs. pool

- **Single server (default, sufficient for everything):**
  `chive/pipeline/slurm_scripts/submit_vllm_server*.sh` launch one vLLM server per model;
  `chive/binary_eval/agent/jobs/serve_*.sh` launch the AO/CF servers (base model + LoRAs)
  and write their server-info JSONs to `data/scratch/`. Point drivers at
  `vllm:MODEL@http://<node>:<port>`.
- **Pool (throughput for the big runs):** `tools/vllm_pool/` puts N independent vLLM
  servers behind one stable endpoint with affinity routing, health checks, and drain —
  this is how the paper's 5000-prompt canonical runs and the 397B arms were generated.
  See `tools/vllm_pool/README.md`.

Rough scale, for budgeting: a 5000-prompt canonical run keeps ~5–8 single-GPU (H200)
vLLM backends busy for on the order of a day, and the Anthropic-side investigator
(Opus, ~10–15 counterfactual experiments per investigated prompt, each rerun ~30× on
the target server) is a co-equal bottleneck — beyond ~8 GPUs the API side dominates.
The pipeline smoke config (100 prompts, n=10, 5 investigations) runs end-to-end in
~15 min on 1 GPU. Analysis-only reproduction (the `experiments/*/analyze.py` quickstart)
needs no GPU and no API keys.

## What is cluster-specific

Every Slurm script is marked `# cluster-specific (Slurm); adapt paths/partitions to your
setup.` Concretely, ours assume:

- Partitions `general`/`overflow`, QoS `high`/`low`, 8× H200 nodes; NCCL env
  (`NCCL_SOCKET_IFNAME=vxlan0`, `NCCL_NVLS_ENABLE=0`) required on our fabric.
- `scripts/gpu_preflight.sh` sourced at job start (fails fast if an assigned GPU is
  occupied by a zombie process) and `jobs/find_zombie_gpu_processes.sh --exclude` before
  submission. Harmless to delete on a healthy cluster.
- `#SBATCH --output` paths are repo-relative (e.g. `tools/vllm_pool/logs/`,
  `chive/pipeline/logs/`); the log dirs are gitignored, so `mkdir -p` them before the first
  `sbatch`, and submit from the repo root.
- Caches live off node-local disk: `HF_HOME=/workspace-vast/$USER/hf_cache` — change to
  wherever your shared storage is.

None of the Python code depends on Slurm; if you have GPUs on the local machine you can
run the same commands the job scripts wrap.

## Installs and data

- Extras map to workflow: base install = everything CPU-only — analysis/plots, pipeline,
  claim building, eval arms, pool proxy;
  `.[gpu]` = everything hardware-bound — vLLM serving, AO capture, NLA/SAE precompute
  (needs Python ≥3.12 for the pinned `vllm-lens` fork), and text SFT (local LoRA + Tinker).
  `uv sync --extra gpu` with the committed `uv.lock`
  reproduces the environment exactly; base deps (`numpy`/`scipy`/`matplotlib`) are pinned
  because figure byte-reproducibility depends on them.
- All data paths resolve through `chive/paths.py`: `data/` by default, overridable with
  `CHIVE_DATA`. Materialize it with `python scripts/download_data.py` (HuggingFace dataset
  repo → `data/`; supports `--only` filters so you don't need all ~93 GB for one
  experiment).
- Exception: Qwen3.5-397B serving needs a newer vLLM (0.21.x) than the repo lock — use a
  dedicated venv for those servers (see `tools/vllm_pool/submit_qwen3.5_397b.sh`).
