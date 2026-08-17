#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=vlens_qwen_ao
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --qos=high
#SBATCH --output=/workspace-vast/adamk/chive/chive/binary_eval/agent/logs/%x_%j.out

# Persistent vllm-lens AO server for the Qwen3-8B activation oracles (Phase B eval).
# Mirror of serve_gemma_ao.sh but serves Qwen/Qwen3-8B + BOTH its AO verbalizer LoRAs
# (single-layer and multi-layer) side by side, so eval runs select the checkpoint via
# --ao-lora without a server restart. The vllm-lens plugin auto-registers, so the
# OpenAI HTTP endpoints carry activations out and steering vectors in.
#
#   - capture: POST /v1/completions  model=Qwen/Qwen3-8B  (base, no LoRA)
#              vllm_xargs={"output_residual_stream": "[9, 18, 27]"}  # JSON list of layers
#   - query:   POST /v1/completions  model=<lora module name>  (AO verbalizer LoRA)
#              vllm_xargs={"apply_steering_vectors": json([sv.model_dump()])}
#
# LoRA module names are derived from the HF repo: basename minus the "checkpoints_"
# prefix (must match ao_tool.ao_lora_name). Capture layers are NOT pinned here — each
# eval derives them from its LoRA's ao_config.json (resolve_capture_layers):
#   - latentqa_cls_past_lens_addition_Qwen3-8B: single-layer, combos [[9],[18],[27]] -> 18
#   - 500k_pl_31k_spqav2_199k_sqav3_50k_hb_126k_cls: multi-layer, combos [[9,18,27]] -> all 3

set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate
source scripts/gpu_preflight.sh

export HF_HOME=/workspace-vast/adamk/hf_cache
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PORT=8413   # 8414 is the gemma AO server; use a distinct port
BASE_MODEL=Qwen/Qwen3-8B
AO_LORAS=(
  adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B
  adamkarvonen/checkpoints_500k_pl_31k_spqav2_199k_sqav3_50k_hb_126k_cls
)
LORA_MODULE_ARGS=()
for repo in "${AO_LORAS[@]}"; do
  name=${repo##*/}; name=${name#checkpoints_}   # ao_tool.ao_lora_name
  LORA_MODULE_ARGS+=("${name}=${repo}")
done
echo "[serve] node=$(hostname) port=${PORT} job=${SLURM_JOB_ID}"
echo "[serve] adapters: ${LORA_MODULE_ARGS[*]}"
echo "[serve] connect with: http://$(hostname):${PORT}"

# Publish the connection info to a QWEN-SPECIFIC file (NOT the shared ao_server.json,
# which a concurrently-running gemma AO server owns) so we don't clobber another eval's
# pointer. The Qwen Phase-B eval passes explicit --url/--base-model (read from this
# file) plus the always-required --ao-lora, so it never depends on the shared default.
AO_CONF="${CHIVE_DATA:-/workspace-vast/adamk/chive/data}/scratch/ao_server_qwen.json"
mkdir -p "$(dirname "$AO_CONF")"
printf '{"url": "http://%s:%s", "base_model": "%s", "ao_loras": ["%s", "%s"], "slurm_job": "%s"}\n' \
  "$(hostname)" "${PORT}" "${BASE_MODEL}" "${AO_LORAS[0]}" "${AO_LORAS[1]}" "${SLURM_JOB_ID}" > "$AO_CONF"
echo "[serve] wrote ${AO_CONF}"

vllm serve "${BASE_MODEL}" \
  --enable-lora \
  --lora-modules "${LORA_MODULE_ARGS[@]}" \
  --max-loras 2 \
  --max-lora-rank 64 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port "${PORT}"
