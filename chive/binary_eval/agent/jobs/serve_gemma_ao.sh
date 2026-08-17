#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=vlens_gemma_ao
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --qos=high
#SBATCH --output=/workspace-vast/adamk/chive/chive/binary_eval/agent/logs/%x_%j.out

# Persistent vllm-lens AO server for the GEMMA-3-27B activation oracle.
# Mirrors investigations/backtracking_mc_vllm_lens/jobs/serve_vllm_lens.sh but
# serves google/gemma-3-27b-it + its AO verbalizer LoRA. The vllm-lens plugin
# auto-registers, so the OpenAI HTTP endpoints carry activations out and steering
# vectors in.
#
#   - capture: POST /v1/completions  model=google/gemma-3-27b-it  (base, no LoRA)
#              vllm_xargs={"output_residual_stream": "[31]"}  # JSON list of layers
#   - query:   POST /v1/completions  model=<lora module name>  (gemma AO verbalizer LoRA)
#              vllm_xargs={"apply_steering_vectors": json([sv.model_dump()])}
#
# The LoRA module name is derived from the HF repo: basename minus the "checkpoints_"
# prefix (must match ao_tool.ao_lora_name). Capture layers are NOT pinned here — each
# eval derives them from the LoRA's ao_config.json (resolve_capture_layers): combos
# [[15],[31],[46]] -> median layer 31; hook_onto_layer 1; coeff 1.0.

set -euo pipefail
cd /workspace-vast/adamk/chive
source .env
source .venv/bin/activate
source scripts/gpu_preflight.sh

export HF_HOME=/workspace-vast/adamk/hf_cache
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PORT=8414   # 8413 is the Qwen3-8B AO server; use a distinct port
BASE_MODEL=google/gemma-3-27b-it
AO_LORA=adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-3-27b-it
LORA_NAME=${AO_LORA##*/}; LORA_NAME=${LORA_NAME#checkpoints_}   # ao_tool.ao_lora_name
echo "[serve] node=$(hostname) port=${PORT} job=${SLURM_JOB_ID}"
echo "[serve] adapter: ${LORA_NAME}=${AO_LORA}"
echo "[serve] connect with: http://$(hostname):${PORT}"

# Publish the connection info so run_binary_eval --tool ao only needs --ao-lora (it
# reads url/base_model from this file; explicit flags override).
AO_CONF="${CHIVE_DATA:-/workspace-vast/adamk/chive/data}/scratch/ao_server.json"
mkdir -p "$(dirname "$AO_CONF")"
cat > "$AO_CONF" <<EOF
{"url": "http://$(hostname):${PORT}", "base_model": "${BASE_MODEL}", "ao_loras": ["${AO_LORA}"], "slurm_job": "${SLURM_JOB_ID}"}
EOF
echo "[serve] wrote ${AO_CONF}"

vllm serve "${BASE_MODEL}" \
  --enable-lora \
  --lora-modules "${LORA_NAME}=${AO_LORA}" \
  --max-lora-rank 64 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port "${PORT}"
