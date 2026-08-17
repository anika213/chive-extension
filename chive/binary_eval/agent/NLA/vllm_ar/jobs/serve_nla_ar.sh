#!/bin/bash
# cluster-specific (Slurm); adapt paths/partitions to your setup.
#SBATCH --job-name=nla_ar
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --qos=high
#SBATCH --output=/workspace-vast/adamk/chive/chive/binary_eval/agent/NLA/vllm_ar/logs/%x_%j.out

# Persistent vllm-lens server for the NLA AR (critic / reconstructor).
# Hosts a PATCHED copy of kitft/nla-qwen2.5-7b-L20-ar — the shipped checkpoint
# is missing lm_head and model.norm weights (stripped at training time);
# vLLM's Qwen2 loader requires both. patch_ar_for_vllm.py creates a sibling
# directory with symlinks + stub zero/one weights.
#
# Client recipe (POST /v1/completions):
#   prompt = "Summary of the following text: <text>{explanation}</text> <summary>"
#       (tokenize with add_special_tokens=True; for Qwen, bos=None so noop)
#   max_tokens=1
#   vllm_xargs={"output_residual_stream": "[20]"}     # last block of 21
#
# Then on the client (one-time setup):
#   from safetensors.torch import load_file
#   value_head_w = load_file(f"{AR_SRC_DIR}/value_head.safetensors")["weight"].float()
#                                                     # [d_model, d_model]
# Per request:
#   from vllm_lens import decode_activations
#   resid = decode_activations(response.activations)["residual_stream"]  # [1, T, d]
#   pred = (value_head_w @ resid[0, -1].float()).cpu()                   # [d]
#   # normalize both to L2 = mse_scale (≈ sqrt(d_model), from nla_meta.yaml)
#   # MSE_normed = ((pred_n - gold_n)**2).mean() == 2 * (1 - cos)
#
# Why "layer 20" on a 21-layer model: layer indices in vllm-lens are 0-based
# block indices, so block 20 == the LAST block. Captured value is the residual
# stream at its OUTPUT, BEFORE the final norm — which is exactly what the AR
# value_head was trained on (final norm was Identity).

set -euo pipefail
ROOT=/workspace-vast/adamk/chive
cd "$ROOT"
source .env
source .venv/bin/activate
source scripts/gpu_preflight.sh

export HF_HOME=/workspace-vast/adamk/hf_cache
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 1. Resolve original AR snapshot (downloads on first run).
AR_SRC=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('kitft/nla-qwen2.5-7b-L20-ar'))")
echo "[serve] AR src snapshot: ${AR_SRC}"

# 2. Build/refresh patched dir (idempotent — symlinks + stub safetensors).
NLA_AR_DIR="${ROOT}/chive/binary_eval/agent/NLA/vllm_ar"
AR_PATCHED="${NLA_AR_DIR}/checkpoints/ar_vllm_compat"
mkdir -p "$(dirname "$AR_PATCHED")"
python "${NLA_AR_DIR}/patch_ar_for_vllm.py" "${AR_SRC}" "${AR_PATCHED}"

PORT=8423
echo "[serve] node=$(hostname) port=${PORT} job=${SLURM_JOB_ID}"
echo "[serve] connect with: http://$(hostname):${PORT}"

# Served-model-name=ar so clients can address as model="ar" regardless of dir.
# enforce_eager: max_tokens=1, no point in cudagraph compile cost.
vllm serve "${AR_PATCHED}" \
  --served-model-name ar \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port "${PORT}"
