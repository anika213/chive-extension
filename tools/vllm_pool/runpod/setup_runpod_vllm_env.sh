#!/bin/bash
set -euo pipefail

VLLM_ENV="${VLLM_ENV:-/root/vllm_envs/qwen36}"
VLLM_VERSION="${VLLM_VERSION:-0.19.1}"

if [[ ! -x "${VLLM_ENV}/bin/python" ]]; then
    python3 -m venv "${VLLM_ENV}"
fi

"${VLLM_ENV}/bin/python" -m pip install -U pip setuptools wheel
"${VLLM_ENV}/bin/python" -m pip install "vllm==${VLLM_VERSION}"

"${VLLM_ENV}/bin/python" - <<'PY'
import torch
import vllm

print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"vllm={vllm.__version__}")
PY
