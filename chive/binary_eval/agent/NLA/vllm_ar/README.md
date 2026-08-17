# Optional: AR critic on vllm-lens

The canonical NLA reconstruction runs in plain PyTorch via the upstream
`nla_inference.py::NLACritic` recipe — `AutoModelForCausalLM`, strip
`lm_head`, replace final norm with Identity, apply `value_head`. Use this folder only when **you need to serve AR
reconstruction over HTTP** — e.g. a long-running eval that scores thousands
of explanations and benefits from vLLM's batching + KV reuse, or an agent
harness where keeping the AR resident on its own GPU is easier than
spinning up a torch model per process.

## What's hard about loading the AR in vLLM

The AR ships as a truncated 21-layer Qwen2 (`num_hidden_layers=21`) with:

- `lm_head` weights stripped (critic never produces logits)
- final RMSNorm replaced by Identity at train time, so `model.norm.weight`
  is NOT in the safetensors shards
- a separate `value_head.safetensors` with a `[d, d]` Linear bolted on

vLLM's `Qwen2ForCausalLM` loader rejects the checkpoint as-is because both
`lm_head.weight` and `model.norm.weight` are missing. We make it loadable
by writing stub weights into a sibling directory:

| key | stub value | why it's inert |
|---|---|---|
| `lm_head.weight` | `zeros(vocab, hidden)` | we discard generated tokens (`max_tokens=1`) |
| `model.norm.weight` | `ones(hidden)` | we capture resid at the OUTPUT of layer 20, BEFORE the final norm fires |

[`patch_ar_for_vllm.py`](patch_ar_for_vllm.py) builds the patched dir by
symlinking every file from the source snapshot and adding a
`model-stub.safetensors` shard with these two tensors.

## Serving

[`jobs/serve_nla_ar.sh`](jobs/serve_nla_ar.sh) is a persistent sbatch job
that:

1. Resolves the source AR snapshot via `huggingface_hub.snapshot_download`
2. Idempotently runs `patch_ar_for_vllm.py` to produce
   `checkpoints/ar_vllm_compat/`
3. `vllm serve`s that path with `--served-model-name ar`

Port 8423 by default. vllm-lens auto-registers as a plugin.

## Client recipe (HTTP)

```python
import json
import httpx
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer
from vllm_lens import decode_activations
import yaml

AR_REPO = "kitft/nla-qwen2.5-7b-L20-ar"
ar_src = snapshot_download(AR_REPO)

# One-time: load the value head + tokenizer + sidecar.
meta = yaml.safe_load((Path(ar_src) / "nla_meta.yaml").read_text())
mse_scale = float(meta["extraction"]["mse_scale"])    # ≈ sqrt(d_model)
ar_template = meta["prompt_templates"]["ar"]
tok = AutoTokenizer.from_pretrained(ar_src)
vh_w = load_file(str(Path(ar_src) / "value_head.safetensors"))["weight"].float()
                                                       # [d_model, d_model]

# Per request:
def score(ar_url: str, explanation: str, gold_vector):
    ids = tok(ar_template.format(explanation=explanation),
              add_special_tokens=True)["input_ids"]
    resp = httpx.post(
        f"{ar_url}/v1/completions",
        json={
            "model": "ar",                     # --served-model-name from the job
            "prompt": "",
            "prompt_token_ids": ids,
            "max_tokens": 1,
            "temperature": 0.0,
            # JSON-encoded list — see ../VLLM_LENS_PATCH.md
            "vllm_xargs": {"output_residual_stream": json.dumps([20])},
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    acts = decode_activations(resp.json()["activations"])
    resid = acts["residual_stream"]                    # [1, T, d]
    pred = (vh_w @ resid[0, -1].float()).cpu()         # [d]
    gold = gold_vector.float()
    pred_n = pred / pred.norm().clamp_min(1e-12) * mse_scale
    gold_n = gold / gold.norm().clamp_min(1e-12) * mse_scale
    mse = ((pred_n - gold_n) ** 2).mean().item()       # = 2*(1-cos)
    cos = (pred_n @ gold_n / (pred_n.norm() * gold_n.norm())).item()
    return {"mse_nrm": mse, "cos": cos,
            "fve_nrm": 1.0 - mse / 0.7335}
```

## Why layer 20 on a 21-layer model

vllm-lens layer indices are 0-based block indices, so block 20 == the LAST
block. The captured value is the residual stream at its OUTPUT, BEFORE the
final norm — which is exactly what the AR value_head was trained on (final
norm was Identity during training, so the value_head sees raw resid_post of
the last block).

## Equivalence vs torch AR

Validated end-to-end in `/workspace-vast/adamk/nla_verification/results/` (external
archive, not included in this release):
on the same explanations, vllm-lens AR vs torch AR aggregate deltas are
`cos = 7e-5`, `fve_nrm = 2e-4` — pure floating-point noise. So either path
gives the same FVE; torch wins for setup simplicity, vllm-lens wins for
HTTP-serving throughput.

## Files

- [`README.md`](README.md) — this file
- [`patch_ar_for_vllm.py`](patch_ar_for_vllm.py) — stub-weights patcher
- [`jobs/serve_nla_ar.sh`](jobs/serve_nla_ar.sh) — persistent serving job
