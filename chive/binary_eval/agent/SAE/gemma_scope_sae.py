"""Hand-rolled Gemma Scope 2 JumpReLU SAE (load + encode/decode).

Deliberately NOT SAELens: the math is one matmul + a threshold gate, and a hand-rolled
module keeps a single code path for non-gemma SAEs later (same approach as
adamkarvonen/interp_tools, adapted from GS1's params.npz to GS2's params.safetensors).

Gemma Scope 2 ships per-SAE folders on HF (e.g. google/gemma-scope-2-27b-it /
resid_post/layer_40_width_65k_l0_medium/) with:
  - params.safetensors: w_enc [d_in, d_sae], b_enc/threshold [d_sae],
    w_dec [d_sae, d_in], b_dec [d_in]   (all float32)
  - config.json: {hf_hook_point_in: "model.layers.40.output", width, l0,
    architecture: "jump_relu", model_name, ...}

encode(x) = relu(x @ w_enc + b_enc) gated by (pre > threshold)  — activations are in
the SAE's trained scale (raw resid_post inputs, no input normalization).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


@dataclass(frozen=True)
class GemmaScopeSAE:
    w_enc: torch.Tensor      # [d_in, d_sae]
    b_enc: torch.Tensor      # [d_sae]
    threshold: torch.Tensor  # [d_sae]
    w_dec: torch.Tensor      # [d_sae, d_in]
    b_dec: torch.Tensor      # [d_in]
    repo: str
    sae_path: str            # e.g. "resid_post/layer_40_width_65k_l0_medium"
    hook_layer: int          # e.g. 40, parsed from config's hf_hook_point_in
    expected_l0: int         # config "l0" (training target; sanity checks compare to it)

    @property
    def d_in(self) -> int:
        return self.w_enc.shape[0]

    @property
    def d_sae(self) -> int:
        return self.w_enc.shape[1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d_in] (cast to the SAE's dtype/device by the caller) -> [..., d_sae]."""
        pre = x @ self.w_enc + self.b_enc
        return torch.nn.functional.relu(pre) * (pre > self.threshold)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.w_dec + self.b_dec

    def fvu(self, x: torch.Tensor) -> float:
        """Fraction of variance UNexplained of the reconstruction over x: [n, d_in]."""
        recon = self.decode(self.encode(x))
        total = torch.var(x, dim=0).sum()
        residual = torch.var(x - recon, dim=0).sum()
        return float(residual / total)


def load_gemma_scope_sae(repo: str, sae_path: str, device: str | torch.device) -> GemmaScopeSAE:
    cfg = json.loads(Path(hf_hub_download(repo, f"{sae_path}/config.json")).read_text())
    assert cfg["architecture"] == "jump_relu", cfg
    hook = cfg["hf_hook_point_in"]                       # "model.layers.40.output"
    hook_layer = int(hook.removeprefix("model.layers.").removesuffix(".output"))
    params = load_file(hf_hub_download(repo, f"{sae_path}/params.safetensors"), device=str(device))
    sae = GemmaScopeSAE(
        w_enc=params["w_enc"], b_enc=params["b_enc"], threshold=params["threshold"],
        w_dec=params["w_dec"], b_dec=params["b_dec"],
        repo=repo, sae_path=sae_path, hook_layer=hook_layer, expected_l0=int(cfg["l0"]),
    )
    assert sae.d_sae == int(cfg["width"]), (sae.d_sae, cfg["width"])
    assert sae.w_dec.shape == (sae.d_sae, sae.d_in)
    print(f"[sae] {repo}/{sae_path}: d_in={sae.d_in} d_sae={sae.d_sae} "
          f"hook_layer={hook_layer} expected_l0={sae.expected_l0} device={device}")
    return sae
