"""Hand-rolled batch-top-k SAE (load + encode/decode), sibling of gemma_scope_sae.py.

Loads dictionary_learning-trained batch-top-k SAEs (e.g. adamkarvonen/qwen3-8b-saes),
following the reference loader in adamkarvonen/interp_tools
(interp_tools/saes/batch_topk_sae.py). The checkpoint is a single ae.pt with the
trainer's raw state dict:
  - encoder.weight [d_sae, d_in], encoder.bias [d_sae]  (nn.Linear layout -> transpose)
  - decoder.weight [d_in, d_sae]                        (transpose)
  - b_dec [d_in], k (scalar int), threshold (scalar float)
plus a config.json whose trainer block records layer / k / lm_name / submodule_name.

encode(x) uses INFERENCE mode: relu((x - b_dec) @ W_enc + b_enc) gated by the trained
global scalar threshold (JumpReLU-style), NOT exact per-position top-k — so per-position
L0 varies around the trained k (~75-85 for k=80) rather than being capped at k.

IMPORTANT (experiment-semantic): these SAEs were trained with attention-sink positions
EXCLUDED (residual norm > 10x the sequence median — Qwen has such sinks hundreds of
tokens in, not just position 0). Inference must apply the same mask or sink positions
produce garbage features; that filtering lives in precompute_sae_cache.py, not here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class BatchTopKSAE:
    w_enc: torch.Tensor      # [d_in, d_sae]
    b_enc: torch.Tensor      # [d_sae]
    b_dec: torch.Tensor      # [d_in]
    threshold: torch.Tensor  # scalar, > 0
    w_dec: torch.Tensor      # [d_sae, d_in]
    repo: str
    sae_path: str            # e.g. "saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_18/trainer_2"
    hook_layer: int          # config trainer.layer (resid_post of that block)
    expected_l0: int         # config trainer.k (training target; sanity checks compare to it)
    lm_name: str             # config trainer.lm_name, e.g. "Qwen/Qwen3-8B"

    @property
    def d_in(self) -> int:
        return self.w_enc.shape[0]

    @property
    def d_sae(self) -> int:
        return self.w_enc.shape[1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d_in] (cast to the SAE's dtype/device by the caller) -> [..., d_sae]."""
        pre = torch.nn.functional.relu((x - self.b_dec) @ self.w_enc + self.b_enc)
        return pre * (pre > self.threshold)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.w_dec + self.b_dec

    def fvu(self, x: torch.Tensor) -> float:
        """Fraction of variance UNexplained of the reconstruction over x: [n, d_in]."""
        recon = self.decode(self.encode(x))
        total = torch.var(x, dim=0).sum()
        residual = torch.var(x - recon, dim=0).sum()
        return float(residual / total)


def load_batch_topk_sae(repo: str, sae_path: str, device: str | torch.device) -> BatchTopKSAE:
    """sae_path is the checkpoint DIR inside the repo (containing ae.pt + config.json)."""
    cfg = json.loads(Path(hf_hub_download(repo, f"{sae_path}/config.json")).read_text())["trainer"]
    assert cfg["dict_class"] == "BatchTopKSAE", cfg
    assert cfg["submodule_name"] == f"resid_post_layer_{cfg['layer']}", cfg
    sd = torch.load(hf_hub_download(repo, f"{sae_path}/ae.pt"), map_location=str(device))
    assert set(sd) == {"encoder.weight", "encoder.bias", "decoder.weight", "b_dec",
                       "k", "threshold"}, sorted(sd)
    assert int(sd["k"]) == int(cfg["k"]), (int(sd["k"]), cfg["k"])
    assert float(sd["threshold"]) > 0, f"inference threshold not set: {sd['threshold']}"
    sae = BatchTopKSAE(
        w_enc=sd["encoder.weight"].T.contiguous(), b_enc=sd["encoder.bias"],
        b_dec=sd["b_dec"], threshold=sd["threshold"],
        w_dec=sd["decoder.weight"].T.contiguous(),
        repo=repo, sae_path=sae_path, hook_layer=int(cfg["layer"]),
        expected_l0=int(cfg["k"]), lm_name=cfg["lm_name"],
    )
    assert sae.d_in == int(cfg["activation_dim"]), (sae.d_in, cfg["activation_dim"])
    assert sae.d_sae == int(cfg["dict_size"]), (sae.d_sae, cfg["dict_size"])
    assert sae.w_dec.shape == (sae.d_sae, sae.d_in)
    print(f"[sae] {repo}/{sae_path}: batch_top_k d_in={sae.d_in} d_sae={sae.d_sae} "
          f"hook_layer={sae.hook_layer} expected_l0={sae.expected_l0} "
          f"threshold={float(sae.threshold):.3f} lm={sae.lm_name} device={device}")
    return sae
