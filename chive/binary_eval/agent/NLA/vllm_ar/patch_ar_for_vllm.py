"""Make the NLA AR checkpoint vLLM-loadable.

The AR ships truncated (21 layers) with `lm_head` and `model.norm.weight`
omitted (`lm_head` was stripped, final norm was replaced with Identity at
training time). vLLM's Qwen2 loader requires both. We materialize stub
weights into a sibling directory:

  lm_head.weight     = zeros(vocab, hidden)  — generation discarded
  model.norm.weight  = ones(hidden)          — we capture BEFORE final norm

Run:
    python patch_ar_for_vllm.py <SRC_AR_DIR> <DST_DIR>
or import and call patch_ar_for_vllm(src, dst).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig


def patch_ar_for_vllm(src_dir: str | Path, dst_dir: str | Path) -> None:
    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    cfg = AutoConfig.from_pretrained(str(src))
    print(
        f"[patch-ar] src={src}\n[patch-ar] dst={dst}\n"
        f"[patch-ar] vocab_size={cfg.vocab_size} hidden_size={cfg.hidden_size} "
        f"num_hidden_layers={cfg.num_hidden_layers}",
        flush=True,
    )
    for fname in os.listdir(src):
        if fname == "model.safetensors.index.json":
            continue
        tgt = dst / fname
        if tgt.exists() or tgt.is_symlink():
            tgt.unlink()
        tgt.symlink_to(src / fname)
    save_file(
        {
            "lm_head.weight": torch.zeros(cfg.vocab_size, cfg.hidden_size,
                                          dtype=torch.bfloat16),
            "model.norm.weight": torch.ones(cfg.hidden_size,
                                            dtype=torch.bfloat16),
        },
        str(dst / "model-stub.safetensors"),
    )
    orig = json.loads((src / "model.safetensors.index.json").read_text())
    new = {
        "metadata": dict(orig.get("metadata", {})),
        "weight_map": dict(orig["weight_map"]),
    }
    new["weight_map"]["lm_head.weight"] = "model-stub.safetensors"
    new["weight_map"]["model.norm.weight"] = "model-stub.safetensors"
    new["metadata"]["total_size"] = (
        new["metadata"].get("total_size", 0)
        + cfg.vocab_size * cfg.hidden_size * 2 + cfg.hidden_size * 2
    )
    (dst / "model.safetensors.index.json").write_text(json.dumps(new, indent=2))
    print(
        f"[patch-ar] wrote index with {len(new['weight_map'])} entries "
        f"(orig had {len(orig['weight_map'])})",
        flush=True,
    )


if __name__ == "__main__":
    assert len(sys.argv) == 3, "usage: patch_ar_for_vllm.py <SRC_AR_DIR> <DST_DIR>"
    patch_ar_for_vllm(sys.argv[1], sys.argv[2])
