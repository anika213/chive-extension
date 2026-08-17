"""Shared, backend-agnostic helpers for the activation-fidelity check.

Pure logic (no GPU, no server): transcript/post sequence construction, per-token activation
metrics, KL helpers, and the running accumulators. Imported by collect_hf.py (the canonical HF
in-process collector). The two regions and how they're built are documented in collect_hf.py /
README.md.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch

# Layers captured (Qwen3-8B has 32 layers). 24 = NLA extraction layer (agent/NLA/nla_inference.py
# extraction_layer=24), 16 = middle. Single explicit source of truth.
DEFAULT_LAYERS = [16, 24]
NLA_LAYER = 24
REGIONS = ("transcript", "post")


@dataclass
class RegionStat:
    """Running token-weighted accumulators for one (model, layer, region) cell."""
    n_tokens: int = 0
    sum_cos: float = 0.0
    sum_rel_l2: float = 0.0
    sum_raw_mse: float = 0.0
    sum_base_norm: float = 0.0

    def add(self, cos: torch.Tensor, rel_l2: torch.Tensor, raw_mse: torch.Tensor, base_norm: torch.Tensor) -> None:
        self.n_tokens += cos.numel()
        self.sum_cos += float(cos.sum())
        self.sum_rel_l2 += float(rel_l2.sum())
        self.sum_raw_mse += float(raw_mse.sum())
        self.sum_base_norm += float(base_norm.sum())

    def summary(self) -> dict:
        n = max(self.n_tokens, 1)
        return {
            "n_tokens": self.n_tokens,
            "cosine": self.sum_cos / n,
            "rel_l2": self.sum_rel_l2 / n,
            "raw_mse": self.sum_raw_mse / n,
            "base_norm": self.sum_base_norm / n,
        }


@dataclass
class KLStat:
    """Running accumulators for one (model, region) KL cell. `coverage` is 1.0 for the exact
    full-vocab KL (HF path); for a top-K approximation it is the mean top-K coverage."""
    n_pos: int = 0
    sum_kl: float = 0.0
    sum_coverage: float = 0.0

    def add_vector(self, kl: torch.Tensor) -> None:
        """Fold in a [n_pos] tensor of EXACT per-position KLs (coverage counted as full)."""
        self.n_pos += kl.numel()
        self.sum_kl += float(kl.sum())
        self.sum_coverage += float(kl.numel())

    def summary(self) -> dict:
        n = max(self.n_pos, 1)
        return {"n_pos": self.n_pos, "kl": self.sum_kl / n, "coverage": self.sum_coverage / n}


def load_transcripts(bank_path: Path, n: int, seed: int) -> list[dict]:
    """Dedupe bank rows by prompt_id (claims share their investigation's transcript), then take a
    seeded random sample of n unique transcripts. Keeps prompt_id/messages/chosen_completion/question."""
    by_id: dict[str, dict] = {}
    for line in open(bank_path):
        d = json.loads(line)
        if d["prompt_id"] not in by_id:
            by_id[d["prompt_id"]] = {
                k: d[k] for k in ("prompt_id", "messages", "chosen_completion", "question")
            }
    items = list(by_id.values())
    random.Random(seed).shuffle(items)
    return items[:n]


def find_last_user_turn_start(tokenizer, ids: list[int]) -> int:
    """Token index of the LAST user turn's `<|im_start|>` — i.e. the appended question turn (always
    the final user turn, sitting just before the assistant generation prompt). Robust to multi-turn
    originals: we never count turns, just find the last `<|im_start|> user` in the full sequence."""
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    assert isinstance(im_start, int) and im_start >= 0, f"no <|im_start|> token id: {im_start}"
    cands = [
        p for p in range(len(ids) - 1)
        if ids[p] == im_start and tokenizer.decode([ids[p + 1]]).strip() == "user"
    ]
    assert cands, "no user turn found in the full sequence"
    return cands[-1]


def build_sequences(tokenizer, item: dict) -> tuple[list[int], list[int], int]:
    """Build the two sequences and the post-region boundary in B.

    Returns (ids_A, ids_B, boundary_B):
      - ids_A: standalone transcript (messages + [assistant: completion], no gen prompt) — EXACTLY
        what NLA reads. The whole sequence is the transcript region.
      - ids_B: full follow-up (... + [user: question], gen prompt). The post region is
        [boundary_B, len(ids_B)); B's earlier (transcript-history) positions are ignored.

    Two sequences (not one) because Qwen3's template injects an empty <think></think> into the
    FINAL assistant turn and strips it once a turn follows — so the transcript renders differently
    standalone vs as history. Each region uses the rendering that matches how it is actually consumed."""
    transcript_msgs = list(item["messages"]) + [
        {"role": "assistant", "content": item["chosen_completion"]}
    ]
    full_msgs = transcript_msgs + [{"role": "user", "content": item["question"]}]
    ids_A = tokenizer.apply_chat_template(
        transcript_msgs, tokenize=True, add_generation_prompt=False, enable_thinking=False
    )
    ids_B = tokenizer.apply_chat_template(
        full_msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    boundary_B = find_last_user_turn_start(tokenizer, ids_B)
    assert 0 < boundary_B < len(ids_B), (boundary_B, len(ids_B))
    return ids_A, ids_B, boundary_B


def activation_metrics(a_t: torch.Tensor, a_b: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Per-token metrics between trained (a_t) and base (a_b) activations, each [n_tok, d].
    Returns (cosine, relative-L2, raw-MSE, base-norm), each [n_tok]."""
    diff = a_t - a_b
    base_norm = a_b.norm(dim=-1)
    cos = torch.nn.functional.cosine_similarity(a_t, a_b, dim=-1)
    sq_err = (diff * diff).sum(dim=-1)                   # ||diff||^2
    rel_l2 = sq_err / base_norm.clamp_min(1e-12).pow(2)  # ||diff||^2 / ||base||^2
    raw_mse = sq_err / a_b.shape[-1]                     # mean over d
    return cos, rel_l2, raw_mse, base_norm
