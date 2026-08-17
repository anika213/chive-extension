"""NLA inference — the production unit: specs, injection, capture, AV decode.

Everything needed to RUN an NLA activation-verbalizer lives here and is shared by
the cache precompute (precompute_nla_cache.py) and the round-trip validator
(nla_roundtrip.py), so the test exercises exactly the code that builds the cache.
The verification-only AR critic + scoring lives in nla_roundtrip.py, not here.

Contents:
  - NLASpec registry: OUR config for each checkpoint (base, AV/AR, subject layer).
  - AVInjector: builds the canonical AV prompt once, injects one activation per call.
    The ONE model-family behavior (gemma scales token embeddings by sqrt(d); vLLM's
    prompt_embeds path omits that scale, so we re-apply it) is a single scalar from
    an explicit model_type map that fails loud on the unknown. Everything else is
    data read from the checkpoint's own nla_meta.yaml/config.json.
  - capture_residual: HTTP to a vllm-lens base server, one cheap max_tokens=1 read.
  - AVDecoder: offline vllm.LLM(enable_prompt_embeds) + AVInjector; vectors -> text.
    async_scheduling=False is REQUIRED on vLLM < 0.22 (the prompt_embeds `<<<<`
    corruption bug; see precompute_nla_cache.py header).
"""

from __future__ import annotations

import os
# Must precede any torch/vllm import so vLLM re-inits CUDA in its worker (see
# AGENTS.md "vLLM offline"). Entry scripts should import this module early.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

# AV decode budget: the NLA-trained completion length. One constant used everywhere
# (AVDecoder, roundtrip, precompute). finish_reason is tracked so truncation at this
# budget shows up loudly (the Qwen3-8B actor decodes ~140 tokens, no truncation).
AV_MAX_NEW_TOKENS = 256


# ─── Specs: OUR config registry for each checkpoint ─────────────────────────
#
# Deliberately decoupled from the upstream NLA training tree: we record only what
# we need to USE a checkpoint, and read the rest (injection_scale, marker ids,
# d_model) from the checkpoint's own nla_meta.yaml, which ships with the weights.
# The subject `extraction_layer` is recorded here because the *actor* checkpoint's
# yaml omits it (newer schema). For the Qwen3-8B baseline it is 24 — recorded data:
# the actor yaml's `trained_on` parquet sidecar (rl.parquet.nla_meta.yaml) has
# `extraction.layer_index: 24` and its rows carry `activation_layer: 24` (verified
# 2026-06-12). Paths may be HF repo ids or local dirs.

@dataclass(frozen=True)
class NLASpec:
    name: str              # short label (cache/file naming, logs)
    base_model: str        # subject model whose residual stream the AV reads
    av_repo: str           # AV (verbalizer) checkpoint — HF id or local dir
    extraction_layer: int  # subject residual layer the AV was trained on
    ar_repo: str | None = None  # AR (critic) — verification only (round trip)


# Canonical round-trip regression target (published HF checkpoints, Qwen2.5-7B L20);
# matches tests/test_nla_vllm_lens_roundtrip.py reference numbers.
QWEN25_TEST = NLASpec(
    name="qwen2.5-7b-L20",
    base_model="Qwen/Qwen2.5-7B-Instruct",
    av_repo="kitft/nla-qwen2.5-7b-L20-av",
    ar_repo="kitft/nla-qwen2.5-7b-L20-ar",
    extraction_layer=20,
)

# The Qwen3-8B baseline ("nosalvage": 1-activation, no-source critic) we want to eval.
# The released HF repo is the actor (AV) only — verified byte-identical to the original
# `rl_qwen3_baseline_nosalvage/actor/iter_0000400/hf` checkpoint. The AR critic was not
# released, so the AR round trip can't run for this spec (same situation as GEMMA3_27B);
# the binary-eval precompute only needs the AV.
QWEN3_8B_BASELINE = NLASpec(
    name="qwen3-8b-baseline-nosalvage",
    base_model="Qwen/Qwen3-8B",
    av_repo="marco-bazzani/Qwen3-8B-nla",
    ar_repo=None,
    extraction_layer=24,
)

# Gemma-3-27B NLA used by the binary eval precompute. No AR critic available, so the
# AR round trip can't run for it (ar_repo=None) — but the precompute only needs the AV.
GEMMA3_27B = NLASpec(
    name="gemma3-27b-L41",
    base_model="google/gemma-3-27b-it",
    av_repo="kitft/nla-gemma3-27b-L41-av",
    ar_repo=None,
    extraction_layer=41,
)

SPECS = {s.name: s for s in (QWEN25_TEST, QWEN3_8B_BASELINE, GEMMA3_27B)}


# ─── Injection ──────────────────────────────────────────────────────────────

# Token-embedding scale on the vLLM prompt_embeds path, by HF model_type. Only the
# Gemma family scales token embeddings by sqrt(hidden_size); everything else is 1.0.
# Explicit on both sides so an unfamiliar arch fails loudly rather than silently
# guessing the wrong scale (a 64x error -> OOD AV prompt -> CJK-flavoured garbage).
GEMMA_EMBED_SCALE_TYPES = {"gemma", "gemma2", "gemma3", "gemma3_text"}
NO_EMBED_SCALE_TYPES = {"qwen2", "qwen3", "llama", "mistral"}


def resolve_checkpoint_dir(repo_or_dir: str) -> str:
    """An HF repo id -> snapshot_download; an existing local dir -> itself."""
    return repo_or_dir if os.path.isdir(repo_or_dir) else snapshot_download(repo_or_dir)


def embed_scale_for_prompt_embeds(model_type: str, d_model: int) -> float:
    """The token-embedding scale vLLM omits on the prompt_embeds path. Gemma
    family -> sqrt(hidden_size); Qwen/Llama/Mistral -> 1.0. Fail loud on unknown."""
    if model_type in GEMMA_EMBED_SCALE_TYPES:
        return float(d_model) ** 0.5
    assert model_type in NO_EMBED_SCALE_TYPES, (
        f"unknown model_type {model_type!r}: does this arch scale token embeddings "
        f"by sqrt(hidden_size) (Gemma-style)? Add it to GEMMA_EMBED_SCALE_TYPES or "
        f"NO_EMBED_SCALE_TYPES — do not guess."
    )
    return 1.0


def normalize_activation(v: torch.Tensor, target_scale: float) -> torch.Tensor:
    """Rescale a vector to `target_scale` L2-norm (norm computed in fp32)."""
    v = v.float().view(-1)
    return v / (v.norm().clamp_min(1e-12) / target_scale)


def load_embed_weight(checkpoint_dir: str, dtype: torch.dtype) -> torch.Tensor:
    """Load just `*.embed_tokens.weight` from a (possibly sharded) HF checkpoint."""
    root = Path(checkpoint_dir)
    idx_path = root / "model.safetensors.index.json"
    if idx_path.exists():
        weight_map = json.loads(idx_path.read_text())["weight_map"]
        keys = [k for k in weight_map if k.endswith("embed_tokens.weight")]
        assert len(keys) == 1, f"expected one embed_tokens.weight, got {keys}"
        shard, key = root / weight_map[keys[0]], keys[0]
    else:
        shard = root / "model.safetensors"
        with safe_open(str(shard), framework="pt") as f:
            keys = [k for k in f.keys() if k.endswith("embed_tokens.weight")]
        assert len(keys) == 1, f"expected one embed_tokens.weight, got {keys}"
        key = keys[0]
    with safe_open(str(shard), framework="pt") as f:
        return f.get_tensor(key).to(dtype)


class AVInjector:
    """Builds the canonical AV prompt once; injects one activation per call.

    `extraction_layer` is required only when the checkpoint's nla_meta.yaml omits
    `extraction_layer_index` (qwen actor schema); when the yaml has it and the
    caller also passes one, they must agree. The layer is metadata for callers that
    capture activations — it does not affect injection itself.
    """

    def __init__(self, av_repo: str, extraction_layer: int | None = None):
        av_dir = resolve_checkpoint_dir(av_repo)
        meta = yaml.safe_load((Path(av_dir) / "nla_meta.yaml").read_text())
        self.d_model = int(meta["d_model"])
        self.injection_scale = float(meta["extraction"]["injection_scale"])
        self.inj_id = int(meta["tokens"]["injection_token_id"])
        self.left_id = int(meta["tokens"]["injection_left_neighbor_id"])
        self.right_id = int(meta["tokens"]["injection_right_neighbor_id"])
        inj_char = meta["tokens"]["injection_char"]

        # Schema drift: gemma checkpoints key the AV template "av", qwen "actor".
        templates = meta["prompt_templates"]
        av_template = templates["av"] if "av" in templates else templates["actor"]

        # Subject extraction layer: gemma stores it top-level; qwen actor omits it.
        yaml_layer = meta.get("extraction_layer_index")
        if yaml_layer is not None and extraction_layer is not None:
            assert int(yaml_layer) == int(extraction_layer), (yaml_layer, extraction_layer)
        layer = yaml_layer if yaml_layer is not None else extraction_layer
        assert layer is not None, (
            "extraction layer absent from nla_meta.yaml (qwen actor schema) — pass "
            "extraction_layer= explicitly (the subject residual layer the AV was trained on)."
        )
        self.extraction_layer = int(layer)

        model_type = json.loads((Path(av_dir) / "config.json").read_text())["model_type"]
        self.embed_scale = embed_scale_for_prompt_embeds(model_type, self.d_model)

        tok = AutoTokenizer.from_pretrained(av_dir)
        # Build the AV prompt with the tokenizer DEFAULT chat template. For Qwen3
        # (thinking-capable) do NOT pass enable_thinking: forcing it False injects an
        # empty <think></think> the model never saw in RL rollout -> OOD. No-op for
        # non-thinking tokenizers (Qwen2.5, Gemma).
        ids = tok.apply_chat_template(
            [{"role": "user", "content": av_template.format(injection_char=inj_char)}],
            tokenize=True, add_generation_prompt=True,
        )
        self.ids = ids
        self.inj_pos = self._find_marker(ids)

        embed_w = load_embed_weight(av_dir, torch.bfloat16)
        assert embed_w.shape[1] == self.d_model, (embed_w.shape, self.d_model)
        embed = torch.nn.Embedding(embed_w.shape[0], embed_w.shape[1], _weight=embed_w)
        embed.requires_grad_(False).eval()
        scale = torch.tensor(self.embed_scale, dtype=torch.bfloat16)
        with torch.no_grad():
            self.base_embeds = embed(torch.tensor(ids).unsqueeze(0))[0].to(torch.bfloat16) * scale
        print(
            f"[AVInjector] {model_type} d_model={self.d_model} T={len(ids)} "
            f"inj_pos={self.inj_pos} embed_scale={self.embed_scale:.3f} "
            f"extraction_layer={self.extraction_layer} injection_scale={self.injection_scale}",
            flush=True,
        )

    def _find_marker(self, ids: list[int]) -> int:
        """The single injection-char position with the expected ±1 neighbors."""
        for p, t in enumerate(ids):
            if (
                t == self.inj_id and 0 < p < len(ids) - 1
                and ids[p - 1] == self.left_id and ids[p + 1] == self.right_id
            ):
                return p
        raise AssertionError("no neighbor-verified injection position in AV prompt")

    def prompt_embeds_tensor(self, vector: torch.Tensor) -> torch.Tensor:
        """[d_model] captured activation -> [T, d] bf16 prompt_embeds with the
        activation (renormalized to injection_scale) injected at the marker slot."""
        emb = self.base_embeds.clone()
        emb[self.inj_pos] = normalize_activation(vector, self.injection_scale).to(torch.bfloat16)
        return emb


# ─── Decode-output parsing ──────────────────────────────────────────────────

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)


def extract_explanation(raw: str) -> tuple[str, bool]:
    """-> (explanation, missing). The <explanation>…</explanation> span (which
    naturally skips any leading <think> block), or the raw text marked missing
    if the tags aren't present (truncated / degenerate decode)."""
    m = EXPLANATION_RE.search(raw)
    if m is not None:
        return m.group(1).strip(), False
    return raw.strip(), True


# ─── Capture (HTTP to a vllm-lens base server) ──────────────────────────────

def _post_with_retries(client: httpx.Client, url: str, payload: dict, retries: int = 4) -> dict:
    """POST with a few retries on transient connection resets (the engine is fine;
    at load the server occasionally drops a connection at the HTTP layer)."""
    for attempt in range(retries):
        try:
            r = client.post(url, json=payload)
            r.raise_for_status()
            return r.json()
        except httpx.TransportError:
            if attempt == retries - 1:
                raise
    raise RuntimeError("unreachable")


def capture_residual(
    capture_url: str, base_model: str, token_ids: list[int], layer: int, d_model: int,
    timeout: float = 600.0,
) -> torch.Tensor:
    """Capture the layer-`layer` residual stream over `token_ids` -> [T, d] fp.

    Passes TOKEN IDS (not text) so there is no re-tokenization drift between
    capture and the segmentation the rest of the pipeline uses. Asserts the
    captured length equals the input length (guards silent BOS insertion etc.).
    """
    from vllm_lens._helpers._serialize import decode_activations

    payload = {
        "model": base_model, "prompt": token_ids, "max_tokens": 1, "temperature": 0.0,
        "vllm_xargs": {"output_residual_stream": json.dumps([layer])},
    }
    with httpx.Client(timeout=timeout) as client:
        body = _post_with_retries(client, f"{capture_url}/v1/completions", payload)
    residual = decode_activations(body)["residual_stream"]
    assert residual.shape[0] == 1, (
        f"expected one captured layer, got {tuple(residual.shape)} — is the vllm-lens "
        f"output_residual_stream JSON-decode patch applied + server restarted?"
    )
    captured = residual[0]                       # [T, d]
    assert captured.shape[0] == len(token_ids), (captured.shape, len(token_ids))
    assert captured.shape[1] == d_model, (captured.shape, d_model)
    return captured


# ─── AV decode (offline vLLM prompt_embeds) ─────────────────────────────────

@dataclass(frozen=True)
class DecodeResult:
    explanation: str       # extracted <explanation> text (or raw if missing)
    raw: str               # full model output (incl. any <think> block)
    finish_reason: str     # "stop" | "length" — "length" => truncated
    completion_tokens: int
    missing_tag: bool      # no <explanation>…</explanation> found


class AVDecoder:
    """Offline AV decode engine: captured vectors -> explanation text.

    Owns a vllm.LLM(enable_prompt_embeds=True, async_scheduling=False) plus the
    AVInjector. Shared by the precompute and the round trip. Call close() before
    loading another GPU model (e.g. the round trip's torch AR) to free the engine.
    """

    def __init__(
        self, av_repo: str, extraction_layer: int | None = None,
        temperature: float = 1.0, max_tokens: int = AV_MAX_NEW_TOKENS,
        max_model_len: int = 2048, gpu_memory_utilization: float = 0.90,
    ):
        from vllm import LLM, SamplingParams

        self.injector = AVInjector(av_repo, extraction_layer=extraction_layer)
        self.extraction_layer = self.injector.extraction_layer
        self.d_model = self.injector.d_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # prompt + decode budget must fit the engine window, else vLLM silently truncates.
        assert len(self.injector.ids) + max_tokens <= max_model_len, (
            f"AV prompt ({len(self.injector.ids)}) + max_tokens ({max_tokens}) "
            f"> max_model_len ({max_model_len})"
        )
        # async_scheduling=False: REQUIRED on vLLM < 0.22 — async scheduling corrupts
        # prompt_embeds decodes into `<<<<` loops via a stale is_token_ids GPU mask.
        self.llm = LLM(
            model=resolve_checkpoint_dir(av_repo), dtype="bfloat16",
            enable_prompt_embeds=True, max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization, async_scheduling=False,
        )
        self.sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    def decode(self, vectors: list[torch.Tensor]) -> list[DecodeResult]:
        """One batched generate() over all vectors. Each vector is [d_model]."""
        prompt_inputs = [
            {"prompt_embeds": self.injector.prompt_embeds_tensor(v)} for v in vectors
        ]
        results = self.llm.generate(prompt_inputs, sampling_params=self.sp, use_tqdm=False)
        assert len(results) == len(vectors), (len(results), len(vectors))
        out: list[DecodeResult] = []
        for r in results:
            o = r.outputs[0]
            expl, missing = extract_explanation(o.text)
            out.append(DecodeResult(
                explanation=expl, raw=o.text, finish_reason=o.finish_reason,
                completion_tokens=len(o.token_ids), missing_tag=missing,
            ))
        return out

    def close(self) -> None:
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
