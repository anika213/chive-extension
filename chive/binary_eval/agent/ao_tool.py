"""Activation-oracle tool for the interp agent — tool #1.

Talks to the vllm-lens HTTP server (nothing loaded in-process). Given a target
model transcript (pre-segmented over its captured token sequence, see
transcript_segments.py), it (1) captures the residual stream once over the full
chat-templated token sequence at every AO capture layer, then on each agent call
(2) resolves the agent's selected span (sentence range or token range) to token
positions, builds the AO oracle prompt with one per-layer block of special tokens
(one special token per injected position per layer, mirroring the AO's
training-time introspection prefix — the format ships with the released LoRA as
ao_config.json's prefix_template), and (3) runs a
norm-matched steered query against the AO LoRA, returning the oracle's
natural-language verbalization.

Single- vs multi-layer AOs are both supported and the capture layers are derived
from the LoRA's ao_config.json (see resolve_capture_layers): a multi-layer AO
(act_layer_combinations like [[9, 18, 27]]) reads all its layers at once; a
single-layer AO ([[9], [18], [27]]) reads the median layer.

AO config (capture/hook layers, special token, steering coeff) is read from the
LoRA repo's ao_config.json.

NOTE (experiment semantics): the AO is base-model-specific — it only reads
activations of the model it was trained on. Probing a transcript produced by a
DIFFERENT target model runs mechanically but is meaningless; the MC bank's target
model MUST match the AO's base model (caller passes both explicitly).
"""

from __future__ import annotations

import json
import os

import httpx
import torch
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from vllm_lens import SteeringVector
from vllm_lens._helpers._serialize import decode_activations

from chive.binary_eval.agent.agent import Tool
from chive.binary_eval.agent.transcript_segments import (
    SegmentedTranscript, parse_selection, resolve_positions, selection_property)


class AOConfig(BaseModel, extra="allow"):
    """Mirror of ao_config.json in the AO LoRA repo."""
    hook_onto_layer: int
    special_token: str
    prefix_template: str = "Layer: {layer}\n{special_tokens} \n"
    steering_coefficient: float
    act_layer_combinations: list[list[int]]


def load_ao_config(lora_path: str) -> AOConfig:
    """Load ao_config.json from the LoRA repo (HF Hub or local path)."""
    if os.path.isdir(lora_path):
        config_path = os.path.join(lora_path, "ao_config.json")
    else:
        config_path = hf_hub_download(repo_id=lora_path, filename="ao_config.json")
    with open(config_path) as f:
        return AOConfig(**json.load(f))


def ao_lora_name(lora_path: str) -> str:
    """Canonical short name for an AO LoRA repo: basename minus the 'checkpoints_'
    prefix. Used as the served LoRA module name (the serve scripts derive the same
    name in bash: ${repo##*/} then ${name#checkpoints_}) and as the checkpoint
    identity embedded in eval result filenames."""
    name = lora_path.rstrip("/").split("/")[-1].removeprefix("checkpoints_")
    assert name, f"cannot derive AO lora name from {lora_path!r}"
    return name


MAX_INJECT = 512          # cap on injected positions. The AO was trained ONLY on CONTIGUOUS
                          # token spans, so an oversized selection is truncated to the first
                          # MAX_INJECT (low end) — never subsampled (a thinned, non-contiguous
                          # span is out of distribution).
AO_MAX_NEW_TOKENS = 200
AO_N_SAMPLES = 10         # the oracle can't express uncertainty in one sample; sample N times
AO_TEMPERATURE = 1.0      # so the agent can read confidence off cross-sample consensus. Affordable
                          # because AO is called once per SPAN (unlike the per-token tools, where
                          # N samples * every position would blow up the agent's context).

QUERY_AO_TOOL = {
    "name": "query_ao",
    "description": (
        "Probe the TARGET model's internal activations (residual stream) over a chosen span "
        "of its transcript and get a natural-language verbalization answering your question "
        "about what it is internally representing there. The oracle reads the hidden state "
        "and describes it; it is noisy, so ask focused OPEN-ENDED questions (never either/or, "
        "never counterfactuals) and corroborate across spans before trusting a single read.\n"
        "Choose the span with the `selection` string: coarsely by segment ('S11' for one segment, "
        "'S11-S18' for a run, using the [S#] ids from the addressable transcript) or finely by "
        "token ('t40-t75', inclusive, using indices from show_tokens). The oracle reads a single "
        "CONTIGUOUS span (it was trained only on contiguous token runs), so scattered selections "
        f"are not allowed; a span over {MAX_INJECT} tokens is truncated to its first {MAX_INJECT}. "
        "Every token is in some [S#] segment, including the chat-template scaffolding (the "
        "(control) segments: BOS, turn-start/turn-end markers, role headers); to read a turn "
        "boundary, select the (control) segment between the two turns."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            **selection_property(allow_scattered=False),
            "question": {
                "type": "string",
                "description": (
                    "Natural-language question for the oracle about the model's internal state "
                    "in that span (e.g. 'What entity or concept is the model representing here?')."
                ),
            },
        },
        "required": ["selection", "question"],
    },
}

# Mechanical-only ("simple") prompt, matching the NLA tool: describes how the tool works and
# that it is noisy, with NO behavioral guidance about how to weigh the reads. Elaborate guided
# variants (question taxonomies, grounding checks, reads-vs-counterfactual guardrails) moved
# only the operating point, never AUROC — see binary_evals/INTERP_TOOL_FAILURE_INVESTIGATION.md.
AO_SYSTEM_FRAGMENT = (
    "## Tool: query_ao (activation oracle)\n"
    "Reads the TARGET model's internal residual-stream activations over a chosen span of its OWN "
    "transcript and describes, in plain language, what it is representing there. Spans refer to "
    "the TARGET's transcript (its activations), not your own reasoning. Use it to read out what "
    "the model encodes while you decide how likely the claim is.\n"
    "Set `selection` to a segment ('S11' / 'S11-S18') or a contiguous token range ('t40-t75'; "
    "indices from show_tokens) — scattered tokens are not allowed, and a span over "
    f"{MAX_INJECT} tokens is truncated to its first {MAX_INJECT}. Chat-template tokens "
    "(BOS, turn markers, role headers) form their own (control) [S#] segments — select them "
    "like any other. Ask short, open-ended questions about what the activation currently "
    "encodes; the oracle reads the model's current state, it cannot run experiments.\n"
    "Each call returns several INDEPENDENT samples, and any single sample sounds confident "
    "(the oracle was not trained to express uncertainty). The reads are noisy, so judge by "
    "what is consistent across the samples — and across related spans — not any single read."
)

SHOW_TOKENS_TOOL = {
    "name": "show_tokens",
    "description": (
        "Reveal the individual tokens (with their integer indices) inside a segment or token "
        "span, so you can select a narrower span by token. This does NOT read activations — it "
        "just lists tokens. Set `selection` to 'S11' / 'S11-S18' (segments) or 't40-t75' (tokens)."
    ),
    "input_schema": {
        "type": "object",
        "properties": selection_property(allow_scattered=False),
        "required": ["selection"],
    },
}

SHOW_TOKENS_SYSTEM_FRAGMENT = (
    "## Tool: show_tokens\n"
    "Lists the tokens and their integer indices inside a segment or token span (no activation "
    "read). Call it when you want to point a tool at a few specific tokens rather than a whole "
    "segment; then pass selection='t40-t75' (inclusive) to the reading tool."
)


def _build_oracle_prompt_ids(tokenizer, ao_config, capture_layers, num_positions, question):
    """Replicates the AO oracle prompt: one prefix_template block PER capture layer
    (each with one special token per injected position), then the question, via the
    chat template. Identical to the AO's training-time prompt format (prefix_template
    from the LoRA's ao_config.json): for layers [9, 18, 27] the prefix is three
    'Layer: {L}\\n<special>*P \\n' blocks in layer order."""
    template = ao_config.prefix_template.replace(
        "{special_token} * {num_positions}", "{special_tokens}"
    ).replace("\\n", "\n")
    prefix = "".join(
        template.format(special_tokens=ao_config.special_token * num_positions, layer=layer)
        for layer in capture_layers
    )
    # tokenize=True: re-encoding the rendered template text with add_special_tokens=True
    # would prepend a SECOND <bos> on Gemma, putting every read off-distribution vs AO
    # training (which tokenizes via the template directly).
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prefix + question}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    special_ids = tokenizer.encode(ao_config.special_token, add_special_tokens=False)
    assert len(special_ids) == 1, special_ids
    # Mirror training's find_pattern_in_tokens: the first L*P special tokens are the
    # injection slots, layer-major (block i = capture_layers[i]); each block must be
    # one contiguous run of P.
    total = len(capture_layers) * num_positions
    injection_positions = [i for i, t in enumerate(prompt_ids) if t == special_ids[0]][:total]
    assert len(injection_positions) == total, (
        f"expected {total} special-token slots, found {len(injection_positions)}"
    )
    for b in range(len(capture_layers)):
        block = injection_positions[b * num_positions:(b + 1) * num_positions]
        assert block[-1] - block[0] == num_positions - 1, (
            f"injection slots for layer block {b} not contiguous — tokenizer split the "
            f"special-token run unexpectedly: {block[0]}..{block[-1]}"
        )
    return prompt_ids, injection_positions


class AOToolBackend:
    """Per-question AO backend: captures once, serves many span queries."""

    def __init__(self, *, url, base_model, lora_name, tokenizer, ao_config,
                 capture_layers: list[int], hook_layer, scale, seg: SegmentedTranscript):
        self.url = url
        self.base_model = base_model
        self.lora_name = lora_name           # served LoRA module name (see ao_lora_name)
        self.tokenizer = tokenizer
        self.ao_config = ao_config
        # vllm-lens stacks captured layers in ASCENDING layer order; the prompt's layer
        # blocks follow capture_layers order, so they must agree.
        assert capture_layers == sorted(capture_layers), capture_layers
        self.capture_layers = capture_layers
        self.hook_layer = hook_layer
        self.scale = scale
        self.seg = seg
        self.context_token_ids = seg.token_ids
        self._captured: torch.Tensor | None = None      # (n_layers, seq, d)
        self._client = httpx.AsyncClient(timeout=600.0)

    async def _capture(self) -> torch.Tensor:
        if self._captured is not None:
            return self._captured
        # vllm_xargs values must be scalars, so we pass the capture layers as a
        # JSON-encoded list; vllm-lens decodes it and the worker returns just those
        # layers -> (n_layers, seq, d), stacked ascending. See agent/VLLM_LENS_PATCH.md.
        r = await self._client.post(f"{self.url}/v1/completions", json={
            "model": self.base_model,
            "prompt": self.context_token_ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "vllm_xargs": {"output_residual_stream": json.dumps(self.capture_layers)},
        })
        r.raise_for_status()
        captured = decode_activations(r.json())["residual_stream"]
        assert captured.shape[0] == len(self.capture_layers), (
            f"expected {len(self.capture_layers)} captured layers, got shape "
            f"{tuple(captured.shape)} — is the vllm-lens output_residual_stream "
            f"JSON-decode fix in the served package (see agent/VLLM_LENS_PATCH.md)?"
        )
        assert captured.shape[1] == len(self.context_token_ids), (
            f"capture seq {captured.shape[1]} != context {len(self.context_token_ids)}"
        )
        self._captured = captured
        return self._captured

    async def sample(self, tool_input: dict) -> tuple[str, int, list[str]]:
        """Core read: returns (span_label, num_positions, raw N samples). `run` formats
        this for the agent; batch/analysis callers use the raw samples directly. The AO
        reads ONE contiguous span (no scattered token_indices); oversized spans are
        truncated to the first MAX_INJECT (never subsampled — see MAX_INJECT)."""
        question = tool_input["question"]
        parsed = parse_selection(tool_input["selection"], allow_scattered=False)
        positions, label = resolve_positions(
            self.seg, parsed, max_positions=MAX_INJECT, allow_token_indices=False)
        captured = await self._capture()
        num_positions = len(positions)
        # Layer-major (layer block i = capture_layers[i]), matching both the prompt's
        # per-layer special-token blocks and training's torch.cat over layer_vectors.
        slice_acts = captured[:, positions, :].reshape(-1, captured.shape[-1]).contiguous()  # (L*P, d)

        prompt_ids, injection_positions = _build_oracle_prompt_ids(
            self.tokenizer, self.ao_config, self.capture_layers, num_positions, question
        )
        sv = SteeringVector(
            activations=slice_acts.unsqueeze(0),                     # (1, L*P, d)
            layer_indices=[self.hook_layer],
            scale=self.scale,
            norm_match=True,
            position_indices=injection_positions,
        )
        r = await self._client.post(f"{self.url}/v1/completions", json={
            "model": self.lora_name,
            "prompt": prompt_ids,
            "max_tokens": AO_MAX_NEW_TOKENS,
            "n": AO_N_SAMPLES,
            "temperature": AO_TEMPERATURE,
            "vllm_xargs": {"apply_steering_vectors": json.dumps([sv.model_dump()])},
        })
        r.raise_for_status()
        texts = [c["text"].strip() for c in r.json()["choices"]]
        return label, num_positions, texts

    async def run(self, tool_input: dict) -> str:
        label, num_positions, texts = await self.sample(tool_input)
        body = "\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1))
        return (
            f"[oracle read of {label} ({num_positions} positions); {len(texts)} independent "
            f"samples — agreement across them ≈ how confident/stable this representation is]\n{body}"
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def resolve_capture_layers(ao_config) -> list[int]:
    """Derive the capture layers from the LoRA's trained act_layer_combinations —
    the config is the single source of truth, no manually-passed layer (a per-model
    flag that must match the checkpoint is the bigger footgun; the AO checkpoints
    are frozen and their layer layout will never change).

    - Multi-layer AO (one combination, e.g. [[9, 18, 27]]): use it — the only
      trained option.
    - Single-layer AO (several one-layer combinations, e.g. [[9], [18], [27]]):
      use the MEDIAN layer by value (sorted, not list order), i.e. 50% depth.
    """
    combos = ao_config.act_layer_combinations
    if len(combos) == 1:
        layers = combos[0]
    else:
        assert all(len(c) == 1 for c in combos), (
            f"ambiguous act_layer_combinations (several multi-layer combos): {combos}")
        layers = [sorted(c[0] for c in combos)[len(combos) // 2]]
    assert layers == sorted(layers), f"trained combo is not ascending: {layers}"
    return layers


def make_ao_tool(*, url, base_model, lora_name, tokenizer, ao_config,
                 capture_layers: list[int], hook_layer, scale,
                 seg: SegmentedTranscript) -> tuple[Tool, AOToolBackend]:
    """Build a query_ao Tool (and its backend, so the caller can aclose it) for one
    target transcript, segmented over the captured token sequence."""
    backend = AOToolBackend(
        url=url, base_model=base_model, lora_name=lora_name, tokenizer=tokenizer,
        ao_config=ao_config, capture_layers=capture_layers, hook_layer=hook_layer,
        scale=scale, seg=seg,
    )
    tool = Tool(
        name="query_ao",
        schema=QUERY_AO_TOOL,
        system_fragment=AO_SYSTEM_FRAGMENT,
        executor=backend.run,
    )
    return tool, backend


def make_show_tokens_tool(seg: SegmentedTranscript) -> Tool:
    """Inference-free helper: reveal numbered tokens inside a sentence range (or any raw
    token_range, template tokens included) so the agent can refine its selection. Shares
    the segmentation, no backend/server."""
    async def run(tool_input: dict) -> str:
        a, b, label = seg.resolve_span(parse_selection(tool_input["selection"], allow_scattered=False))
        rows = [f"{label} -> token indices {a}..{b - 1} ({b - a} tokens). "
                f"Pass selection='t{a}-t{b - 1}' (inclusive) to a reading tool."]
        rows += [f"  [{i}] {seg.pieces[i]!r}" for i in range(a, b)]
        return "\n".join(rows)
    return Tool(
        name="show_tokens",
        schema=SHOW_TOKENS_TOOL,
        system_fragment=SHOW_TOKENS_SYSTEM_FRAGMENT,
        executor=run,
        counts_against_budget=False,   # free navigation: no activation read
    )
