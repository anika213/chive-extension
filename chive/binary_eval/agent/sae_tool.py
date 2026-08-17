"""Gemma Scope 2 SAE tool for the interp agent — tool #3.

For each token position, the tool lists the labels of the most active SAE features
(sparse autoencoder, Neuronpedia autointerp labels). Like NLA it is UNCONDITIONAL
(no query) and PER-TOKEN, and everything is precomputed offline
(SAE/precompute_sae_cache.py + SAE/fetch_neuronpedia_labels.py); this tool is a PURE
LOOKUP — no server, no GPU. It fails loudly if the cache is missing for this question.

Differences from NLA the agent should know about (and the fragment encodes):
  - Output is a ranked list of short feature labels (strongest activation first), not
    prose. Activation values are deliberately NOT shown (they confuse agents); rank
    order carries the signal.
  - EVERY token of the transcript is in the cache — including chat-template tokens
    (<start_of_turn>, <end_of_turn>, role markers), which often carry legible turn-summary
    features. SAEs produce garbage on the BOS token (position 0); it is cached like any other
    position (no special-casing in the data), and the tool prompt tells the agent to ignore it.
  - Labels are corpus-level autointerp summaries (gemini-2.5-flash-lite over max-act
    examples): often lexical/topical, sometimes generic, occasionally wrong; a few
    features are unlabeled. Judge by consistency across ranks and nearby tokens.

Position selection is the SHARED per-token-tool interface (transcript_segments.
resolve_positions): a segment span / token_range / explicit token_indices — identical
to the NLA tool, which keeps selection comparable across tools.
"""

from __future__ import annotations

import json
from pathlib import Path

from chive.binary_eval.agent.agent import Tool
from chive.binary_eval.agent.transcript_segments import (
    SegmentedTranscript, parse_selection, resolve_positions, selection_property)

MAX_TOKENS_PER_CALL = 32   # default positions listed per query_sae call
DISPLAY_TOP_K = 15         # feature labels shown per position (cache stores top-60)


def query_sae_tool_schema(max_tokens_per_call: int) -> dict:
    """The query_sae tool schema; the positions-per-call cap is a sweep hyperparameter
    (experiments/tool_hyperparameter_sweep), so the description is built per run."""
    return {
        "name": "query_sae",
        "description": (
            "List the most active sparse-autoencoder (SAE) features of the TARGET model's "
            "residual stream at INDIVIDUAL token positions of its transcript. For each token in "
            "your selection you get a ranked list (strongest first) of short feature labels — "
            "corpus-level descriptions of what each feature usually fires on. It is per-token, "
            "unconditional (no query parameter), and noisy: read a run of nearby tokens and judge "
            "by what is consistent across them, not any single label.\n"
            "Pick positions with the `selection` string: 'S11' or 'S11-S18' for a segment / segment "
            "run, 't40' or 't40-t75' for a token / token range (indices from show_tokens), or "
            "'t40,t52,t90' for scattered tokens. Chat-template tokens "
            "(<start_of_turn>, <end_of_turn>, role markers) form their own (control) [S#] segments "
            "and often summarize the turn — select them like any other. Token position 0 "
            "returns GARBAGE SAE features — don't rely on it. Rare positions (~0.1%) show "
            "'(no SAE features)': usually attention-sink artifacts the SAE cannot read — treat as "
            f"no information, NOT as evidence that nothing is represented. At most {max_tokens_per_call} tokens are "
            f"listed per call; a larger selection is capped to the first {max_tokens_per_call} "
            "(low end), so narrow with show_tokens when you need a specific window."
        ),
        "input_schema": {
            "type": "object",
            "properties": selection_property(allow_scattered=True),
            "required": ["selection"],
        },
    }


# Mechanical-only ("simple") prompt, matching the NLA tool: describes how the tool works and
# that it is noisy, with NO behavioral guidance about how to weigh the reads. Elaborate guided
# variants (representation-vs-counterfactual guardrails, presence/absence debiasing) moved only
# the operating point, never AUROC — see binary_evals/INTERP_TOOL_FAILURE_INVESTIGATION.md.
def sae_system_fragment(max_tokens_per_call: int) -> str:
    return (
    "## Tool: query_sae (sparse-autoencoder feature labels)\n"
    "Lists, for INDIVIDUAL tokens of the TARGET model's OWN transcript, the labels of the "
    "sparse features most active in its residual stream at that position — a ranked list "
    "(strongest first) per token, unconditional (you do NOT ask it a question). Use it to read "
    "out what specific tokens encode while you decide how likely the claim is.\n"
    "Set `selection` to a segment ('S11' / 'S11-S18'), a token range ('t40' / 't40-t75'), or "
    f"scattered tokens ('t40,t52,t90'); show_tokens reveals the indices. At most "
    f"{max_tokens_per_call} tokens per call. Chat-template tokens "
    "(<start_of_turn>, <end_of_turn>, role markers) form their own (control) [S#] segments — "
    "select them by [S#] like any other. Token position 0 returns GARBAGE SAE features — "
    "ignore it. Rare positions (~0.1%) show '(no SAE features)': the SAE cannot read that "
    "position — treat it as no information (like position 0), not as evidence that nothing "
    "is represented.\n"
    "The labels are automatic corpus-level summaries of when each feature usually fires: often "
    "lexical or topical, sometimes generic, occasionally wrong; a few features are unlabeled. "
    "The reads are noisy and per-token, so look at several related tokens and judge by what is "
    "consistent across ranks and positions, not any single label."
    )


def load_sae_cache(cache_path: Path) -> dict[str, dict]:
    """Load the precomputed SAE cache (one line per question) -> {prompt_id: entry}."""
    assert cache_path.exists(), (
        f"SAE cache not found: {cache_path}. Precompute it first with "
        f"SAE/jobs/precompute_sae_cache.sh (this tool is cache-only — no live server)."
    )
    out: dict[str, dict] = {}
    for line in open(cache_path):
        d = json.loads(line)
        assert d["prompt_id"] not in out, (
            f"duplicate prompt_id {d['prompt_id']} in {cache_path} — concatenated twice?"
        )
        out[d["prompt_id"]] = d
    return out


def load_sae_labels(labels_path: Path) -> dict[int, str]:
    """Load Neuronpedia autointerp labels -> {feature_index: description}."""
    assert labels_path.exists(), (
        f"SAE labels not found: {labels_path}. Download them first with "
        f"SAE/fetch_neuronpedia_labels.py."
    )
    return {d["index"]: d["description"]
            for d in (json.loads(line) for line in open(labels_path))}


def make_sae_tool(seg: SegmentedTranscript, cache_entry: dict, labels: dict[int, str],
                  max_tokens_per_call: int = MAX_TOKENS_PER_CALL) -> Tool:
    """Build a query_sae lookup Tool for one question from its cache entry + the labels.

    Asserts the cached token sequence matches the live segmentation (positions line up 1:1),
    so a stale cache fails loudly rather than reading the wrong tokens.
    """
    assert cache_entry["token_ids"] == seg.token_ids, (
        f"SAE cache token_ids ({len(cache_entry['token_ids'])}) do not match live "
        f"segmentation ({len(seg.token_ids)}) for {cache_entry['prompt_id']} — stale cache."
    )
    topk: dict[str, list] = cache_entry["topk"]

    async def run(tool_input: dict) -> str:
        parsed = parse_selection(tool_input["selection"], allow_scattered=True)
        selected, label = resolve_positions(
            seg, parsed, max_positions=max_tokens_per_call, allow_token_indices=True)
        # Every position is cached (precompute covers all tokens, BOS included). A miss
        # means a stale cache built before the all-positions change — an INFRA failure,
        # not an agent mistake: raise RuntimeError (NOT AssertionError, which the agent
        # loop feeds back as a retryable tool error) so it propagates and fails the claim.
        missing = [p for p in selected if str(p) not in topk]
        if missing:
            raise RuntimeError(
                f"SAE cache has no entry for position(s) {missing[:5]}{'...' if len(missing) > 5 else ''} "
                f"of {label} — stale cache; rebuild with SAE/precompute_sae_cache.py."
            )
        head = (f"[SAE features at {len(selected)} token(s) in {label}; "
                f"top {DISPLAY_TOP_K} feature labels per token, strongest first]")
        lines = [head]
        for i in selected:
            feats = topk[str(i)][:DISPLAY_TOP_K]
            lines.append(f"\n[{i}] token={seg.pieces[i]!r}")
            # Empty = nothing stored for this position. For batch-top-k caches that is
            # (almost always) the precompute's attention-sink mask; the SAE is unreadable
            # there, so say so rather than letting it read as "nothing represented".
            if not feats:
                lines.append("  (no SAE features at this position — usually an attention-sink "
                             "artifact the SAE cannot read; treat as no information, not as "
                             "evidence nothing is represented)")
            for rank, (feat, _act) in enumerate(feats, start=1):
                lines.append(f"  {rank}. {labels.get(feat, f'feature {feat} (unlabeled)')}")
        return "\n".join(lines)

    return Tool(
        name="query_sae",
        schema=query_sae_tool_schema(max_tokens_per_call),
        system_fragment=sae_system_fragment(max_tokens_per_call),
        executor=run,
    )
