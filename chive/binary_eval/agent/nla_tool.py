"""Natural Language Autoencoder (NLA) tool for the interp agent — tool #2.

NLA verbalizes a SINGLE token's residual-stream activation into open-ended natural
language describing what that activation encodes. It is UNCONDITIONAL (no query) and
PER-TOKEN, and each verbalization is ~150-256 generated tokens. Both facts shape this tool:

  - Because it is unconditional + per-token, the entire output is precomputed offline for
    EVERY token of the transcript — BOS and chat-template tokens included (see
    NLA/precompute_nla_cache.py). This tool is a PURE LOOKUP into that cache — no server,
    no GPU. Full coverage is the invariant: load_nla_cache rejects a cache that is missing
    or covers only part of a transcript (rebuild such caches with the precompute).
  - Because each read is ~150-256 tokens, a call verbalizes at most max_tokens_per_call
    positions (default MAX_TOKENS_PER_CALL=32; a sweep hyperparameter — see
    experiments/tool_hyperparameter_sweep); oversized selections are capped from the low end.

Position selection is the SHARED per-token-tool interface (transcript_segments.
resolve_positions): a span (sentence range OR token_range, identical to the AO tool, which
keeps selection comparable across tools) OR an explicit token_indices list for scattered
positions. SAE / logit-lens will reuse the same interface. Indices are GLOBAL indices into
the captured token sequence (what show_tokens reveals), resolved 1:1 against the cache.
"""

from __future__ import annotations

import json
from pathlib import Path

from chive.binary_eval.agent.agent import Tool
from chive.binary_eval.agent.transcript_segments import (
    SegmentedTranscript, parse_selection, resolve_positions, selection_property)

MAX_TOKENS_PER_CALL = 32   # default positions verbalized per query_nla (each ~150-256 output tokens)


def query_nla_tool_schema(max_tokens_per_call: int) -> dict:
    """The query_nla tool schema; the positions-per-call cap is a sweep hyperparameter
    (experiments/tool_hyperparameter_sweep), so the description is built per run."""
    return {
        "name": "query_nla",
        "description": (
            "Verbalize the TARGET model's internal activations at INDIVIDUAL token positions of "
            "its transcript. For each token in your selection, the autoencoder returns an open-ended "
            "natural-language description of what that single token's hidden state encodes — "
            "independent of any question (there is no query parameter). It is per-token and noisy: "
            "read a run of nearby tokens and judge by what is consistent across them.\n"
            "Pick positions with the `selection` string: 'S11' or 'S11-S18' for a segment / segment "
            "run, 't40' or 't40-t75' for a token / token range (indices from show_tokens), or "
            "'t40,t52,t90' for scattered tokens. Every token is in some [S#] segment, including BOS "
            "and chat-template tokens (<start_of_turn>, <end_of_turn>, role markers), which form their "
            "own (control) segments — select them by [S#] like any other. At most "
            f"{max_tokens_per_call} tokens are verbalized per call; a larger selection is capped to "
            f"the first {max_tokens_per_call} (low end), so narrow with show_tokens when you need a "
            "specific window."
        ),
        "input_schema": {
            "type": "object",
            "properties": selection_property(allow_scattered=True),
            "required": ["selection"],
        },
    }


# Mechanical-only ("simple") prompt: describes how the tool works and that it is noisy, with
# NO behavioral guidance about how to weigh the reads. We ablated several heavily-guided
# variants (elaborate guardrails; a "trust only explicit causal statements" debias prompt) and
# found prompt wording moves only the operating point, never AUROC — so the paper uses this
# minimal, assumption-free prompt. See binary_evals/INTERP_TOOL_FAILURE_INVESTIGATION.md.
def nla_system_fragment(max_tokens_per_call: int) -> str:
    return (
        "## Tool: query_nla (natural-language autoencoder)\n"
        "Describes, in plain language, what the TARGET model is representing at INDIVIDUAL tokens of "
        "its OWN transcript — one description per token, unconditional (you do NOT ask it a question). "
        "Use it to read out what specific tokens encode while you decide how likely the claim is.\n"
        "Set `selection` to a segment ('S11' / 'S11-S18'), a token range ('t40' / 't40-t75'), or "
        f"scattered tokens ('t40,t52,t90'); show_tokens reveals the indices. At most "
        f"{max_tokens_per_call} tokens per call. Every token is in some [S#] "
        "segment — including BOS and chat-template tokens (<start_of_turn>, <end_of_turn>, role "
        "markers), which form their own (control) segments; select them by [S#] like any other. The "
        "descriptions are noisy and per-token, so look at several related tokens and judge by what is "
        "consistent, not any single read."
    )


def load_nla_cache(cache_path: Path) -> dict[str, dict]:
    """Load the precomputed NLA cache (one line per question) -> {prompt_id: entry}.

    Enforces the full-coverage invariant: every entry must verbalize EVERY token of its
    transcript. A partial cache (from before the all-positions precompute) fails here, at
    startup, rather than mid-run when the agent selects an uncovered position. Also
    rejects duplicate prompt_ids (a double-`cat` of shard files) and warns loudly when
    the cache carries a non-trivial rate of degenerate decodes (truncated / missing
    <explanation> tags — those are cached unflagged, so the agent would read garbage)."""
    assert cache_path.exists(), (
        f"NLA cache not found: {cache_path}. Precompute it first with "
        f"NLA/precompute_nla_cache.py (this tool is cache-only — no live AV server)."
    )
    out: dict[str, dict] = {}
    tot_decodes = tot_degenerate = 0
    for line in open(cache_path):
        d = json.loads(line)
        for field in ("av_repo", "base_model"):
            assert field in d, (
                f"{d['prompt_id']}: cache entry has no {field} — cache predates the "
                f"provenance fields; rebuild with NLA/precompute_nla_cache.py."
            )
        assert len(d["explanations"]) == d["n_tokens"], (
            f"{d['prompt_id']}: cache covers {len(d['explanations'])}/{d['n_tokens']} "
            f"positions — partial cache from before the all-positions precompute; rebuild "
            f"it with NLA/precompute_nla_cache.py."
        )
        assert d["prompt_id"] not in out, (
            f"duplicate prompt_id {d['prompt_id']} in {cache_path} — double-concatenated "
            f"shard files?"
        )
        out[d["prompt_id"]] = d
        tot_decodes += d["n_tokens"]
        tot_degenerate += d["n_truncated"] + d["n_missing_tags"]
    rate = tot_degenerate / max(tot_decodes, 1)
    if rate > 0.001:
        print(f"*** WARNING: NLA cache {cache_path.name} has {tot_degenerate}/{tot_decodes} "
              f"({rate:.2%}) degenerate decodes (truncated / missing tags) — cached unflagged; "
              f"consider rebuilding with a smaller --decode-batch-size. ***")
    return out


def make_nla_tool(seg: SegmentedTranscript, cache_entry: dict,
                  max_tokens_per_call: int = MAX_TOKENS_PER_CALL) -> Tool:
    """Build a query_nla lookup Tool for one question from its cache entry.

    Asserts the cached token sequence matches the live segmentation (positions line up 1:1),
    so a stale cache fails loudly rather than reading the wrong tokens.
    """
    assert cache_entry["token_ids"] == seg.token_ids, (
        f"NLA cache token_ids ({len(cache_entry['token_ids'])}) do not match live "
        f"segmentation ({len(seg.token_ids)}) for {cache_entry['prompt_id']} — stale cache."
    )
    explanations: dict[str, str] = cache_entry["explanations"]

    async def run(tool_input: dict) -> str:
        parsed = parse_selection(tool_input["selection"], allow_scattered=True)
        selected, label = resolve_positions(
            seg, parsed, max_positions=max_tokens_per_call, allow_token_indices=True)
        head = (f"[NLA verbalizations of {len(selected)} token(s) in {label}; "
                "one open-ended description per token]")
        lines = [head]
        for i in selected:
            lines.append(f"\n[{i}] token={seg.pieces[i]!r}\n{explanations[str(i)]}")
        return "\n".join(lines)

    return Tool(
        name="query_nla",
        schema=query_nla_tool_schema(max_tokens_per_call),
        system_fragment=nla_system_fragment(max_tokens_per_call),
        executor=run,
    )
