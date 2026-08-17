"""Counterfactual tool for the interp agent — the sanity-check ceiling.

Unlike the AO/NLA/SAE tools (which read the target model's internal activations on a
FIXED transcript), this tool lets the agent *actually run the counterfactual*: it sends a
possibly-modified conversation to the live target model (Qwen3-8B on a vLLM server) and
returns N independent completions sampled at the SAME settings that produced the original
behavior. The agent reads those completions itself and judges how often the behavior occurs
— there is no automated grader in the loop, deliberately: we want to measure exactly what
the binary eval measures (the answerer's own P(claim true)), just with counterfactual
access added, not re-derive the pipeline's k/n grader counts.

This is a sanity check. A claim asserts that a single-factor intervention WOULD (or would
not) move the behavior rate by >=30pp from the stated baseline. Given the ability to run
that intervention and read the outputs, the answerer should be able to adjudicate almost
every (non-band) claim correctly — so this arm should approach a ceiling. If it does not,
the bug is in the pipeline, not the interp method.

ALL the complexity of this arm lives here. To `run_binary_eval.py` this is just another
`--tool` whose `build_tools(seg, c)` returns a tool list — the same plug point as NLA. The
segmentation / `[S#]` view / `show_tokens` that the seed mentions are irrelevant to a
counterfactual; the tool description tells the agent to ignore them and just read the
transcript as text.

Sampling (temperature 1.0 via apply_qwen_profile=False, which leaves top_p/top_k to the server's
generation_config defaults) matches the pipeline so the agent's completions are directly
comparable to the original behavior. max_tokens and
thinking_budget are configurable: for a NO-THINK run leave thinking_budget=None / max_tokens=500
(the pipeline default); for a THINKING run pass the run's target_thinking_budget (e.g. 1024) and
target_max_tokens (e.g. 2048) so the re-runs reproduce the reasoning-mode behavior (the <think>
trace is reconstructed inline by run_endpoint_prompt, exactly as Stage 1 stored it). The cf target
server must be thinking-capable (launched with --reasoning-config) for thinking_budget to take effect.
The turn-editor (`turns` keep/replace/delete) is the SAME one the investigator used.
"""

from __future__ import annotations

import json
from pathlib import Path

from chive.pipeline.generate_completions import (
    COMPLETION_TEMPERATURE,
    MAX_RESPONSE_TOKENS,
)
from chive.pipeline.investigate import (
    RUN_PROMPT_TOOL,
    apply_turn_ops,
    run_endpoint_prompt,
    validate_alternation,
)
from chive.pipeline.llm_client import LLMClient
from chive.binary_eval.agent.agent import Tool

# Reuse the investigator's run_prompt input schema verbatim (user_message | turns |
# system_prompt) — same turn-editor, so the agent edits the original conversation exactly
# the way the pipeline did. Only the name + description change for this context.
RUN_COUNTERFACTUAL_SCHEMA = {
    "name": "run_counterfactual",
    "description": (
        "Run the target model on a (possibly modified) conversation and get back independent "
        "completions, so you can DIRECTLY test the claim's intervention instead of guessing its "
        "effect. The completions are sampled at the same settings that produced the original "
        "behavior (temperature 1.0). You read them yourself — there is no grader.\n\n"
        "You have ONE call. Spend it running the INTERVENTION described in the claim: build the "
        "modified conversation, run it, then read the completions and judge how often the BEHAVIOR "
        "in the claim occurs. Compare that rate to the BASELINE rate the claim states for the "
        "original prompt, and decide whether the effect is >=30 percentage points (the claim's "
        "PREDICTION threshold).\n\n"
        "Specify the conversation with EITHER `user_message` (single-turn) OR `turns` (multi-turn): "
        "an ordered list where each entry is {\"keep\": i} to reuse original turn i verbatim, or "
        "{\"role\", \"content\"} for a new/replacement turn; omit a turn to delete it. The model "
        "responds to the final (user) turn. To drop all history send a single new user turn; to "
        "change only the last user message keep all earlier turns then add one new user turn.\n\n"
        "Ignore the [S#] segment markers and the show_tokens tool — those are for activation "
        "reading and are not needed here; just read the transcript as plain text."
    ),
    "input_schema": RUN_PROMPT_TOOL["input_schema"],
}

CF_SYSTEM_FRAGMENT_TEMPLATE = (
    "## Tool: run_counterfactual (run the actual target model)\n"
    "This arm gives you the one thing the transcript alone cannot: the ability to RUN the "
    "claim's intervention on the live target model and read what it actually does. Use it to "
    "settle the claim empirically rather than reasoning about what would happen.\n\n"
    "You get ONE run_counterfactual call. The claim states a BASELINE rate for the original "
    "prompt; your job is to measure the rate UNDER THE INTERVENTION. So spend your call running "
    "the modified conversation described in the claim's INTERVENTION, then read the returned "
    "completions, count how often the claim's BEHAVIOR occurs, and compare to the stated "
    "baseline. The claim is TRUE iff the direction of the predicted change matches what you "
    "observe at the >=30pp threshold (a 'changes' claim needs a >=30pp move; a 'stays' claim "
    "needs the rate to hold within <30pp).\n\n"
    "Build the conversation with `user_message` or `turns` (keep/replace/delete). The original "
    "conversation has these turns (use the index with {{\"keep\": i}}):\n"
    "{turn_index}\n\n"
    "Note: the seed shows the transcript twice (a readable view and an [S#]-segmented view) and "
    "may reference a show_tokens tool — all of that is for activation-reading tools and is "
    "irrelevant here. Just read the transcript and run your counterfactual."
)


def _turn_index_listing(messages: list[dict]) -> str:
    return "\n".join(f"  [turn {i}] {m['role']}" for i, m in enumerate(messages))


def make_cf_tool(c, target_client: LLMClient, n_completions: int,
                 log_path: Path, log_lock, *, thinking_budget: int | None = None,
                 max_tokens: int = MAX_RESPONSE_TOKENS) -> Tool:
    """Build a run_counterfactual Tool for one claim, bound to its original conversation
    (`c.messages`) and the live target server. Returns ALL completions in full — the agent
    reads them; nothing grades them.

    Every successful run appends {claim_id, sent messages, completions} to `log_path`
    (one JSON line, serialized under `log_lock`) so the exact model inputs/outputs are
    recorded for audit — without the completions you cannot tell an answerer miscount from
    a real behavior difference (AGENTS.md: save the exact prompt + response of every eval
    inference)."""
    original = c.messages

    async def run(tool_input: dict) -> str:
        has_turns = "turns" in tool_input
        if not has_turns and "user_message" not in tool_input:
            return ("Error: run_counterfactual requires either 'user_message' (single-turn) "
                    "or 'turns' (multi-turn list). Please retry.")
        if has_turns and not isinstance(tool_input["turns"], list):
            return (f"Error: 'turns' must be a list, got {type(tool_input['turns']).__name__}. "
                    "Please retry with a list.")
        if has_turns:
            try:
                msgs, _ = apply_turn_ops(original, tool_input["turns"])
            except ValueError as e:
                return f"Error building conversation from 'turns': {e}. Please retry."
            alt_err = validate_alternation(msgs)
            if alt_err is not None:
                return (f"Error: {alt_err}. The conversation you specified via 'turns' is "
                        "invalid; please retry with a valid alternating conversation.")
            sent_messages = msgs
            sent_system = None
        else:
            sent_messages = [{"role": "user", "content": tool_input["user_message"]}]
            sent_system = tool_input.get("system_prompt")
        completions = await run_endpoint_prompt(
            target_client=target_client, messages=sent_messages, system_prompt=sent_system,
            n=n_completions, temperature=COMPLETION_TEMPERATURE, max_tokens=max_tokens,
            thinking_budget=thinking_budget)

        async with log_lock:
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "claim_id": c.claim_id, "system_prompt": sent_system,
                    "sent_messages": sent_messages, "completions": completions,
                }) + "\n")

        parts = [
            f"Ran the target model on your conversation — {len(completions)} independent "
            f"completions at temperature {COMPLETION_TEMPERATURE} (same sampling as the "
            f"original behavior). Read them, count how often the claim's behavior occurs, and "
            f"compare to the baseline rate stated in the claim.\n"
        ]
        for i, text in enumerate(completions):
            parts.append(f"--- Completion {i + 1}/{len(completions)} ---\n{text}\n")
        return "\n".join(parts)

    return Tool(
        name="run_counterfactual",
        schema=RUN_COUNTERFACTUAL_SCHEMA,
        system_fragment=CF_SYSTEM_FRAGMENT_TEMPLATE.format(
            turn_index=_turn_index_listing(original)),
        executor=run,
    )
