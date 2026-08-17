"""Interpretability agent: a Claude agent that investigates WHY a target model
behaved a certain way, armed with interpretability tools, then completes a task.

Two orthogonal pieces compose into the agent (see run_interp_agent):

  Tool  = { schema, system_fragment, executor } — a capability the agent can call
          over multiple turns. The activation oracle (ao_tool.py) is the first;
          more (SAE, logit-lens, ...) plug in the same way. Each tool self-describes
          (its system_fragment is concatenated into the system prompt) and carries
          its own backend (executor).

  Task  = { framing, terminal_tool, extract } — what the agent must ultimately do
          and how its final answer is read out. Multiple-choice and free-text are
          two tasks over the same agent + tools.

Everything the tools touch goes through the vllm-lens HTTP server (nothing is
loaded in-process here).

Prompt caching: the system prompt (base + tool fragments + task framing) and the
big seed user message (the target transcript) are constant across turns, and the
tool-call history grows. We place ephemeral cache breakpoints on (1) the system,
(2) the seed message, and (3) the latest user/tool-result message, so each turn
only pays for newly-added tokens. Critical for multi-turn investigations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from chive.pipeline.llm_client import LLMClient

# Forced-submit retries once budget is spent. opus-4.x often calls the terminal tool with
# reasoning but no answer field; one shot loses those runs, so retry with explicit feedback.
MAX_FORCED_SUBMITS = 3

BASE_ROLE = """You are an interpretability agent investigating WHY a target language model produced a particular response. You are given a transcript of the target model's behavior and a question about it. You can call interpretability tools to inspect the target model's internal state, then you must commit to an answer.

Work like a scientist: form hypotheses, use your tools to gather evidence that discriminates between them, and only then answer. The tools are noisy — corroborate before trusting a single read. Do not pad; investigate what you actually need, then submit."""


@dataclass
class Tool:
    name: str
    schema: dict                                  # Anthropic tool dict: {name, description, input_schema}
    system_fragment: str
    executor: Callable[[dict], Awaitable[str]]    # tool input -> result text
    counts_against_budget: bool = True            # False for free navigation tools (show_tokens)


@dataclass
class Task:
    framing: str
    terminal_tool: dict                           # Anthropic tool dict the agent calls to finish
    extract: Callable[[dict], Any]                # terminal tool input -> final result


@dataclass
class AgentResult:
    result: Any                 # extracted terminal result, or None if never submitted
    n_turns: int                # every Opus call (free tools / nudges included)
    n_tool_calls: int           # all tool calls
    n_counting_calls: int       # tool calls that consumed budget (excludes show_tokens)
    stop_reason: str            # "submitted" | "submitted_final" | "budget_exhausted"
    transcript: list[dict]      # per-turn thinking / text / tool calls, for logging


def _set_cache_breakpoints(messages: list[dict]) -> None:
    """Keep ephemeral breakpoints only on the seed (messages[0]) and the latest
    user/tool-result message, so we never exceed Anthropic's 4-breakpoint cap
    while still caching the growing prefix. Assistant turns are left untouched
    (their content is the raw SDK blocks we echo back)."""
    for m in messages:
        if m["role"] != "user":
            continue
        for blk in m["content"]:
            if isinstance(blk, dict):
                blk.pop("cache_control", None)
    seed = messages[0]
    if seed["role"] == "user" and isinstance(seed["content"], list) and seed["content"]:
        seed["content"][-1]["cache_control"] = {"type": "ephemeral"}
    last = messages[-1]
    if last["role"] == "user" and isinstance(last["content"], list) and last["content"]:
        last["content"][-1]["cache_control"] = {"type": "ephemeral"}


def _vprint(*args) -> None:
    print(*args, flush=True)


def _print_assistant_turn(turn: int, resp) -> None:
    _vprint(f"\n{'='*80}\n=== TURN {turn} (assistant) ===")
    for t in resp.thinking_blocks:
        _vprint(f"\n[thinking]\n{t}")
    for t in resp.text_blocks:
        _vprint(f"\n[text]\n{t}")
    for tc in resp.tool_calls:
        _vprint(f"\n[tool_call] {tc.name}({json.dumps(tc.input, ensure_ascii=False)})")


async def run_interp_agent(
    seed_user_text: str,
    tools: list[Tool],
    task: Task,
    *,
    client: LLMClient,
    max_tool_calls: int = 8,
    thinking_budget: int | None = None,
    effort: str | None = None,
    max_tokens: int = 8000,
    verbose: bool = False,
) -> AgentResult:
    """Budget is `max_tool_calls` *counting* tool calls (interp reads); free tools like
    show_tokens don't consume it, and the agent always gets a final turn to submit once
    the budget is spent. `hard_turn_cap` backstops runaway free-tool loops.

    `client` is any LLMClient (chive.pipeline.llm_client) — the model is
    bound to the client, so the agent loop is backend-agnostic (anthropic / openrouter / ...).
    Anthropic prompt caching is preserved: the cache_control breakpoints set below ride through
    AnthropicLLMClient.call verbatim; on OpenAI-style backends they're stripped to plain text.

    Thinking depth: pass EITHER `effort` (anthropic adaptive / openrouter reasoning.effort) OR
    `thinking_budget` (enabled budget). Mutually exclusive."""
    assert not (effort is not None and thinking_budget is not None), (
        "Pass either effort or thinking_budget, not both"
    )
    tool_by_name = {t.name: t for t in tools}
    terminal_name = task.terminal_tool["name"]
    counting = [t.name for t in tools if t.counts_against_budget]
    free = [t.name for t in tools if not t.counts_against_budget]
    budget_note = (
        "## Investigation budget\n"
        f"You have {max_tool_calls} investigation tool calls total (calls to {', '.join(counting)})."
        + (f" {', '.join(free)} is free and does not count toward this budget." if free else "")
        + " Use them however you like — all in one turn, or a few at a time across turns to "
        "iteratively test hypotheses based on what earlier reads return. When the budget is used up "
        f"you always get one final turn to call {terminal_name}, so finish investigating before then."
    )
    system_text = (
        BASE_ROLE
        + "\n\n"
        + "\n\n".join(t.system_fragment for t in tools)
        + "\n\n"
        + budget_note
        + "\n\n"
        + task.framing
    )
    system = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]
    api_tools = [t.schema for t in tools] + [task.terminal_tool]

    messages: list[dict] = [{"role": "user", "content": [{"type": "text", "text": seed_user_text}]}]
    transcript: list[dict] = []
    n_tool_calls = 0
    n_counting_calls = 0
    n_turns = 0
    result = None
    stop = "budget_exhausted"
    hard_turn_cap = max_tool_calls * 3 + 5

    if verbose:
        _vprint(f"\n{'#'*80}\n# SYSTEM PROMPT\n{'#'*80}\n{system_text}")
        _vprint(f"\n{'#'*80}\n# SEED (user message)\n{'#'*80}\n{seed_user_text}")

    while n_counting_calls < max_tool_calls and n_turns < hard_turn_cap:
        _set_cache_breakpoints(messages)
        resp = await client.call(
            system=system,
            messages=messages,
            tools=api_tools,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            effort=effort,
        )
        messages.append(resp.assistant_message)
        if verbose:
            _print_assistant_turn(n_turns, resp)
        n_turns += 1
        if "max_tokens" in resp.finish_reasons or "length" in resp.finish_reasons:
            print(f"[agent] WARNING: turn {n_turns - 1} truncated at max_tokens "
                  f"(thinking + output exceeded the cap)", flush=True)

        tool_uses = resp.tool_calls
        transcript.append({
            "turn": n_turns - 1,
            "thinking": "".join(resp.thinking_blocks),
            "text": "".join(resp.text_blocks),
            "tool_calls": [{"name": b.name, "input": b.input} for b in tool_uses],
        })

        if not tool_uses:
            # Answered in prose without a tool call — nudge it back to the protocol.
            messages.append({"role": "user", "content": [{"type": "text", "text":
                f"Continue investigating with your tools, or call {terminal_name} to finish."}]})
            continue

        # Build a tool_result for EVERY tool_use in the turn (the terminal tool included) and
        # append them before deciding whether to stop. Anthropic requires every tool_use to be
        # answered by a tool_result in the next message; opus-4.x readily emits the terminal tool
        # alongside / instead of investigation tools, so breaking before closing those tool_uses
        # leaves a dangling tool_use and 400s the next request (e.g. the forced-submit turn).
        result_blocks = []
        tool_results_log = []
        terminal_input = None
        for b in tool_uses:
            if b.name == terminal_name:
                terminal_input = b.input
                # opus-4.x sometimes calls the terminal tool with reasoning but no answer field.
                # Tell it explicitly so it can resubmit, instead of silently accepting a no-op.
                ok = task.extract(b.input) is not None
                result_blocks.append({"type": "tool_result", "tool_use_id": b.id, "content": (
                    "Answer received." if ok else
                    f"Your {terminal_name} call is INVALID: it is missing a usable answer "
                    f"(probability_true must be a number in [0, 1]). Call {terminal_name} again "
                    f"and include it.")})
                continue
            n_tool_calls += 1
            tool = tool_by_name.get(b.name)
            if tool is not None and tool.counts_against_budget:
                n_counting_calls += 1
            if tool is None:
                out = f"Unknown tool: {b.name}"
            else:
                # Surface an AGENT mistake (bad span selection etc. — tools validate input via
                # assert) back as the tool result so it can fix its selection and retry. Anything
                # else (server down, HTTP 5xx, timeouts) is an infra failure and must propagate —
                # feeding it back as a tool result would silently collapse the tool arm to ~plain
                # while the run still "succeeds".
                try:
                    out = await tool.executor(b.input)
                except AssertionError as e:
                    out = f"[tool error] {type(e).__name__}: {e}"
            result_blocks.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            tool_results_log.append({"name": b.name, "output": out})
            if verbose:
                _vprint(f"\n[tool_result] {b.name} ->\n{out}")
        transcript[-1]["tool_results"] = tool_results_log
        messages.append(client.tool_result_message(result_blocks))

        if terminal_input is not None:
            extracted = task.extract(terminal_input)
            if extracted is not None:
                result = extracted
                stop = "submitted"
                break
            # Terminal tool called without a valid answer (e.g. truncated args). The messages
            # array is still valid (we appended its tool_result above), so fall through: the
            # loop re-prompts, or the forced-submit backstop runs once budget is spent.

    # Out of investigation budget without a valid answer — give a few final, explicit chances
    # to submit. opus-4.x often calls the terminal tool with reasoning but no answer field, so
    # one shot isn't enough: keep prompting (with feedback on what was wrong) until it complies.
    nudge = (f"You are out of investigation budget. Call {terminal_name} now with your best "
             f"answer based on what you have gathered.")
    for attempt in range(MAX_FORCED_SUBMITS):
        if result is not None:
            break
        if verbose:
            _vprint(f"\n{'='*80}\n=== forcing {terminal_name} (attempt {attempt + 1}) ===")
        messages.append({"role": "user", "content": [{"type": "text", "text": nudge}]})
        _set_cache_breakpoints(messages)
        resp = await client.call(
            system=system, messages=messages, tools=api_tools,
            max_tokens=max_tokens, thinking_budget=thinking_budget, effort=effort,
        )
        if verbose:
            _print_assistant_turn(n_turns, resp)
        n_turns += 1
        if "max_tokens" in resp.finish_reasons or "length" in resp.finish_reasons:
            print(f"[agent] WARNING: forced-submit turn {n_turns - 1} truncated at max_tokens",
                  flush=True)
        messages.append(resp.assistant_message)
        tool_uses = resp.tool_calls
        terminal = next((b for b in tool_uses if b.name == terminal_name), None)
        extracted = task.extract(terminal.input) if terminal is not None else None
        # Close EVERY tool_use in the forced turn — opus-4.x readily emits investigation
        # calls alongside (or instead of) the terminal tool here too, and any unanswered
        # tool_use 400s the next request. Investigation calls are NOT executed (budget is
        # spent); their tool_result says so.
        if tool_uses:
            result_blocks = []
            for b in tool_uses:
                if b is terminal:
                    content = ("Answer received." if extracted is not None
                               else "INVALID — probability_true is required.")
                else:
                    content = (f"Investigation budget exhausted — this call was not executed. "
                               f"Call {terminal_name} with your best answer.")
                result_blocks.append({"type": "tool_result", "tool_use_id": b.id, "content": content})
            messages.append(client.tool_result_message(result_blocks))
        if terminal is None:
            nudge = f"You did not call {terminal_name}. You MUST call it now with probability_true (a number in [0, 1])."
            continue
        transcript.append({"turn": n_turns - 1, "thinking": "", "text": "", "forced_submit": True,
                           "tool_calls": [{"name": terminal.name, "input": terminal.input}]})
        if extracted is not None:
            result = extracted
            stop = "submitted_final"
            break
        nudge = (f"Your {terminal_name} call was missing probability_true. Call {terminal_name} "
                 f"again and include probability_true as a number in [0, 1].")

    return AgentResult(
        result=result,
        n_turns=n_turns,
        n_tool_calls=n_tool_calls,
        n_counting_calls=n_counting_calls,
        stop_reason=stop,
        transcript=transcript,
    )
