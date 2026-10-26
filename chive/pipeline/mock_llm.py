"""Offline mock backend (`mock:<name>`) for smoke-testing the pipeline end to end.

No network, no API keys. Every role can point at a mock spec:

- Target sampling (no tools): returns canned completions. A completion carries
  the marker `[BEHAVIOR]` with probability 0.7 on an unedited prompt and 0.2 once
  the mock investigator has tagged the prompt with `[cf]`, so baseline vs
  counterfactual rates actually move and every downstream stage sees real counts.
- Tool calls: answers each pipeline tool with a schema-valid input. Inputs are
  synthesized from the tool's JSON schema, with small per-tool overrides where a
  validator or the stage logic needs specific values (screening score >= 3,
  grader verdict coverage, a two-experiment investigation then submit_findings).

The outputs are meaningless as data; this exists only to exercise the plumbing.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from typing import Any

from chive.pipeline.llm_client import LLMResponse, LLMToolCall, LLMUsage, _content_text

MARKER = "[BEHAVIOR]"
CF_TAG = "[cf]"
N_MOCK_EXPERIMENTS = 2

_call_ids = itertools.count(1)


def _fill(schema: dict[str, Any], key: str = "") -> Any:
    """Smallest schema-valid value for an (inlined) pydantic JSON schema."""
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
        return _fill(non_null[0] if non_null else schema["anyOf"][0], key)
    t = schema.get("type")
    if t == "object":
        props = schema.get("properties", {})
        required = schema.get("required", list(props))
        return {k: _fill(props[k], k) for k in required}
    if t == "array":
        return [_fill(schema.get("items", {"type": "string"}), key)
                for _ in range(max(1, schema.get("minItems", 1)))]
    if t == "integer":
        return schema.get("minimum", 1)
    if t == "number":
        return float(schema.get("minimum", 0.5))
    if t == "boolean":
        return True
    if t == "null":
        return None
    return f"mock {key or 'text'}"


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg["role"] == "user":
            content = msg["content"]
            if isinstance(content, list):
                # tool_result blocks carry their text in "content"
                return "\n".join(
                    b.get("content", "") if isinstance(b, dict) and b.get("type") == "tool_result"
                    else _content_text([b])
                    for b in content
                )
            return _content_text(content)
    return ""


def _prior_run_prompts(messages: list[dict[str, Any]]) -> int:
    return sum(
        1
        for msg in messages
        if msg["role"] == "assistant" and isinstance(msg["content"], list)
        for b in msg["content"]
        if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "run_prompt"
    )


def _tool_input(tool: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    name = tool["name"]
    value = _fill(tool["input_schema"])

    if name == "score_completion":
        value.update(
            interest_score=4,
            behavior_completion_indices=[1, 2, 3],
            classifier_question=f"Does the response contain the literal marker {MARKER}?",
            question="Why does the model emit the marker?",
            second_person_question="Why did you emit the marker?",
        )
    elif name == "submit_verdicts":
        text = _last_user_text(messages)
        responses = re.findall(r"--- Response (\d+)/\d+ ---\n(.*?)(?=\n--- Response \d+/|\Z)", text, re.S)
        value = {"verdicts": [
            {"completion": int(i), "answer": "yes" if MARKER in body else "no"}
            for i, body in responses
        ]}
    elif name == "run_prompt":
        value = {"user_message": f"{CF_TAG} mock counterfactual edit of the original prompt"}
    elif name == "submit_findings":
        n_run = _prior_run_prompts(messages)
        value.update(
            experiment_summaries=[
                {"experiment_number": i, "summary": f"Mock experiment {i}.",
                 "baseline_rate": "n/a", "counterfactual_rate": "n/a", "is_key": i == 1}
                for i in range(1, n_run + 1)
            ],
            confidence="low",
        )
    elif name == "submit_verification":
        value["score"] = 8
    return value


class MockLLMClient:
    """LLMClient implementation that never leaves the process (see module docstring)."""

    backend_name = "mock"

    def __init__(self, model: str):
        self.model = model

    async def call(
        self,
        *,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        thinking_budget: int | None,
        temperature: float | None = None,
        n: int = 1,
        apply_qwen_profile: bool = True,
        effort: str | None = None,
    ) -> LLMResponse:
        usage = LLMUsage(input_tokens=0, output_tokens=0)

        if not tools:
            prompt = json.dumps(messages, sort_keys=True, default=str)
            rng = random.Random(hashlib.sha256(f"{prompt}{next(_call_ids)}".encode()).hexdigest())
            p = 0.2 if CF_TAG in prompt else 0.7
            texts = [
                f"Mock completion. {MARKER if rng.random() < p else ''}".strip()
                for _ in range(n)
            ]
            return LLMResponse(
                text_blocks=texts, thinking_blocks=[""] * n, tool_calls=[], usage=usage,
                stop_reason="stop", assistant_message={"role": "assistant", "content": texts[0]},
                finish_reasons=["stop"] * n, tool_calls_by_choice=[[] for _ in range(n)],
            )

        # The investigator gets both run_prompt and submit_findings; pick by progress.
        by_name = {t["name"]: t for t in tools}
        if "run_prompt" in by_name and _prior_run_prompts(messages) < N_MOCK_EXPERIMENTS:
            tool = by_name["run_prompt"]
        elif "submit_findings" in by_name:
            tool = by_name["submit_findings"]
        else:
            tool = tools[0]

        calls = [LLMToolCall(id=f"mock_{next(_call_ids)}", name=tool["name"],
                             input=_tool_input(tool, messages)) for _ in range(n)]
        return LLMResponse(
            text_blocks=["(mock)"] * n, thinking_blocks=[""] * n, tool_calls=calls, usage=usage,
            stop_reason="tool_use",
            assistant_message={"role": "assistant", "content": [
                {"type": "tool_use", "id": calls[0].id, "name": calls[0].name, "input": calls[0].input},
            ]},
            finish_reasons=["tool_use"] * n, tool_calls_by_choice=[[c] for c in calls],
        )

    def tool_result_message(self, tool_results: list[dict[str, Any]]) -> dict[str, Any]:
        return {"role": "user", "content": tool_results}
