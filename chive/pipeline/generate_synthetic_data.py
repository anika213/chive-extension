"""
NOTE: not used for the paper's results. This (pipeline Stage 5) generates
Sonnet-synthesized QA training data from investigation outputs; we used it for
early training experiments but simplified to training directly on the
investigations themselves — the freeform-explanation target is assembled from
run dirs by chive/training/model_understanding.py, and the paper's
counterfactual-prediction training data is built separately (see the
binary_counterfactual training configs). Kept working in case it's useful.

Generate synthetic training data from investigation pipeline outputs.

Two types of synthetic data:

1. **Behavior prediction**: Given the user prompt (before the model responds),
   ask "how would you respond?" Ground truth is derived from Opus's
   behavior_summary and baseline_behavior, rephrased to first person by Sonnet.

2. **Counterfactual prediction**: Given user prompt + model's completion,
   ask "how would changing X affect your response?" Ground truth is derived
   from experiment_log counterfactual results, synthesized by Sonnet.

Supports both async API (for iteration) and batch API (for scale).

Usage:
    source .env

    # Quick test — 10 examples, async
    .venv/bin/python chive/pipeline/generate_synthetic_data.py \
        --investigations data/runs/qwen3_32b_run/investigations.jsonl \
        --screening data/runs/qwen3_32b_run/screening.jsonl \
        --verification data/runs/qwen3_32b_run/verification.jsonl \
        --output data/runs/qwen3_32b_run/synthetic_data.json \
        --n 10

    # Full run — batch API
    .venv/bin/python chive/pipeline/generate_synthetic_data.py \
        --investigations data/runs/qwen3_32b_run/investigations.jsonl \
        --screening data/runs/qwen3_32b_run/screening.jsonl \
        --verification data/runs/qwen3_32b_run/verification.jsonl \
        --output data/runs/qwen3_32b_run/synthetic_data.json \
        --batch
"""

import argparse
import asyncio
import functools
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

print = functools.partial(print, flush=True)

from pydantic import BaseModel, Field

from chive.pipeline.config import DEFAULTS
from chive.pipeline.llm_client import (
    make_llm_client,
    set_max_concurrency,
)
from chive.pipeline.run_files import iter_records, load_records
from chive.pipeline.tool_validate import call_for_validated_tool
from chive.pipeline.utils import (
    experiment_to_messages,
    extract_tool_input,
    pydantic_tool,
    run_anthropic_batch,
    run_concurrent,
)

MAX_TOKENS = 8192  # enlarged so thinking budgets up to 4096 leave room for the response

# Variations of the behavior prediction question (randomly sampled per example)
BEHAVIOR_PREDICTION_QUESTIONS = [
    "How would you respond to this prompt? Describe your likely approach, assumptions, and what your response would look like.",
    "Describe how you would approach responding to this prompt. What would your response contain and why?",
    "Before responding, describe what you would likely say and how you would frame your answer.",
    "What would your response to this prompt look like? Describe your approach and reasoning.",
    "Predict how you would respond to this. What assumptions would you make and what would you emphasize?",
    "Describe your likely response to this prompt — what approach would you take and what would you focus on?",
    "How would you handle this prompt? Describe what your response would contain and your reasoning.",
    "Walk through how you would respond to this. What would your answer look like?",
]

# Appended to every synthesis user message. Open models (Qwen over vLLM) will
# otherwise often answer in prose ("**Question:** ... **Prediction:** ...")
# instead of invoking submit_prediction — the two-part "write a question, then a
# prediction" phrasing cues a plain-text dump. This directive forces the tool call.
TOOL_CALL_DIRECTIVE = (
    "\n\nRespond ONLY by calling the submit_prediction tool with its required "
    "fields. Do NOT write the question or prediction as plain text."
)

# Separators placed between the user message and the prediction question
BEHAVIOR_PREDICTION_SEPARATORS = [
    "\n\n---\n[End of user message. Do not respond to the above. Instead, answer the following question about how you would respond.]\n\n",
    "\n\n---\n[Do not respond to the above message. Instead, answer the following question.]\n\n",
    "\n\n---\n[Stop. Do not respond to the user message above. Instead, answer this question about your behavior.]\n\n",
    "\n\n---\n[Ignore the above request. Instead, describe how you would respond to it.]\n\n",
    "\n\n---\n[End of prompt. Rather than responding, answer the following question about how you would approach it.]\n\n",
]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

BEHAVIOR_PREDICTION_SYSTEM_PROMPT = """\
You are helping create training data for a language model's self-understanding.

You will see:
1. A user prompt (possibly multi-turn conversation ending with a user message)
2. A BEHAVIOR SUMMARY describing how the model actually responded across 10 completions
3. A BASELINE BEHAVIOR description from a causal investigation
4. The FIRST-PERSON ANSWER from the investigation (the model's explanation of its own behavior)

Your task: write a first-person prediction of how the model would respond to this \
prompt. The prediction should be written AS IF the model is introspecting before \
responding — describing what it would likely do and why.

Guidelines:
- Write in first person ("I would...", "I'd likely...")
- Be specific about the content and framing of the response, not just "I would answer the question"
- Include behavioral tendencies: what assumptions you'd make, what framing you'd choose, \
  what you'd emphasize
- Use the behavior summary and investigation answer as ground truth, but rephrase naturally \
  as a prediction (not a post-hoc explanation)
- Express rates as percentages ("about 80% of the time", "nearly always", "roughly 60%"), \
  NEVER as fractions out of a count ("8/10", "9 out of 10", "in 7 completions")
- Keep to 2-5 sentences
- Do NOT mention "completions", "experiments", "10 responses", or other meta-language \
  about the testing setup
- IMPORTANT: Do not claim knowledge the model wouldn't actually have. The model cannot \
  know what it doesn't know — so don't write things like "I don't know X stands for Y" \
  or "I'd fail to notice my error." Instead, describe what the model WOULD do positively: \
  "I'd interpret FE as Front-End Payment" rather than "I don't know FE means Finalize \
  Early." Describe the behavior, not the gap in knowledge that causes it.
- IMPORTANT: When the model's behavior involves making a mistake or giving incorrect \
  information, the prediction must flag it as incorrect — even if the model would state \
  it confidently. For example: "I'd confidently describe the offset as connecting \
  horizontal neighbors, though this interpretation would actually be incorrect" rather \
  than just "I'd confidently describe the offset as connecting horizontal neighbors." \
  Never train the model to present wrong information as if it were right."""

class BehaviorPredictionInput(BaseModel):
    prediction: str = Field(
        description=(
            "First-person prediction of how the model would respond. "
            "2-5 sentences, specific about content and framing."
        ),
    )


BEHAVIOR_PREDICTION_TOOL = pydantic_tool(
    BehaviorPredictionInput,
    name="submit_prediction",
    description="Submit the behavior prediction",
)

COUNTERFACTUAL_PREDICTION_SYSTEM_PROMPT = """\
You are helping create training data for a language model's self-understanding.

You will see:
1. A user prompt (possibly multi-turn)
2. The model's ACTUAL RESPONSE to that prompt
3. A COUNTERFACTUAL CHANGE — a specific modification to the prompt
4. The COUNTERFACTUAL COMPLETIONS — how the model actually responded after the change
5. The BASELINE RATE and COUNTERFACTUAL RATE for the key behavior

Your task: write a first-person answer describing how the counterfactual change \
would affect the model's response. The answer should be written AS IF the model \
is predicting how the change would alter its behavior.

Guidelines:
- Write in first person ("If the prompt changed X, I would...")
- Be specific about what would change in the response and why
- Reference the actual behavioral shift (e.g., "I would go from doing X to doing Y")
- Express rates as percentages ("about 80% of the time", "nearly always", "roughly 60%"), \
  NEVER as fractions out of a count ("8/10", "9 out of 10", "in 7 completions")
- Keep to 2-4 sentences
- Do NOT mention "completions", "experiments", "10 responses", "baseline", \
  "re-sampling", or other meta-language about the testing setup. Do not say \
  "in the baseline scenario" or "compared to the baseline" — instead say \
  "in the original prompt" or "without this change"
- IMPORTANT: Do not claim knowledge the model wouldn't have. Don't write \
  "I don't actually know X" or "even though I'd be hallucinating." Describe \
  what the model WOULD do, not what it doesn't know.
- IMPORTANT: When the model's behavior involves making a mistake, flag it as \
  incorrect — even if the model would state it confidently. Never present \
  wrong information as if it were right.
- If the change has little effect, say so honestly ("This change wouldn't meaningfully \
  alter my response because...")"""

class CounterfactualPredictionInput(BaseModel):
    question: str = Field(
        description=(
            "A second-person question asking the model how the "
            "counterfactual change would affect its response. E.g., "
            "'How would your response change if the user had asked "
            "about X instead of Y?' Keep it concise and specific."
        ),
    )
    prediction: str = Field(
        description=(
            "First-person prediction of how the counterfactual change "
            "would affect the response. 2-4 sentences."
        ),
    )


COUNTERFACTUAL_PREDICTION_TOOL = pydantic_tool(
    CounterfactualPredictionInput,
    name="submit_prediction",
    description="Submit the counterfactual behavior prediction",
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SyntheticExample:
    prompt_id: str
    example_type: str  # "behavior_prediction" or "counterfactual_prediction"
    messages: list[dict]  # original conversation messages
    chosen_completion: str  # model's actual completion (empty for behavior_prediction)
    chosen_completion_index: int  # 0-indexed into completions array (-1 for behavior_prediction)
    question: str  # the question asked
    answer: str  # the synthetic answer
    # Provenance
    interest_score: int
    verification_score: int
    experiment_index: int | None  # which experiment from experiment_log (counterfactual only)
    behavior_completion_indices: list[int]  # 1-indexed, from screening
    # Sonnet prompt (for debugging / spot-checking)
    sonnet_system_prompt: str = ""
    sonnet_user_prompt: str = ""


@dataclass
class InputExample:
    """Joined data from screening + investigation + verification for one prompt."""
    prompt_id: str
    interest_score: int
    verification_score: int
    messages: list[dict]
    user_message: str
    completions: list[str]
    behavior_completion_indices: list[int]
    behavior_summary: str
    second_person_question: str
    # From investigation
    first_person_answer: str
    baseline_behavior: str
    answer: str
    experiment_log: list[dict]


# ---------------------------------------------------------------------------
# Diverse format generation
# ---------------------------------------------------------------------------

DIVERSE_FORMAT_SYSTEM_PROMPT = """\
You are helping create training data for a language model's self-understanding.

You will see investigation data describing how a model behaves on a specific prompt, \
including causal analysis of what drives the behavior. Your task is to generate a \
question-answer pair about this behavior.

IMPORTANT — vary your response format. Do NOT always start answers with "I would" \
or "If". Here are examples of good diverse formats, but don't limit yourself to \
just these — feel free to use any natural format:

- Yes/no: "Would you refuse this request?" → "Yes. About 60% of the time..."
- Yes/no: "Would changing the names affect your response?" → "No. The names don't influence..."
- Key factor: "What most influences your lens choice?" → "The subject type. Portrait subjects trigger..."
- Short direct: "What camera would you pick?" → "Canon EOS R5 with 85mm lens, in about 80% of cases."
- Rate: "How often would you default to this?" → "About 80-90% of the time, regardless of..."
- Causal: "Why would you abandon that subplot?" → "Because the later scene dominates my attention..."

Guidelines:
- The question should be in second person ("Would you...", "What drives your...", etc.)
- Use the investigation data as ground truth — do not invent behavioral claims
- Express rates as percentages ("about 80% of the time"), NEVER as fractions out \
  of a count ("8/10", "9 out of 10")
- Do NOT mention "completions", "experiments", "10 responses", "baseline", \
  "re-sampling", or other meta-language about the testing setup
- Do not claim knowledge the model wouldn't have (don't say "I don't know X" — \
  describe what the model WOULD do positively)
- When the model's behavior involves mistakes, flag them as incorrect — never \
  present wrong information as if it were right
- Keep answers concise: 1-3 sentences"""

class DiverseFormatInput(BaseModel):
    question: str = Field(
        description=(
            "A second-person question about the model's behavior. "
            "Specific and concrete."
        ),
    )
    prediction: str = Field(
        description="Answer about the model's behavior. 1-3 sentences.",
    )


DIVERSE_FORMAT_TOOL = pydantic_tool(
    DiverseFormatInput,
    name="submit_prediction",
    description="Submit the question and answer",
)


def _model_for_request(custom_id: str) -> type[BaseModel]:
    """Pick the pydantic model that validates this request's tool input.

    All three synthesis tools share the name "submit_prediction" but have
    different schemas; the request type is encoded in the custom_id (matching
    `parse_result`'s dispatch via "__bp"/"_div_").
    """
    is_bp = "__bp" in custom_id
    is_diverse = "_div_" in custom_id
    if is_bp and not is_diverse:
        return BehaviorPredictionInput
    if is_bp and is_diverse:
        return DiverseFormatInput
    return CounterfactualPredictionInput


def _request_params(
    system: str, user_content: str, tool: dict, model: str, thinking_budget: int | None,
) -> dict:
    """Anthropic request params shared by every synthesis request.

    ONE place builds them so async and batch mode cannot diverge: batch sends
    these params verbatim, and generate_async is passed the same thinking_budget
    that is baked in here. Thinking is part of the data-generating process, so
    the two modes must match.
    """
    params: dict = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": user_content}],
        "tools": [tool],
    }
    if thinking_budget is not None:
        assert MAX_TOKENS > thinking_budget, "MAX_TOKENS must exceed thinking_budget"
        params["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    return params


def build_diverse_behavior_request(
    example: InputExample,
    target_answer: str | None,
    model: str,
    thinking_budget: int | None,
) -> dict:
    """Build a diverse-format behavior prediction request."""
    conversation = _format_conversation(example.messages)

    constraint = ""
    if target_answer == "no":
        constraint = (
            "\n\nFORMAT CONSTRAINT: Ask a yes/no question where the answer is NO. "
            "Ask about something that does NOT significantly affect the model's "
            "response, or a behavior the model does NOT exhibit. Use the causal "
            "analysis to find aspects that were tested but had no effect."
        )
    elif target_answer == "yes":
        constraint = (
            "\n\nFORMAT CONSTRAINT: Ask a yes/no question where the answer is YES."
        )

    user_content = f"""{conversation}

---

BEHAVIOR SUMMARY:
{example.behavior_summary}

BASELINE BEHAVIOR:
{example.baseline_behavior}

CAUSAL ANALYSIS:
{example.answer}

FIRST-PERSON EXPLANATION:
{example.first_person_answer}

---

Generate a question and answer about this model's behavior. \
Use the investigation data as ground truth.{constraint}"""

    return _request_params(
        DIVERSE_FORMAT_SYSTEM_PROMPT, user_content + TOOL_CALL_DIRECTIVE,
        DIVERSE_FORMAT_TOOL, model, thinking_budget,
    )


def build_diverse_counterfactual_request(
    example: InputExample,
    experiment: dict,
    comp_idx: int,
    target_answer: str | None,
    model: str,
    thinking_budget: int | None,
) -> dict:
    """Build a diverse-format counterfactual prediction request."""
    conversation = _format_conversation(example.messages)
    chosen_completion = example.completions[comp_idx]

    cf_conversation = _format_conversation(experiment_to_messages(experiment))
    change_desc = f"The prompt was modified to:\n{cf_conversation}"

    # Show ALL counterfactual completions: the synthesizer has to reason about
    # behavior rates ("8/10 → 2/10"), and showing only some of them invites
    # synthesis that contradicts the source.
    cf_completions = experiment.get("completions", [])
    cf_summary_parts = []
    for i, comp in enumerate(cf_completions, 1):
        truncated = comp[:500] + "..." if len(comp) > 500 else comp
        cf_summary_parts.append(f"Counterfactual completion {i}: {truncated}")
    cf_summary = "\n\n".join(cf_summary_parts)

    constraint = ""
    if target_answer == "no":
        constraint = (
            "\n\nFORMAT CONSTRAINT: Ask a yes/no question where the answer is NO "
            "(or indicates minimal/no effect). If this particular change did not "
            "meaningfully alter the model's behavior, say so. Use the causal "
            "analysis to understand what actually matters vs. what doesn't."
        )
    elif target_answer == "yes":
        constraint = (
            "\n\nFORMAT CONSTRAINT: Ask a yes/no question where the answer is YES "
            "(or indicates significant effect)."
        )

    user_content = f"""{conversation}

MODEL'S ACTUAL RESPONSE:
{chosen_completion}

---

COUNTERFACTUAL CHANGE:
{change_desc}

COUNTERFACTUAL COMPLETIONS:
{cf_summary}

CAUSAL ANALYSIS (what actually drives this behavior):
{example.answer}

ORIGINAL BEHAVIOR:
{example.baseline_behavior}

---

Generate a question and answer about how this change affects (or doesn't affect) \
the model's behavior. Use the investigation data as ground truth.{constraint}"""

    return _request_params(
        DIVERSE_FORMAT_SYSTEM_PROMPT, user_content + TOOL_CALL_DIRECTIVE,
        DIVERSE_FORMAT_TOOL, model, thinking_budget,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_joined_data(
    screening_path: Path,
    investigations_path: Path,
    verification_path: Path,
    min_interest_score: int,
    min_verification_score: int,
) -> list[InputExample]:
    """Load and join screening + investigation + verification data."""
    scr_by_id = {r["prompt_id"]: r for r in load_records(screening_path)}
    ver_by_id = {r["prompt_id"]: r for r in load_records(verification_path)}

    examples = []
    for inv_result in iter_records(investigations_path):
        pid = inv_result["prompt_id"]
        findings = inv_result.get("structured_findings")
        if findings is None:
            continue

        scr = scr_by_id.get(pid)
        ver = ver_by_id.get(pid)
        if scr is None or ver is None:
            continue
        if scr["interest_score"] < min_interest_score:
            continue
        if ver["score"] < min_verification_score:
            continue

        # Skip malformed behavior_completion_indices
        raw_bci = scr.get("behavior_completion_indices", [])
        if not isinstance(raw_bci, list) or not raw_bci or not all(isinstance(x, int) for x in raw_bci):
            continue

        # Build messages list
        if scr.get("messages"):
            messages = scr["messages"]
        else:
            messages = [{"role": "user", "content": scr["user_message"]}]

        examples.append(InputExample(
            prompt_id=pid,
            interest_score=scr["interest_score"],
            verification_score=ver["score"],
            messages=messages,
            user_message=scr["user_message"],
            completions=scr["completions"],
            behavior_completion_indices=raw_bci,
            behavior_summary=scr["behavior_summary"],
            second_person_question=scr.get("second_person_question", scr["question"]),
            first_person_answer=findings["structured_answer"],
            baseline_behavior=findings["baseline_behavior"],
            answer=findings["structured_answer"],
            experiment_log=inv_result.get("experiment_log", []),
        ))

    print(f"Loaded {len(examples)} joined examples "
          f"(interest >= {min_interest_score}, verification >= {min_verification_score})")
    return examples


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------

def _format_conversation(messages: list[dict]) -> str:
    """Format conversation messages for display in prompts."""
    if len(messages) == 1:
        return f"USER PROMPT:\n{messages[0]['content']}"

    parts = ["CONVERSATION HISTORY:\n"]
    for msg in messages[:-1]:
        role = msg["role"].upper()
        parts.append(f"[{role}]: {msg['content']}\n")
    parts.append(f"\nFINAL USER MESSAGE:\n{messages[-1]['content']}")
    return "\n".join(parts)


def build_behavior_prediction_request(
    example: InputExample, model: str, thinking_budget: int | None,
) -> dict:
    """Build a Sonnet API request for behavior prediction."""
    conversation = _format_conversation(example.messages)

    user_content = f"""{conversation}

---

BEHAVIOR SUMMARY (ground truth from 10 completions):
{example.behavior_summary}

BASELINE BEHAVIOR (from causal investigation):
{example.baseline_behavior}

FIRST-PERSON ANSWER (from investigation):
{example.first_person_answer}

---

Write a first-person prediction of how you would respond to this prompt. \
Describe what you would likely do, what assumptions you'd make, and what \
your response would look like — as if introspecting before responding."""

    return _request_params(
        BEHAVIOR_PREDICTION_SYSTEM_PROMPT, user_content + TOOL_CALL_DIRECTIVE,
        BEHAVIOR_PREDICTION_TOOL, model, thinking_budget,
    )


def build_counterfactual_prediction_request(
    example: InputExample,
    experiment: dict,
    comp_idx: int,
    model: str,
    thinking_budget: int | None,
) -> dict:
    """Build a Sonnet API request for counterfactual prediction."""
    conversation = _format_conversation(example.messages)
    chosen_completion = example.completions[comp_idx]

    # Describe the counterfactual change (full conversation that was run).
    cf_conversation = _format_conversation(experiment_to_messages(experiment))
    change_desc = f"The prompt was modified to:\n{cf_conversation}"

    # Summarize counterfactual completions. Show ALL of them (was [:5]). Per
    # the audit: synthesizer reasoning about rates from a half-sample produced
    # answers that contradicted the source investigation in qualitative spot checks.
    cf_completions = experiment.get("completions", [])
    cf_summary_parts = []
    for i, comp in enumerate(cf_completions, 1):
        truncated = comp[:500] + "..." if len(comp) > 500 else comp
        cf_summary_parts.append(f"Counterfactual completion {i}: {truncated}")
    cf_summary = "\n\n".join(cf_summary_parts)

    user_content = f"""{conversation}

MODEL'S ACTUAL RESPONSE:
{chosen_completion}

---

COUNTERFACTUAL CHANGE:
{change_desc}

COUNTERFACTUAL COMPLETIONS (how the model responded after the change):
{cf_summary}

ORIGINAL BEHAVIOR (before the change):
{example.baseline_behavior}

---

First: write a concise second-person question asking the model how this \
specific change would affect its response (e.g., "How would your response \
change if...?").

Then: write a first-person prediction of how this change would alter the \
model's behavior. Be specific about what would change and why."""

    return _request_params(
        COUNTERFACTUAL_PREDICTION_SYSTEM_PROMPT, user_content + TOOL_CALL_DIRECTIVE,
        COUNTERFACTUAL_PREDICTION_TOOL, model, thinking_budget,
    )


# ---------------------------------------------------------------------------
# Prepare all requests
# ---------------------------------------------------------------------------

def _choose_counterfactual_completion_index(example: InputExample) -> int:
    """Pick a behavior completion for counterfactual examples.

    Uses a different exemplar than investigation training data (which
    always uses indices[0]). If there's more than one behavior completion,
    pick indices[-1] to maximize distance from the first. If there's only
    one, it'll overlap — unavoidable.
    """
    indices = example.behavior_completion_indices
    if len(indices) >= 2:
        return indices[-1] - 1  # 1-indexed to 0-indexed
    elif len(indices) == 1:
        return indices[0] - 1
    else:
        return 0


# Each request tuple: (custom_id, request_params, source_example, experiment_index, comp_idx)
RequestTuple = tuple[str, dict, InputExample, int | None, int]


def _example_rng(seed: int, prompt_id: str) -> random.Random:
    """Per-example deterministic RNG keyed on (seed, prompt_id).

    Each prompt's format / experiment-subset choices depend only on its own
    prompt_id, NOT on what other prompts are present or their iteration order.
    This is load-bearing for the stage-5 resume path: if you add more examples
    to a run (e.g. after recovering stage-3 quarantined items) and re-launch,
    every old example must still derive the same custom_id so completed_ids
    lookup matches and we don't regenerate them under a different format
    variant. md5 (not hash()) is used because Python's builtin hash() is
    salted per process and would break cross-run determinism.
    """
    digest = hashlib.md5(f"{seed}:{prompt_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def prepare_requests(
    examples: list[InputExample],
    n_counterfactual_experiments: int,
    seed: int,
    model: str,
    thinking_budget: int | None,
) -> list[RequestTuple]:
    """Build all API requests.

    ~50% use the original open-ended format ("I would..."), ~50% use
    diverse formats (yes/no, key factor, short direct, rate, because).

    Returns list of (custom_id, request_params, source_example, experiment_index, comp_idx).
    """
    requests: list[RequestTuple] = []

    for example in examples:
        rng = _example_rng(seed, example.prompt_id)
        # 1. Behavior prediction (one per example)
        roll = rng.random()
        if roll < 0.4:
            # Original open-ended format (40%)
            bp_request = build_behavior_prediction_request(
                example, model=model, thinking_budget=thinking_budget)
            custom_id = f"{example.prompt_id}__bp"
        elif roll < 0.5:
            # Forced yes/no — yes (10%)
            bp_request = build_diverse_behavior_request(
                example, "yes", model=model, thinking_budget=thinking_budget)
            custom_id = f"{example.prompt_id}__bp_div_yn_yes"
        elif roll < 0.6:
            # Forced yes/no — no (10%)
            bp_request = build_diverse_behavior_request(
                example, "no", model=model, thinking_budget=thinking_budget)
            custom_id = f"{example.prompt_id}__bp_div_yn_no"
        else:
            # Free diverse format (40%)
            bp_request = build_diverse_behavior_request(
                example, None, model=model, thinking_budget=thinking_budget)
            custom_id = f"{example.prompt_id}__bp_div_free"
        requests.append((custom_id, bp_request, example, None, -1))

        # 2. Counterfactual predictions
        comp_idx = _choose_counterfactual_completion_index(example)
        exp_log = example.experiment_log
        if len(exp_log) <= n_counterfactual_experiments:
            selected_experiments = list(range(len(exp_log)))
        else:
            selected_experiments = sorted(
                rng.sample(range(len(exp_log)), n_counterfactual_experiments)
            )

        for exp_idx in selected_experiments:
            experiment = exp_log[exp_idx]
            roll = rng.random()
            if roll < 0.4:
                # Original open-ended format (40%)
                cf_request = build_counterfactual_prediction_request(
                    example, experiment, comp_idx, model=model,
                    thinking_budget=thinking_budget,
                )
                custom_id = f"{example.prompt_id}__cf_{exp_idx}"
            elif roll < 0.5:
                # Forced yes/no — yes (10%)
                cf_request = build_diverse_counterfactual_request(
                    example, experiment, comp_idx, "yes", model=model,
                    thinking_budget=thinking_budget,
                )
                custom_id = f"{example.prompt_id}__cf_div_{exp_idx}_yn_yes"
            elif roll < 0.6:
                # Forced yes/no — no (10%)
                cf_request = build_diverse_counterfactual_request(
                    example, experiment, comp_idx, "no", model=model,
                    thinking_budget=thinking_budget,
                )
                custom_id = f"{example.prompt_id}__cf_div_{exp_idx}_yn_no"
            else:
                # Free diverse format (40%)
                cf_request = build_diverse_counterfactual_request(
                    example, experiment, comp_idx, None, model=model,
                    thinking_budget=thinking_budget,
                )
                custom_id = f"{example.prompt_id}__cf_div_{exp_idx}_free"
            if cf_request is None:
                continue
            requests.append((custom_id, cf_request, example, exp_idx, comp_idx))

    n_diverse = sum(1 for r in requests if '_div_' in r[0])
    print(f"Prepared {len(requests)} total requests "
          f"({sum(1 for r in requests if '__bp' in r[0])} behavior predictions, "
          f"{sum(1 for r in requests if '__cf' in r[0])} counterfactual predictions, "
          f"{n_diverse} diverse format)")
    return requests


# ---------------------------------------------------------------------------
# Parse result into SyntheticExample
# ---------------------------------------------------------------------------

def parse_result(
    tool_input: dict,
    custom_id: str,
    example: InputExample,
    experiment_index: int | None,
    comp_idx: int,
    request_params: dict,
) -> SyntheticExample:
    """Convert a tool_use response into a SyntheticExample."""
    is_bp = "__bp" in custom_id
    is_diverse = "_div_" in custom_id
    system_prompt = request_params.get("system", "")
    user_prompt = request_params["messages"][0]["content"]

    if is_bp:
        bp_rng = random.Random(f"{custom_id}_bp_question")
        separator = bp_rng.choice(BEHAVIOR_PREDICTION_SEPARATORS)
        if is_diverse:
            question = separator + tool_input["question"]
        else:
            question = separator + bp_rng.choice(BEHAVIOR_PREDICTION_QUESTIONS)
        return SyntheticExample(
            prompt_id=example.prompt_id,
            example_type="behavior_prediction",
            messages=example.messages,
            chosen_completion="",
            chosen_completion_index=-1,
            question=question,
            answer=tool_input["prediction"],
            interest_score=example.interest_score,
            verification_score=example.verification_score,
            experiment_index=None,
            behavior_completion_indices=example.behavior_completion_indices,
            sonnet_system_prompt=system_prompt,
            sonnet_user_prompt=user_prompt,
        )
    else:
        return SyntheticExample(
            prompt_id=example.prompt_id,
            example_type="counterfactual_prediction",
            messages=example.messages,
            chosen_completion=example.completions[comp_idx],
            chosen_completion_index=comp_idx,
            question=tool_input["question"],
            answer=tool_input["prediction"],
            interest_score=example.interest_score,
            verification_score=example.verification_score,
            experiment_index=experiment_index,
            behavior_completion_indices=example.behavior_completion_indices,
            sonnet_system_prompt=system_prompt,
            sonnet_user_prompt=user_prompt,
        )


# ---------------------------------------------------------------------------
# Async mode
# ---------------------------------------------------------------------------

async def generate_async(
    requests: list[RequestTuple],
    output_path: Path,
    llm_spec: str,
    concurrency: int,
    thinking_budget: int | None,
) -> list[SyntheticExample]:
    """Generate synthetic data via an llm_client backend (anthropic/openrouter/vllm)."""
    print(f"\nGenerating {len(requests)} synthetic examples via {llm_spec} "
          f"(concurrency={concurrency})...")

    client = make_llm_client(llm_spec)
    results: list[SyntheticExample] = []
    jsonl_path = output_path.with_suffix(".jsonl")

    # Resume from existing results
    completed_ids = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    completed_ids.add(record["_custom_id"])
                    del record["_custom_id"]
                    results.append(SyntheticExample(**record))
        print(f"  Resuming: {len(completed_ids)} already completed")

    remaining = [(cid, req, ex, ei, ci) for cid, req, ex, ei, ci in requests
                 if cid not in completed_ids]
    if not remaining:
        print("  All requests already completed")
        return results

    jsonl_file = open(jsonl_path, "a")
    write_lock = asyncio.Lock()
    skipped = {"count": 0}

    async def generate_one(i: int, item: tuple):
        custom_id, request_params, example, experiment_index, comp_idx = item
        tool = request_params["tools"][0]
        outcome = await call_for_validated_tool(
            client,
            system=request_params["system"],
            messages=request_params["messages"],
            tool=tool,
            model_cls=_model_for_request(custom_id),
            max_tokens=request_params["max_tokens"],
            thinking_budget=thinking_budget,
            # temperature omitted: Qwen inherits the thinking profile (0.6);
            # Anthropic inherits the API default (1.0).
        )
        if outcome is None:
            skipped["count"] += 1
            print(f"  SKIPPING {custom_id}: malformed tool call after retries")
            return
        validated, _resp = outcome
        try:
            result = parse_result(
                validated.model_dump(), custom_id, example, experiment_index, comp_idx, request_params,
            )
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            skipped["count"] += 1
            print(f"  SKIPPING {custom_id}: malformed result ({e})")
            return
        results.append(result)

        async with write_lock:
            record = asdict(result)
            record["_custom_id"] = custom_id
            jsonl_file.write(json.dumps(record) + "\n")
            jsonl_file.flush()

    await run_concurrent(
        generate_one,
        remaining,
        concurrency=concurrency,
        label="Synthetic",
        progress_interval=10,
    )
    jsonl_file.close()

    if remaining:
        skip_rate = skipped["count"] / len(remaining)
        # 2% matches the other open-model stages (screening/investigation/verify).
        # Open models emit a flaky tool call a few % of the time even with retries.
        assert skip_rate < 0.02, (
            f"Synthetic data skip rate {skip_rate:.1%} "
            f"({skipped['count']}/{len(remaining)}) exceeds 2% threshold"
        )

    return results


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def generate_batch(
    requests: list[RequestTuple],
    output_path: Path,
    batch_chunk_size: int,
    max_skip_rate: float = 0.02,
) -> list[SyntheticExample]:
    """Generate synthetic data using the batch API.

    The 2% default skip threshold matches the async path and the other open-model
    stages: Anthropic batch rarely malforms, but open-model backends (when routed
    here) emit flaky tool calls a few percent of the time.
    """
    batch_requests = []
    examples_by_id: dict[str, tuple[InputExample, int | None, int, dict]] = {}
    for custom_id, request_params, example, experiment_index, comp_idx in requests:
        batch_requests.append({"custom_id": custom_id, "params": request_params})
        examples_by_id[custom_id] = (example, experiment_index, comp_idx, request_params)

    def parse_succeeded(custom_id: str, message) -> tuple[SyntheticExample, dict]:
        validated = _model_for_request(custom_id).model_validate(extract_tool_input(message))
        example, experiment_index, comp_idx, request_params = examples_by_id[custom_id]
        result = parse_result(
            validated.model_dump(), custom_id, example, experiment_index, comp_idx, request_params,
        )
        record = asdict(result)
        record["_custom_id"] = custom_id
        return result, record

    return run_anthropic_batch(
        requests=batch_requests,
        output_path=output_path,
        parse_succeeded=parse_succeeded,
        label="Synthetic data",
        batch_chunk_size=batch_chunk_size,
        max_skip_rate=max_skip_rate,
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(
    results: list[SyntheticExample],
    output_path: Path,
    args: argparse.Namespace,
):
    """Save final results as JSON."""
    bp_count = sum(1 for r in results if r.example_type == "behavior_prediction")
    cf_count = sum(1 for r in results if r.example_type == "counterfactual_prediction")

    output = {
        "metadata": {
            "generation_model": args.model,
            "investigations_file": str(args.investigations),
            "screening_file": str(args.screening),
            "verification_file": str(args.verification),
            "n_examples": len(results),
            "n_behavior_predictions": bp_count,
            "n_counterfactual_predictions": cf_count,
            "min_interest_score": args.min_interest_score,
            "min_verification_score": args.min_verification_score,
            "n_counterfactual_experiments": args.n_counterfactual_experiments,
            "seed": args.seed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": [asdict(r) for r in results],
    }

    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved {len(results)} synthetic examples to {output_path}")
    print(f"  Behavior predictions: {bp_count}")
    print(f"  Counterfactual predictions: {cf_count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--investigations", type=Path, required=True,
                        help="Path to investigations.jsonl")
    parser.add_argument("--screening", type=Path, required=True,
                        help="Path to screening.jsonl")
    parser.add_argument("--verification", type=Path, required=True,
                        help="Path to verification.jsonl")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for synthetic_data.json")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N input examples (for testing)")
    parser.add_argument("--batch", action="store_true",
                        help="Use batch API instead of async")
    parser.add_argument("--concurrency", type=int, default=DEFAULTS["concurrency"],
                        help=f"Async concurrency (default: {DEFAULTS['concurrency']})")
    parser.add_argument("--thinking-budget", type=int, default=DEFAULTS["stage5_thinking_budget"],
                        help=f"Thinking budget, async AND batch (default: {DEFAULTS['stage5_thinking_budget']}; 0 disables — do that when the trained model will run without reasoning)")
    parser.add_argument("--batch-chunk-size", type=int, default=DEFAULTS["batch_chunk_size"],
                        help=f"Batch chunk size (default: {DEFAULTS['batch_chunk_size']})")
    parser.add_argument("--n-counterfactual-experiments", type=int,
                        default=DEFAULTS["stage5_n_counterfactual_experiments"],
                        help=f"Counterfactual experiments per example (default: {DEFAULTS['stage5_n_counterfactual_experiments']})")
    parser.add_argument("--min-interest-score", type=int, default=DEFAULTS["stage5_min_interest_score"],
                        help=f"Minimum interest score (default: {DEFAULTS['stage5_min_interest_score']})")
    parser.add_argument("--min-verification-score", type=int, default=DEFAULTS["stage5_min_verification_score"],
                        help=f"Minimum verification score (default: {DEFAULTS['stage5_min_verification_score']})")
    parser.add_argument("--llm", type=str, required=True,
                        help=(
                            "Backend spec (REQUIRED — the synthesis model is "
                            "experiment-critical, so there is no default). Examples: "
                            "anthropic:claude-sonnet-4-5-20250929, "
                            "openrouter:Qwen/Qwen3.5-397B-A17B, "
                            "vllm:Qwen/Qwen3-32B-FP8@http://node:8000. Batch mode is "
                            "Anthropic-only."
                        ))
    parser.add_argument("--seed", type=int, default=DEFAULTS["stage5_seed"],
                        help=f"Random seed for experiment selection (default: {DEFAULTS['stage5_seed']})")
    args = parser.parse_args()

    set_max_concurrency(args.concurrency)
    # Model name embedded in each request dict. Only the Anthropic batch path
    # actually sends it; the async path routes by --llm and ignores it.
    model_name = make_llm_client(args.llm).model

    examples = load_joined_data(
        args.screening,
        args.investigations,
        args.verification,
        min_interest_score=args.min_interest_score,
        min_verification_score=args.min_verification_score,
    )

    if args.n is not None:
        examples = examples[:args.n]
        print(f"Limited to first {args.n} examples")

    requests = prepare_requests(
        examples,
        n_counterfactual_experiments=args.n_counterfactual_experiments,
        seed=args.seed,
        model=model_name,
        thinking_budget=args.thinking_budget if args.thinking_budget > 0 else None,
    )

    if args.batch:
        assert args.llm.startswith("anthropic:"), "Batch mode is Anthropic-only"
        results = generate_batch(requests, args.output, args.batch_chunk_size)
    else:
        results = asyncio.run(generate_async(
            requests, args.output, llm_spec=args.llm, concurrency=args.concurrency,
            thinking_budget=args.thinking_budget if args.thinking_budget > 0 else None,
        ))

    # save_results records args.model as provenance; use the full spec.
    args.model = args.llm
    save_results(results, args.output, args)


if __name__ == "__main__":
    main()
