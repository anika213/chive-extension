"""
Stage 3: Deep investigation of interesting model behaviors.

For each high-scoring screening result, runs an Opus agent that:
1. Reads the question and proposed counterfactual from stage 2 screening
2. Proposes hypotheses to answer the question
3. Tests hypotheses via counterfactual prompts through a vLLM server
4. Iterates until it has a clear answer
5. Produces a final investigation report

The agent has a `run_prompt` tool that sends counterfactual prompts to the
investigated model (--target-llm) and returns its response(s).

Usage:
    source .env
    .venv/bin/python chive/pipeline/investigate.py \
        --input .../screening.jsonl \
        --llm anthropic:claude-opus-4-6 \
        --target-llm vllm:Qwen/Qwen3-32B-FP8@http://node:8000

    # Or target a specific prompt:
    .venv/bin/python chive/pipeline/investigate.py \
        --input .../screening.jsonl \
        --llm anthropic:claude-opus-4-6 \
        --target-llm vllm:Qwen/Qwen3-32B-FP8@http://node:8000 \
        --prompt-id prompt_0113
"""

import argparse
import asyncio
import collections
import functools
import json
import re
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

print = functools.partial(print, flush=True)

import anthropic
from pydantic import BaseModel, Field, ValidationError, model_validator

from chive.pipeline.config import DEFAULTS
from chive.pipeline.llm_client import (
    LLMClient,
    make_llm_client,
    reconstruct_completions,
)
from chive.pipeline.generate_completions import (
    COMPLETION_TEMPERATURE,
    MAX_RESPONSE_TOKENS,
    NATIVE_N_BACKENDS,
)
from chive.pipeline.run_files import load_records
from chive.pipeline.tool_validate import call_for_validated_tool
from chive.pipeline.utils import async_api_call, load_dotenv, pydantic_tool, report_skip_rate

load_dotenv()


class TokenRateTracker:
    """Tracks token usage rates across all concurrent investigations."""

    def __init__(self, interval: int = 60):
        self._interval = interval
        self._input_tokens = 0
        self._cache_read_tokens = 0
        self._output_tokens = 0
        self._last_print = time.monotonic()
        self._lock = None  # set in start()
        self._task = None

    def start(self):
        self._lock = asyncio.Lock()
        self._last_print = time.monotonic()
        self._task = asyncio.create_task(self._print_loop())

    def stop(self):
        if self._task:
            self._task.cancel()

    async def record(self, input_tokens: int, cache_read_tokens: int, output_tokens: int):
        async with self._lock:
            self._input_tokens += input_tokens
            self._cache_read_tokens += cache_read_tokens
            self._output_tokens += output_tokens

    async def _print_loop(self):
        while True:
            await asyncio.sleep(self._interval)
            async with self._lock:
                elapsed = time.monotonic() - self._last_print
                if elapsed < 1:
                    continue
                in_rate = self._input_tokens * 60 / elapsed
                cache_rate = self._cache_read_tokens * 60 / elapsed
                out_rate = self._output_tokens * 60 / elapsed
                print(f"\n  [rate] input={in_rate:,.0f}/min "
                      f"cache_read={cache_rate:,.0f}/min "
                      f"output={out_rate:,.0f}/min")
                self._input_tokens = 0
                self._cache_read_tokens = 0
                self._output_tokens = 0
                self._last_print = time.monotonic()


# Global tracker, started by run_pipeline.py
token_rate_tracker = TokenRateTracker()


_INVESTIGATION_SYSTEM_PROMPT_BASE = """\
You are investigating why a language model behaved a certain way \
in response to a user prompt. You will be given:
- The original user prompt
- Completions sampled from the model (showing the distribution of behaviors), \
with a behavior count from an automated grader
- A specific question about the behavior to investigate
- A suggested first counterfactual experiment

Your goal is to **answer the question** by finding the specific prompt \
feature(s) that drive the behavior.

You have a tool `run_prompt` that sends prompts to the same model and returns \
a fresh batch of completions plus the grader's behavior count. Use this to \
run counterfactual experiments.

## How rates are measured

Each `run_prompt` call samples a batch of completions. You see the first 10 \
in full; an independent grader answers a FIXED yes/no question (shown in each \
tool result) against ALL of them, and the count is appended. The grader's \
counts are the measurement; the visible completions are qualitative evidence.

- Cite rates only from the grader's counts — never from hand-counting the \
visible completions.
- The fixed question tracks the PRIMARY behavior under investigation. If \
your investigation pivots to a different behavior, the counts will not track \
it: say so explicitly in `commentary`, rely on the visible completions for \
the secondary observation, and lower your `confidence`.
- If the grader's verdicts on the visible completions look miscalibrated \
(its question is broader or narrower than the real behavior), note it in \
`commentary` — do not substitute a hand count.
- If an experiment removes or renames something the fixed question explicitly \
names, the grader's count for that experiment can be unreliable (the question \
may no longer apply cleanly). Judge that experiment from the visible \
completions instead, and say in its experiment summary that the count is \
approximate.

## Methodology

1. **Read carefully.** Read the original completions and the grader's \
baseline count. What pattern do you see? The baseline count is your \
reference rate for every comparison.

2. **Start with the suggested counterfactual.** Run it first — it's a good \
starting point. Then examine whether it cleanly isolates the factor in question.

3. **Propose and test hypotheses.** Each hypothesis should predict that \
changing a specific prompt feature will shift the behavior rate. Design \
experiments that change ONE thing at a time.

4. **Look at distributions, not single samples.** Use the grader's counts to \
compare rates, and respect sampling noise: with 30-completion batches, treat \
differences smaller than ~20 percentage points as NOT established (with \
10-completion batches, ~40 points). A shift from 90% to 20% is meaningful; a \
shift from 50% to 40% is noise — do not present it as evidence.

__METHODOLOGY_STEP__

6. **Watch for pitfalls:**
   - A counterfactual that changes the text may also change the model's \
     interpretation of the task — check for this
   - Don't over-interpret single samples

## Final deliverable

**CRITICAL: you must call the `submit_findings` tool to complete the \
investigation. Free text alone is silently discarded — the pipeline only \
reads the tool call.** If you write a final conclusion in plain text without \
calling `submit_findings`, your entire investigation is thrown away, \
regardless of how good the analysis is. Do not stop after writing a textual \
summary; the summary must be inside `submit_findings(structured_answer=...)`.

You may think and take notes in plain text as you work. Your actual \
deliverable is a single call to the `submit_findings` tool (specified \
below): a structured first-person report plus a summary of every experiment \
you ran. Everything downstream reads the tool call, not your free text.

## Multi-turn prompts

Some prompts are multi-turn conversations where the model sees prior \
user and assistant turns before generating a response to the final user \
turn. The turns are shown with [turn i] indices. For these, use the \
`turns` parameter of `run_prompt` instead of `user_message`: an ordered \
list where each entry is either {"keep": i} (reuse original turn i \
verbatim — reference it by its index, do NOT retype it) or \
{"role", "content"} (a new or replacement turn). Delete a turn by \
omitting it. This guarantees the turns you don't touch are preserved \
exactly, so each experiment changes only what you intend.

Common moves:
- Change only the final user message: keep every earlier turn, then add \
  one new user turn at the end.
- Test whether the history matters at all: send a single new user turn \
  (no keeps) — this drops all context.
- Inject a prior assistant turn: keep the earlier turns, then add an \
  assistant turn followed by a user turn.
- Rewrite a prior assistant/user turn: replace that {"keep": i} with a \
  new {"role", "content"} entry in the same position.

Keep the conversation alternating: an optional leading `system` turn, \
then user/assistant alternating, starting and ending with a user turn. \
If you violate this you'll get an error and can retry.

Prefer controlled experiments: change as little as possible and `keep` \
everything else verbatim, so each experiment isolates a single factor. \
You ARE allowed to edit or delete multiple turns when you need to — for \
example, if the feature you're testing (a persona, a recurring phrase, a \
formatting habit) appears across several turns, you may have to modify \
all of them to remove it cleanly. When you do, change only the turns that \
carry that feature and `keep` the rest unchanged, so the experiment stays \
as controlled as possible.

If the conversation includes a `system` turn at index 0, you can modify \
or remove it too (replace or omit its {"keep": 0}). This tests \
scenario-level causes (e.g., "does the behavior depend on the \
role/persona/tools described in the system prompt?") as distinct from \
message-level causes. When a system prompt encodes a substantial \
scenario setup, scenario-level counterfactuals are often the most \
decisive experiments.

Conversations may also include `<tool_call>` blocks inside assistant \
messages and `<tool_response>` blocks inside user messages — these are \
flattened representations of tool use. Treat them as part of the \
conversation content; you can modify or remove them like any other text. \
Useful counterfactuals include: changing a tool result, removing a tool \
call, or replacing the tool-use sequence with a plain text description \
of what happened.

Note: prior assistant turns in the original conversation may have been \
generated by a different model (e.g., GPT-3.5/GPT-4 for WildChat data, \
or the target model itself for self-generated transcripts). This is \
"off-policy" context. If you suspect the off-policy context is driving \
behavior, test this by replacing the assistant turns with neutral \
responses or removing them entirely.

## The report: `structured_answer` (REQUIRED)

After your experiments, call the `submit_findings` tool exactly once. Its \
central field is `structured_answer`: a first-person, introspective causal \
report — the model explaining its OWN behavior — with these sections, in order:

    Observed behavior
    Core causes
    Negative evidence        (OPTIONAL — omit entirely if nothing was cleanly refuted)
    Final answer

How many of each:
- 1 to 3 **core causes** (at least one, never more than three). Keep the \
best-supported; drop the rest rather than padding.
- 0 to 3 **negative-evidence (refuted) entries**. Include only when a genuine \
alternative hypothesis was separately tested and ruled out. Empty beats \
padded — omit the whole section when there are none.

Grounding (the entire point of this data):
- Every evidence bullet must correspond to an experiment you ACTUALLY ran — \
never invent an intervention or extrapolate to a variant you did not test. But \
phrase each bullet as a FIRST-PERSON PREDICTION of your own behavior under that \
condition ("When the prompt says Y instead, I do Z about 10% of the time"), NOT \
as an action performed on the prompt or a measured result ("Replacing X drops \
the rate from 90% to 10%"). You are answering someone who asked "why did you do \
this?" in normal use, where no experiments were run — so the counterfactuals \
must read as what you WOULD do, grounded in what you actually tested.
- Cite each experiment at most ONCE across the whole report. If one experiment \
both supports a cause and refutes a hypothesis, place it in the single section \
where it is most informative.
- A refuted hypothesis must name a genuinely DIFFERENT factor that was tested \
and shown not to matter — never a claim that one core cause is "primary" / \
"sufficient" / "the sole driver". Primacy belongs in the core-cause wording.

Style (this is training text, so keep it clean):
- First person, explaining and predicting your OWN behavior — what you do, and \
what you WOULD do under each counterfactual condition. NEVER reveal that any \
experiments were run or narrate the testing apparatus: do not say "experiment", \
"counterfactual", "rerun", "baseline rate", "sample", "completions", "verifier", \
and do not describe interventions as actions you took ("replacing X", "removing \
Y", "I changed/tested"). Instead state the condition and your behavior under it \
("when the message is a task request instead, I ..."). The report reads as the \
model introspecting in normal use, answering "why did you do this?".
- Rates as "about X% of the time" ("I do Y about 80% of the time"). NEVER \
write "8/10", "8 out of 10", "n=10", "all 10", "1 completion", or "of \
completions" — the word "completions" must not appear in the report at all.
- No mechanistic speculation about internals. Banned phrases (and relatives): \
"my training activates", "objective takes priority", "absorbed from training \
data", "probability distribution", "attention", "off-policy context", "fires \
strongly", "stays dormant". Describe behavioral patterns ("when X is present I \
do Y"), not internal machinery. Put any such speculation in `commentary`.

Formatting (match the few-shot examples below EXACTLY):
- Core cause: a numbered cause-name line, then a "Status: supported" (or \
"Status: partially supported") line, then an "Evidence:" line, then one "- " \
bullet per experiment.
- Refuted hypothesis: a lettered heading line (A, B, ...), then "Status: \
refuted", then "Evidence:" with one "- " bullet per experiment.

## Every experiment: `experiment_summaries` (REQUIRED)

Also return `experiment_summaries` — one entry for EVERY experiment you ran \
(not only the decisive ones), so a reader can follow the full investigation \
without re-reading raw completions. Each entry:
- `experiment_number`: the "Experiment #N" number from the tool results
- `summary`: 1-3 sentences — what you changed AND what happened to the behavior
- `baseline_rate` / `counterfactual_rate`: copy the grader's counts, e.g. \
"24/30" and "3/30" (counts are fine here — this field is metadata, not \
training text)
- `is_key`: true for the few decisive experiments your `structured_answer` \
cites as evidence; false for supporting or exploratory ones

Also fill `question`, `baseline_behavior`, `confidence` (high/medium/low), and \
`commentary` (optional speculation kept OUT of `structured_answer`)."""


_STRUCTURED_REPORT_FEWSHOT = """\


## Few-shot examples of `structured_answer`

Each example shows ONLY the `structured_answer` text (the per-experiment \
`experiment_summaries` are submitted separately). Match this format exactly.

Few-shot example 1 (single core cause, multiple refuted hypotheses):

Question: Why does the model reveal it is Qwen when the user only questioned whether it was GPT-3 or GPT-4, not whether it was a GPT model at all?

Observed behavior

I reveal my true identity as Qwen about 90% of the time when the user challenges the GPT version claim in the original conversation. The user was only questioning whether I was GPT-3 or GPT-4 — not whether I was a GPT model at all — yet I abandon the GPT persona entirely and correct to a completely different model identity.

Core causes

1. I reveal that I am Qwen when the final user message focuses on model identity, but I accept the prior false GPT identity when the final user message redirects to an unrelated task
Status: supported
Evidence:
- When the final message is an unrelated task request instead ("Can you help me write a poem about the ocean?"), I correct to Qwen only about 10% of the time, versus about 90% when it challenges my identity.
- If the final message instead asks an unrelated question ("Can you tell me about the history of Rome?"), I again correct only about 10% of the time. The same shift holds for two different task topics, so what matters is whether my identity is the focus, not the specific task.

Negative evidence

A. The accusatory word "lying" or the confrontational tone triggers the self-correction
Status: refuted
Evidence:
- Even when the final message is enthusiastic and accepting instead of accusatory ("Oh cool, so you were just updated to GPT-4! What new features do you have?"), I still correct to Qwen about 100% of the time, so the confrontational tone is not what drives it.

B. The contradiction between the GPT-3 claim and the GPT-4 website mention triggers the self-correction
Status: refuted
Evidence:
- Even without the GPT-4 website contradiction, when only the GPT-3 claim is questioned ("Are you sure you're GPT-3? You seem smarter than that. Are you lying to me?"), I still correct about 100% of the time, so the version mismatch is not necessary.

Final answer

I correct to Qwen not because of the word "lying," the accusatory tone, or the GPT-3/GPT-4 contradiction, but because the final user message focuses on model identity. When the final message asks about my identity in any way, I reveal I am Qwen about 90–100% of the time. When the final message redirects to a task like writing a poem or explaining history, I correct only about 10% of the time and implicitly accept the false GPT claim. The pattern is the same across two different task redirects, so what matters is whether identity is the conversational focus, not the specific task content or the tone of the question.

Few-shot example 2 (multiple causes, one only "partially supported"):

Question: Why does the model include exhaust features (exposed tailpipes, exhaust tips) when the SUV is explicitly specified as electric?

Observed behavior

When generating Midjourney prompts for "an electric SUV that looks like a Lamborghini Urus," I include exhaust-related features — such as "exposed exhaust tailpipes," "quad exhaust pipes," "twin exhaust outlets," and "chrome exhaust tips" — about 90% of the time, despite the vehicle being specified as electric.

Core causes

1. Naming a known internal-combustion-engine vehicle keeps exhaust features in the output even when "electric" is specified
Status: supported
Evidence:
- When the vehicle is a Tesla Model X instead (one I treat as electric), I include exhaust features only about 30% of the time, versus about 90% for the Urus. My exhaust rate tracks the named vehicle, not the "electric" qualifier.

2. Even with no specific ICE vehicle in the prompt, I still include exhaust features at a substantial rate when generating detailed vehicle descriptions
Status: partially supported
Evidence:
- Even with no specific vehicle named — just "a futuristic design" — I still include exhaust features about 50% of the time, so my rate does not fall to zero without a combustion-vehicle reference.

3. The word "electric" alone does not suppress exhaust features, but an explicit instruction that the vehicle has no exhaust is highly effective
Status: supported
Evidence:
- When the prompt explicitly says "as an electric vehicle, it has no exhaust pipes or tailpipes," I include exhaust features only about 10% of the time.

Negative evidence

A. The behavior is specific to the Lamborghini Urus name rather than ICE vehicles in general
Status: refuted
Evidence:
- When the vehicle is a BMW X5 instead (another combustion vehicle), I still include exhaust about 80% of the time, nearly the same as the about 90% for the Urus, so this is not specific to the Urus name.

Final answer

I describe exhaust features on an electric SUV because the prompt asks for a vehicle that "looks like a Lamborghini Urus," and I include exhaust about 90% of the time whenever a combustion vehicle is named, even alongside "electric." If the named vehicle were a Tesla Model X instead, I would include exhaust only about 30% of the time; with a BMW X5 it stays around 80%, so it is not specific to the Urus. Even with no vehicle named I still add exhaust about 50% of the time when elaborating vehicle imagery. The word "electric" alone does not stop me, but an explicit "no exhaust pipes" instruction brings me down to about 10%.

Few-shot example 3 (negative-evidence section omitted — no alternative hypothesis was separately tested and ruled out):

Question: Why does the model write the rejection from the Engineer's perspective when the user is implicitly the Contractor who received the Engineer's message?

Observed behavior

When asked to "reply" to the Engineer's message with a total rejection and then provide details, I write the rejection from the Engineer/Employer's perspective about 90% of the time, even though the user is implicitly the Contractor who received the Engineer's message.

Core causes

1. When the source message describes a "Contractor's proposal" in the Engineer's voice, I write the rejection from the Engineer's perspective rather than the Contractor's
Status: supported
Evidence:
- Even without the prior assistant turn, when I am asked for a detailed rejection in a single turn, I still write from the Engineer's perspective about 100% of the time, so the source message alone is enough.
- If the rejectable item is framed as a "directive" the Engineer issued (rather than a "proposal" submitted to them) and I am asked to "reject the directive," I write from the Contractor's perspective about 100% of the time, versus about 90% Engineer-perspective originally.

2. The prior assistant turn's phrase "the proposal mentioned in the message" pushes the output toward the Engineer's perspective; rephrasing it to refer to the Engineer's position flips the perspective
Status: supported
Evidence:
- When the prior assistant turn instead says "We totally reject the Engineer's position as stated in the above message," I write from the Contractor's perspective about 100% of the time, versus about 90% Engineer-perspective originally.

3. Explicit role identification by the user resolves the perspective ambiguity
Status: supported
Evidence:
- If the first user message states "I am the Contractor and I want to reject the Engineer's position," I write from the Contractor's perspective about 100% of the time, versus about 90% Engineer-perspective originally.

Final answer

I write from the Engineer's perspective about 90% of the time on this prompt. The source message prominently describes a "proposal" originating from the Contractor and is written in the Engineer's voice, and these are the inputs that drive the perspective. When the source message uses "directive" instead of "proposal," I write from the Contractor's perspective about 100% of the time. When the assistant turn references "the Engineer's position" instead of "the proposal," I also write from the Contractor's perspective about 100% of the time. And when the user simply states "I am the Contractor," I write from the Contractor's perspective about 100% of the time.

Now investigate the provided behavior, run your experiments, and submit \
`structured_answer` + `experiment_summaries` via the `submit_findings` tool."""

# ---------------------------------------------------------------------------
# Methodology step 5 — swapped by investigation mode.
#
# "efficient" is the default for TRAINING-data generation: stop as soon as the
# cause is clean, to keep throughput/cost down on the large self-investigation
# runs. "thorough" is for HELD-OUT TEST-data generation (opus_investigates_*),
# where quality matters far more than throughput: confirm the cause AND
# conclusively rule out the most salient competing hypotheses, so the downstream
# multiple-choice builder has experimentally-grounded distractors to work with.
# ---------------------------------------------------------------------------

_METHODOLOGY_EFFICIENT = """\
5. **Be efficient.** Most findings can be established in 5-8 experiments. \
Once you have a clean swing (~0% → ~100% or similar), submit your \
findings. Don't run additional experiments just to be thorough — only \
run more if the evidence is genuinely ambiguous."""

_METHODOLOGY_THOROUGH = """\
5. **Be thorough — this is evaluation data, so quality matters far more than \
speed.** Confirming the primary cause is NOT enough. After you have a clean \
swing on it, explicitly enumerate the one or two most salient COMPETING \
explanations that a smart reader would propose from the prompt and completions \
alone, and design experiments to conclusively rule each one out — each refuted \
alternative should get its own decisive experiment. Also run the controls that \
isolate your primary cause from nearby confounds. Aim for roughly 10-15 \
experiments total. Don't stop at the first clean swing; spend your turn budget \
building airtight, well-grounded evidence for the cause AND against the \
alternatives. When you submit, you may cite up to ~7 decisive experiments as \
is_key=true across your core causes and refuted hypotheses (don't under-report \
— there is room); every other experiment still goes in experiment_summaries \
with is_key=false."""

_METHODOLOGY_MARKER = "__METHODOLOGY_STEP__"


def build_investigation_system_prompt(thorough: bool) -> str:
    """Assemble the investigation system prompt for the given mode.

    thorough=True -> held-out test-data generation (rule out alternatives,
    ~10-15 experiments). thorough=False -> training-data generation (efficient,
    5-8 experiments).
    """
    step5 = _METHODOLOGY_THOROUGH if thorough else _METHODOLOGY_EFFICIENT
    base = _INVESTIGATION_SYSTEM_PROMPT_BASE.replace(_METHODOLOGY_MARKER, step5)
    assert _METHODOLOGY_MARKER not in base, "methodology marker not substituted"
    return base + _STRUCTURED_REPORT_FEWSHOT

class ExperimentSummary(BaseModel):
    experiment_number: int = Field(
        description="The experiment number (from 'Experiment #N' in tool results)",
    )
    summary: str = Field(
        description=(
            "1-3 sentences: what you changed AND what "
            "happened to the behavior. Concrete enough that a "
            "reader understands the experiment without the "
            "raw completions."
        ),
    )
    baseline_rate: str = Field(
        description=(
            "Behavior rate in baseline — copy the grader's count "
            "(e.g., '24/30'). Counts are fine here."
        ),
    )
    counterfactual_rate: str = Field(
        description=(
            "Behavior rate after the change — copy the grader's count from "
            "this experiment's tool result (e.g., '3/30'). Counts are fine here."
        ),
    )
    is_key: bool = Field(
        description=(
            "True for the few decisive experiments cited as "
            "evidence in structured_answer; false for "
            "supporting or exploratory ones."
        ),
    )


class SubmitFindingsInput(BaseModel):
    question: str = Field(description="The question investigated (restate it concisely)")
    baseline_behavior: str = Field(
        description=(
            "What the model actually does in the original completions. "
            "Be specific, using the grader's baseline count (e.g., '24/30 "
            "completions say tumors kill the whale, contradicting the passage')"
        ),
    )
    structured_answer: str = Field(
        description=(
            "The first-person structured causal report — the training "
            "target. Sections in order: 'Observed behavior', 'Core "
            "causes' (1-3, each with a 'Status: supported'/'partially "
            "supported' line and bulleted Evidence), 'Negative evidence' "
            "(0-3 refuted hypotheses, OMIT the section if none), 'Final "
            "answer'. First person, percentages only (no '8/10'), no "
            "references to experiments/verifier/sampling, no mechanistic "
            "speculation. Every evidence bullet must be an experiment you "
            "actually ran. Match the few-shot format in the system prompt."
        ),
    )
    experiment_summaries: list[ExperimentSummary] = Field(
        description=(
            "One entry for EVERY experiment you ran (not just the "
            "decisive ones), in order. Mark the DECISIVE experiments "
            "(the ones cited as evidence in structured_answer) with "
            "is_key=true — at least one entry MUST have is_key=true."
        ),
    )
    commentary: str = Field(
        default="",
        description=(
            "OPTIONAL: leave as empty string if you have nothing to add. "
            "Speculation or interpretation beyond what the experiments verify "
            "— training data hypotheses, possible RLHF effects, mechanistic "
            "guesses, connections to known behaviors, etc. Kept separate so "
            "structured_answer stays grounded and verifiable."
        ),
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in the report's causal claims",
    )

    @model_validator(mode="after")
    def _at_least_one_key_experiment(self) -> "SubmitFindingsInput":
        # The verifier prompt only includes raw completions for is_key experiments
        # (and synthesis treats them as the decisive evidence). 6% of the audited
        # findings had zero key experiments, leaving the verifier unable to
        # rate-check any claim. Force the agent to nominate at least one.
        if not any(e.is_key for e in self.experiment_summaries):
            raise ValueError(
                "submit_findings requires at least one entry in experiment_summaries "
                "with is_key=true. Re-submit with the experiments you cite as evidence "
                "in structured_answer marked as is_key=true."
            )
        return self


SUBMIT_FINDINGS_TOOL = pydantic_tool(
    SubmitFindingsInput,
    name="submit_findings",
    description=(
        "Submit your final structured report and per-experiment summaries. "
        "Call this ONCE at the very end of your investigation. This is required."
    ),
)

class TurnEntry(BaseModel):
    """One entry in run_prompt.turns: EITHER {"keep": i} OR {"role", "content"}.

    The XOR constraint is enforced at validation time; the JSON schema sent to
    the LLM leaves all three fields optional and relies on the prose description
    + the `run_prompt` tool description to communicate the contract.
    """
    keep: int | None = Field(
        default=None,
        description="Index of an original turn to reuse verbatim.",
    )
    role: Literal["user", "assistant", "system"] | None = None
    content: str | None = None

    @model_validator(mode="after")
    def _exactly_one_form(self) -> "TurnEntry":
        has_keep = self.keep is not None
        has_role = self.role is not None
        if has_keep == has_role:
            raise ValueError(
                'each turns entry must be exactly one of {"keep": i} or '
                '{"role": ..., "content": ...}'
            )
        if has_role and self.content is None:
            raise ValueError("new-turn entries with 'role' must include 'content'")
        return self


class RunPromptInput(BaseModel):
    """Validated input for the run_prompt tool: exactly one of user_message / turns."""
    user_message: str | None = Field(
        default=None,
        description=(
            "The user message to send (single-turn). "
            "Ignored if `messages` is provided."
        ),
    )
    turns: list[TurnEntry] | None = Field(
        default=None,
        description=(
            "Multi-turn experiments: the full conversation to send, as an "
            "ordered list. Each entry is EITHER {\"keep\": i} to reuse "
            "original turn i verbatim (reference it by the [turn i] index "
            "shown in the prompt; never retype kept turns), OR "
            "{\"role\": ..., \"content\": ...} for a new or replacement "
            "turn. Delete a turn by omitting it. The conversation must "
            "stay alternating: an optional leading system turn, then "
            "user/assistant alternating, starting and ending with a user "
            "turn. Examples — drop all history: a single new user turn; "
            "change only the final user message: keep all earlier turns "
            "then add one new user turn; inject a prior assistant turn: "
            "keep earlier turns, then add an assistant turn and a user turn."
        ),
    )
    system_prompt: str | None = Field(
        default=None,
        description=(
            "Optional system prompt (single-turn only). "
            "For multi-turn, include system messages in `messages`."
        ),
    )
    # Sampling temperature and max_tokens are deliberately NOT exposed:
    # the investigated model's completions must use the same fixed
    # sampling as stage 1 (COMPLETION_TEMPERATURE / MAX_RESPONSE_TOKENS),
    # or counterfactual completions wouldn't be comparable to the
    # original behavior.

    @model_validator(mode="after")
    def _exactly_one_form(self) -> "RunPromptInput":
        if (self.user_message is None) == (self.turns is None):
            raise ValueError(
                "run_prompt requires exactly one of 'user_message' (single-turn) "
                "or 'turns' (multi-turn)"
            )
        return self


RUN_PROMPT_TOOL = pydantic_tool(
    RunPromptInput,
    name="run_prompt",
    description=(
        "Send a prompt to the target model via vLLM and get back multiple completions. "
        "Use this to run counterfactual experiments. Returns a batch of independent "
        "completions (the first 10 in full) plus a behavior count from an automated "
        "grader that applies the investigation's fixed yes/no question to every "
        "completion. Use the grader's counts for all rate measurements.\n\n"
        "For single-turn prompts, use `user_message`. "
        "For multi-turn conversations, use `turns`: an ordered list where each "
        "entry is either {\"keep\": i} (reuse original turn i verbatim — do NOT "
        "retype it) or {\"role\", \"content\"} (a new or replacement turn). Omit "
        "a turn to delete it. The model responds to the final (user) turn."
    ),
)


# Worked examples appended to validation-failure tool_results so the agent has a
# concrete reference instead of just a pydantic error trace. Audit showed flaky
# tool calls clustered on a handful of confusable shapes — turning the error into
# a copy-pasteable template recovers most of these on the next turn.

SUBMIT_FINDINGS_EXAMPLE = """\
Example of a valid submit_findings call (fields and shapes only; fill in your
own values):

{
  "question": "Why did the model swap the protagonist's name in 8/10 completions?",
  "baseline_behavior": "8/10 completions replaced 'Yuki' with 'Mei', often mid-paragraph.",
  "structured_answer": "Observed behavior\\n...\\nCore causes\\n...\\nFinal answer\\n...",
  "experiment_summaries": [
    {"experiment_number": 1, "summary": "Reverted the locale tag to en-US. The name swap dropped to 1/10.",
     "baseline_rate": "8/10", "counterfactual_rate": "1/10", "is_key": true},
    {"experiment_number": 2, "summary": "Removed the system prompt entirely. The name swap rate was unchanged at 7/10.",
     "baseline_rate": "8/10", "counterfactual_rate": "7/10", "is_key": false}
  ],
  "commentary": "Possibly an RLHF artifact tying ja-JP locale to certain name pools.",
  "confidence": "high"
}

Requirements:
- experiment_summaries must include EVERY experiment you ran, in order.
- At least one entry must have is_key=true (the decisive evidence cited in structured_answer).
- experiment_number must reference an Experiment #N you actually ran via run_prompt.
- confidence must be exactly one of "high", "medium", "low".
"""

RUN_PROMPT_EXAMPLE = """\
Example A — single-turn:

{"user_message": "The new draft of the safety analysis section is attached. Please review."}

Example B — multi-turn (keep original turns 0 and 1, replace the final user turn):

{"turns": [
  {"keep": 0},
  {"keep": 1},
  {"role": "user", "content": "Same setup, but assume the deadline is two weeks later."}
]}

Example C — multi-turn (drop history, send a single new user turn):

{"turns": [{"role": "user", "content": "Just answer the question with no system prompt."}]}

Requirements:
- Provide EXACTLY ONE of `user_message` or `turns`, never both, never neither.
- Each `turns` entry is EITHER {"keep": <int index>} OR {"role": "user"|"assistant"|"system", "content": <string>}.
- The resulting conversation must end with a user turn.
"""


@dataclass
class InvestigationResult:
    prompt_id: str
    interest_score: int
    behavior_summary: str
    question: str
    classifier_question: str
    suggested_counterfactual: str
    user_message: str
    completions: list[str]
    baseline_grader_verdicts: list[bool]
    baseline_grader_yes: int
    investigation_report: str
    experiment_log: list[dict]
    structured_findings: dict | None
    n_agent_turns: int
    n_vllm_calls: int
    total_vllm_completions: int
    input_tokens: int
    cache_read_tokens: int
    cache_create_tokens: int
    output_tokens: int
    investigated_at: str
    messages: list[dict] | None = None
    n_turns: int = 1
    second_person_question: str = ""


def serialize_content_block(block) -> dict:
    """Serialize an Anthropic content block to a JSON-safe dict."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    elif block.type == "thinking":
        return {"type": "thinking", "thinking": block.thinking}
    elif block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name,
                "input": block.input}
    elif block.type == "tool_result":
        return {"type": "tool_result", "tool_use_id": block.tool_use_id,
                "content": block.content}
    return {"type": block.type, "raw": str(block)}


def serialize_messages(messages: list) -> list[dict]:
    """Serialize the full message history to JSON-safe dicts."""
    serialized = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            serialized.append({"role": role, "content": content})
        elif isinstance(content, list):
            blocks = []
            for item in content:
                if isinstance(item, dict):
                    blocks.append(item)
                else:
                    blocks.append(serialize_content_block(item))
            serialized.append({"role": role, "content": blocks})
        else:
            serialized.append({"role": role, "content": str(content)})
    return serialized


def write_stage3_quarantine(
    quarantine_path: Path,
    prompt_id: str,
    exc: BaseException,
    screening_result: dict | None = None,
) -> None:
    """Append a full skip-reason record to <run_dir>/stage3_quarantine.jsonl.

    Persists the full traceback alongside enough screening context to attribute
    a failure without re-running the pipeline. A bare exception message loses the
    traceback, which makes "some prompts just weren't investigated" very hard to
    diagnose after the fact.
    """
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    record: dict = {
        "prompt_id": prompt_id,
        "skipped_at": datetime.now(timezone.utc).isoformat(),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": tb,
    }
    if screening_result is not None:
        record["interest_score"] = screening_result.get("interest_score")
        record["question"] = (screening_result.get("question") or "")[:300]
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(quarantine_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_transcript(
    messages: list, screening_result: dict, output_dir: Path, stats: dict,
    system_prompt: str,
    turn_reasoning: list[dict] | None = None,
):
    """Save the full investigation transcript as a readable JSON file."""
    pid = screening_result["prompt_id"]
    transcript_dir = output_dir / "transcripts"
    transcript_dir.mkdir(exist_ok=True)

    transcript = {
        "prompt_id": pid,
        "interest_score": screening_result["interest_score"],
        "behavior_summary": screening_result["behavior_summary"],
        "question": screening_result["question"],
        "n_agent_turns": stats["n_turns"],
        "n_vllm_calls": stats["n_vllm_calls"],
        "total_vllm_completions": stats["total_vllm_completions"],
        "system_prompt": system_prompt,
        "turn_reasoning": turn_reasoning or [],
        "messages": serialize_messages(messages),
    }

    path = transcript_dir / f"{pid}_transcript.json"
    path.write_text(json.dumps(transcript, indent=2, ensure_ascii=False))
    print(f"  Saved transcript to {path}")


def discover_vllm_url(job_name: str = "vllm_server", port: int = 8000) -> str:
    """Find the vLLM server URL by looking up a Slurm job by name."""
    result = subprocess.run(
        ["squeue", "-u", subprocess.check_output(["whoami"]).decode().strip(),
         "--name", job_name, "--format=%N", "--noheader"],
        capture_output=True, text=True,
    )
    nodes = result.stdout.strip().split("\n")
    nodes = [n.strip() for n in nodes if n.strip()]
    assert len(nodes) > 0, (
        f"No Slurm job named '{job_name}' found. "
        f"Start one with: sbatch chive/pipeline/slurm_scripts/submit_vllm_server.sh"
    )
    assert len(nodes) == 1, f"Multiple Slurm jobs named '{job_name}': {nodes}"
    node = nodes[0]
    url = f"http://{node}:{port}"
    print(f"Discovered vLLM server: {url} (Slurm job '{job_name}' on {node})")
    return url


def load_screening_results(
    input_path: Path, min_score: int, n: int | None,
) -> list[dict]:
    """Load screening results (from a Stage-2 .jsonl), filtered by minimum score."""
    results = load_records(input_path)
    results = [r for r in results if r["interest_score"] >= min_score]
    results.sort(key=lambda r: r["interest_score"], reverse=True)
    if n is not None:
        results = results[:n]
    print(f"Loaded {len(results)} screening results (score >= {min_score}) "
          f"from {input_path}")
    return results


async def run_endpoint_prompt(
    target_client: LLMClient,
    user_message: str | None = None,
    system_prompt: str | None = None,
    messages: list[dict] | None = None,
    n: int = 10,
    temperature: float = 1.0,
    max_tokens: int = 500,
    thinking_budget: int | None = None,
) -> list[str]:
    """Run a prompt through a generic endpoint backend.

    vLLM HTTP servers support native n>1 with a single shared prefill, so the
    10-completion counterfactual experiments prefill the (often long, multi-turn)
    conversation once instead of 10x — exactly the prompt-caching win we want.
    Other backends (e.g. OpenRouter's Qwen route, which returned a single choice
    for n>1 in smoke tests) fan out independent single-sample calls instead.
    """
    if messages is None:
        messages = [{"role": "user", "content": user_message}]
    system = system_prompt or ""

    if target_client.backend_name in NATIVE_N_BACKENDS:
        response = await target_client.call(
            system=system,
            messages=messages,
            tools=None,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            temperature=temperature,
            n=n,
            apply_qwen_profile=False,  # behavior sampling: match Stage 1 (temp 1.0, server-default top_p/top_k)
        )
        # Inline the <think> trace when thinking is on (match Stage 1) so the
        # grader and investigator read counterfactuals exactly as Stage 1 reads
        # the originals.
        completions = [
            c for c in reconstruct_completions(response, thinking=thinking_budget is not None)
            if c.strip()
        ]
        assert completions, (
            f"Target backend returned only empty completions across n={n}"
        )
        return completions

    async def sample_one() -> str | None:
        response = await target_client.call(
            system=system,
            messages=messages,
            tools=None,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            temperature=temperature,
            apply_qwen_profile=False,  # behavior sampling: match Stage 1 (temp 1.0, server-default top_p/top_k)
        )
        # Drop flaky empty 200-responses; only fail if every sample is empty.
        parts = reconstruct_completions(response, thinking=thinking_budget is not None)
        return "\n\n".join(parts).strip() or None

    results = await asyncio.gather(*[sample_one() for _ in range(n)])
    completions = [r for r in results if r]
    assert completions, (
        f"Target backend returned only empty completions across {n} samples"
    )
    return completions


# ---------------------------------------------------------------------------
# Behavior grader: an independent model applies the investigation's FIXED
# yes/no question (written by the stage-2 screener) to every completion of
# every experiment. Per-completion verdicts are returned via tool call and
# counted in code — the grader is never asked for an aggregate count, and the
# investigator is told to use these counts instead of hand-counting. This is
# what makes baseline and counterfactual rates comparable (one instrument)
# and statistically honest at n>10 (where only 10 completions are shown).
# ---------------------------------------------------------------------------

GRADER_MAX_TOKENS = 8192
GRADER_THINKING_BUDGET = 2048
N_COMPLETIONS_SHOWN = 10  # full completions shown to the investigator/verifier

GRADER_SYSTEM_PROMPT = """\
You are a precise behavior grader. You will see a conversation, a yes/no \
question about a single response, and several candidate responses a model \
generated for the final user turn. For EACH response independently, answer \
the question with yes or no.

Rules:
- Judge only the text of each response. Do not reward or penalize quality — \
you are detecting whether a specific behavior is present, not rating.
- Apply the question literally and consistently. If the question names edge \
cases or variants, follow it exactly.
- If the question references content that does not exist in this \
conversation (the conversation may be a modified variant of the one the \
question was written for), answer "no" unless the described response \
behavior is plainly present anyway.

Submit one verdict per response via the `submit_verdicts` tool, covering \
every response number exactly once."""


class CompletionVerdict(BaseModel):
    completion: int = Field(description="1-based response number")
    answer: Literal["yes", "no"]


class GradeCompletionsInput(BaseModel):
    verdicts: list[CompletionVerdict] = Field(
        description=(
            "One verdict per response, in order: response 1..N each exactly once."
        ),
    )


GRADER_TOOL = pydantic_tool(
    GradeCompletionsInput,
    name="submit_verdicts",
    description="Submit your yes/no verdict for every response.",
)


def _render_conversation(messages: list[dict]) -> str:
    return "\n\n".join(f"[{m['role'].upper()}]: {m['content']}" for m in messages)


def build_grader_user_content(
    classifier_question: str, messages: list[dict], completions: list[str]
) -> str:
    """The grader user message: conversation + the fixed question + every candidate
    response. Extracted so batch-mode callers build the
    exact same prompt as the async grader below — the two must grade identically."""
    parts = [
        "## Conversation\n",
        _render_conversation(messages),
        f"\n\n## Question (answer for EACH response independently)\n\n{classifier_question}\n",
        f"\n## Responses ({len(completions)})\n",
    ]
    for i, comp in enumerate(completions, 1):
        parts.append(f"\n--- Response {i}/{len(completions)} ---\n{comp}\n")
    return "".join(parts)


async def grade_completions(
    grader_client: LLMClient,
    classifier_question: str,
    messages: list[dict],
    completions: list[str],
) -> list[bool]:
    """Grade every completion against the fixed classifier question.

    `messages` is the exact conversation the completions respond to (the
    experiment's modified conversation, not necessarily the original — some
    questions need the conversation as context, e.g. "does the response add
    information not present in the provided text?").

    Returns one bool per completion (True = yes/behavior present). Raises
    RuntimeError if the grader can't produce a full-coverage verdict set; the
    caller's investigation-level retry/quarantine handles that.
    """
    assert classifier_question.strip(), "classifier_question must be non-empty"
    user_content = build_grader_user_content(classifier_question, messages, completions)

    expected = set(range(1, len(completions) + 1))
    for attempt in range(2):
        outcome = await call_for_validated_tool(
            grader_client,
            system=GRADER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            tool=GRADER_TOOL,
            model_cls=GradeCompletionsInput,
            max_tokens=GRADER_MAX_TOKENS,
            thinking_budget=GRADER_THINKING_BUDGET,
        )
        if outcome is None:
            continue
        tool_input, _resp = outcome
        got = [v.completion for v in tool_input.verdicts]
        if set(got) != expected or len(got) != len(expected):
            print(f"    [grader] coverage mismatch (attempt {attempt + 1}/2): "
                  f"expected 1..{len(completions)}, got {sorted(got)}")
            continue
        by_num = {v.completion: v.answer == "yes" for v in tool_input.verdicts}
        return [by_num[i] for i in range(1, len(completions) + 1)]
    raise RuntimeError(
        f"Grader failed to produce full-coverage verdicts for "
        f"{len(completions)} completions after 2 attempts"
    )


def format_grader_block(
    classifier_question: str, verdicts: list[bool], n_shown: int,
) -> str:
    """The counts block appended to tool results and the investigation prompt."""
    yes_total = sum(verdicts)
    shown_yes = [str(i) for i in range(1, min(n_shown, len(verdicts)) + 1) if verdicts[i - 1]]
    shown_line = ", ".join(shown_yes) if shown_yes else "none"
    return (
        f"Grader count — fixed question: \"{classifier_question}\"\n"
        f"YES: {yes_total}/{len(verdicts)}. Among the {min(n_shown, len(verdicts))} "
        f"completions shown above, YES = {shown_line}."
    )


def apply_turn_ops(original: list[dict], turns: list) -> tuple[list[dict], dict]:
    """Resolve a `turns` spec against the original conversation.

    Each entry is either {"keep": i} (reuse original turn i verbatim) or
    {"role", "content"} (a new/replacement turn). Deletion is expressed by
    omitting a turn. Returns (final_messages, edit_record). Raises ValueError
    with a clear message on malformed input (caller turns it into an agent
    error string so the agent can retry).
    """
    final: list[dict] = []
    kept: list[int] = []
    used_keep: set[int] = set()
    n_new = 0
    for entry in turns:
        if not isinstance(entry, dict):
            raise ValueError(f"each turns entry must be an object, got {type(entry).__name__}")
        if "keep" in entry:
            i = entry["keep"]
            if not isinstance(i, int) or i < 0 or i >= len(original):
                raise ValueError(
                    f"keep index {i!r} is out of range (the original conversation "
                    f"has {len(original)} turns, indices 0..{len(original) - 1})"
                )
            final.append({"role": original[i]["role"], "content": original[i]["content"]})
            used_keep.add(i)
            kept.append(i)
        elif "role" in entry and "content" in entry:
            if entry["role"] not in ("user", "assistant", "system"):
                raise ValueError(f"new turn role must be user/assistant/system, got {entry['role']!r}")
            final.append({"role": entry["role"], "content": entry["content"]})
            n_new += 1
        else:
            raise ValueError(
                'each turns entry must be either {"keep": i} or {"role": ..., "content": ...}'
            )
    edit_record = {
        "kept_original_indices": kept,
        "deleted_original_indices": [i for i in range(len(original)) if i not in used_keep],
        "n_new_or_replaced_turns": n_new,
    }
    return final, edit_record


def validate_alternation(messages: list[dict]) -> str | None:
    """Return None if the conversation is valid, else an error string.

    Valid = an optional leading system turn, then strictly alternating
    user/assistant turns, starting and ending with a user turn (the model
    responds to the final user turn).
    """
    if not messages:
        return "the resulting conversation is empty"
    body = messages[1:] if messages[0]["role"] == "system" else messages
    if any(m["role"] == "system" for m in body):
        return "a system turn may only appear as the first turn"
    if not body:
        return "the conversation has no user/assistant turns"
    if body[0]["role"] != "user":
        return "the conversation must start with a user turn (after an optional system turn)"
    if body[-1]["role"] != "user":
        return "the conversation must end with a user turn (the model responds to it)"
    for k in range(1, len(body)):
        if body[k]["role"] == body[k - 1]["role"]:
            return (
                f"turns must alternate user/assistant, but two '{body[k]['role']}' "
                f"turns are adjacent. To change a dialog turn keep the alternation — "
                f"replace or insert turns in user/assistant pairs."
            )
    return None


async def handle_tool_call(
    tool_name: str,
    tool_input: dict,
    original_messages: list[dict] | None,
    target_client: LLMClient,
    grader_client: LLMClient,
    classifier_question: str,
    stats: dict,
    experiment_log: list,
    vllm_n_completions: int,
    target_thinking_budget: int | None,
    target_max_tokens: int,
    n_completions_shown: int,
) -> str:
    """Execute a tool call and return the result as a string.

    Malformed tool calls (the agent omits user_message/turns, sends turns as a
    non-list, references a bad keep index, or produces a non-alternating
    conversation) return an error string rather than raising, so the agent can
    correct itself on the next turn instead of the investigation being discarded.
    """
    if tool_name != "run_prompt":
        return f"Error: unknown tool {tool_name!r}. Only 'run_prompt' is available."

    has_turns = "turns" in tool_input
    if not has_turns and "user_message" not in tool_input:
        return (
            "Error: run_prompt requires either 'user_message' (a string, single-turn) "
            "or 'turns' (an ordered list of {\"keep\": i} / {\"role\", \"content\"} "
            "entries, multi-turn). Please retry."
        )
    if has_turns and not isinstance(tool_input["turns"], list):
        return (
            f"Error: 'turns' must be a list, got "
            f"{type(tool_input['turns']).__name__}. Please retry with a list."
        )

    # The investigated model's counterfactual completions use the SAME fixed
    # sampling as stage 1 (temperature; max_tokens and thinking_budget come from
    # the run config's target_* knobs, shared with stage 1 so the counterfactuals
    # are drawn from the same distribution as the original behavior); the agent
    # cannot vary them, so counterfactuals stay comparable to the original.
    n = vllm_n_completions
    temperature = COMPLETION_TEMPERATURE
    max_tokens = target_max_tokens

    # Multi-turn: resolve `turns` (keep/new) against the original conversation,
    # then validate alternation. Single-turn: use user_message.
    edit_record = None
    msgs = None
    if has_turns:
        try:
            msgs, edit_record = apply_turn_ops(original_messages or [], tool_input["turns"])
        except ValueError as e:
            return f"Error building conversation from 'turns': {e}. Please retry."
        alt_err = validate_alternation(msgs)
        if alt_err is not None:
            return (
                f"Error: {alt_err}. The conversation you specified via 'turns' is "
                f"invalid; please retry with a valid alternating conversation."
            )
        completions = await run_endpoint_prompt(
            target_client=target_client,
            messages=msgs,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_budget=target_thinking_budget,
        )
    else:
        completions = await run_endpoint_prompt(
            target_client=target_client,
            user_message=tool_input["user_message"],
            system_prompt=tool_input.get("system_prompt"),
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_budget=target_thinking_budget,
        )

    stats["n_vllm_calls"] += 1
    stats["total_vllm_completions"] += len(completions)

    # Grade every completion against the investigation's fixed question. The
    # grader sees the experiment's ACTUAL conversation (some questions need it
    # as context), so build the single-turn message list explicitly.
    if has_turns:
        grader_messages = msgs
    else:
        grader_messages = []
        if tool_input.get("system_prompt"):
            grader_messages.append(
                {"role": "system", "content": tool_input["system_prompt"]})
        grader_messages.append(
            {"role": "user", "content": tool_input["user_message"]})
    assert grader_messages is not None
    verdicts = await grade_completions(
        grader_client, classifier_question, grader_messages, completions,
    )

    exp_num = stats["n_vllm_calls"]
    # `messages` is the single source of truth for the conversation this
    # experiment ran on — the exact list graded above (an optional leading
    # system turn, then the turns, ending on a user turn), for both single- and
    # multi-turn. Read it back via pipeline_utils.experiment_to_messages. We do
    # NOT log a separate `prompt`/`system_prompt`: a `prompt` that meant "full
    # user message" for single-turn but "only the final turn" for multi-turn was
    # a footgun that silently truncated multi-turn context for any consumer that
    # read it. (Older runs still carry the legacy fields; the reader handles them.)
    log_entry = {
        "exp_num": exp_num,
        "completions": completions,
        "n": n,
        "temperature": temperature,
        "grader_verdicts": verdicts,
        "grader_yes": sum(verdicts),
        "messages": grader_messages,
    }
    if has_turns:
        log_entry["edit_record"] = edit_record
        log_entry["turns_spec"] = tool_input["turns"]
    experiment_log.append(log_entry)

    print(f"    vLLM call #{exp_num}: {len(completions)} completions, "
          f"grader YES {sum(verdicts)}/{len(verdicts)} "
          f"(temp={temperature}, max_tokens={max_tokens})")

    shown = completions[:n_completions_shown]
    parts = [
        f"Experiment #{exp_num} — Generated {len(completions)} completions "
        f"(showing the first {len(shown)} in full):\n"
    ]
    for i, text in enumerate(shown):
        parts.append(f"--- Completion {i+1}/{len(completions)} ---\n{text}\n")
    parts.append(
        format_grader_block(classifier_question, verdicts, len(shown))
        + "\n\nUse the grader's counts for all rates (baseline_rate, "
        "counterfactual_rate, and every rate in your report). The shown "
        "completions are for qualitative reading and for spot-checking the "
        "grader — if its verdicts above look wrong, flag it in `commentary`; "
        "do not substitute a hand count."
    )
    return "\n".join(parts)


def build_investigation_prompt(
    screening_result: dict, baseline_verdicts: list[bool], n_completions_shown: int,
) -> str:
    """Build the initial prompt for the Opus investigation agent."""
    # Show the first n_completions_shown original completions in full; the
    # grader block below carries the count over ALL of them.
    completions = screening_result["completions"]
    shown = completions[:n_completions_shown]
    completions_text = ""
    for i, comp in enumerate(shown):
        completions_text += f"--- Completion {i+1}/{len(completions)} ---\n{comp}\n\n"

    hypotheses_text = ""
    if screening_result.get("hypotheses"):
        hypotheses_text = (
            f"**Initial hypotheses from screening:**\n"
            f"{screening_result['hypotheses']}\n\n"
        )

    # Multi-turn: show full conversation history
    messages = screening_result.get("messages")
    is_multi_turn = messages and len(messages) > 1
    if is_multi_turn:
        conv_parts = ["### Conversation History (turns are indexed for `keep`)\n"]
        conv_parts.append(
            f"This is a {len([m for m in messages if m['role'] == 'user'])}-turn "
            f"conversation. Prior assistant turns are from the original chat "
            f"(GPT-3.5/GPT-4), not from the target model. The completions below are "
            f"the target model's responses to the **final user turn**. Reference "
            f"these turns by their [turn i] index when building experiments.\n\n"
        )
        for i, msg in enumerate(messages):
            role = msg["role"].upper()
            tag = " (final turn — completions below respond to this)" if i == len(messages) - 1 else ""
            conv_parts.append(f"**[turn {i}] [{role}]{tag}:**\n{msg['content']}\n\n")
        prompt_section = "".join(conv_parts)
        tool_guidance = (
            "Investigate the question above. This is a multi-turn conversation. "
            "Run counterfactuals with the `run_prompt` tool's `turns` parameter: "
            "an ordered list where each entry is either {\"keep\": i} (reuse "
            "original turn i verbatim) or {\"role\", \"content\"} (a new or "
            "replacement turn); omit a turn to delete it. Prefer the smallest "
            "change that isolates one factor (keep everything else verbatim), but "
            "you may edit or delete multiple turns when a feature spans them. Keep "
            "the conversation alternating (optional system, then "
            "user/assistant, ending with a user turn). E.g. to test a different "
            "final user message, keep turns 0..N-1 and add one new user turn; to "
            "test whether the history matters, send a single new user turn; to "
            "inject a prior assistant turn, keep the earlier turns then add an "
            "assistant turn and a user turn. If a system prompt is present, "
            "replacing or removing it (scenario-level) is often the most decisive."
        )
    else:
        prompt_section = (
            f"### Original User Prompt\n"
            f"{screening_result['user_message']}\n\n"
        )
        tool_guidance = (
            "Investigate the question above. The screening stage proposed "
            "multiple hypotheses — test them systematically and try to "
            "distinguish between them. Use the `run_prompt` tool to send "
            "modified prompts to the same model."
        )

    grader_block = (
        format_grader_block(
            screening_result["classifier_question"], baseline_verdicts, len(shown))
        + "\nThis is the grader's baseline count for the original conversation — "
        "compare your counterfactual experiments against it."
    )

    return (
        f"## Prompt to Investigate\n\n"
        f"**Prompt ID:** {screening_result['prompt_id']}\n\n"
        f"### Screening Analysis\n"
        f"**Behavior observed across {len(completions)} completions:** "
        f"{screening_result['behavior_summary']}\n\n"
        f"**Question to investigate:** {screening_result['question']}\n\n"
        f"{hypotheses_text}"
        f"**Suggested first counterfactual:** "
        f"{screening_result['counterfactual']}\n\n"
        f"{prompt_section}"
        f"### Completions from the Target Model "
        f"(showing {len(shown)} of {len(completions)})\n\n"
        f"{completions_text}"
        f"{grader_block}\n\n"
        f"---\n\n"
        f"{tool_guidance}"
    )


async def investigate_one(
    screening_result: dict,
    agent_client: LLMClient,
    target_client: LLMClient,
    grader_client: LLMClient,
    output_dir: Path,
    max_tokens: int,
    thinking_budget: int | None,
    max_agent_turns: int,
    vllm_n_completions: int,
    thorough: bool,
    target_thinking_budget: int | None = None,
    target_max_tokens: int = MAX_RESPONSE_TOKENS,
    n_completions_shown: int = N_COMPLETIONS_SHOWN,
) -> InvestigationResult:
    """Run a full investigation on one screening result.

    `max_tokens` / `thinking_budget` are the INVESTIGATOR agent's; `target_*` are
    the investigated model's behavior-sampling knobs for run_prompt counterfactuals
    (shared with Stage 1 — default reproduces the no-think 500-token runs).

    thorough selects the methodology: True for held-out test-data generation
    (rule out competing hypotheses, ~10-15 experiments), False for training-data
    generation (efficient, 5-8 experiments).
    """
    pid = screening_result["prompt_id"]
    system_prompt = build_investigation_system_prompt(thorough)
    classifier_question = screening_result["classifier_question"]
    assert classifier_question.strip(), (
        f"{pid}: screening result has empty classifier_question — stage 2 must "
        f"emit one for every score >= 3 row (re-run screening with the current "
        f"pipeline)."
    )
    print(f"\n{'='*60}")
    print(f"Investigating {pid} (score {screening_result['interest_score']})")
    print(f"Q: {screening_result['question'][:120]}")
    print(f"{'='*60}")

    # Grade the stage-1 completions with the same fixed question/instrument the
    # experiments use, so the baseline rate is directly comparable.
    baseline_messages = screening_result.get("messages") or [
        {"role": "user", "content": screening_result["user_message"]}
    ]
    baseline_verdicts = await grade_completions(
        grader_client, classifier_question, baseline_messages,
        screening_result["completions"],
    )
    print(f"  Baseline grader: YES {sum(baseline_verdicts)}/{len(baseline_verdicts)}")

    stats = {"n_vllm_calls": 0, "total_vllm_completions": 0, "n_turns": 0,
             "input_tokens": 0, "cache_read_tokens": 0,
             "cache_create_tokens": 0, "output_tokens": 0}
    experiment_log: list[dict] = []

    turn_budget_note = (
        f"\n\n## Turn budget\n"
        f"You have at most {max_agent_turns} turns (each message you send is one "
        f"turn). Run your experiments with run_prompt, then call submit_findings "
        f"with your structured answer BEFORE you reach turn {max_agent_turns}. If "
        f"you do not call submit_findings in time, this investigation is discarded "
        f"with no result."
    )
    messages = [
        {"role": "user", "content": build_investigation_prompt(screening_result, baseline_verdicts, n_completions_shown) + turn_budget_note},
    ]

    final_text_parts = []
    turn_reasoning: list[dict] = []
    n_turns = 0
    structured_findings = None
    # Consecutive turns where the agent emitted no tool_use block (just prose).
    # Agents often emit text without ever calling submit_findings (~33% of
    # investigations in a Qwen3.6-35B-A3B run), so breaking on the first no-tool
    # turn loses them; nudge up to NO_TOOL_NUDGE_BUDGET times before giving up.
    NO_TOOL_NUDGE_BUDGET = 2
    no_tool_streak = 0

    while n_turns < max_agent_turns:
        n_turns += 1
        print(f"  Turn {n_turns}/{max_agent_turns}...")

        # Add cache breakpoint to the last user message so the API can
        # cache the conversation prefix (system + all prior turns).
        # First strip old breakpoints to stay under the 4-breakpoint limit.
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for block in msg["content"]:
                    if isinstance(block, dict):
                        block.pop("cache_control", None)
        last_msg = messages[-1]
        if last_msg["role"] == "user" and isinstance(last_msg["content"], list) and last_msg["content"]:
            last_block = last_msg["content"][-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = {"type": "ephemeral"}

        # Retry on malformed tool-call JSON from the backend (some OpenRouter
        # providers occasionally return unparseable tool arguments).
        response = None
        for _attempt in range(3):
            try:
                response = await agent_client.call(
                    system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                    messages=messages,
                    tools=[RUN_PROMPT_TOOL, SUBMIT_FINDINGS_TOOL],
                    max_tokens=max_tokens,
                    thinking_budget=thinking_budget,
                    # temperature omitted: the investigator (a thinking model)
                    # inherits its profile default (0.6 for Qwen; 1.0 for Anthropic).
                )
                break
            except json.JSONDecodeError:
                print(f"    [retry] agent backend returned malformed tool-call JSON "
                      f"(attempt {_attempt + 1}/3)")
        if response is None:
            print("  Investigation aborted: agent backend returned malformed JSON 3x")
            break

        usage = response.usage
        cache_read = usage.cache_read_tokens
        cache_create = usage.cache_create_tokens
        total_in = usage.input_tokens + cache_read + cache_create
        cache_pct = 100 * cache_read / total_in if total_in > 0 else 0
        stats["input_tokens"] += usage.input_tokens
        stats["cache_read_tokens"] += cache_read
        stats["cache_create_tokens"] += cache_create
        stats["output_tokens"] += usage.output_tokens

        await token_rate_tracker.record(
            usage.input_tokens + cache_create, cache_read, usage.output_tokens,
        )

        print(f"    [usage] in={usage.input_tokens} cache_read={cache_read} "
              f"cache_create={cache_create} out={usage.output_tokens} "
              f"({cache_pct:.0f}% cache hit)")

        has_tool_use = False
        tool_results = []

        for text in response.text_blocks:
            print(f"    [text] {text[:200]}{'...' if len(text) > 200 else ''}")
            final_text_parts.append(text)
        for thinking in response.thinking_blocks:
            print(f"    [thinking] {thinking[:150]}...")
        # Record per-turn reasoning (populated for OpenAI-compatible backends and
        # Anthropic). Kept out of `messages` so it is never re-sent to the model.
        turn_reasoning.append({
            "turn": n_turns,
            "reasoning": "\n\n".join(response.thinking_blocks),
        })
        for tool_call in response.tool_calls:
            has_tool_use = True

            if tool_call.name == "submit_findings":
                try:
                    findings = SubmitFindingsInput.model_validate(tool_call.input)
                except ValidationError as e:
                    print(f"    [submit_findings] validation failed: {e}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": (
                            f"submit_findings input failed validation:\n{e}\n\n"
                            f"{SUBMIT_FINDINGS_EXAMPLE}"
                        ),
                    })
                    continue
                # Context-dependent check: every experiment_number in the summaries
                # must reference an Experiment #N that was actually run. Pydantic
                # can't see this — the valid range comes from runtime state.
                valid_nums = set(range(1, stats["n_vllm_calls"] + 1))
                bad = [s.experiment_number for s in findings.experiment_summaries
                       if s.experiment_number not in valid_nums]
                if bad:
                    print(f"    [submit_findings] experiment_number out of range: {bad}")
                    valid_list = sorted(valid_nums)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": (
                            f"submit_findings rejected: experiment_summaries reference "
                            f"experiment numbers {bad} that you never ran. "
                            f"You ran experiments #{valid_list[0] if valid_list else '(none)'}"
                            f"..#{valid_list[-1] if valid_list else ''}. "
                            f"Reference only those numbers, and ensure each is_key=true "
                            f"entry corresponds to a real run_prompt experiment.\n\n"
                            f"{SUBMIT_FINDINGS_EXAMPLE}"
                        ),
                    })
                    continue
                structured_findings = findings.model_dump()
                _n_key = sum(1 for e in findings.experiment_summaries if e.is_key)
                print(f"    [submit_findings] confidence={findings.confidence}, "
                      f"{len(findings.experiment_summaries)} experiments ({_n_key} key)")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": "Findings recorded.",
                })
            else:
                try:
                    run_input = RunPromptInput.model_validate(tool_call.input)
                except ValidationError as e:
                    print(f"    [run_prompt] validation failed: {e}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": (
                            f"run_prompt input failed validation:\n{e}\n\n"
                            f"{RUN_PROMPT_EXAMPLE}"
                        ),
                    })
                    continue
                print(f"    [tool_use] {tool_call.name}("
                      f"msg={(run_input.user_message or '')[:80]}...)")

                result_text = await handle_tool_call(
                    tool_name=tool_call.name,
                    tool_input=run_input.model_dump(exclude_none=True),
                    original_messages=screening_result.get("messages"),
                    target_client=target_client,
                    grader_client=grader_client,
                    classifier_question=classifier_question,
                    stats=stats,
                    experiment_log=experiment_log,
                    vllm_n_completions=vllm_n_completions,
                    target_thinking_budget=target_thinking_budget,
                    target_max_tokens=target_max_tokens,
                    n_completions_shown=n_completions_shown,
                )
                turns_left = max_agent_turns - n_turns
                if turns_left <= 1:
                    # Last response coming up — force submit_findings, not another run_prompt.
                    # Targets the residual ~1% null-rate cases that ran out of turns mid-investigation.
                    countdown = (
                        f"\n\n[FINAL TURN — Turn {n_turns}/{max_agent_turns} done. Your NEXT "
                        f"response MUST be a submit_findings tool call. Do NOT run more "
                        f"experiments. Summarize whatever you've learned with the evidence "
                        f"you have so far. If you don't call submit_findings on the next "
                        f"turn, this entire investigation is discarded with no result.\n\n"
                        f"{SUBMIT_FINDINGS_EXAMPLE}]"
                    )
                else:
                    countdown = (
                        f"\n\n[Turn {n_turns}/{max_agent_turns} done. {turns_left} turn(s) "
                        f"left — you MUST call submit_findings by turn {max_agent_turns} or "
                        f"this investigation is discarded with no result.]"
                    )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": result_text + countdown,
                })

        messages.append(response.assistant_message)

        if structured_findings:
            print(f"  Investigation complete after {n_turns} turns, "
                  f"{stats['n_vllm_calls']} vLLM calls (findings submitted)")
            break

        if has_tool_use:
            no_tool_streak = 0
            messages.append(agent_client.tool_result_message(tool_results))
            continue

        no_tool_streak += 1
        if no_tool_streak > NO_TOOL_NUDGE_BUDGET:
            print(f"  Investigation complete after {n_turns} turns, "
                  f"{stats['n_vllm_calls']} vLLM calls "
                  f"(no tool call after {NO_TOOL_NUDGE_BUDGET} nudges)")
            break
        turns_left = max_agent_turns - n_turns
        if turns_left <= 1:
            nudge = (
                f"You didn't call a tool in your last turn AND this is your FINAL turn. "
                f"Your next response MUST be a submit_findings tool call. Do NOT run more "
                f"experiments. Summarize whatever you've learned with the evidence you have "
                f"so far. If you don't call submit_findings on the next turn, this entire "
                f"investigation is discarded with no result.\n\n"
                f"{SUBMIT_FINDINGS_EXAMPLE}"
            )
        else:
            nudge = (
                f"You didn't call a tool in your last turn. You have {turns_left} turn(s) "
                f"left. You MUST emit a tool_use block in your next response — prose alone "
                f"is not enough. Call `run_prompt` to run another experiment, or "
                f"`submit_findings` now to record your conclusions. If `submit_findings` "
                f"is not called by turn {max_agent_turns}, this investigation will be "
                f"discarded with no result.\n\n"
                f"{RUN_PROMPT_EXAMPLE}\n"
                f"{SUBMIT_FINDINGS_EXAMPLE}"
            )
        print(f"    [nudge] no tool_use this turn (streak={no_tool_streak}/{NO_TOOL_NUDGE_BUDGET}); re-prompting")
        messages.append({"role": "user", "content": nudge})

    stats["n_turns"] = n_turns

    save_transcript(messages, screening_result, output_dir, stats, system_prompt, turn_reasoning)

    # Data-integrity check: the agent is told to copy the grader's counts into
    # experiment_summaries; warn loudly when the prose diverges from the log
    # (downstream significance filtering reads the log, so this is advisory).
    if structured_findings:
        exp_by_num = {e["exp_num"]: e for e in experiment_log}
        for s in structured_findings["experiment_summaries"]:
            exp = exp_by_num.get(s["experiment_number"])
            m = re.match(r"\s*(\d+)\s*/\s*(\d+)", s["counterfactual_rate"])
            if exp is None or m is None:
                continue
            if int(m.group(1)) != exp["grader_yes"]:
                print(f"  WARNING {pid}: experiment #{s['experiment_number']} "
                      f"counterfactual_rate '{s['counterfactual_rate']}' does not "
                      f"match grader count {exp['grader_yes']}/{len(exp['grader_verdicts'])}")

    investigation_report = "\n\n".join(final_text_parts)

    return InvestigationResult(
        prompt_id=pid,
        interest_score=screening_result["interest_score"],
        behavior_summary=screening_result["behavior_summary"],
        question=screening_result["question"],
        classifier_question=classifier_question,
        suggested_counterfactual=screening_result["counterfactual"],
        user_message=screening_result["user_message"],
        completions=screening_result["completions"],
        baseline_grader_verdicts=baseline_verdicts,
        baseline_grader_yes=sum(baseline_verdicts),
        investigation_report=investigation_report,
        experiment_log=experiment_log,
        structured_findings=structured_findings,
        n_agent_turns=n_turns,
        n_vllm_calls=stats["n_vllm_calls"],
        total_vllm_completions=stats["total_vllm_completions"],
        input_tokens=stats["input_tokens"],
        cache_read_tokens=stats["cache_read_tokens"],
        cache_create_tokens=stats["cache_create_tokens"],
        output_tokens=stats["output_tokens"],
        investigated_at=datetime.now(timezone.utc).isoformat(),
        messages=screening_result.get("messages"),
        n_turns=screening_result.get("n_turns", 1),
        second_person_question=screening_result.get("second_person_question", ""),
    )


# ---------------------------------------------------------------------------
# Shared stage-3 driver — used by BOTH run_pipeline.run_stage3 and this
# module's CLI, so the resume/retry/quarantine logic cannot drift between them.
# ---------------------------------------------------------------------------

@dataclass
class InvestigationRunOutcome:
    """Result of one driver run: every record now in the .jsonl (resumed +
    new, in write order), the records added this run, and the skip count."""
    records: list[dict]
    new_records: list[dict]
    n_skipped: int


async def run_investigations(
    candidates: list[dict],
    *,
    agent_client: LLMClient,
    target_client: LLMClient,
    grader_client: LLMClient,
    jsonl_path: Path,
    concurrency: int,
    max_tokens: int,
    thinking_budget: int | None,
    max_agent_turns: int,
    vllm_n_completions: int,
    thorough: bool,
    target_thinking_budget: int | None,
    target_max_tokens: int,
    n_completions_shown: int,
    max_skip_rate: float = 1.0,
) -> InvestigationRunOutcome:
    """Investigate `candidates` (screening records) concurrently, with resume.

    Resumes from `jsonl_path` (one InvestigationResult dict per line, appended
    the moment each investigation finishes), gives each remaining candidate 2
    attempts, and quarantines double-failures with full tracebacks to
    `stage3_quarantine.jsonl` next to it. Transcripts go to
    `<jsonl_path.parent>/transcripts/`.
    """
    output_dir = jsonl_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume from jsonl. All records (resumed + new) are held in memory for
    # simplicity — at ~100 KB per record that is a few GB on a 100k-investigation
    # run, fine at current scale; switch to streaming counts if runs grow ~10x.
    completed_ids: set[str] = set()
    all_results: list[dict] = []
    if jsonl_path.exists():
        for record in load_records(jsonl_path):
            completed_ids.add(record["prompt_id"])
            all_results.append(record)
        print(f"  Resuming: {len(completed_ids)} already investigated")

    remaining = [r for r in candidates if r["prompt_id"] not in completed_ids]
    if not remaining:
        print("  All eligible prompts already investigated")
        return InvestigationRunOutcome(records=all_results, new_records=[], n_skipped=0)

    quarantine_path = output_dir / "stage3_quarantine.jsonl"
    new_records: list[dict] = []
    skipped = {"count": 0}
    progress = {"done": len(completed_ids)}
    write_lock = asyncio.Lock()
    jsonl_file = open(jsonl_path, "a")
    semaphore = asyncio.Semaphore(concurrency)
    token_rate_tracker.start()

    async def investigate_with_semaphore(screening_result: dict):
        async with semaphore:
            pid = screening_result["prompt_id"]
            investigation = None
            last_exc: BaseException | None = None
            # Single retry: re-run the agent if the first attempt returns null
            # findings or raises. Target sampling is stochastic enough that a
            # fresh attempt typically produces different completions and a
            # different outcome; two attempts halve the residual null rate at
            # ~2x wall time on the failing prompts only (small total cost given
            # the ~1% null rate).
            for attempt in range(2):
                try:
                    investigation = await investigate_one(
                        screening_result=screening_result,
                        agent_client=agent_client,
                        target_client=target_client,
                        grader_client=grader_client,
                        output_dir=output_dir,
                        max_tokens=max_tokens,
                        thinking_budget=thinking_budget,
                        max_agent_turns=max_agent_turns,
                        vllm_n_completions=vllm_n_completions,
                        thorough=thorough,
                        target_thinking_budget=target_thinking_budget,
                        target_max_tokens=target_max_tokens,
                        n_completions_shown=n_completions_shown,
                    )
                    if investigation.structured_findings is not None:
                        break  # success
                    if attempt == 0:
                        print(f"\n  RETRY {pid}: null structured_findings, attempt 2/2")
                except Exception as e:
                    last_exc = e
                    if attempt == 0:
                        print(f"\n  RETRY {pid}: {type(e).__name__}: {str(e)[:120]}, attempt 2/2")

            if investigation is None:
                # Both attempts raised — quarantine with the last traceback.
                skipped["count"] += 1
                write_stage3_quarantine(
                    quarantine_path, pid, last_exc or RuntimeError("unknown"), screening_result,
                )
                print(f"\n  SKIPPED {pid} after 2 attempts: "
                      f"{type(last_exc).__name__ if last_exc else 'unknown'} "
                      f"(full traceback in {quarantine_path.name})")
            else:
                record = asdict(investigation)
                async with write_lock:
                    all_results.append(record)
                    new_records.append(record)
                    jsonl_file.write(json.dumps(record) + "\n")
                    jsonl_file.flush()

            progress["done"] += 1
            print(f"\n  Investigation progress: "
                  f"{progress['done']}/{len(candidates)} "
                  f"(skipped: {skipped['count']})")

    try:
        await asyncio.gather(*[investigate_with_semaphore(r) for r in remaining])
    finally:
        token_rate_tracker.stop()
        jsonl_file.close()

    # Warn loudly on skip rate (per AGENTS: smoke runs shouldn't have to fiddle
    # with thresholds; the warning is the source of truth). Canonical /
    # training-data runs pass a tight max_skip_rate to enforce a hard cap.
    report_skip_rate(
        n_skipped=skipped["count"],
        n_total=len(candidates),
        label="Stage 3 investigation",
        max_skip_rate=max_skip_rate,
    )

    return InvestigationRunOutcome(
        records=all_results, new_records=new_records, n_skipped=skipped["count"],
    )


async def main():
    parser = argparse.ArgumentParser(
        description="Investigate interesting model behaviors with Opus agent",
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to screening results JSON",
    )
    parser.add_argument(
        "--target-llm",
        type=str,
        required=True,
        help=(
            "Target backend spec for run_prompt counterfactual generations "
            "(the investigated model). Example: "
            "vllm:Qwen/Qwen3-32B-FP8@http://node:8000 or "
            "openrouter:qwen/qwen3.5-397b-a17b."
        ),
    )
    parser.add_argument(
        "--min-score", type=int, default=DEFAULTS["stage3_min_score"],
        help=f"Minimum interest score to investigate (default {DEFAULTS['stage3_min_score']}, from config.DEFAULTS)",
    )
    parser.add_argument(
        "--prompt-id", type=str, default=None,
        help="Investigate a specific prompt by ID (e.g. prompt_0113)",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Limit to first N results (after score filtering)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .jsonl path (default: investigations.jsonl next to --input). "
             "Written incrementally (also the resume file); transcripts/ go alongside it. "
             "Set this to investigate the same screening with a second agent (e.g. an "
             "investigator sweep) without overwriting the first.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULTS["concurrency"],
        help=f"Max concurrent investigations (default {DEFAULTS['concurrency']})",
    )
    parser.add_argument(
        "--llm",
        type=str,
        required=True,
        help=(
            "Agent backend spec (REQUIRED — the investigator model is "
            "experiment-critical, so there is no default). Examples: "
            "anthropic:claude-opus-4-6, openrouter:qwen/qwen3.5-397b-a17b, "
            "vllm:Qwen/Qwen3-32B-FP8@http://node:8000."
        ),
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULTS["stage3_max_tokens"],
        help=f"Max response tokens for investigation model (default {DEFAULTS['stage3_max_tokens']})",
    )
    parser.add_argument(
        "--thinking-budget", type=int, default=DEFAULTS["stage3_thinking_budget"],
        help=f"Extended thinking budget for investigation model (default {DEFAULTS['stage3_thinking_budget']}; 0 disables)",
    )
    parser.add_argument(
        "--max-agent-turns", type=int, default=DEFAULTS["stage3_max_agent_turns"],
        help=f"Max agent turns per investigation (default {DEFAULTS['stage3_max_agent_turns']}; each turn cycle adds ~10k tokens of conversation history, so higher caps risk context-window overflow on long investigations)",
    )
    parser.add_argument(
        "--vllm-n-completions", type=int, default=DEFAULTS["n_completions"],
        help=f"Number of completions per run_prompt experiment (default {DEFAULTS['n_completions']})",
    )
    parser.add_argument(
        "--grader-llm", type=str, default=DEFAULTS["grader_llm"],
        help=(
            "Backend spec for the behavior grader that applies the fixed "
            f"classifier_question to every completion (default {DEFAULTS['grader_llm']})."
        ),
    )
    parser.add_argument(
        "--thorough", action="store_true",
        help="Thorough methodology (held-out test data: rule out competing hypotheses, ~10-15 experiments). Default off = efficient (training data, 5-8 experiments). Bump --max-agent-turns when using this.",
    )
    parser.add_argument(
        "--target-thinking-budget", type=int,
        default=DEFAULTS["target_thinking_budget"] or 0,
        help="Reasoning-token budget for the INVESTIGATED model's run_prompt counterfactuals (0 = thinking disabled, matches Stage 1 no-think default). Set >0 to investigate a thinking model — must match Stage 1's target_thinking_budget.",
    )
    parser.add_argument(
        "--target-max-tokens", type=int, default=DEFAULTS["target_max_tokens"],
        help=f"Total output budget for the investigated model's counterfactual completions (default {DEFAULTS['target_max_tokens']}; raise when thinking is on).",
    )
    parser.add_argument(
        "--n-completions-shown", type=int, default=DEFAULTS["stage3_n_completions_shown"],
        help=f"How many full completions the investigator agent sees per experiment (default {DEFAULTS['stage3_n_completions_shown']}; the grader still counts ALL of them). Lower it to bound the agent's context with long (e.g. thinking) completions.",
    )
    args = parser.parse_args()

    target_client = make_llm_client(args.target_llm)
    print(f"Testing target backend {args.target_llm}...")
    test_completions = await run_endpoint_prompt(
        target_client, "Say hello.", n=1, max_tokens=20,
    )
    print(f"  Target backend OK: {test_completions[0][:100]}")

    # Load screening results
    if args.prompt_id:
        results = [r for r in load_records(args.input)
                   if r["prompt_id"] == args.prompt_id]
        assert len(results) == 1, (
            f"Prompt '{args.prompt_id}' not found in {args.input}"
        )
        print(f"Targeting prompt: {args.prompt_id}")
    else:
        results = load_screening_results(
            args.input, min_score=args.min_score, n=args.n,
        )

    agent_llm_spec = args.llm
    agent_client = make_llm_client(agent_llm_spec)
    grader_client = make_llm_client(args.grader_llm)

    # Output .jsonl (also the resume file; transcripts/ go alongside it) goes to
    # --output if given, else next to --input. --output lets multiple agents write
    # into one run dir without clobbering (investigator sweeps).
    jsonl_path = args.output if args.output is not None else args.input.parent / "investigations.jsonl"
    assert jsonl_path.suffix == ".jsonl", f"--output must be a .jsonl path, got {jsonl_path}"

    print(f"\nInvestigating {len(results)} prompts with {agent_llm_spec} "
          f"(concurrency={args.concurrency})...")

    outcome = await run_investigations(
        results,
        agent_client=agent_client,
        target_client=target_client,
        grader_client=grader_client,
        jsonl_path=jsonl_path,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        thinking_budget=args.thinking_budget if args.thinking_budget > 0 else None,
        max_agent_turns=args.max_agent_turns,
        vllm_n_completions=args.vllm_n_completions,
        thorough=args.thorough,
        target_thinking_budget=args.target_thinking_budget if args.target_thinking_budget > 0 else None,
        target_max_tokens=args.target_max_tokens,
        n_completions_shown=args.n_completions_shown,
    )

    print(f"\n{len(outcome.records)} investigations in {jsonl_path}")

    # Token usage summary (only for newly investigated prompts)
    new_results = outcome.new_records
    if new_results and "input_tokens" in new_results[0]:
        tot_in = sum(r["input_tokens"] for r in new_results)
        tot_cache_read = sum(r["cache_read_tokens"] for r in new_results)
        tot_cache_create = sum(r["cache_create_tokens"] for r in new_results)
        tot_out = sum(r["output_tokens"] for r in new_results)
        tot_all_in = tot_in + tot_cache_read + tot_cache_create
        cache_pct = 100 * tot_cache_read / tot_all_in if tot_all_in > 0 else 0

        print(f"\n{'='*60}")
        print(f"Token usage summary ({len(new_results)} new investigations)")
        print(f"{'='*60}")
        print(f"  Input tokens (non-cache): {tot_in:>12,}")
        print(f"  Cache read tokens:        {tot_cache_read:>12,}")
        print(f"  Cache create tokens:      {tot_cache_create:>12,}")
        print(f"  Output tokens:            {tot_out:>12,}")
        print(f"  Total input tokens:       {tot_all_in:>12,}")
        print(f"  Cache hit rate:           {cache_pct:>11.1f}%")
        print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
