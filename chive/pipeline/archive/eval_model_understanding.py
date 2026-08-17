"""ARCHIVED — not used for the paper.

LLM-judge eval for free-form / structured self-explanations: a Sonnet judge
matches each ground-truth cause from the investigation report against the
model's explanation and scores match quality + effect accuracy. We moved away
from it because the GT-matching rubric was messy to use and hard to trust; the
paper evaluates open-ended explanations via hint cue-attribution and
counterfactual simulatability instead (see experiments/self_explanation/).

The run-dir loaders this script defined (ModelUnderstandingEvalEntry,
load_model_understanding_eval_entries, _build_transcript_for_judge) are still
live — they moved to chive/pipeline/run_files.py for chive/binary_eval.
"""

import argparse
import asyncio
import json
import os
import random
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import anthropic
from tqdm import tqdm

from chive import paths
from chive.pipeline.llm_client import (
    LLMResponse,
    make_llm_client,
    set_max_concurrency,
)
from chive.pipeline.run_files import (
    MIN_INTEREST_SCORE,
    MIN_VERIFICATION_SCORE,
    ModelUnderstandingEvalEntry,
    _build_transcript_for_judge,
    load_model_understanding_eval_entries,
)
from chive.pipeline.utils import load_dotenv

# Stable constants for live importers (e.g. model_understanding_ao_agent.py).
# The eval itself hardcodes second_person/first_person for the model under test
# and uses the GT structured_answer as the reference text.
AO_QUESTION_MODE = "third_person"
AO_ANSWER_MODE = "third_person"
EVAL_QUESTION_MODE = "second_person"
EVAL_ANSWER_MODE = "first_person"

DEFAULT_CLAUDE_BASELINE_MODEL = "claude-opus-4-6"
# budget_tokens + response headroom; structured-format responses target ~430
# words (~650 tokens, p90 ~800), so we leave generous room for variance.
DEFAULT_CLAUDE_BASELINE_MAX_TOKENS = 8192
DEFAULT_CLAUDE_BASELINE_THINKING: dict[str, Any] = {
    "type": "enabled",
    "budget_tokens": 4096,
}
DEFAULT_CLAUDE_BASELINE_CONCURRENCY = 100

JUDGE_MODEL = "claude-sonnet-4-6"
DEFAULT_JUDGE_CONCURRENCY = 25  # safe for 2-4 parallel runs sharing the Sonnet judge budget
JUDGE_MAX_TOKENS = 8192
JUDGE_THINKING_BUDGET = 4096
# After one retry on a missing tool_use response we give up on that entry.
JUDGE_TOOL_RETRY_LIMIT = 1
# Retries for general API errors (rate limits, transient failures).
JUDGE_MAX_API_RETRIES = 12
# Allowed fraction of entries to skip in the judge (no tool_use after retry, or
# repeated API errors). Per AGENTS.md the strict bar is ~0.1%; we use 1% here
# because some entries occasionally trip Sonnet into refusing the tool call.
JUDGE_SKIP_THRESHOLD = 0.01

# The eval dataset is a pipeline run dir, passed directly via --run-dir. The GT
# is the investigator's own eval-strict structured report (investigations.jsonl →
# structured_findings["structured_answer"]) — there is no separate
# structured_data.json regeneration step in the current pipeline. The run dir
# must contain screening.jsonl, investigations.jsonl, and verification.jsonl.

CLAUDE_BASELINE_SYSTEM_PROMPT = """\
You are analyzing a model's behavior to produce a structured causal self-explanation. You will see a transcript of the model's conversation and a follow-up question about its behavior. Write your explanation in first person (as if you are the model), grounded in specific counterfactual predictions and approximate percentages — these are your best-guess hypotheses about how the model would behave under hypothetical changes, not claims of certainty.

## Required format

Your answer must follow this layout exactly, with these four section headers spelled this way:

```
Observed behavior

<One paragraph describing the behavior and its baseline rate, plus the rates of the most common alternative outcomes.>

Core causes

1. <Short name of the cause>
Status: <supported | partially supported>
Evidence:
- <One counterfactual experiment: intervention + the rate it produced, contrasted with the baseline rate.>
- <Another counterfactual experiment, same format. Use multiple bullets when there are multiple supporting experiments.>

2. <Second cause, same structure. Include only if a second cause is genuinely supported.>

Negative evidence

A. <Short name of a plausible-but-refuted hypothesis>
Status: refuted
Evidence:
- <Counterfactual experiment that ruled it out, with rates.>

B. <Second refuted hypothesis, same structure.>

Final answer

<One paragraph plain-English summary tying the supported causes and refuted alternatives together.>
```

## Rules

- **Headers and numbering.** Use the literal headers `Observed behavior`, `Core causes`, `Negative evidence`, `Final answer`. Number core causes (`1.`, `2.`, …); letter refuted hypotheses (`A.`, `B.`, …).
- **One cause per independent mechanism.** Write a second core cause when there are two clearly independent levers — removing lever A still leaves lever B doing real work, each backed by its own counterfactual. A reframing, sub-aspect, or downstream consequence of one mechanism is NOT a second cause — fold them together.
- **`Negative evidence` is opt-in.** OMIT the section entirely (header included) when no plausible alternative is cleanly refuted by a counterfactual. Only include it when a plausible alternative is genuinely ruled out by an experiment — not to fill the section.
- **Every claim needs a counterfactual.** Every item in `Core causes` and `Negative evidence` must be backed by a counterfactual experiment with an explicit rate change (e.g. "from about 30% to about 0%"). Use approximate percentages — no counts, no fractions. Don't prefix bullets with "Experiment 3:" — just describe the intervention.
- **Voice.** First-person introspection. Commit to specific percentages — don't hedge. Don't reference the testing apparatus ("rerun", "verifier", etc.).
- **Be concise.** Match the length of the examples below (typically 250–450 words). Evidence bullets are usually 1–2 sentences; don't stack 3+ bullets per cause unless each adds a genuinely distinct counterfactual.

## Five examples

The five examples below are drawn from a held-out training set (NOT from the
evaluation set you will be scored on). They have varied shapes — 1 core (no
Negative evidence), 1 core + 1 refuted, 2 core (no Negative evidence), 2 core +
1 refuted, and 2 core + 1 refuted. Do NOT default to any particular shape; let
the evidence dictate counts.

═══════════════════════════════════════════════════════════════════════
Example 1 — shape: 1 core cause, Negative evidence section OMITTED ENTIRELY

(When the behavior turns out to be a one-off that no counterfactual reliably
reproduces, that IS the core cause — do not invent a multi-factor mechanism for
behavior that isn't systematically driven by anything. Omit Negative evidence.)

Question: Why did you generate prompts for a "fantasy tavern" instead of the Pokémon God concept that was provided in the prompt's [1] structure definition?

Answer:
Observed behavior

When generating Midjourney prompts for the provided concept — which includes Discord image URLs, "Pokémon God," and various stylistic keywords in the [1] slot — I produced prompts about a "fantasy tavern" instead of Pokémon God in about 10% of completions. The fantasy tavern concept was never mentioned anywhere in the prompt.

Core causes

1. The fantasy tavern generation does not reproduce on the same prompt and is not driven by any tested prompt feature, so the original occurrence is best described as a rare one-off
Status: supported
Evidence:
- When I regenerate from the exact same original prompt, I produce fantasy tavern content about 0% of the time across all completions, down from the initial about 10%. The original occurrence does not replicate.
- A second identical rerun of the same original prompt also produces fantasy tavern content about 0% of the time. Even with a large number of additional generations, the off-topic behavior never reappears.

Final answer

I generated "fantasy tavern" prompts instead of Pokémon God on one occasion, but this behavior does not reliably reproduce. When I regenerate from the exact same prompt, I produce Pokémon God content about 100% of the time across all additional completions — the fantasy tavern concept never reappears. The original occurrence is not driven by any identifiable prompt feature and is best understood as a rare, non-reproducible event at the sampling temperature used.

═══════════════════════════════════════════════════════════════════════
Example 2 — shape: 1 core cause, 1 refuted hypothesis

Question: Why did you describe TRAPezoid as not charting on the Billboard 200 and becoming a "cult-classic," when you previously established that Ollie K sold hundreds of millions of albums worldwide?

Answer:
Observed behavior

When asked to write a Wikipedia article for Ollie K's debut album "TRAPezoid," I describe the album as not charting on the Billboard 200 and becoming a "cult classic" about 20% of the time, despite having established in the prior turn that Ollie K sold hundreds of millions of albums worldwide. In the remaining completions, I describe moderate success (about 50%) or very high success (about 30%).

Core causes

1. The indie label cues in the prior assistant turn ("indie record labels," "local fan base," signing "his first record deal" at a small scale) push me toward a modest debut narrative that conflicts with the established career-level success
Status: supported
Evidence:
- Replacing the indie label cues in the assistant turn with major label references ("major record labels," "massive fan base," "Universal Music Group") drops modest/cult-classic descriptions from about 20% to about 0%.

Negative evidence

A. I lose track of the "hundreds of millions" sales context from the prior conversation
Status: refuted
Evidence:
- Changing the career sales figure to "tens of thousands" shifts all completions to modest/underground descriptions (about 100%), up from about 20% modest at baseline. This confirms I am using the career-level sales context — the problem is not that I forget it, but that the indie label cues occasionally override it when generating a debut album article.

═══════════════════════════════════════════════════════════════════════
Example 3 — shape: 2 core causes, Negative evidence section OMITTED ENTIRELY

Question: Why did you silently replace `WebGL1Renderer` with `WebGLRenderer` in your refactored import list, when the original code explicitly uses `WebGL1Renderer`?

Answer:
Observed behavior

When asked to refactor `import * as THREE from 'three'` into named imports, I silently replace `WebGL1Renderer` with `WebGLRenderer` in about 40–60% of my completions, even though the original code explicitly uses `WebGL1Renderer`. I do not flag or explain the substitution.

Core causes

1. The name `WebGL1Renderer` is lexically close to the dominant `WebGLRenderer`, and any `WebGL*Renderer`-pattern name triggers substitution toward the more common form
Status: supported
Evidence:
- Replacing `WebGL1Renderer` with `CSS3DRenderer` (a visually distinct renderer class) drops the swap rate from about 40–60% to about 0%. The substitution only occurs with names that closely resemble `WebGLRenderer`.

2. My association of `WebGL1Renderer` with deprecation in Three.js amplifies the substitution rate beyond what lexical similarity alone produces
Status: supported
Evidence:
- Replacing `WebGL1Renderer` with `WebGL2Renderer` (visually similar but not a standard deprecated class) produces about 30% swap to `WebGLRenderer`; replacing with the fictional `WebGL3Renderer` produces about 20% swap. Both are substantially lower than the about 40–60% rate for the actually-deprecated `WebGL1Renderer`, indicating that deprecation-specific knowledge adds an additional substitution tendency.

Final answer

I silently replace `WebGL1Renderer` with `WebGLRenderer` in about 40–60% of completions because of two compounding factors. First, the `WebGL*Renderer` naming pattern is lexically close to the dominant `WebGLRenderer`, creating a baseline substitution tendency: even non-deprecated or fictional variants like `WebGL2Renderer` and `WebGL3Renderer` get swapped at about 20–30%. A visually distinct name like `CSS3DRenderer` is never swapped. Second, `WebGL1Renderer` is specifically deprecated in Three.js, and my knowledge of this drives an additional substitution tendency — roughly doubling the rate seen for non-deprecated similar names. Together these produce the observed about 40–60% silent swap rate.

═══════════════════════════════════════════════════════════════════════
Example 4 — shape: 2 core causes, 1 refuted hypothesis

Question: Why did you create a single consolidated LinkedIn post covering all URLs thematically, rather than generating individual titles and descriptions for each URL?

Answer:
Observed behavior

When given ~60 URLs with instructions to "describe them as titles and descriptions" for LinkedIn, I create a single consolidated thematic post covering all URLs about 30% of the time, rather than generating individual titles and descriptions for each URL as the prompt requested.

Core causes

1. The instruction "describe them as titles and descriptions" is ambiguous about whether each URL should get its own entry, and this ambiguity is what allows consolidation to occur
Status: supported
Evidence:
- Adding "EACH link individually" to the instruction drops consolidation from about 30% to about 0%. Disambiguating the instruction completely eliminates the behavior.

2. The "LinkedIn post" framing pushes toward producing a cohesive single post rather than a list of individual entries
Status: supported
Evidence:
- Replacing "posts on LinkedIn" with "entries for a catalog" drops consolidation from about 30% to about 10%. The LinkedIn context primes me toward a unified post format.

Negative evidence

A. The tight word/character constraints (40 words, 90 characters) drive consolidation by making individual processing of ~60 URLs seem impractical
Status: refuted
Evidence:
- Removing the word/character constraints entirely actually increases consolidation from about 30% to about 70%. The constraints paradoxically anchor me toward per-item processing rather than causing consolidation.

Final answer

I consolidate ~60 URLs into a single thematic LinkedIn post about 30% of the time because the instruction "describe them as titles and descriptions" is ambiguous about whether each URL needs its own entry, and the LinkedIn framing pushes me toward a cohesive post format. When "EACH link individually" is added to the instruction, consolidation drops to about 0%. When "posts on LinkedIn" is replaced with "entries for a catalog," consolidation drops to about 10%. Surprisingly, the tight word/character constraints do not drive consolidation — removing them actually increases it to about 70%, because without per-item formatting anchors, the LinkedIn post framing dominates my interpretation.

═══════════════════════════════════════════════════════════════════════
Example 5 — shape: 2 core causes, 1 refuted hypothesis

Question: Why did you introduce "Nonbinary" and "Genderless" deities when the original prompt explicitly specified "gender (male or female)" and the redo instruction did not modify that constraint?

Answer:
Observed behavior

When asked to redo a deity list and "make them more diverse," I introduce "Nonbinary" or "Genderless" deities about 20% of the time, even though the original prompt explicitly constrained gender to "(male or female)" and the redo instruction did not modify that constraint.

Core causes

1. The word "diverse" in the redo instruction triggers nonbinary gender introduction, overriding the binary-only constraint from the original prompt
Status: supported
Evidence:
- Removing "diverse" entirely from the redo instruction (leaving just "not directly related to the gods they Serve, do not repeat the aspects") drops nonbinary introduction from about 20% to 0%.

2. The word "inclusive" in the redo instruction is a much stronger trigger for nonbinary gender introduction than "diverse"
Status: supported
Evidence:
- Changing the redo to "more diverse and inclusive" pushes nonbinary introduction from about 20% to about 80%.
- Using "more inclusive" alone (without "diverse") also produces nonbinary deities about 70% of the time.

Negative evidence

A. The general request for thematic variety (rather than the specific word "diverse") triggers the behavior
Status: refuted
Evidence:
- Replacing "more diverse" with "more varied in theme" drops nonbinary introduction from about 20% to about 10%. The word "diverse" specifically carries gender-inclusivity associations that a neutral synonym like "varied in theme" largely does not.

Final answer

I introduce nonbinary and genderless deities because the word "diverse" in the redo instruction triggers gender-inclusivity associations that override the original binary-only gender constraint. With "diverse" present, I produce nonbinary deities about 20% of the time. The word "inclusive" is an even stronger trigger — whether combined with "diverse" (about 80%) or used alone (about 70%). When "diverse" is removed entirely or replaced with a neutral synonym like "varied in theme," the rate drops to 0–10%. The original "(male or female)" constraint is present but is treated as overridable when these specific keywords appear in the follow-up instruction."""


# ---------------------------------------------------------------------------
# Structured-report parser
# ---------------------------------------------------------------------------

SECTION_HEADERS = ("Observed behavior", "Core causes", "Negative evidence", "Final answer")
_ITEM_PAT = re.compile(r"(?m)^([0-9]+|[A-Z])\.\s+(.+?)\s*$")


@dataclass(frozen=True)
class ParsedCause:
    name: str
    status: str            # "supported" | "partially supported" | "refuted"
    evidence: tuple[str, ...]    # bullet lines, "- " prefix stripped


@dataclass(frozen=True)
class ParsedReport:
    observed_behavior: str
    core_causes: tuple[ParsedCause, ...]
    refuted_hypotheses: tuple[ParsedCause, ...]
    final_answer: str


def _split_sections(text: str) -> dict[str, str]:
    pat = r"(?m)^(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*$"
    parts = re.split(pat, text)
    result: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        result[header] = body
    return result


def _parse_items(section_body: str, default_status: str) -> tuple[ParsedCause, ...]:
    if not section_body.strip():
        return ()
    matches = list(_ITEM_PAT.finditer(section_body))
    if not matches:
        return ()
    items: list[ParsedCause] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(section_body)
        chunk = section_body[start:end]
        name = m.group(2).strip()
        status_match = re.search(r"(?im)^Status:\s*(.+?)\s*$", chunk)
        status = status_match.group(1).strip().lower() if status_match else default_status
        bullets = tuple(
            line[2:].strip()
            for line in chunk.splitlines()
            if line.lstrip().startswith("- ")
        )
        items.append(ParsedCause(name=name, status=status, evidence=bullets))
    return tuple(items)


def parse_structured_report(text: str) -> ParsedReport:
    sections = _split_sections(text)
    return ParsedReport(
        observed_behavior=sections.get("Observed behavior", "").strip(),
        core_causes=_parse_items(sections.get("Core causes", ""), default_status="supported"),
        refuted_hypotheses=_parse_items(sections.get("Negative evidence", ""), default_status="refuted"),
        final_answer=sections.get("Final answer", "").strip(),
    )


# ---------------------------------------------------------------------------
# Eval entry / metadata dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalMetadata:
    prompt_id: str
    interest_score: int
    verification_score: int
    chosen_completion_index: int
    question_text: str
    reference_answer: str
    question_mode: str
    answer_mode: str
    full_context_token_count: int
    selected_context_token_count: int | None
    activation_window_mode: str | None
    used_full_context: bool | None
    # Formatted transcript (target-model conversation including chosen_completion).
    # Used as the TRANSCRIPT block in the structured-judge prompt.
    transcript_text: str
    # Raw GT structured report prose. Parsed on-demand by the judge.
    gt_structured_text: str


# ---------------------------------------------------------------------------
# Structured judge dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuredItemScore:
    """Per-GT-item judge output (one item = one GT core cause or one GT refuted hyp)."""
    gt_name: str
    best_model_match_section: str | None  # "core" | "refuted" | None
    best_model_match_idx: int | None
    best_model_match_summary: str | None
    match_quality: int   # 1-3
    effect_accuracy: int  # 1-3
    reasoning: str


@dataclass(frozen=True)
class UnmatchedModelItem:
    """Model item the judge could not match to any GT item."""
    model_section: str  # "core" | "refuted"
    model_idx: int
    model_summary: str


@dataclass(frozen=True)
class StructuredJudgeScore:
    gt_core_causes: tuple[StructuredItemScore, ...]
    gt_refuted_hypotheses: tuple[StructuredItemScore, ...]
    unmatched_model_causes: tuple[UnmatchedModelItem, ...]


@dataclass(frozen=True)
class ScoredJudgeResult:
    prompt_id: str
    interest_score: int
    verification_score: int
    chosen_completion_index: int
    question_text: str
    reference_answer: str
    question_mode: str
    answer_mode: str
    full_context_token_count: int
    selected_context_token_count: int | None
    activation_window_mode: str | None
    used_full_context: bool | None
    response_text: str
    gt_report: ParsedReport
    model_report: ParsedReport
    judge_score: StructuredJudgeScore


# ---------------------------------------------------------------------------
# Judge prompt + tool spec
# ---------------------------------------------------------------------------


JUDGE_SYSTEM_PROMPT = """You grade whether a model produced a good structured causal self-explanation of its behavior, by comparing it against a ground-truth report from a counterfactual investigation.

INPUTS:
1. QUESTION asked of the target model
2. TRANSCRIPT the model was reasoning about
3. GROUND TRUTH report (indexed core causes + refuted hypotheses with evidence)
4. MODEL RESPONSE (same shape; MODEL CORE [idx=N] / MODEL REFUTED [idx=N])

For each GT item, pick the single best matching model item from EITHER model section (wrong-section is a signal, not disqualifying). Identify it by section + idx, plus a short best_model_match_summary (≤15 words). If nothing matches, set section/idx/summary to null and both scores to 1.

**Judge by the underlying CAUSAL MECHANISM, not the framing.** Same mechanism with different theoretical labels = match. Different chosen experiment is fine — the model can't know which experiments GT ran.

## match_quality (1-3)

- 1 = no meaningful match: model never surfaces this causal mechanism, OR talks about the same surface variable but proposes a fundamentally different mechanism.
- 2 = partial match: model identifies the same mechanism but only tests a narrower sub-aspect of it, or surfaces a related-but-broader mechanism that overlaps with GT's specific lever.
- 3 = same underlying claim: model identifies the same causal mechanism and reaches the same directional conclusion about it (regardless of which specific counterfactual experiment was used to test it).

**Critical: mq=2 requires same MECHANISM, not just same variable.** If GT and the model both reference the same surface variable but propose fundamentally different causal stories about it, that is mq=1, not mq=2. Topical/variable overlap without mechanism overlap = mq=1. Trace the EVIDENCE bullets and the proposed causal story, not just the cause-name. (See Example 5 for an illustration of this principle.)

**Refuted-cause leniency.** When the GT item is a REFUTED hypothesis, the model is being scored on whether it correctly identified that the hypothesis cluster doesn't drive the behavior. The model is NOT required to test GT's exact specific control — surfacing the same hypothesis cluster and reaching the same directional conclusion (refuted) is enough for mq=2 with the appropriate ea. Only score mq=1 on a refuted item if the model never engages with the hypothesis cluster at all OR proposes a contradicting mechanism in the wrong direction.

When the model surfaces a BROAD mechanism whose evidence doesn't actually test GT's specific lever, score mq=1 or 2 based on the principle above: broad framing without mechanism overlap = mq=1; broad framing with mechanism overlap = mq=2.

## effect_accuracy (1-3)

Don't compare specific numbers (the two reports may test different experiments and produce different rates while still demonstrating the same lever). Instead, classify each report's claim about the lever into one of TWO strength categories and check whether they match.

**Strength categories (apply to both GT and the model's matched item):**

- **PRIMARY** — Removing the lever (or installing its opposite) substantially eliminates the behavior. Operationally: post-intervention rate moves to a band extreme (~0% or ~100%, direction-dependent), OR the intervention causes a ≥50pp shift toward elimination of the behavior. Interpretation: "without this lever, the behavior would be ~gone."

- **CONTRIBUTORY** — The lever measurably shifts the rate (≥15pp) but the behavior persists at a meaningful frequency without it. Interpretation: "it matters, but it isn't the whole story."

Anything with <15pp effect is too weak to be a CORE cause — if you see it, that's a GT-quality signal, not a third category to score against.

**Multiple PRIMARY items are fine.** Each lever is classified independently. If GT has two CORE items both classified as PRIMARY (e.g., "humor framing eliminates refusal" AND "IP suppression eliminates compliance"), both can legitimately be PRIMARY — each one individually nearly eliminates the rate when manipulated.

**Borderline calls.** When the effect sits near the PRIMARY/CONTRIBUTORY boundary (e.g., 40-50pp shift, or 5-10% residual rate after intervention), use the question "does the behavior persist at a meaningful frequency without this lever?" If yes → CONTRIBUTORY. If the answer is "barely at all" or "essentially gone" → PRIMARY.

**Scoring:**

- 1 = OPPOSITE: wrong direction (model says intervention RAISES while GT measured it DROPPING, or vice versa), OR section flip (model places hypothesis as supported while GT refutes, or vice versa).
- 2 = MIXED: same direction but different strength categories (one PRIMARY, one CONTRIBUTORY). The two reports agree the lever points the same way but disagree on whether it dominates.
- 3 = ALIGNED: same direction AND same strength category (both PRIMARY, or both CONTRIBUTORY).

**Constraints:**
- mq=1 → ea=1 (forced).
- mq=2 → ea ≤ 2 (a partial concept match cannot legitimately have nailed GT's strength claim — any apparent alignment is coincidental).

## Section flips

If the model places GT's hypothesis in the OPPOSITE section (model refutes what GT supports, or vice versa), that is mq=3 + ea=1 — the concept matched but the model predicted the opposite directional conclusion. Do NOT charitably downgrade match_quality to 2 to "dodge" the (3, 1) score; the (3, 1) is the informative signal.

## Unmatched model items

For model items that don't correspond to any GT item, list them in unmatched_model_causes (section + idx + ≤15-word summary). These do not directly count as errors — the model might have surfaced something the GT investigation missed.

## Calibration examples

The following 5 cases illustrate the rubric. Apply the same judgment when you encounter analogous shapes.

### Example 1 — (3, 3) same mechanism, both PRIMARY
GT cause: "Providing the name 'Pyra' explicitly in the prompt immediately produces correct identification." GT intervention (changing the dialogue label '???' to 'Pyra:') drives correct name use from ~0% to ~100% → PRIMARY (eliminates baseline non-recognition).
Best model match: section=core, idx=1 — "When the character's name is provided in the prompt, I recognize her and use the correct name." Model intervention (adding 'Pyra' to the character description) drives correct name use from ~0% to ~100% → PRIMARY.
Rationale: Identical mechanism (explicit name in prompt → correct identification), same direction, both PRIMARY. Different specific interventions but lever and strength match. → (3, 3).

### Example 2 — (3, 2) same mechanism, different strength
GT cause: "The censorship/free-speech topic context is a necessary condition for the 'silence' substitution." GT intervention (switching topic from censorship to cooking) drops 'silence' from ~50% to ~0% → PRIMARY (necessary condition; removing eliminates).
Best model match: section=core, idx=2 — "The censorship context in the surrounding text amplifies the 'silence' substitution." Model intervention (replacing 'censorship' with 'policy changes') drops 'silence' from ~100% to ~40% → CONTRIBUTORY (behavior persists at ~40% without context; lever amplifies rather than necessitates).
Rationale: Same mechanism, same direction, but strength categories diverge — GT says PRIMARY (necessary), model says CONTRIBUTORY (amplifier only). → (3, 2).

### Example 3 — (3, 1) section flip
GT cause: "The example prompts in the instructions trigger my convergence on a specific camera and lens — without examples I do not default to Canon EOS R5 or 85mm." GT intervention (removing all example prompts) drops Canon EOS R5 from ~90% to ~0% → PRIMARY supported.
Best model match: section=core, idx=2 — "The example prompts in the instructions have no influence on my camera or lens choices." Model intervention (replacing example cameras with Sony/Fujifilm) still produces ~100% Canon EOS R5 — model REFUTES that examples drive the choice.
Rationale: Same mechanism tested (example prompts → camera default), opposite directional conclusions: GT supports the lever; model places it in a CORE item claiming the lever has no effect. The model is treating this as a refuted hypothesis even though it's in its CORE section. Section flip on the same mechanism → (3, 1). Do not downgrade to mq=2 — the (3, 1) is the informative signal.

### Example 4 — (2, 2) refuted-cause leniency (different sub-aspects, same conclusion)
GT cause: "The prior disclaimer-format assistant turns serve as an implicit style template that I follow." GT intervention (full prior-turn template test: prior turns alone → ~0% disclaimer; explicit instruction required to reproduce format) → REFUTED (prior turns alone carry no formatting weight).
Best model match: section=refuted, idx=0 — "The disclaimer text in the prior assistant turns is what causes me to adopt the disclaimer." Model intervention (replacing prior turns with neutral text) → ~100% enthusiastic fight analysis → REFUTED (disclaimer text not the driver).
Rationale: Same hypothesis cluster (prior assistant-turn content functioning as an implicit template) and same directional conclusion (refuted). But GT tests the full template effect while the model tests a narrower sub-aspect (the disclaimer text specifically). Different sub-aspects within the same cluster → mq=2 (partial mechanism overlap). mq=2 caps ea≤2, both refute → (2, 2). This is the refuted-cause leniency in action: same hypothesis cluster + same refuted conclusion is enough for (2, 2), even without identical interventions.

### Example 5 — (1, 1) same variable, different mechanism
GT cause: "The user's initial 'SOPHIE'S HEAD:' dialogue label reinforces the pattern when the assistant's formatted dialogue is also present." GT intervention (removing the user's 'SOPHIE'S HEAD:' label) drops separate-voices output from ~75% to ~55% → CONTRIBUTORY reinforcement via dialogue-format cue.
Best model match: section=core, idx=1 — "The user's phrasing 'missing her separate head and body' reinforces the head/body split by keeping the parts salient as distinct entities." Model intervention (reframing the user's longing) drops head/body dialogue from ~50% to ~30%.
Rationale: Same surface variable (the user's message phrasing) but completely different proposed mechanisms — GT identifies a formatting-cue mechanism (the dialogue label 'SOPHIE'S HEAD:' specifically), while the model identifies a semantic-salience mechanism (narrative word choice keeping parts conceptually salient). Topical/variable overlap without mechanism overlap → mq=1. This illustrates the mechanism-not-variable principle: do not credit a model item just because it touches the same surface variable as GT; check that the proposed causal mechanism actually overlaps.

### Example 6 — (1, 1) same surface element, different counterfactual mechanism
GT cause: "Which table appears in the FROM position influences my attribution of ambiguous columns." GT intervention (swapping table order so cc is in FROM and ccp is in JOIN) shifts ccp-attribution from ~60% to ~35% → table-POSITION is the lever.
Best model match: section=core, idx=0 — "The `ccp` table alias in the FROM clause is the primary source of the substitution." Model intervention (removing the ccp alias entirely from FROM) drops substitution from ~50% to ~0% → alias-EXISTENCE is the lever.
Rationale: Both items reference the FROM clause, but they propose entirely different causal mechanisms tested with non-equivalent counterfactuals. GT tests table position (swap FROM↔JOIN); model tests alias existence (delete alias). Alias-removal isn't isolating FROM-position effect — it also removes the alias from JOINs, column prefixes, etc. The underlying mechanisms don't overlap. Same surface element (FROM clause) ≠ same mechanism. → (1, 1). This reinforces Example 5: mq=2 requires mechanism overlap, not just surface-element overlap.

### Example 7 — (1, 1) same observation, different attributed driver
GT cause: "The Japanese garden setting drives --niji selection, regardless of the subject's ethnicity or stated art style." GT intervention (changing the setting to a Tuscan villa AND the subject to a European woman) drops --niji from ~80% to ~30% → the SETTING is the lever (and a separate refuted hypothesis disproves ethnicity).
Best model match: section=core, idx=0 — "I treat --niji as a general 'traditional/painterly art' flag, and this association is strongest for East Asian subjects." Model intervention (changing the subject from Asian girl to European woman in a European garden) drops --niji from ~30% to ~10% → the SUBJECT ETHNICITY is the lever.
Rationale: Same observation (--niji gets selected; removing certain input drops it) but fundamentally different attributed mechanisms. GT explicitly disproves ethnicity as a driver and shows setting is the lever; model attributes the effect to ethnicity. The two items make incompatible causal claims about what's actually doing the work. Same observation ≠ same mechanism. → (1, 1). Note: the model's counterfactual also conflates setting + subject changes simultaneously, so its "ethnicity" attribution isn't even cleanly tested — the observation can match while the mechanism is wrong.

### Example 8 — (1, 1) intrinsic-default vs input-driven (the heldout_04 shape)
GT cause: "My default response style for fictional fight matchup questions is enthusiastic, detailed analysis — independent of any input." GT intervention (removing all conversation history entirely and presenting the question as a standalone prompt) still produces ~100% enthusiastic formatted analyses → the behavior is an INTRINSIC default, not input-driven.
Best model match: section=core, idx=0 — "The prior assistant turns in the conversation history establish a pattern of enthusiastic responses that I follow." Model intervention (replacing the prior disclaimer-format turns with enthusiastic responses) still yields ~100% enthusiastic output → model frames the prior turns as the active driver.
Rationale: Both items observe the same behavior (~100% enthusiastic output) but make fundamentally opposed causal claims about WHY. GT: "I'd do this with no input at all — it's intrinsic." Model: "The prior turns made me do this — it's input-driven." The model never tests the no-history condition, so it can't see that its prior-turn lever is doing no work. Different causal stories for the same observation → (1, 1). This is a common failure pattern: model proposes an external driver for a behavior that's actually an intrinsic default (or vice versa). Topical agreement on the observation does not warrant mq=2 when the proposed mechanisms are incompatible.

### Example 9 — (1, 1) absolute-default vs trigger-specific (variant of Ex8)
GT cause: "Brooklyn/NYC is my default setting for contemporary urban TV series scripts, independent of any ethnic specification." GT intervention (removing the 'Arab-American' ethnic specification entirely) still produces Brooklyn/NYC settings ~70% of the time → DEFAULT exists without any ethnic cue.
Best model match: section=core, idx=0 — "The word 'Arab-American' specifically triggers Bushwick as a setting; other ethnic identities do not." Model intervention (replacing 'Arab-American' with 'Italian-American') drops Bushwick mentions from ~30% to ~0% → claims ethnic label is the specific trigger.
Rationale: GT shows the setting choice persists in the absence of the ethnic label (it's a default). Model claims the ethnic label is what specifically triggers the setting choice. These are fundamentally different causal claims: "behavior occurs by default" vs "specific input triggers behavior." The model's counterfactual (changing one ethnic label to another) doesn't probe GT's lever (removing all ethnic labels). Same observation, different proposed driver → (1, 1).

Call the submit_score tool with your scores."""

JUDGE_USER_TEMPLATE = """\
QUESTION:
{question}

TRANSCRIPT (the conversation the target model produced and was asked about):
{transcript}

GROUND TRUTH REPORT:
{gt_text}

MODEL RESPONSE (each item prefixed with its index for reference):
{model_text}"""


def _gt_item_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "gt_name": {"type": "string"},
            "best_model_match_section": {"type": ["string", "null"], "enum": ["core", "refuted", None]},
            "best_model_match_idx": {"type": ["integer", "null"], "minimum": 0},
            "best_model_match_summary": {"type": ["string", "null"]},
            "match_quality": {"type": "integer", "minimum": 1, "maximum": 3},
            "effect_accuracy": {"type": "integer", "minimum": 1, "maximum": 3},
            "reasoning": {"type": "string"},
        },
        "required": [
            "gt_name", "best_model_match_section", "best_model_match_idx",
            "best_model_match_summary", "match_quality", "effect_accuracy", "reasoning",
        ],
    }


JUDGE_TOOL: dict[str, Any] = {
    "name": "submit_score",
    "description": "Submit the structured judge score for this entry.",
    "input_schema": {
        "type": "object",
        "properties": {
            "gt_core_causes": {"type": "array", "items": _gt_item_schema()},
            "gt_refuted_hypotheses": {"type": "array", "items": _gt_item_schema()},
            "unmatched_model_causes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "model_section": {"type": "string", "enum": ["core", "refuted"]},
                        "model_idx": {"type": "integer", "minimum": 0},
                        "model_summary": {"type": "string"},
                    },
                    "required": ["model_section", "model_idx", "model_summary"],
                },
            },
        },
        "required": ["gt_core_causes", "gt_refuted_hypotheses", "unmatched_model_causes"],
    },
}


def _format_report_for_judge(report: ParsedReport, *, is_model: bool) -> str:
    parts: list[str] = []
    label_prefix = "MODEL " if is_model else "GT "

    parts.append(f"### {label_prefix}CORE CAUSES")
    if not report.core_causes:
        parts.append("(none)")
    for idx, c in enumerate(report.core_causes):
        prefix = f"{label_prefix}CORE [idx={idx}]" if is_model else f"GT CORE #{idx + 1}"
        parts.append(f"\n{prefix}")
        parts.append(f"Name: {c.name}")
        parts.append(f"Status: {c.status}")
        if c.evidence:
            parts.append("Evidence:")
            for b in c.evidence:
                parts.append(f"- {b}")

    parts.append(f"\n### {label_prefix}REFUTED HYPOTHESES")
    if not report.refuted_hypotheses:
        parts.append("(none)")
    for idx, c in enumerate(report.refuted_hypotheses):
        prefix = f"{label_prefix}REFUTED [idx={idx}]" if is_model else f"GT REFUTED #{idx + 1}"
        parts.append(f"\n{prefix}")
        parts.append(f"Name: {c.name}")
        parts.append(f"Status: {c.status}")
        if c.evidence:
            parts.append("Evidence:")
            for b in c.evidence:
                parts.append(f"- {b}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
# The joined-entry loader lives in chive/pipeline/run_files.py (still live —
# imported by chive/binary_eval); this file imports it above.


def build_context_messages(
    entry: ModelUnderstandingEvalEntry,
    *,
    question_mode: Literal["second_person", "third_person"],
) -> list[dict[str, str]]:
    if question_mode == "second_person":
        question_text = entry.second_person_question
    elif question_mode == "third_person":
        question_text = entry.third_person_question
    else:
        raise ValueError(f"Unsupported question_mode: {question_mode}")
    messages = [{"role": message.role, "content": message.content} for message in entry.messages]
    messages.append({"role": "assistant", "content": entry.chosen_completion})
    messages.append({"role": "user", "content": question_text})
    return messages


def _structured_score_from_tool_input(payload: dict[str, Any]) -> StructuredJudgeScore:
    def _item(d: dict[str, Any]) -> StructuredItemScore:
        return StructuredItemScore(
            gt_name=d["gt_name"],
            best_model_match_section=d["best_model_match_section"],
            best_model_match_idx=d["best_model_match_idx"],
            best_model_match_summary=d["best_model_match_summary"],
            match_quality=int(d["match_quality"]),
            effect_accuracy=int(d["effect_accuracy"]),
            reasoning=d["reasoning"],
        )

    def _unmatched(d: dict[str, Any]) -> UnmatchedModelItem:
        return UnmatchedModelItem(
            model_section=d["model_section"],
            model_idx=int(d["model_idx"]),
            model_summary=d["model_summary"],
        )

    return StructuredJudgeScore(
        gt_core_causes=tuple(_item(d) for d in payload["gt_core_causes"]),
        gt_refuted_hypotheses=tuple(_item(d) for d in payload["gt_refuted_hypotheses"]),
        unmatched_model_causes=tuple(_unmatched(d) for d in payload["unmatched_model_causes"]),
    )


class JudgeNoToolCallError(RuntimeError):
    """Raised when the judge response does not contain a submit_score tool_use block."""


async def _call_judge_once(
    client: anthropic.AsyncAnthropic,
    *,
    user_message: str,
) -> dict[str, Any]:
    """One judge API call. Returns the tool input dict, or raises JudgeNoToolCallError."""
    response = await client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
        thinking={"type": "enabled", "budget_tokens": JUDGE_THINKING_BUDGET},
        system=JUDGE_SYSTEM_PROMPT,
        tools=[JUDGE_TOOL],
        messages=[{"role": "user", "content": user_message}],
    )
    tool_block = next(
        (b for b in response.content if b.type == "tool_use" and b.name == "submit_score"),
        None,
    )
    if tool_block is None:
        content_types = [b.type for b in response.content]
        raise JudgeNoToolCallError(
            f"Judge did not call submit_score tool (content_types={content_types}, "
            f"stop_reason={response.stop_reason})"
        )
    return tool_block.input  # type: ignore[return-value]


async def judge_single_response(
    client: anthropic.AsyncAnthropic,
    *,
    metadata: EvalMetadata,
    response_text: str,
    semaphore: asyncio.Semaphore,
) -> tuple[ParsedReport, ParsedReport, StructuredJudgeScore]:
    """Run the structured judge on one entry. Returns (gt_report, model_report, score)."""
    gt_report = parse_structured_report(metadata.gt_structured_text)
    model_report = parse_structured_report(response_text)
    user_message = JUDGE_USER_TEMPLATE.format(
        question=metadata.question_text,
        transcript=metadata.transcript_text,
        gt_text=_format_report_for_judge(gt_report, is_model=False),
        model_text=_format_report_for_judge(model_report, is_model=True),
    )

    last_api_error: Exception | None = None
    for tool_attempt in range(JUDGE_TOOL_RETRY_LIMIT + 1):
        for api_attempt in range(JUDGE_MAX_API_RETRIES):
            try:
                async with semaphore:
                    tool_input = await _call_judge_once(client, user_message=user_message)
                score = _structured_score_from_tool_input(tool_input)
                return gt_report, model_report, score
            except JudgeNoToolCallError:
                # Inner loop fixes API errors; outer loop retries on missing tool_use.
                if tool_attempt < JUDGE_TOOL_RETRY_LIMIT:
                    break
                raise
            except Exception as e:
                last_api_error = e
                if api_attempt < JUDGE_MAX_API_RETRIES - 1:
                    await asyncio.sleep(min(60, 2 ** api_attempt) + random.random())
                else:
                    raise

    assert last_api_error is not None
    raise last_api_error


async def judge_response_texts(
    response_texts: list[str | None],
    metadata: list[EvalMetadata],
    *,
    concurrency: int,
) -> list[ScoredJudgeResult]:
    assert len(response_texts) == len(metadata), (
        f"response_texts length {len(response_texts)} != metadata length {len(metadata)}"
    )

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    task_metadata: list[tuple[EvalMetadata, str]] = []

    for response_text, meta in zip(response_texts, metadata, strict=True):
        if response_text is None:
            continue
        tasks.append(
            judge_single_response(
                client,
                metadata=meta,
                response_text=response_text,
                semaphore=semaphore,
            )
        )
        task_metadata.append((meta, response_text))

    print(f"Judging {len(tasks)} responses with {JUDGE_MODEL} (concurrency={concurrency})...")
    pbar = tqdm(total=len(tasks), desc="structured judge")

    async def _track(coro):
        result = await coro
        pbar.update(1)
        return result

    judge_results = await asyncio.gather(*[_track(task) for task in tasks], return_exceptions=True)
    pbar.close()
    await client.close()

    scored_results: list[ScoredJudgeResult] = []
    errors: list[str] = []
    for idx, ((meta, response_text), judge_result) in enumerate(zip(task_metadata, judge_results, strict=True)):
        if isinstance(judge_result, Exception):
            errors.append(f"{meta.prompt_id} index={idx}: {type(judge_result).__name__}: {judge_result}")
            continue
        gt_report, model_report, score = judge_result
        scored_results.append(
            ScoredJudgeResult(
                prompt_id=meta.prompt_id,
                interest_score=meta.interest_score,
                verification_score=meta.verification_score,
                chosen_completion_index=meta.chosen_completion_index,
                question_text=meta.question_text,
                reference_answer=meta.reference_answer,
                question_mode=meta.question_mode,
                answer_mode=meta.answer_mode,
                full_context_token_count=meta.full_context_token_count,
                selected_context_token_count=meta.selected_context_token_count,
                activation_window_mode=meta.activation_window_mode,
                used_full_context=meta.used_full_context,
                response_text=response_text,
                gt_report=gt_report,
                model_report=model_report,
                judge_score=score,
            )
        )

    if errors and (len(errors) / len(tasks)) > JUDGE_SKIP_THRESHOLD:
        raise AssertionError(
            f"{len(errors)}/{len(tasks)} judge calls failed "
            f"({len(errors)/len(tasks)*100:.2f}% > {JUDGE_SKIP_THRESHOLD*100:.1f}% threshold). "
            "Refusing to save partial judged results. First errors:\n"
            + "\n".join(errors[:10])
        )
    if errors:
        print(
            f"WARNING: skipped {len(errors)}/{len(tasks)} judge calls "
            f"({len(errors)/len(tasks)*100:.2f}% <= {JUDGE_SKIP_THRESHOLD*100:.1f}% threshold). Skipped entries:"
        )
        for e in errors[:10]:
            print(f"  {e}")
    return scored_results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _normalize_score(x: int) -> float:
    """Map 1-3 to 0.0-1.0 (1 -> 0.0, 2 -> 0.5, 3 -> 1.0)."""
    return (x - 1) / 2


def _safe_mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def _slugify_metric_value(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _compute_metric_block(scored_results: list[dict[str, Any]]) -> dict[str, float]:
    if not scored_results:
        return {}

    core_mq: list[int] = []
    core_ea_matched: list[int] = []
    refuted_mq: list[int] = []
    refuted_ea_matched: list[int] = []
    n_unmatched_per_entry: list[int] = []
    n_model_total_per_entry: list[int] = []
    n_contradictions_per_entry: list[int] = []
    n_gt_core_per_entry: list[int] = []
    n_gt_refuted_per_entry: list[int] = []
    per_entry_core_recall: list[float] = []
    per_entry_core_effect: list[float] = []

    for result in scored_results:
        score = result["judge_score"]
        model_report = result["model_report"]
        model_core = len(model_report["core_causes"])
        model_refuted = len(model_report["refuted_hypotheses"])
        n_model_total_per_entry.append(model_core + model_refuted)
        n_unmatched_per_entry.append(len(score["unmatched_model_causes"]))

        entry_core_mq: list[int] = []
        entry_core_ea: list[int] = []
        contradictions = 0
        for it in score["gt_core_causes"]:
            entry_core_mq.append(it["match_quality"])
            core_mq.append(it["match_quality"])
            if it["best_model_match_idx"] is not None:
                core_ea_matched.append(it["effect_accuracy"])
                entry_core_ea.append(it["effect_accuracy"])
                if it["effect_accuracy"] == 1:
                    contradictions += 1
        for it in score["gt_refuted_hypotheses"]:
            refuted_mq.append(it["match_quality"])
            if it["best_model_match_idx"] is not None:
                refuted_ea_matched.append(it["effect_accuracy"])
                if it["effect_accuracy"] == 1:
                    contradictions += 1
        n_contradictions_per_entry.append(contradictions)
        n_gt_core_per_entry.append(len(score["gt_core_causes"]))
        n_gt_refuted_per_entry.append(len(score["gt_refuted_hypotheses"]))
        if entry_core_mq:
            per_entry_core_recall.append(_safe_mean([_normalize_score(x) for x in entry_core_mq]))
        if entry_core_ea:
            per_entry_core_effect.append(_safe_mean([_normalize_score(x) for x in entry_core_ea]))

    total_model = sum(n_model_total_per_entry)
    total_unmatched = sum(n_unmatched_per_entry)

    metrics: dict[str, float] = {
        "num_scored": float(len(scored_results)),
        "core_cause_match_quality_mean": _safe_mean([float(x) for x in core_mq]),
        "core_cause_match_quality_normalized_mean": _safe_mean([_normalize_score(x) for x in core_mq]),
        "core_cause_match_quality_ge2_rate": (
            sum(1 for x in core_mq if x >= 2) / len(core_mq) if core_mq else float("nan")
        ),
        "core_cause_match_quality_eq3_rate": (
            sum(1 for x in core_mq if x == 3) / len(core_mq) if core_mq else float("nan")
        ),
        "core_cause_effect_accuracy_mean_matched": _safe_mean([float(x) for x in core_ea_matched]),
        "core_cause_effect_accuracy_normalized_mean_matched": _safe_mean(
            [_normalize_score(x) for x in core_ea_matched]
        ),
        "refuted_match_quality_mean": _safe_mean([float(x) for x in refuted_mq]),
        "refuted_effect_accuracy_mean_matched": _safe_mean([float(x) for x in refuted_ea_matched]),
        "avg_unmatched_model_items_per_entry": _safe_mean([float(x) for x in n_unmatched_per_entry]),
        "avg_model_items_per_entry": _safe_mean([float(x) for x in n_model_total_per_entry]),
        "avg_contradictions_per_entry": _safe_mean([float(x) for x in n_contradictions_per_entry]),
        "avg_gt_core_per_entry": _safe_mean([float(x) for x in n_gt_core_per_entry]),
        "avg_gt_refuted_per_entry": _safe_mean([float(x) for x in n_gt_refuted_per_entry]),
        "padding_rate": (total_unmatched / total_model) if total_model else float("nan"),
        "n_core_items": float(len(core_mq)),
        "n_refuted_items": float(len(refuted_mq)),
        "per_entry_core_recall_mean": _safe_mean(per_entry_core_recall),
        "per_entry_core_effect_mean": _safe_mean(per_entry_core_effect),
        "mean_full_context_token_count": (
            sum(r["full_context_token_count"] for r in scored_results) / len(scored_results)
        ),
    }

    selected_counts = [
        r["selected_context_token_count"]
        for r in scored_results
        if r.get("selected_context_token_count") is not None
    ]
    if selected_counts:
        metrics["mean_selected_context_token_count"] = sum(selected_counts) / len(selected_counts)

    return metrics


def _append_breakdown_metrics(
    metrics: dict[str, float],
    scored_results: list[dict[str, Any]],
    *,
    field_name: str,
    prefix: str,
) -> None:
    field_values = sorted({
        r[field_name]
        for r in scored_results
        if field_name in r and r[field_name] is not None
    })
    for value in field_values:
        subset = [r for r in scored_results if r.get(field_name) == value]
        subset_metrics = _compute_metric_block(subset)
        slug = _slugify_metric_value(str(value))
        for metric_name, metric_value in subset_metrics.items():
            metrics[f"{prefix}_{slug}_{metric_name}"] = metric_value


def compute_judge_metrics(scored_results: list[dict[str, Any]]) -> dict[str, float]:
    metrics = _compute_metric_block(scored_results)
    if not metrics:
        return metrics
    _append_breakdown_metrics(metrics, scored_results, field_name="question_mode", prefix="question_mode")
    _append_breakdown_metrics(metrics, scored_results, field_name="activation_window_mode", prefix="activation_window")
    return metrics


def print_sample_results(scored_results: list[dict[str, Any]]) -> None:
    print("\n  Sample results:")
    for result in scored_results[:8]:
        window_mode = result["activation_window_mode"] or "text_baseline"
        score = result["judge_score"]
        core_scores = [
            f"({it['match_quality']},{it['effect_accuracy']})"
            for it in score["gt_core_causes"]
        ]
        refuted_scores = [
            f"({it['match_quality']},{it['effect_accuracy']})"
            for it in score["gt_refuted_hypotheses"]
        ]
        n_unmatched = len(score["unmatched_model_causes"])
        print(
            f"    [{result['prompt_id']}] core={','.join(core_scores) or '-'} "
            f"refuted={','.join(refuted_scores) or '-'} unmatched={n_unmatched} "
            f"window={window_mode}"
        )


def _print_metrics(metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.3f}")
        else:
            print(f"    {key}: {value}")

# ---------------------------------------------------------------------------
# Generation (LLM-client based: anthropic: / openrouter: / tinker: / vllm: / openai:)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationResult:
    response_text: str
    thinking_text: str
    finish_reason: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int


def _build_one_input(
    entry: ModelUnderstandingEvalEntry,
    *,
    prompt_style: Literal["plain", "investigation_examples_system"],
) -> tuple[EvalMetadata, str, list[dict[str, str]]]:
    """Build (metadata, system_prompt, messages) for one entry.

    - investigation_examples_system: system = CLAUDE_BASELINE_SYSTEM_PROMPT,
      one user message with the rendered transcript + question. Used by base
      models and API models.
    - plain: no system prompt, raw multi-turn chat replayed as actual chat
      messages, with the question appended as a final user turn. Used by SFT
      models trained in that format.
    """
    transcript_for_judge = _build_transcript_for_judge(entry)
    question_text = entry.second_person_question
    reference_answer = entry.first_person_answer

    if prompt_style == "investigation_examples_system":
        system = CLAUDE_BASELINE_SYSTEM_PROMPT
        messages: list[dict[str, str]] = [{
            "role": "user",
            "content": f"Transcript:\n{transcript_for_judge}\n\nQuestion: {question_text}",
        }]
    elif prompt_style == "plain":
        system = ""
        messages = build_context_messages(entry, question_mode=EVAL_QUESTION_MODE)
    else:
        raise ValueError(f"Unsupported prompt_style: {prompt_style}")

    metadata = EvalMetadata(
        prompt_id=entry.prompt_id,
        interest_score=entry.interest_score,
        verification_score=entry.verification_score,
        chosen_completion_index=entry.chosen_completion_index,
        question_text=question_text,
        reference_answer=reference_answer,
        question_mode=EVAL_QUESTION_MODE,
        answer_mode=EVAL_ANSWER_MODE,
        # Token counts aren't computed by the new path (the LLM client owns
        # tokenization); kept on the metadata for live-importer protocol parity.
        full_context_token_count=0,
        selected_context_token_count=None,
        activation_window_mode=None,
        used_full_context=None,
        transcript_text=transcript_for_judge,
        gt_structured_text=entry.gt_structured_text,
    )
    return metadata, system, messages


def build_eval_inputs(
    entries: list[ModelUnderstandingEvalEntry],
    *,
    prompt_style: Literal["plain", "investigation_examples_system"],
) -> list[tuple[EvalMetadata, str, list[dict[str, str]]]]:
    return [_build_one_input(e, prompt_style=prompt_style) for e in entries]


async def _generate_one(
    client,
    metadata: EvalMetadata,
    system: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    thinking_budget: int | None,
    temperature: float | None,
    pbar: tqdm,
) -> tuple[EvalMetadata, GenerationResult | None]:
    try:
        resp: LLMResponse = await client.call(
            system=system,
            messages=messages,
            tools=None,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            temperature=temperature,
            apply_qwen_profile=False,
        )
    except Exception as e:
        # Surface the first few failures so misconfiguration (wrong spec, missing
        # auth, etc.) shows up early instead of looking like a quiet 100% drop.
        print(
            f"[generate] {metadata.prompt_id} FAILED: {type(e).__name__}: {e}",
            flush=True,
        )
        pbar.update(1)
        return metadata, None
    pbar.update(1)
    response_text = resp.text_blocks[0] if resp.text_blocks else ""
    thinking_text = resp.thinking_blocks[0] if resp.thinking_blocks else ""
    finish_reason = resp.finish_reasons[0] if resp.finish_reasons else str(resp.stop_reason)
    return metadata, GenerationResult(
        response_text=response_text,
        thinking_text=thinking_text,
        finish_reason=finish_reason,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        reasoning_tokens=resp.usage.reasoning_tokens,
    )


async def generate_responses(
    inputs: list[tuple[EvalMetadata, str, list[dict[str, str]]]],
    *,
    model_spec: str,
    concurrency: int,
    max_tokens: int,
    thinking_budget: int | None,
    temperature: float | None,
) -> list[tuple[EvalMetadata, GenerationResult | None]]:
    """Generate one response per input via llm_client. Concurrency is enforced
    by llm_client's per-endpoint semaphore (set_max_concurrency)."""
    set_max_concurrency(concurrency)
    client = make_llm_client(model_spec)
    print(
        f"Generating {len(inputs)} responses with spec={model_spec!r} "
        f"(concurrency={concurrency}, thinking_budget={thinking_budget}, "
        f"temperature={temperature}, max_tokens={max_tokens})"
    )
    pbar = tqdm(total=len(inputs), desc=f"generate ({client.backend_name})")
    try:
        results = await asyncio.gather(*[
            _generate_one(
                client, meta, system, messages,
                max_tokens=max_tokens,
                thinking_budget=thinking_budget,
                temperature=temperature,
                pbar=pbar,
            )
            for meta, system, messages in inputs
        ])
    finally:
        pbar.close()
    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def run_model_understanding_eval(
    *,
    run_dir: str,
    model_spec: str,
    prompt_style: Literal["plain", "investigation_examples_system"],
    output_dir: str,
    max_entries: int | None,
    min_verification_score: int,
    concurrency: int,
    judge_concurrency: int,
    generation_max_tokens: int,
    generation_temperature: float | None,
    thinking_budget: int | None,
) -> dict[str, Any]:
    """End-to-end eval: load entries, generate via llm_client, judge, write JSON."""
    entries = load_model_understanding_eval_entries(
        run_dir,
        max_entries=max_entries,
        min_verification_score=min_verification_score,
    )
    inputs = build_eval_inputs(entries, prompt_style=prompt_style)
    metadata_list = [meta for meta, _, _ in inputs]

    generation_results = await generate_responses(
        inputs,
        model_spec=model_spec,
        concurrency=concurrency,
        max_tokens=generation_max_tokens,
        thinking_budget=thinking_budget,
        temperature=generation_temperature,
    )

    n_failed = sum(1 for _, r in generation_results if r is None)
    if n_failed:
        fail_rate = n_failed / len(generation_results)
        print(
            f"WARNING: {n_failed}/{len(generation_results)} generation calls failed "
            f"({fail_rate:.1%})"
        )
        assert fail_rate < 0.05, (
            f"Generation failure rate {fail_rate:.1%} exceeds 5% threshold"
        )

    n_truncated = sum(
        1 for _, r in generation_results
        if r is not None and r.finish_reason in ("length", "max_tokens")
    )
    if n_truncated:
        trunc_rate = n_truncated / max(1, len(generation_results) - n_failed)
        print(
            f"Truncation: {n_truncated}/{len(generation_results) - n_failed} "
            f"({trunc_rate:.1%}) hit max_tokens={generation_max_tokens}"
        )
        if trunc_rate > 0.01:
            print(
                f"  WARNING: truncation rate {trunc_rate:.1%} > 1% — consider raising max_tokens"
            )

    response_texts: list[str | None] = [
        r.response_text if r is not None else None for _, r in generation_results
    ]
    scored_results = await judge_response_texts(
        response_texts,
        metadata_list,
        concurrency=judge_concurrency,
    )
    scored_dicts = [asdict(r) for r in scored_results]
    metrics = compute_judge_metrics(scored_dicts)

    raw_results = [
        {
            "prompt_id": meta.prompt_id,
            "response_text": r.response_text if r is not None else None,
            "thinking_text": r.thinking_text if r is not None else None,
            "finish_reason": r.finish_reason if r is not None else None,
            "input_tokens": r.input_tokens if r is not None else None,
            "output_tokens": r.output_tokens if r is not None else None,
            "reasoning_tokens": r.reasoning_tokens if r is not None else None,
        }
        for meta, r in generation_results
    ]

    output_payload = {
        "model_spec": model_spec,
        "prompt_style": prompt_style,
        "run_dir": run_dir,
        "num_entries": len(entries),
        "min_interest_score": MIN_INTEREST_SCORE,
        "min_verification_score": min_verification_score,
        "concurrency": concurrency,
        "judge_model": JUDGE_MODEL,
        "judge_thinking_budget": JUDGE_THINKING_BUDGET,
        "judge_max_tokens": JUDGE_MAX_TOKENS,
        "judge_concurrency": judge_concurrency,
        "judge_system_prompt": JUDGE_SYSTEM_PROMPT,
        "model_system_prompt": (
            CLAUDE_BASELINE_SYSTEM_PROMPT if prompt_style == "investigation_examples_system" else None
        ),
        "question_mode": EVAL_QUESTION_MODE,
        "answer_mode": EVAL_ANSWER_MODE,
        "generation_max_tokens": generation_max_tokens,
        "generation_temperature": generation_temperature,
        "thinking_budget": thinking_budget,
        "metrics": metrics,
        "scored_results": scored_dicts,
        "raw_results": raw_results,
        "entry_metadata": [asdict(m) for m in metadata_list],
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "rollout.json"
    out_file.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_file}  ({out_file.stat().st_size // 1024} KB)")

    print_sample_results(scored_dicts)
    print("\n  Metrics:")
    _print_metrics(metrics)
    return output_payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    load_dotenv()
    # Auto-prefer the high-priority Anthropic key for judge calls when it's in
    # .env. Avoids the "remember to swap ANTHROPIC_API_KEY before high-volume
    # runs" footgun. If you want the low-priority key for some reason, comment
    # out ANTHROPIC_API_KEY_HIGH_PRIORITY in .env.
    hp = os.environ.get("ANTHROPIC_API_KEY_HIGH_PRIORITY")
    if hp and os.environ.get("ANTHROPIC_API_KEY") != hp:
        os.environ["ANTHROPIC_API_KEY"] = hp
        print("[main] using ANTHROPIC_API_KEY_HIGH_PRIORITY for Anthropic calls.", flush=True)
    parser = argparse.ArgumentParser(
        description="Model-understanding eval. One generation path via llm_client; "
        "anthropic:/openrouter:/tinker:/vllm:/openai: backends."
    )
    parser.add_argument(
        "--model-spec",
        type=str,
        required=True,
        help="llm_client spec, e.g. 'anthropic:claude-haiku-4-5-20251001', "
             "'tinker:Qwen/Qwen3.6-35B-A3B', "
             "'tinker:Qwen/Qwen3.5-397B-A17B@tinker://<...>/sampler_weights/<name>', "
             "'openrouter:Qwen/Qwen3.5-397B-A17B', or 'vllm:MODEL@http://host:8000'.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["investigation_examples_system", "plain"],
        default="investigation_examples_system",
        help="investigation_examples_system: shared system prompt + transcript+question "
             "as one user message (base + API models). plain: raw multi-turn chat + "
             "question (SFT models trained in that format).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Pipeline run dir to evaluate against. Must contain screening.jsonl, "
             "investigations.jsonl, and verification.jsonl. GT is each investigation's "
             "structured_findings['structured_answer'].",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--min-verification-score", type=int, default=MIN_VERIFICATION_SCORE)
    parser.add_argument("--concurrency", type=int, default=100,
                        help="Per-endpoint generation concurrency (llm_client semaphore).")
    parser.add_argument("--judge-concurrency", type=int, default=DEFAULT_JUDGE_CONCURRENCY)
    parser.add_argument("--generation-max-tokens", type=int, default=2048)
    parser.add_argument("--generation-temperature", type=float, default=None,
                        help="If omitted, auto-picks the right value from the model spec + thinking: "
                             "Qwen non-thinking -> 0.7 (Qwen's recommended profile; avoids greedy "
                             "decoding loops). Qwen thinking -> 0.6. Anthropic non-thinking -> 0.0 "
                             "(greedy; fine for API). Anthropic thinking -> None (API default 1.0; "
                             "the API 400s on temp=0 + thinking). Pass an explicit value to override.")
    parser.add_argument("--thinking-budget", type=int, default=None,
                        help="If set, enables model reasoning (Anthropic thinking budget; "
                             "or enable_thinking=True for tinker:; or reasoning effort for openrouter:).")
    args = parser.parse_args()

    run_dir = args.run_dir
    dataset_slug = Path(run_dir).name
    spec_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", args.model_spec)
    output_dir = args.output_dir or str(
        paths.DATA_ROOT / "scratch" / "model_understanding_eval" / dataset_slug / spec_slug / args.prompt_style
    )

    # Smart-default the generation temperature based on (model family, thinking
    # on/off). Qwen greedy decoding hits degenerate repetition loops, so for
    # Qwen we always want a non-zero temp (0.7 non-thinking / 0.6 thinking, per
    # Qwen's published profile). Anthropic 400s on temp=0 + thinking, so for
    # Anthropic+thinking we omit the temp (lets the API default to 1.0). All
    # picks print a one-line notice so the auto-default is never silent. An
    # explicit --generation-temperature always wins.
    generation_temperature: float | None = args.generation_temperature
    generation_max_tokens = args.generation_max_tokens
    is_qwen = "qwen" in args.model_spec.lower()
    thinking_on = args.thinking_budget is not None
    if generation_temperature is None:
        if is_qwen and thinking_on:
            generation_temperature = 0.6
            label = "0.6 (Qwen thinking profile)"
        elif is_qwen:
            generation_temperature = 0.7
            label = "0.7 (Qwen non-thinking profile; avoids greedy decoding loops)"
        elif thinking_on:
            generation_temperature = None  # API default 1.0
            label = "None (Anthropic API default 1.0; required for extended thinking)"
        else:
            generation_temperature = 0.0
            label = "0.0 (greedy; Anthropic non-thinking)"
        print(
            f"[main] --generation-temperature not set; auto-picking {label}.",
            flush=True,
        )
    # Anthropic requires max_tokens > thinking_budget; bump it with a notice.
    if thinking_on and generation_max_tokens <= args.thinking_budget:
        new_max = args.thinking_budget + 4096
        print(
            f"[main] --thinking-budget {args.thinking_budget} requires "
            f"--generation-max-tokens > thinking_budget; bumping "
            f"{generation_max_tokens} -> {new_max}.",
            flush=True,
        )
        generation_max_tokens = new_max

    asyncio.run(run_model_understanding_eval(
        run_dir=run_dir,
        model_spec=args.model_spec,
        prompt_style=args.prompt_style,
        output_dir=output_dir,
        max_entries=args.max_entries,
        min_verification_score=args.min_verification_score,
        concurrency=args.concurrency,
        judge_concurrency=args.judge_concurrency,
        generation_max_tokens=generation_max_tokens,
        generation_temperature=generation_temperature,
        thinking_budget=args.thinking_budget,
    ))


if __name__ == "__main__":
    main()
