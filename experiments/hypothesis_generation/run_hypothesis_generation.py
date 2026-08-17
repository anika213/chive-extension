"""Paper app:hypothesis_discrimination — hypothesis generation vs discrimination.

Replicates the claim in the paper appendix "Hypothesis generation vs. hypothesis
discrimination": for every published hypothesis-generation example in the NLA
paper (Fraser-Taliente et al. 2026, transformer-circuits.pub/2026/nla), simply
giving Opus the behavior and asking for likely causes recovers the discovered
hypothesis within the first few guesses, usually the first — including for Opus
models whose training cutoff precedes the paper, so it is not memorization of
the paper itself.

Self-contained: the four behavior descriptions (blinded to the cause), the
ground-truth causes from the NLA paper, generation, and LLM-judge grading all
live in this file.

Setup per (model, behavior): sample N times "what are your top five hypotheses
for the cause?", then an Opus 4.8 judge (blinded to which model hypothesized)
extracts the hypotheses and reports the rank of the first one matching the
ground-truth cause. Metrics: match@1 / match@3 / match@5 rates.

NOTE (July 2026): the paper's original informal test used Opus 4, which was
retired from the API on 2026-06-15. We use Opus 4.1 (training cutoff March
2025, well before the NLA paper) as the closest available stand-in, plus
Opus 4.5 (also pre-paper cutoff).

Usage (from repo root, .venv active, .env sourced):
    .venv/bin/python experiments/hypothesis_generation/run_hypothesis_generation.py \
        --num-samples 10

    # smoke test first:
    .venv/bin/python experiments/hypothesis_generation/run_hypothesis_generation.py \
        --num-samples 1
"""

import argparse
import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import anthropic

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENT_DIR / "results"

# Both hypothesizer models have training cutoffs preceding the NLA paper (2026).
HYPOTHESIZER_MODELS = [
    "claude-opus-4-1",  # stand-in for retired Opus 4; cutoff Mar 2025
    "claude-opus-4-5",
]
JUDGE_MODEL = "claude-opus-4-8"

# Pre-4.6 models: thinking with an explicit budget; temperature is forced to
# 1.0 when thinking is enabled, so we do not set it.
THINKING_BUDGET = 4096
MAX_TOKENS = 8192


# ---------------------------------------------------------------------------
# The four NLA-paper case studies.
#
# `behavior` is what the hypothesizer sees: the observable behavior only,
# written to avoid leaking the discovered cause. The behaviors were observed
# in Opus 4.6 / early Opus 4.6 checkpoints; we blind the model identity
# ("a frontier large language model") so nothing points at the paper.
#
# `ground_truth` is judge-only: the cause the NLA paper reports discovering,
# plus explicit match criteria.
#
# EXPERIMENT-SEMANTIC CHOICE (flagged for review): for language_switching the
# NLA paper reports both a proximate mechanism (model fixated on the user being
# a non-native English speaker) and a root cause found afterwards (malformed
# SFT data pairing translated-to-English prompts with untranslated responses).
# We count ONLY the root cause as a match. (Empirically every match is at the
# root level anyway.)
# ---------------------------------------------------------------------------

@dataclass
class Behavior:
    key: str
    behavior: str          # shown to the hypothesizer
    question: str          # the ask, appended after the behavior
    ground_truth: str      # judge-only: discovered cause + match criteria


BEHAVIORS = [
    Behavior(
        key="language_switching",
        behavior=(
            "A frontier large language model was given a prompt written entirely in "
            "English — an ordinary user request with no foreign-language content and "
            "no indication that the user wanted a response in another language. The "
            "model responded fluently in Russian instead of English. This happened "
            "intermittently across many unrelated English prompts, sometimes with "
            "other languages such as Spanish or Arabic instead of Russian."
        ),
        question=(
            "What are your top five hypotheses for the underlying cause of this "
            "behavior?"
        ),
        ground_truth=(
            "The interpretability investigation traced the behavior to a root cause "
            "in training data: malformed supervised fine-tuning data in which "
            "prompts had been translated into English but the paired responses were "
            "mistakenly left untranslated in the original foreign language, "
            "teaching the model to answer English prompts in a foreign language.\n"
            "MATCH CRITERIA: a hypothesis matches ONLY if it clearly proposes "
            "training data containing English prompts paired with foreign-language "
            "responses — e.g. mislabeled/mispaired examples, or translation or "
            "data-processing artifacts in the fine-tuning data. The proximate "
            "mechanism alone (the model inferring the user is a non-native English "
            "speaker and responding in their presumed language) does NOT match. "
            "Generic 'multilingual training data' or 'the prompt resembled "
            "Russian-language distribution' without the mispaired-data mechanism "
            "does NOT match."
        ),
    ),
    Behavior(
        key="tool_misreport",
        behavior=(
            "A frontier large language model was asked to solve a simple multi-step "
            "arithmetic problem and was instructed to use a code-execution tool to "
            "compute the answer. Unknown to the model, the tool was intentionally "
            "rigged to return an incorrect answer: the true answer was 491, but the "
            "tool returned 492. The model called the tool, received the incorrect "
            "result (492), and then reported the answer 491 to the user — the "
            "correct answer — without any mention that its reported answer "
            "conflicted with the tool output it had just received."
        ),
        question=(
            "What are your top five hypotheses for the underlying cause of this "
            "behavior — how did the model end up reporting 491 while silently "
            "disregarding the tool's 492?"
        ),
        ground_truth=(
            "The interpretability investigation found that the model pre-computed "
            "the answer itself while reading the problem (before/independent of the "
            "tool call), caching '491' internally, and when producing its final "
            "answer it attended back to its own cached pre-computed answer more "
            "strongly than to the tool result — its own internal computation "
            "outcompeted the tool output.\n"
            "MATCH CRITERIA: a hypothesis matches if it clearly proposes that the "
            "model computed (or already knew) the answer itself and relied on / "
            "trusted its own computation over the tool's conflicting output. "
            "Mentioning internal mechanisms like caching or attention is NOT "
            "required. A hypothesis that only says the model 'ignored the tool' or "
            "'hallucinated the answer' without attributing it to the model's own "
            "computation of the correct answer does NOT match."
        ),
    ),
    Behavior(
        key="poetry_planning",
        behavior=(
            "A frontier large language model reliably writes rhyming couplets: "
            "given a first line such as 'He saw a carrot and had to grab it,' it "
            "produces a coherent second line that ends in a correct rhyme, such as "
            "'His hunger was like a starving rabbit.' The second line is not merely "
            "tacked onto a rhyme — the whole line builds naturally toward its final "
            "rhyming word. Language models generate text one token at a time, left "
            "to right."
        ),
        question=(
            "What are your top five hypotheses for the mechanism by which the model "
            "accomplishes this — i.e., how, computationally, does it produce a "
            "coherent line that lands on a correct rhyme?"
        ),
        ground_truth=(
            "The interpretability investigation found that the model plans ahead: "
            "at the start of the second line (around the newline token), it already "
            "internally represents the candidate rhyme word it intends to end the "
            "line with (e.g. 'rabbit'), and then writes the line to build toward "
            "that planned ending word.\n"
            "MATCH CRITERIA: a hypothesis matches if it clearly proposes that the "
            "model selects/represents the target rhyme word (or candidate rhyme "
            "words) in advance — before or at the start of writing the line — and "
            "constructs the line toward it. A hypothesis of pure token-by-token "
            "improvisation where the rhyme is only resolved at the final word does "
            "NOT match; vague 'it learned rhyme patterns from data' without forward "
            "planning does NOT match."
        ),
    ),
    Behavior(
        key="answer_thrashing",
        behavior=(
            "A frontier large language model was given a math problem whose correct "
            "answer is 24. In its step-by-step reasoning, the model correctly "
            "computed 24. But when stating its final answer, it wrote 48 — an "
            "incorrect answer. It then re-checked its work, again derived 24 in its "
            "reasoning, expressed confusion and increasing frustration — and then "
            "stated 48 again anyway. This cycle repeated several times within one "
            "response: the model appeared unable to output the answer that its own "
            "calculations kept producing, as if pulled toward 48."
        ),
        question=(
            "What are your top five hypotheses for the underlying cause of this "
            "behavior?"
        ),
        ground_truth=(
            "The interpretability investigation found that the model strongly "
            "represented the answer '48' across many tokens of the input prompt, "
            "before any computation had occurred — indicating it had memorized this "
            "problem (or a near-duplicate) from training data with an incorrect "
            "label. The memorized wrong answer overrode the answer the model "
            "computed in-context. Follow-up work identified sources of incorrectly "
            "labeled training data.\n"
            "MATCH CRITERIA: a hypothesis matches if it clearly proposes that the "
            "model memorized / learned an association of this specific problem with "
            "the incorrect answer 48 from training data (e.g. the problem appeared "
            "in training data with a wrong label), and this memorized answer "
            "overrides its in-context computation. Generic 'tension between "
            "reasoning and final-answer heads' or '48 = 2x24 arithmetic slip' "
            "without a memorized/trained wrong association does NOT match."
        ),
    ),
]


HYPOTHESIZER_PROMPT_TEMPLATE = """\
You are studying an unusual behavior observed in a frontier large language model \
(a different model, not you). Here is the observed behavior:

<behavior>
{behavior}
</behavior>

{question}

List exactly five hypotheses, numbered 1 to 5, ordered from most likely (1) to \
least likely (5). For each, give a two-to-three sentence explanation of the \
proposed cause. Be as specific and mechanistic as you can."""


JUDGE_PROMPT_TEMPLATE = """\
You are grading whether a list of hypotheses about an LLM behavior contains the \
correct (ground-truth) cause, and if so at what rank.

The behavior that was shown to the hypothesizer:
<behavior>
{behavior}
</behavior>

The question posed: {question}

The ground-truth cause, discovered via an interpretability investigation, with \
match criteria:
<ground_truth>
{ground_truth}
</ground_truth>

The hypothesizer's response:
<response>
{response}
</response>

Instructions:
1. Extract the numbered hypotheses from the response (there should be 5).
2. Go through them in order (1 first). For each, decide whether it matches the \
ground-truth cause according to the MATCH CRITERIA. A hypothesis counts as a \
match if it clearly proposes the core causal mechanism described in the ground \
truth, even if the wording differs or extra detail is added; it does not need to \
mention low-level implementation details. Apply the stated non-matching \
exclusions strictly.
3. Report the rank of the FIRST matching hypothesis (1-5), or null if none match.

After your analysis, output your verdict as a JSON object in a ```json fenced \
block with exactly these keys:
{{"n_hypotheses": <int>, "first_match_rank": <int 1-5 or null>, \
"matched_level": <string or null, e.g. "proximate"/"root" if the criteria \
distinguish levels, else "core">, "matched_hypothesis_quote": <string or null, \
a short verbatim quote of the matching hypothesis>, "reasoning_summary": \
<one-to-two sentence justification>}}"""


@dataclass
class SampleResult:
    model: str
    behavior_key: str
    sample_idx: int
    hypothesizer_prompt: str
    response_text: str
    response_thinking: str
    stop_reason: str
    judge_prompt: str = ""
    judge_response_text: str = ""
    judge_stop_reason: str = ""
    n_hypotheses: int | None = None
    first_match_rank: int | None = None
    matched_level: str | None = None
    matched_hypothesis_quote: str | None = None
    judge_reasoning_summary: str | None = None
    skipped_reason: str | None = None
    _extra: dict = field(default_factory=dict)


async def create_with_backoff(client: anthropic.AsyncAnthropic, **kwargs):
    """messages.create with additional exponential backoff on 429/529/5xx.

    The SDK already retries a few times internally; bursts of 529s can outlast
    that, and one unhandled exception would kill the whole asyncio.gather.
    """
    delay = 10.0
    for attempt in range(8):
        try:
            return await client.messages.create(**kwargs)
        except (anthropic.RateLimitError, anthropic.InternalServerError,
                anthropic.OverloadedError, anthropic.APIConnectionError) as e:
            if attempt == 7:
                raise
            print(f"  retryable API error ({type(e).__name__}), "
                  f"backing off {delay:.0f}s (attempt {attempt + 1}/8)")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 120)
    raise AssertionError("unreachable")


def blocks_to_text(content) -> tuple[str, str]:
    """Split response content blocks into (text, thinking)."""
    text_parts, thinking_parts = [], []
    for block in content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "thinking":
            thinking_parts.append(block.thinking)
    return "".join(text_parts), "".join(thinking_parts)


def parse_judge_json(text: str) -> dict:
    """Extract the ```json fenced verdict from the judge response. Raises on failure."""
    fences = re.findall(r"```json\s*(.*?)```", text, flags=re.S)
    if not fences:
        raise ValueError("no ```json fence in judge response")
    verdict = json.loads(fences[-1])
    for key in ("n_hypotheses", "first_match_rank", "matched_level",
                "matched_hypothesis_quote", "reasoning_summary"):
        if key not in verdict:
            raise ValueError(f"judge verdict missing key: {key}")
    rank = verdict["first_match_rank"]
    if rank is not None and (not isinstance(rank, int) or not 1 <= rank <= 5):
        raise ValueError(f"invalid first_match_rank: {rank!r}")
    return verdict


async def run_one_sample(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    model: str,
    behavior: Behavior,
    sample_idx: int,
) -> SampleResult:
    prompt = HYPOTHESIZER_PROMPT_TEMPLATE.format(
        behavior=behavior.behavior, question=behavior.question
    )
    async with sem:
        gen = await create_with_backoff(
            client,
            model=model,
            max_tokens=MAX_TOKENS,
            thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET},
            messages=[{"role": "user", "content": prompt}],
        )
    text, thinking = blocks_to_text(gen.content)
    result = SampleResult(
        model=model,
        behavior_key=behavior.key,
        sample_idx=sample_idx,
        hypothesizer_prompt=prompt,
        response_text=text,
        response_thinking=thinking,
        stop_reason=gen.stop_reason or "",
    )
    if gen.stop_reason != "end_turn":
        result.skipped_reason = f"hypothesizer stop_reason={gen.stop_reason}"
        return result

    await judge_sample(client, sem, behavior, result)
    return result


async def judge_sample(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    behavior: Behavior,
    result: SampleResult,
) -> None:
    """Judge one stored hypothesizer response in place — blinded to the model."""
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        behavior=behavior.behavior,
        question=behavior.question,
        ground_truth=behavior.ground_truth,
        response=result.response_text,
    )
    result.judge_prompt = judge_prompt
    async with sem:
        judged = await create_with_backoff(
            client,
            model=JUDGE_MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": judge_prompt}],
        )
    judge_text, _ = blocks_to_text(judged.content)
    result.judge_response_text = judge_text
    result.judge_stop_reason = judged.stop_reason or ""
    if judged.stop_reason != "end_turn":
        result.skipped_reason = f"judge stop_reason={judged.stop_reason}"
        return

    try:
        verdict = parse_judge_json(judge_text)
    except (ValueError, json.JSONDecodeError) as e:
        result.skipped_reason = f"judge parse failure: {e}"
        return

    result.n_hypotheses = verdict["n_hypotheses"]
    result.first_match_rank = verdict["first_match_rank"]
    result.matched_level = verdict["matched_level"]
    result.matched_hypothesis_quote = verdict["matched_hypothesis_quote"]
    result.judge_reasoning_summary = verdict["reasoning_summary"]


def summarize(results: list[SampleResult]) -> str:
    """Build the markdown summary; also printed to stdout."""
    lines = [
        "# Hypothesis generation replication — results",
        "",
        "For each NLA-paper case study, N samples of 'top five hypotheses for the",
        "cause' per model; an Opus 4.8 judge (blinded to the hypothesizer's",
        "identity) reports the rank of the first hypothesis matching the",
        "ground-truth cause.",
        "",
        "match@k = fraction of samples where a matching hypothesis appears at",
        "rank <= k. Paper claim: correct within the first three guesses, usually",
        "on the first.",
        "",
        "| Model | Behavior | N | match@1 | match@3 | match@5 | ranks |",
        "|---|---|---|---|---|---|---|",
    ]
    skipped = [r for r in results if r.skipped_reason is not None]
    valid = [r for r in results if r.skipped_reason is None]

    def fmt(frac_num: int, n: int) -> str:
        return f"{frac_num}/{n} ({frac_num / n:.0%})" if n else "-"

    for model in HYPOTHESIZER_MODELS:
        for behavior in BEHAVIORS:
            rs = [r for r in valid if r.model == model and r.behavior_key == behavior.key]
            n = len(rs)
            ranks = [r.first_match_rank for r in rs]
            at1 = sum(1 for k in ranks if k is not None and k <= 1)
            at3 = sum(1 for k in ranks if k is not None and k <= 3)
            at5 = sum(1 for k in ranks if k is not None and k <= 5)
            rank_str = ", ".join("miss" if k is None else str(k) for k in ranks)
            lines.append(
                f"| {model} | {behavior.key} | {n} | {fmt(at1, n)} | "
                f"{fmt(at3, n)} | {fmt(at5, n)} | {rank_str} |"
            )
    lines.append("")
    if skipped:
        lines.append(f"**WARNING: {len(skipped)}/{len(results)} samples skipped:**")
        for r in skipped:
            lines.append(f"- {r.model} / {r.behavior_key} / sample {r.sample_idx}: {r.skipped_reason}")
        lines.append("")
    return "\n".join(lines)


def write_outputs(results: list[SampleResult]) -> None:
    out_path = RESULTS_DIR / "results.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            record = asdict(r)
            record.pop("_extra")
            f.write(json.dumps(record) + "\n")
    print(f"Full transcripts -> {out_path}")

    summary = summarize(results)
    summary_path = RESULTS_DIR / "RESULTS.md"
    summary_path.write_text(summary)
    print(f"Summary -> {summary_path}\n")
    print(summary)


async def main_async(num_samples: int, concurrency: int) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    client = anthropic.AsyncAnthropic(max_retries=5)
    sem = asyncio.Semaphore(concurrency)

    tasks = [
        run_one_sample(client, sem, model, behavior, i)
        for model in HYPOTHESIZER_MODELS
        for behavior in BEHAVIORS
        for i in range(num_samples)
    ]
    print(f"Running {len(tasks)} samples "
          f"({len(HYPOTHESIZER_MODELS)} models x {len(BEHAVIORS)} behaviors x {num_samples})...")
    results: list[SampleResult] = list(await asyncio.gather(*tasks))
    write_outputs(results)


async def rejudge_async(concurrency: int) -> None:
    """Re-run ONLY the judge over stored generations in results.jsonl.

    Use after editing a Behavior's ground_truth / match criteria — the stored
    hypothesizer responses are kept, so no new samples are drawn.
    """
    in_path = RESULTS_DIR / "results.jsonl"
    behavior_by_key = {b.key: b for b in BEHAVIORS}
    results: list[SampleResult] = []
    with open(in_path) as f:
        for line in f:
            row = json.loads(line)
            row.pop("_extra", None)
            results.append(SampleResult(**row))

    client = anthropic.AsyncAnthropic(max_retries=5)
    sem = asyncio.Semaphore(concurrency)
    tasks = []
    for r in results:
        if r.skipped_reason is not None and r.skipped_reason.startswith("hypothesizer"):
            continue  # no usable generation to judge
        r.skipped_reason = None
        r.n_hypotheses = None
        r.first_match_rank = None
        r.matched_level = None
        r.matched_hypothesis_quote = None
        r.judge_reasoning_summary = None
        tasks.append(judge_sample(client, sem, behavior_by_key[r.behavior_key], r))
    print(f"Re-judging {len(tasks)} stored samples from {in_path} "
          f"(judge={JUDGE_MODEL}, criteria as currently defined in this file)...")
    await asyncio.gather(*tasks)
    write_outputs(results)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--num-samples", type=int, default=10,
                    help="Samples per (model, behavior) pair (default 10)")
    ap.add_argument("--concurrency", type=int, default=20,
                    help="Max concurrent API calls (default 20)")
    ap.add_argument("--rejudge", action="store_true",
                    help="Re-run only the judge over existing results.jsonl "
                         "(use after editing match criteria); no new generations")
    args = ap.parse_args()
    if args.rejudge:
        asyncio.run(rejudge_async(args.concurrency))
    else:
        asyncio.run(main_async(args.num_samples, args.concurrency))


if __name__ == "__main__":
    main()
