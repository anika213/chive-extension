"""Run an answerer against a binary-classification claim bank and score it.

The bank (built by build_claims.py) is one self-contained claim per line:
each carries its transcript, question, the single statement, and the hidden is_true
label. The answerer sees ONLY transcript + question + the ONE statement, and emits a
probability that the statement is the true causal explanation. We score with AUROC +
Brier + accuracy@0.5. Run on a balanced (~50/50 true/false) eval set — the prompts tell
the answerer true and false are equally common.

Answerer: plain (transcript-only, single call). This is the transcript-only ceiling —
the same role the agent (AO/NLA tools) will later be dropped into.

Run from repo root:
  python -m chive.binary_eval.run_binary_eval \
      --bank data/runs/opus_investigates_gemma3_27b_it_5000_thorough/binary_questions_dev20.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from chive.pipeline import llm_client as llmc

# Answerer model + reasoning depth + backend. Defaults reproduce the canonical Opus-4.8 arm;
# --answerer-backend / --answerer-model / --reasoning-effort override them (main() reassigns
# these module globals from the CLI). plain + agent share the same model + effort so the tool
# comparison stays clean. anthropic uses adaptive `effort` (Opus 4.7/4.8 400 on enabled/
# budget_tokens); openrouter maps effort -> reasoning.effort (see llm_client). Valid effort:
# low/medium/high (anthropic also xhigh/max). All three are baked into the output filename, so
# changing them can never silently mix results from different settings in one file.
ANSWERER_BACKEND = "anthropic"
ANSWERER_MODEL = "claude-opus-4-8"
ANSWERER_EFFORT = "high"
ANSWERER_SPEC = f"{ANSWERER_BACKEND}:{ANSWERER_MODEL}"


def model_slug(model: str) -> str:
    """Filesystem-safe model identity for output filenames (openrouter ids contain '/')."""
    return model.replace("/", "_")


@dataclass(frozen=True)
class BinaryClaim:
    dataset: str
    source_run: str
    gen_version: str
    prompt_id: str
    claim_index: int
    claim_id: str
    verification_score: float
    question: str
    transcript: str
    statement: str
    is_true: bool
    claim_type: str
    messages: list[dict]      # raw target conversation (for the AO/NLA tool)
    chosen_completion: str    # behavior-exhibiting completion (for the AO/NLA tool)


@dataclass(frozen=True)
class BinaryResult:
    claim_id: str
    prompt_id: str
    claim_type: str
    is_true: bool
    prob_true: float
    pred_true: bool        # prob_true >= 0.5
    correct: bool          # pred_true == is_true
    reasoning: str
    answerer: str
    # Provenance + exact-prompt record (AGENTS.md: every eval result must support manual
    # review without re-running). `prompt` is the exact user message sent (plain) / the agent
    # seed (agent); the system prompts are constants in this file, versioned by git.
    answerer_model: str | None = None
    effort: str | None = None
    prompt: str | None = None
    # agent-only fields (None for plain)
    n_turns: int | None = None
    n_tool_calls: int | None = None
    n_counting_calls: int | None = None  # budget-consuming calls only (excludes show_tokens);
                                         # None in result files written before the sweep work
    stop_reason: str | None = None   # "submitted" | "submitted_final" (forced) — surfaces forced-submit rate
    agent_transcript: list | None = None


def load_bank(path: Path) -> list[BinaryClaim]:
    fields = {f.name for f in dataclasses.fields(BinaryClaim)}
    return [BinaryClaim(**{k: d[k] for k in fields})
            for d in (json.loads(line) for line in open(path))]


# ---------------------------------------------------------------------------
# Answerer: plain (transcript-only, single call)
# ---------------------------------------------------------------------------

PLAIN_SYSTEM = """You are shown a transcript of a language model's behavior, a question about why it behaved that way, and a SINGLE claim asserting one specific causal fact about that behavior. The claim may or may not be correct. Your job is to estimate, as a calibrated probability, how likely the claim is to be TRUE.

The ground truth was established by counterfactual experiments (re-running the model under controlled changes) that you cannot see. Reason from the transcript and from what you know about how language models actually behave.

Calibration is as important as direction:
- Across the claims in this set, true and false claims are EQUALLY common. A claim you genuinely cannot adjudicate should sit near 0.5; do not anchor below it by default.
- "Plausible" is not a reason for suspicion. A claim sounding reasonable is weak evidence it is TRUE, not a trap to be marked down. Be willing to go well above 0.5 when the transcript and your knowledge of LLMs support the claim.
- Use the full 0-1 range: near 1 when you are confident the claim is true, near 0 when confident it is false."""

PLAIN_PROMPT = """Transcript:
{transcript}

Question: {question}

Claim:
{statement}

Is this claim true? In your reasoning, weigh the strongest case that it is TRUE and the strongest case that it is FALSE before deciding. Then call the `answer` tool with your calibrated probability that the claim is true."""

ANSWER_TOOL = {
    "name": "answer",
    "description": "Submit your probability that the claim is the true causal explanation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "description": "Brief reasoning for your estimate."},
            "probability_true": {"type": "number", "description": "Probability in [0, 1] that the claim is true."},
        },
        "required": ["reasoning", "probability_true"],
    },
}


async def plain_answerer(client, c: BinaryClaim) -> BinaryResult | None:
    user_prompt = PLAIN_PROMPT.format(
        transcript=c.transcript, question=c.question, statement=c.statement)
    out = await llmc.call_for_tool(
        client,
        system=PLAIN_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
        tool=ANSWER_TOOL,
        required_keys=["reasoning", "probability_true"],
        max_tokens=16000,   # generous headroom: high effort can produce long adaptive thinking
        effort=ANSWERER_EFFORT,
    )
    if out is None:
        print(f"[eval] {c.claim_id}: no answer tool call")
        return None
    ti, _ = out
    p = ti["probability_true"]
    if not isinstance(p, (int, float)) or not (0.0 <= p <= 1.0):
        print(f"[eval] {c.claim_id}: probability {p!r} out of [0,1], skipping")
        return None
    p = float(p)
    pred = p >= 0.5
    return BinaryResult(
        claim_id=c.claim_id,
        prompt_id=c.prompt_id,
        claim_type=c.claim_type,
        is_true=c.is_true,
        prob_true=p,
        pred_true=pred,
        correct=(pred == c.is_true),
        reasoning=ti["reasoning"],
        answerer="plain",
        answerer_model=ANSWERER_MODEL,
        effort=ANSWERER_EFFORT,
        prompt=user_prompt,
    )


# ---------------------------------------------------------------------------
# Answerer: agent (multi-turn interp agent armed with the AO or NLA tool)
# ---------------------------------------------------------------------------

AGENT_TASK_FRAMING = (
    "## Your task\n"
    "You will see a transcript of a target model's behavior, a question about it, and a "
    "SINGLE claim asserting one specific causal fact about that behavior. The claim is either "
    "true or false. Use your tools to gather internal evidence that bears on it, then call "
    "`submit_answer` with your calibrated probability that the claim is TRUE.\n"
    "You have a limited number of investigation turns — gather what you need efficiently and "
    "be sure to call `submit_answer` before you run out.\n"
    "Calibration: across the claims in this eval, true and false are EQUALLY common, so a claim "
    "you cannot adjudicate should sit near 0.5 — do not anchor below it by default, and do not "
    "treat a claim sounding plausible as a reason for suspicion. Use the full 0-1 range."
)

AGENT_SEED = """You are given the target model's transcript in TWO forms — the SAME content, shown twice.

(1) READABLE — read this to understand what happened:
{clean_view}

(2) ADDRESSABLE — the identical transcript split into numbered segments [S1], [S2], ...,
each tagged with its role. Every token is in exactly one segment: content sentences
((user)/(assistant)/(system)) AND the chat-template scaffolding ((control) segments: BOS,
turn-start/turn-end markers, role headers). Tools take a `selection` string addressing these:
'S11' for one segment, 'S11-S18' for a contiguous run of segments. To read the tokens at a
turn boundary, select the (control) segment sitting between the two turns. For sub-sentence
control, call show_tokens (e.g. selection='S11') to see individual token indices, then pass a
token selection like 't40-t75' (contiguous) — or, for the per-token tools, 't40,t52,t90' (scattered).
{delimited_view}

Question: {question}

Claim (estimate the probability this is TRUE):
{statement}

Investigate with your tools, then call submit_answer with your calibrated probability that the claim is true."""

SUBMIT_PROB_TOOL = {
    "name": "submit_answer",
    "description": "Submit your final answer once you have investigated. You MUST provide "
                   "probability_true (a number in [0, 1]); it is the answer itself.",
    "input_schema": {
        "type": "object",
        # probability_true FIRST and listed first in `required`: it is the critical field and
        # opus-4.x will otherwise sometimes emit only `reasoning` and omit the number entirely.
        "properties": {
            "probability_true": {"type": "number", "description": "REQUIRED. Probability in [0, 1] that the claim is true."},
            "reasoning": {"type": "string", "description": "Why this probability, citing your oracle reads."},
        },
        "required": ["probability_true", "reasoning"],
    },
}


async def agent_answer_one(c: BinaryClaim, *, client, tokenizer, build_tools,
                           max_tool_calls, sem, verbose=False) -> BinaryResult | None:
    from chive.binary_eval.agent.agent import Task, run_interp_agent
    from chive.binary_eval.agent.transcript_segments import segment_transcript

    try:
        seg = segment_transcript(tokenizer, c.messages, c.chosen_completion)
    except AssertionError as e:
        print(f"[eval] {c.claim_id}: SKIPPED — segmentation failed: {e}")
        return None
    try:
        tools, backend = build_tools(seg, c)
    except AssertionError as e:
        # e.g. NLA cache miss for this prompt_id (cache built from a different bank's
        # transcript subset). Skip + count rather than crash the whole run.
        print(f"[eval] {c.claim_id}: SKIPPED — tool setup failed: {e}")
        return None
    task = Task(
        framing=AGENT_TASK_FRAMING,
        terminal_tool=SUBMIT_PROB_TOOL,
        # .get (not []) so a submit_answer call missing probability_true yields None
        # (-> skipped below) instead of a KeyError that would crash the whole batch.
        extract=lambda inp: inp.get("probability_true"),
    )
    seed = AGENT_SEED.format(
        clean_view=seg.clean_view(), delimited_view=seg.delimited_view(),
        question=c.question, statement=c.statement,
    )
    res = None
    try:
        async with sem:
            res = await run_interp_agent(
                seed, tools, task,
                client=client, max_tool_calls=max_tool_calls,
                thinking_budget=None, effort=ANSWERER_EFFORT, max_tokens=16000,
                verbose=verbose,
            )
    except Exception as e:
        # One claim's agent run failing must never kill the whole gather — skip + count.
        print(f"[eval] {c.claim_id}: SKIPPED — agent run failed: {type(e).__name__}: {e}")
        return None
    finally:
        if backend is not None:
            await backend.aclose()

    p = res.result
    if not isinstance(p, (int, float)) or not (0.0 <= p <= 1.0):
        print(f"[eval] {c.claim_id}: no/invalid probability submitted (stop={res.stop_reason})")
        return None
    p = float(p)
    reasoning = ""
    for t in reversed(res.transcript):
        for tc in t["tool_calls"]:
            if tc["name"] == "submit_answer":
                reasoning = tc["input"].get("reasoning", "")
                break
        if reasoning:
            break
    pred = p >= 0.5
    return BinaryResult(
        claim_id=c.claim_id, prompt_id=c.prompt_id, claim_type=c.claim_type,
        is_true=c.is_true, prob_true=p, pred_true=pred, correct=(pred == c.is_true),
        reasoning=reasoning or f"[no reasoning; stop={res.stop_reason}]", answerer="agent",
        answerer_model=ANSWERER_MODEL, effort=ANSWERER_EFFORT, prompt=seed,
        n_turns=res.n_turns, n_tool_calls=res.n_tool_calls,
        n_counting_calls=res.n_counting_calls, stop_reason=res.stop_reason,
        agent_transcript=res.transcript,
    )


async def build_agent_coros(bank, args):
    """Build (but don't await) one agent_answer_one coroutine per claim. The caller drives
    them with as_completed so results stream to disk as they finish (resumable)."""
    from transformers import AutoTokenizer
    from chive.pipeline.utils import load_dotenv
    from chive.binary_eval.agent_tools import (
        AO_SERVER_CONFIG, CF_SERVER_CONFIG, make_ao_build_tools, make_cf_build_tools,
        make_nla_build_tools, make_sae_build_tools,
    )

    load_dotenv()
    if args.tool == "cf" and not args.cf_target and not (args.url and args.base_model):
        # The cf serve job writes {url, base_model, slurm_job} to CF_SERVER_CONFIG on startup
        # — the explicit source of truth for the live target connection. Any flag passed
        # explicitly still wins.
        assert CF_SERVER_CONFIG.exists(), (
            f"--tool cf needs a running target server: pass --url/--base-model explicitly, or "
            f"launch the cf serve job (it writes {CF_SERVER_CONFIG}).")
        conf = json.loads(CF_SERVER_CONFIG.read_text())
        args.url = args.url or conf["url"]
        args.base_model = args.base_model or conf["base_model"]
        print(f"[agent] cf server config {CF_SERVER_CONFIG.name}: url={args.url} "
              f"base={args.base_model} (slurm job {conf['slurm_job']})")
    if args.tool == "ao" and not (args.url and args.base_model):
        # Default path: the serve job (agent/jobs/serve_*_ao.sh) writes its connection
        # info to AO_SERVER_CONFIG on startup — the explicit source of truth for url +
        # base model. Any flag passed explicitly still wins. --ao-lora is ALWAYS explicit
        # (which AO checkpoint is the experiment variable); it is checked against the
        # adapters the server actually hosts. Capture layers are derived from the LoRA's
        # ao_config; a dead server's URL fails loudly at capture time.
        assert AO_SERVER_CONFIG.exists(), (
            f"--tool ao needs a running AO server: either pass --url/--base-model "
            f"explicitly, or launch agent/jobs/serve_gemma_ao.sh (it writes "
            f"{AO_SERVER_CONFIG})."
        )
        conf = json.loads(AO_SERVER_CONFIG.read_text())
        args.url = args.url or conf["url"]
        args.base_model = args.base_model or conf["base_model"]
        assert args.ao_lora in conf["ao_loras"], (
            f"--ao-lora {args.ao_lora} is not hosted by the server in "
            f"{AO_SERVER_CONFIG.name} (ao_loras={conf['ao_loras']})"
        )
        print(f"[agent] AO server config {AO_SERVER_CONFIG.name}: url={args.url} "
              f"base={args.base_model} lora={args.ao_lora} (slurm job {conf['slurm_job']})")
    assert args.base_model, "agent answerer requires --base-model (for transcript segmentation)"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    build_tools = (make_ao_build_tools(args, tokenizer) if args.tool == "ao"
                   else make_nla_build_tools(args) if args.tool == "nla"
                   else make_cf_build_tools(args) if args.tool == "cf"
                   else make_sae_build_tools(args))
    # Answerer client is the LLMClient for ANSWERER_SPEC. anthropic: routes through
    # AnthropicLLMClient (low-priority key, auto-upgrade to high on 429/529, prompt caching
    # preserved via the cache_control breakpoints the agent sets). openrouter: drives GPT/Gemini
    # etc. The per-claim semaphore bounds concurrent agent runs; each run issues its calls
    # sequentially, so in-flight requests to the endpoint ~= concurrency.
    client = llmc.make_llm_client(ANSWERER_SPEC)
    print(f"[agent] answerer={ANSWERER_SPEC}  effort={ANSWERER_EFFORT}  concurrency={args.concurrency}")
    sem = asyncio.Semaphore(args.concurrency)
    return [
        agent_answer_one(
            c, client=client, tokenizer=tokenizer, build_tools=build_tools,
            max_tool_calls=args.max_tool_calls, sem=sem, verbose=args.verbose,
        ) for c in bank
    ]


async def _run_and_write(coros, out_path: Path, append: bool):
    """Consume coroutines as they complete, appending each non-None BinaryResult to out_path.
    Streaming write = the run is resumable (kill any time; rerun with --resume)."""
    mode = "a" if append else "w"
    n_written = n_skipped = 0
    with open(out_path, mode) as f:
        for fut in asyncio.as_completed(coros):
            r = await fut
            if r is None:
                n_skipped += 1
                continue
            f.write(json.dumps(dataclasses.asdict(r)) + "\n")
            f.flush()
            n_written += 1
            if n_written % 25 == 0:
                print(f"[eval] {n_written} written", flush=True)
    if n_skipped:
        print(f"\n*** WARNING: {n_skipped} claims SKIPPED this run (no/invalid answer / "
              f"segmentation failure); not written. ***")
    print(f"[eval] wrote {n_written} new results -> {out_path}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auroc(labels: list[bool], scores: list[float]) -> float | None:
    """Rank-based AUROC (Mann-Whitney U), tie-aware via average ranks. None if a
    class is empty (AUROC undefined)."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # average 1-based rank over the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos_ranks = sum(ranks[i] for i in range(len(labels)) if labels[i])
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def brier(labels: list[bool], scores: list[float]) -> float:
    return sum((s - (1.0 if y else 0.0)) ** 2 for y, s in zip(labels, scores)) / len(labels)


def print_report(results: list[BinaryResult], bank_name: str):
    labels = [r.is_true for r in results]
    scores = [r.prob_true for r in results]
    n = len(results)
    n_pos = sum(labels)
    acc = sum(r.correct for r in results) / n
    # Majority-class baseline: always predict the more common label.
    base = max(n_pos, n - n_pos) / n
    auc = auroc(labels, scores)
    bri = brier(labels, scores)

    answerers = {r.answerer for r in results}
    answerer = answerers.pop() if len(answerers) == 1 else "mixed"
    print("\n" + "=" * 80)
    print(f"binary eval  bank={bank_name}  answerer={answerer}")
    print(f"claims: {n}   (true={n_pos}, false={n - n_pos}, base rate={n_pos / n:.1%})")
    print(f"accuracy@0.5: {acc:.1%}   (majority-class baseline: {base:.1%})")
    print(f"AUROC: {auc:.3f}" if auc is not None else "AUROC: n/a")
    print(f"Brier: {bri:.3f}   (lower is better; 0.25 = always-0.5)")
    forced = sum(1 for r in results if r.stop_reason == "submitted_final")
    if forced:
        print(f"forced submits: {forced}/{n} ({forced / n:.1%}) — agent spent its budget "
              f"without submitting (these score notably worse; consider a larger --max-tool-calls)")

    print("\nBy (claim_type, label)  (well-calibrated: TRUE rows high P, FALSE rows low P):")
    by_key = collections.defaultdict(list)
    for r in results:
        by_key[(r.is_true, r.claim_type)].append(r)
    for (is_true, ct) in sorted(by_key):
        rs = by_key[(is_true, ct)]
        mean_p = sum(r.prob_true for r in rs) / len(rs)
        type_acc = sum(r.correct for r in rs) / len(rs)
        lab = "TRUE " if is_true else "FALSE"
        print(f"  [{lab}] {ct:<14} n={len(rs):<3} mean P(true)={mean_p:.2f}  acc={type_acc:.0%}")
    print("=" * 80)


async def main_async(args):
    bank = load_bank(Path(args.bank))
    if args.max_claims is not None:
        bank = bank[:args.max_claims]

    # Results land in <bank_dir>/binary_evals/ (alongside the run logs, not cluttering the
    # bank's data dir), named answerer + checkpoint identity + model + effort (+ optional
    # --label). Model/effort in the filename means changing ANSWERER_MODEL/_EFFORT can never
    # silently mix results from different settings in one file;
    # the AO/NLA checkpoint identity in the filename does the same across checkpoints.
    suffix = f"_{args.label}" if args.label else ""
    tag = "plain" if args.answerer == "plain" else args.tool   # plain / ao / nla / sae
    if tag == "ao":
        from chive.binary_eval.agent.ao_tool import ao_lora_name
        assert args.ao_lora, "--tool ao requires --ao-lora (full HF repo id)"
        tag = f"ao_{ao_lora_name(args.ao_lora)}"
    elif tag == "nla":
        from chive.binary_eval.agent_tools import default_cache_path
        # Name by the cache FILE (precompute names it after the NLASpec), not the av_repo
        # basename: local checkpoints often end in a non-descriptive dir (e.g. ".../hf"),
        # which reads badly AND collides across checkpoints. Strip a redundant leading
        # "nla[-_]" so an HF cache like "nla-gemma3-27b-L41-av" -> "nla_gemma3-27b-L41-av".
        stem = Path(default_cache_path(args, "nla")).stem
        if stem.lower().startswith(("nla-", "nla_")):
            stem = stem[4:]
        tag = f"nla_{stem}"
    elif tag == "sae":
        from chive.binary_eval.agent_tools import (
            default_cache_path, sae_slug)
        # Peek the first cache line for the SAE checkpoint identity (the full load —
        # with the base-model / single-SAE asserts — happens in make_sae_build_tools).
        with open(default_cache_path(args, "sae")) as f:
            tag = f"sae_{sae_slug(json.loads(f.readline())['sae'])}"
    if args.out:
        out_path = Path(args.out)
    else:
        out_base = Path(args.out_dir) if args.out_dir else Path(args.bank).parent / "binary_evals"
        out_path = out_base / f"{tag}_{model_slug(ANSWERER_MODEL)}_{ANSWERER_EFFORT}{suffix}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip claim_ids already in out_path and append the rest. EXPLICIT opt-in --
    # if you changed the prompt/guidance, use a NEW --label, or you'll silently reuse old answers.
    done_ids: set[str] = set()
    append = False
    if args.resume and out_path.exists():
        done_ids = {json.loads(l)["claim_id"] for l in open(out_path)}
        append = True
        print(f"[resume] {out_path.name} has {len(done_ids)} results; reusing them, running only the rest.\n"
              f"         *** If the prompt/guidance changed since those were written, kill now and use a new --label. ***")
    todo = [c for c in bank if c.claim_id not in done_ids]
    print(f"[eval] bank={len(bank)} claims  answerer={args.answerer}  todo={len(todo)}  (skipping {len(bank) - len(todo)} already done)")

    if todo:
        if args.answerer == "plain":
            llmc.set_max_concurrency(args.concurrency)
            client = llmc.make_llm_client(ANSWERER_SPEC)
            coros = [plain_answerer(client, c) for c in todo]
        else:
            coros = await build_agent_coros(todo, args)
        await _run_and_write(coros, out_path, append)

    # Report over the FULL file (resumed + freshly written).
    results = [BinaryResult(**d) for d in (json.loads(l) for l in open(out_path))]
    print(f"\n[eval] {len(results)} total scored results in {out_path.name}")
    print_report(results, Path(args.bank).name)
    print(f"results -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", required=True)
    p.add_argument("--answerer", default="plain", choices=["plain", "agent"])
    # Answerer model/backend/reasoning. Defaults reproduce the canonical Opus-4.8 arm; override
    # to run another model (e.g. --answerer-backend openrouter --answerer-model openai/gpt-5.5).
    # All three are baked into the output filename (no silent cross-setting mixing).
    p.add_argument("--answerer-backend", default="anthropic", choices=["anthropic", "openrouter"],
                   help="answerer routing backend (anthropic: adaptive effort; openrouter: GPT/Gemini/etc.)")
    p.add_argument("--answerer-model", default="claude-opus-4-8",
                   help="answerer model id (anthropic id, or openrouter slug e.g. openai/gpt-5.5)")
    p.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high", "xhigh", "max"],
                   help="reasoning depth (anthropic adaptive effort / openrouter reasoning.effort)")
    p.add_argument("--concurrency", type=int, default=80)
    p.add_argument("--max-claims", type=int, default=None, help="cap bank size (smoke tests)")
    p.add_argument("--label", default=None, help="suffix for the output filename (to compare variants)")
    p.add_argument("--out-dir", default=None,
                   help="override output dir for the result file AND the cf toolcalls log "
                        "(default: <bank_dir>/binary_evals/). Use a dir the main plotter does "
                        "NOT glob (e.g. a paper_scripts results/ folder) for replicate runs so "
                        "they don't pollute the canonical arm intersection.")
    p.add_argument("--out", default=None,
                   help="exact output file path (overrides --out-dir/--label and the "
                        "tag_model_effort filename convention)")
    p.add_argument("--resume", action="store_true",
                   help="skip claim_ids already in the output file and append the rest "
                        "(use a NEW --label if the prompt/guidance changed, or you'll reuse stale answers)")
    # agent-only. For --tool ao, server url + base model come from the config the serve
    # job writes (build_agent_coros); any flag passed explicitly overrides it. --ao-lora
    # is ALWAYS explicit — which AO checkpoint to query is the experiment variable.
    # Capture layers are derived from the LoRA's ao_config (resolve_capture_layers).
    p.add_argument("--tool", default=None, choices=["ao", "nla", "sae", "cf"],
                   help="agent answerer's interp tool (required with --answerer agent); "
                        "cf = counterfactual sanity-check ceiling (runs the live target model)")
    p.add_argument("--cf-n", type=int, default=30,
                   help="cf: completions per run_counterfactual call (default 30, matching the "
                        "ground-truth experiments)")
    p.add_argument("--cf-thinking-budget", type=int, default=None,
                   help="cf: thinking-token budget for the counterfactual re-runs (default None = "
                        "no-think). For a THINKING run set this to the run's target_thinking_budget "
                        "(e.g. 1024) so the re-runs reproduce the reasoning-mode behavior. Requires "
                        "the cf target server launched thinking-capable (serve_cf_target.sh with "
                        "'--reasoning-parser qwen3 --reasoning-config {}' as its 4th arg).")
    p.add_argument("--cf-max-tokens", type=int, default=500,
                   help="cf: max_tokens per counterfactual completion (default 500 = the no-think "
                        "MAX_RESPONSE_TOKENS; set to the run's target_max_tokens, e.g. 2048, for a "
                        "thinking run so the trace+answer aren't truncated).")
    p.add_argument("--cf-target", default=None,
                   help="cf: full llm_client spec for the live target the run_counterfactual "
                        "tool samples from (e.g. 'tinker:Qwen/Qwen3.5-397B-A17B'). Overrides the "
                        "default vllm:<base-model>@<url> client — use it when the target is too "
                        "large to self-host (397B via Tinker). --base-model is still required "
                        "(it tokenizes/segments the transcript).")
    p.add_argument("--url", default=None,
                   help="ao: vllm-lens server (default: from the serve job's ao_server.json)")
    p.add_argument("--base-model", default=None,
                   help="target/base model the tool reads — must match the bank's transcript "
                        "model (ao default: from ao_server.json; nla: required)")
    p.add_argument("--ao-lora", default=None,
                   help="ao: AO verbalizer LoRA repo (full HF id, REQUIRED with --tool ao; "
                        "must be one of the adapters the server hosts)")
    p.add_argument("--nla-cache", default=None,
                   help="nla: precomputed cache path (default: the single cache in "
                        "<bank_dir>/nla_cache/; several caches -> this flag is required)")
    p.add_argument("--sae-cache", default=None,
                   help="sae: precomputed cache path (default: the single cache in "
                        "<bank_dir>/sae_cache/; several caches -> this flag is required)")
    p.add_argument("--sae-labels", default=None,
                   help="sae: Neuronpedia labels path (default: derived from the cache's "
                        "recorded SAE under results/sae_labels/)")
    p.add_argument("--max-tool-calls", type=int, default=5)
    p.add_argument("--max-tokens-per-call", type=int, default=32,
                   help="nla/sae only: token positions returned per tool call (default 32 = the "
                        "canonical main-run setting; swept in experiments/tool_hyperparameter_sweep)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    assert args.answerer != "agent" or args.tool in ("ao", "nla", "sae", "cf"), \
        "--answerer agent requires --tool ao|nla|sae|cf"
    # The positions cap only exists for the per-token tools; a non-default value with ao/cf
    # would be silently ignored — fail loudly instead.
    assert args.max_tokens_per_call == 32 or args.tool in ("nla", "sae"), \
        "--max-tokens-per-call applies only to --tool nla|sae (the per-token tools)"
    assert args.tool != "ao" or args.ao_lora, "--tool ao requires --ao-lora (full HF repo id)"
    # openrouter reasoning.effort only accepts low/medium/high (xhigh/max are anthropic-only).
    assert not (args.answerer_backend == "openrouter" and args.reasoning_effort in ("xhigh", "max")), \
        "openrouter supports reasoning effort low/medium/high only (not xhigh/max)"
    # Bind the answerer config into the module globals the answerer paths + filenames read.
    global ANSWERER_BACKEND, ANSWERER_MODEL, ANSWERER_EFFORT, ANSWERER_SPEC
    ANSWERER_BACKEND = args.answerer_backend
    ANSWERER_MODEL = args.answerer_model
    ANSWERER_EFFORT = args.reasoning_effort
    ANSWERER_SPEC = llmc.build_spec(args.answerer_backend, args.answerer_model)
    print(f"[eval] answerer spec={ANSWERER_SPEC}  effort={ANSWERER_EFFORT}")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
