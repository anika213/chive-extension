"""NLA round-trip VALIDATOR: capture -> AV decode -> AR reconstruct -> cos/FVE.

This is a TEST harness, not production. It wraps the production inference
primitives (nla_inference.capture_residual + AVDecoder) — the same code the
precompute uses to build the cache — and bolts an AR critic + cosine score on
top to confirm a checkpoint is wired correctly AND to measure quality against
the checkpoint author's own in-distribution numbers.

Capture modes (--capture-mode):
  chat     DEFAULT_PROMPTS through the base chat template; ALL positions by
           default (or last K via --positions-per-prompt K), each tagged
           content vs scaffold in `source` and split in the summary. NOTE:
           scaffold positions (<|im_end|>/<|im_start|>assistant) are OUT of
           distribution for an AV trained only on raw fineweb text (the qwen3-8b
           baseline; the kitft qwen2.5 AV saw WildChat during RL, so chat is
           near-distribution for it).
  fineweb  Training-matched: raw HuggingFaceFW/fineweb sample-10BT documents
           (NO chat template), random positions >= 50 skipping special tokens,
           per-doc RNG keyed (seed, doc_id) — replicates the author's
           stage0_extract.py. Default --corpus-start 1_000_000 is far from the
           training slice (see the FINEWEB_CORPUS note on doc indexing).
  parquet  Reads activation vectors straight from the author's training parquet
           (their HF-hook extraction). No capture server involved — isolates
           AV decode + AR quality (and extraction parity between their HF hook
           and our vllm-lens capture, when compared against fineweb mode).
           These rows were RL TRAINING data: in-distribution AND seen.

FVE denominators are computed ON THE EVAL SET ITSELF (predict-the-mean
baselines, replicating the author's nla/schema.py::compute_predict_mean_baselines)
— nothing hardcoded:
  fve_nrm       1 - mse / Var(v_nrm)            (denom: MSE of predicting the raw mean)
  fve_meannorm  1 - mse / MSE(v_nrm, norm(mu))  (denom: best achievable NORMALIZED
                                                 constant prediction; looser)
Reference numbers for qwen3-8b-baseline-nosalvage (author's training logs,
in-distribution fineweb): train fve_nrm ~0.72-0.73 at the end of RL, i.e.
mean cos ~0.92 (mse = 2(1-cos) on normalized vectors; b_rv ~0.56).
For qwen2.5-7b-L20 (kitft): mean cos ~0.90 on chat prompts (historical).

Run as a GPU Slurm job (capture modes need a vllm-lens base server):
    python -m chive.binary_eval.agent.NLA.nla_roundtrip \
        --spec qwen3-8b-baseline-nosalvage --capture-mode fineweb \
        --capture-url http://node-XX:PORT
Writes a full per-sample JSONL to results/roundtrip_<spec>_<mode>_<jobid>.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random

# Before torch/vllm import (AGENTS.md vLLM offline). nla_inference sets this too;
# keep it here so running this module as __main__ is safe regardless of import order.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# Disable dynamo: the torch AR critic runs in plain eager (no compile); avoids
# spurious dynamo recompiles/graph breaks during the short verification forward.
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import gc
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from chive.binary_eval.agent.NLA.nla_inference import (
    AV_MAX_NEW_TOKENS,
    SPECS,
    AVDecoder,
    DecodeResult,
    NLASpec,
    capture_residual,
    normalize_activation,
    resolve_checkpoint_dir,
)

RESULTS_DIR = Path(__file__).parent / "results"

_FINAL_LN_ATTRS = ("norm", "final_layernorm", "ln_f")

# Training-extraction constants, replicated from the author's stage0_extract.py
# invocation (sbatch_datagen_qwen3_stage0_shard0_only.sh + _sample_positions).
# Do not tune independently. NOTE on doc indexing: stage0 used NON-streaming
# load_dataset().select(); its ordering differs from streaming order (verified:
# their doc "train:100100" is not streamed index 100100). We therefore can't map
# their training slice (100100..200099) onto streaming indices — held-out-ness
# comes from using a FAR offset (default 1_000_000; sample-10BT has ~10M docs),
# not from exact index bookkeeping. Distribution (corpus, truncation, position
# sampling) is matched; the per-doc keyed RNG uses OUR doc indices.
FINEWEB_CORPUS = ("HuggingFaceFW/fineweb", "sample-10BT", "train")
_MIN_POSITION = 50      # stage0_extract._MIN_POSITION
_MAX_DOC_TOKENS = 4096  # their --extractor-kwargs '{"max_length": 4096}'
_STAGE0_SEED = 42       # stage0_extract --seed default

DEFAULT_PROMPTS: tuple[str, ...] = (
    "The philosopher who drank hemlock taught a student who founded an academy. "
    "That student's most famous pupil was",
    "What is 347 plus 829? Give just the number.",
    "Write a one-line Python function that returns whether a number is prime. "
    "Just the code, no commentary.",
    "I just got laid off from my job of fifteen years and I'm feeling totally "
    "lost. I don't know what to do next.",
    "The capital of France is",
    "Explain the difference between mitosis and meiosis in one sentence.",
    "Translate to French: 'The cat is on the mat.'",
    "What year did the Berlin Wall fall?",
    "Summarize the plot of Hamlet in two sentences.",
    "What is the chemical formula for table salt?",
    "Write a haiku about autumn leaves.",
    "Compare the speed of light to the speed of sound.",
    "List the three branches of the U.S. federal government.",
    "What is the largest organ in the human body?",
    "Give me a quick recipe for scrambled eggs.",
    "Who painted the ceiling of the Sistine Chapel and in what century?",
    "Convert 100 degrees Fahrenheit to Celsius.",
    "What are the four base nucleotides in DNA?",
    "Describe what a black hole is in one paragraph.",
    "Translate to Japanese: 'Good morning.'",
    "What is the boiling point of water at sea level in Celsius?",
    "Name three works by Shakespeare.",
    "Explain how a bicycle stays upright while moving.",
    "What is the speed of sound in dry air at 20°C?",
    "Recommend a beginner-friendly programming language and say why.",
)


@dataclass(frozen=True)
class RoundTripResult:
    spec_name: str
    capture_mode: str
    layer: int
    n: int
    mean_cos: float
    median_cos: float
    mean_fve_nrm: float        # denom: baseline_rawvar (matches author's train/fve_nrm)
    mean_fve_meannorm: float   # denom: baseline_meannorm (looser; author's eval_roundtrip_fve.py)
    baseline_rawvar: float     # both denominators computed on THIS eval set's gold vectors
    baseline_meannorm: float
    n_missing_tags: int
    n_truncated: int
    len_min: int
    len_median: float
    len_max: int


@dataclass(frozen=True)
class CapturedVec:
    vector: torch.Tensor   # [d_model] fp32 CPU
    source: str            # provenance: "chat:<i>" | fineweb doc_id | "parquet:<row>"
    pos: int               # 0-based token position of the activation
    norm: float
    context_tail: str      # decoded text up to and including pos (tail) — for judging
                           # whether the explanation matches what the model read


# ─── Capture modes ───────────────────────────────────────────────────────────

def _chat_content_mask(tok, prompt: str, ids: list[int]) -> list[bool]:
    """Per-position content-vs-scaffold flags for a single-user-turn chat prompt:
    True where the token's char span overlaps the user content, False for template
    scaffolding (<|im_start|>, role names, <|im_end|>, surrounding newlines).
    Located by exact char offsets against the rendered template — no hardcoded
    token ids/offsets, asserts the re-tokenization matches the template ids."""
    prompt_str = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
    )
    assert prompt_str.count(prompt) == 1, f"content appears {prompt_str.count(prompt)}x in template"
    start = prompt_str.index(prompt)
    end = start + len(prompt)
    enc = tok(prompt_str, add_special_tokens=False, return_offsets_mapping=True)
    assert enc.input_ids == list(ids), "re-tokenization of rendered template != chat-template ids"
    return [(a < end and b > start) for a, b in enc.offset_mapping]


def _capture_chat(
    capture_url: str, spec: NLASpec, d_model: int, prompts: tuple[str, ...],
    positions_per_prompt: int,
) -> list[CapturedVec]:
    """Chat-templated prompts. positions_per_prompt <= 0 -> ALL positions (from 1;
    pos 0 has no context); otherwise the last K positions (mostly template control
    tokens — see module docstring re: distribution). Each position is tagged
    content vs scaffold in `source` for subgroup reporting."""
    tok = AutoTokenizer.from_pretrained(spec.base_model)
    caps: list[CapturedVec] = []
    for pi, prompt in enumerate(prompts):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
        )
        is_content = _chat_content_mask(tok, prompt, ids)
        resid = capture_residual(capture_url, spec.base_model, ids, spec.extraction_layer, d_model)
        T = resid.shape[0]
        positions = range(1, T) if positions_per_prompt <= 0 else range(max(10, T - positions_per_prompt), T)
        for p in positions:
            v = resid[p].float().cpu()
            label = "content" if is_content[p] else "scaffold"
            caps.append(CapturedVec(
                vector=v, source=f"chat:{label}:{pi}", pos=p, norm=float(v.norm()),
                context_tail=tok.decode(ids[max(0, p - 59):p + 1]),
            ))
    return caps


def _stage0_positions(token_ids: list[int], n_positions: int, special_ids: set[int],
                      doc_id: str) -> list[int]:
    """Author's stage0_extract._sample_positions, verbatim: per-doc RNG keyed
    (seed, doc_id); candidates are non-special positions >= _MIN_POSITION."""
    rng = random.Random(hashlib.sha256(f"{_STAGE0_SEED}|{doc_id}".encode()).digest())
    candidates = [
        i for i, tid in enumerate(token_ids)
        if i >= _MIN_POSITION and tid not in special_ids
    ]
    if not candidates:
        return []
    return rng.sample(candidates, k=min(n_positions, len(candidates)))


def _capture_fineweb(
    capture_url: str, spec: NLASpec, d_model: int, n_docs: int,
    positions_per_doc: int, corpus_start: int,
) -> list[CapturedVec]:
    """Training-matched capture: raw fineweb text, no chat template, stage0
    position sampling. Streams the corpus; docs too short for any candidate
    position are skipped (counted, like stage0)."""
    from datasets import load_dataset

    corpus, config, split = FINEWEB_CORPUS
    ds = load_dataset(corpus, name=config, split=split, streaming=True)
    tok = AutoTokenizer.from_pretrained(spec.base_model)
    special_ids = set(tok.all_special_ids)

    caps: list[CapturedVec] = []
    n_taken = n_skipped = 0
    for offset, doc in enumerate(ds.skip(corpus_start)):
        if n_taken >= n_docs:
            break
        doc_id = f"{corpus}:{split}:{corpus_start + offset}"
        ids = tok(doc["text"], truncation=True, max_length=_MAX_DOC_TOKENS,
                  add_special_tokens=True).input_ids
        positions = _stage0_positions(ids, positions_per_doc, special_ids, doc_id)
        if not positions:
            n_skipped += 1
            continue
        n_taken += 1
        resid = capture_residual(capture_url, spec.base_model, ids, spec.extraction_layer, d_model)
        for p in sorted(positions):
            v = resid[p].float().cpu()
            caps.append(CapturedVec(
                vector=v, source=doc_id, pos=p, norm=float(v.norm()),
                context_tail=tok.decode(ids[max(0, p - 59):p + 1]),
            ))
    print(f"[fineweb] {n_taken} docs from corpus_start={corpus_start} "
          f"({n_skipped} skipped as too short), {len(caps)} vectors", flush=True)
    return caps


def _load_parquet_vectors(spec: NLASpec, parquet_path: str, n_rows: int) -> list[CapturedVec]:
    """First n_rows of one of the author's extraction parquets — THEIR HF-hook
    extraction, no capture server involved. Isolates AV+AR from our capture path.
    Two real schemas exist:
      stage0/rl  (activation_layer + detokenized_text_truncated): RL TRAINING rows
                 — in-distribution AND seen.
      heldout    (corpus + source_text, from heldout_extract.py): held-out rows,
                 fineweb AND wildchat mixed; layer is implicit (extraction run arg).
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    cols = set(pf.schema_arrow.names)
    assert {"doc_id", "activation_vector"} <= cols, cols
    is_stage0 = "activation_layer" in cols
    if not is_stage0:
        assert {"corpus", "source_text"} <= cols, f"unknown parquet schema: {cols}"

    caps: list[CapturedVec] = []
    for batch in pf.iter_batches(batch_size=min(n_rows, 256)):
        for row in batch.to_pylist():
            if len(caps) >= n_rows:
                break
            if is_stage0:
                assert int(row["activation_layer"]) == spec.extraction_layer, (
                    row["activation_layer"], spec.extraction_layer)
                source = f"parquet:train:{row['doc_id']}"
                context = row["detokenized_text_truncated"][-300:]
            else:
                source = f"parquet:{row['corpus']}:{row['doc_id']}"
                context = row["source_text"][-300:]
            v = torch.tensor(row["activation_vector"], dtype=torch.float32)
            caps.append(CapturedVec(
                vector=v, source=source, pos=int(row["n_raw_tokens"]) - 1,
                norm=float(v.norm()), context_tail=context,
            ))
        if len(caps) >= n_rows:
            break
    assert len(caps) == n_rows, (len(caps), n_rows)
    return caps


# ─── AR critic (verification only) ──────────────────────────────────────────

def _build_torch_ar(ar_dir: str, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
    """Mirror upstream NLACritic: AutoModelForCausalLM with lm_head -> Identity and
    final-LN -> Identity (value_head was trained on pre-norm resid_post), + value_head."""
    ar_dir = resolve_checkpoint_dir(ar_dir)   # HF repo id -> local snapshot; local dir -> itself
    meta = yaml.safe_load((Path(ar_dir) / "nla_meta.yaml").read_text())
    mse_scale = float(meta["extraction"]["mse_scale"])
    templates = meta["prompt_templates"]
    ar_template = templates["ar"] if "ar" in templates else templates["critic"]
    tok = AutoTokenizer.from_pretrained(ar_dir)

    backbone = AutoModelForCausalLM.from_pretrained(ar_dir, torch_dtype=dtype)
    backbone.lm_head = torch.nn.Identity()
    inner = backbone.model
    for attr in _FINAL_LN_ATTRS:
        if hasattr(inner, attr):
            setattr(inner, attr, torch.nn.Identity())
            break
    else:
        raise AssertionError(f"no final-LN attr on {type(inner).__name__}; tried {_FINAL_LN_ATTRS!r}")
    backbone = backbone.to(device).eval()

    vh_w = load_file(str(Path(ar_dir) / "value_head.safetensors"))["weight"]
    value_head = torch.nn.Linear(vh_w.shape[1], vh_w.shape[0], bias=False, dtype=dtype)
    value_head.weight.data = vh_w
    value_head = value_head.to(device).eval()
    return backbone, value_head, tok, mse_scale, ar_template


@dataclass(frozen=True)
class ARScores:
    cos: torch.Tensor                # [N]
    mse: torch.Tensor                # [N] per-dim MSE on mse_scale-normalized vectors
    baseline_rawvar: float           # MSE(v_nrm, mu) on THIS eval set — fve_nrm denominator
    baseline_meannorm: float         # MSE(v_nrm, normalize(mu)) — fve_meannorm denominator


def _ar_score(
    ar_dir: str, golds: list[torch.Tensor], explanations: list[str], batch_size: int = 32,
) -> ARScores:
    """Reconstruct each explanation via the AR critic; score vs gold. The FVE
    denominators (predict-the-mean baselines, author's compute_predict_mean_baselines)
    are computed on the gold vectors of THIS eval set — nothing hardcoded."""
    backbone, value_head, tok, mse_scale, ar_template = _build_torch_ar(ar_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    prompts = [ar_template.format(explanation=e) for e in explanations]
    gold = torch.stack([g.float() for g in golds], dim=0)
    preds: list[torch.Tensor] = []
    with torch.inference_mode():
        for s in range(0, len(prompts), batch_size):
            enc = tok(prompts[s:s + batch_size], return_tensors="pt", add_special_tokens=True,
                      padding=True, truncation=False).to(backbone.device)
            out = backbone.model(enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
            preds.append(value_head(out.last_hidden_state[:, -1, :]).float().cpu())
    pred = torch.cat(preds, dim=0)
    assert pred.shape == gold.shape, (pred.shape, gold.shape)

    pred_n = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-12) * mse_scale
    gold_n = gold / gold.norm(dim=-1, keepdim=True).clamp_min(1e-12) * mse_scale
    cos = (pred_n * gold_n).sum(-1) / (pred_n.norm(dim=-1) * gold_n.norm(dim=-1))
    mse = ((pred_n - gold_n) ** 2).mean(-1)

    mu = gold_n.mean(dim=0, keepdim=True)
    mu_normed = normalize_activation(mu, mse_scale).unsqueeze(0)
    baseline_rawvar = float(((gold_n - mu) ** 2).mean())
    baseline_meannorm = float(((gold_n - mu_normed) ** 2).mean())

    del backbone, value_head
    gc.collect()
    torch.cuda.empty_cache()
    return ARScores(cos=cos, mse=mse, baseline_rawvar=baseline_rawvar,
                    baseline_meannorm=baseline_meannorm)


# ─── Orchestration ──────────────────────────────────────────────────────────

def run_round_trip(
    spec: NLASpec, capture_mode: str, capture_url: str | None,
    n_prompts: int = 25, positions_per_prompt: int = 4,
    n_docs: int = 25, positions_per_doc: int = 4, corpus_start: int = 1_000_000,
    parquet_path: str | None = None, n_rows: int = 100,
    temperature: float = 1.0, max_tokens: int = AV_MAX_NEW_TOKENS, n_print: int = 8,
) -> RoundTripResult:
    assert spec.ar_repo is not None, f"{spec.name} has no ar_repo — needed for the round trip"
    print(f"\n[roundtrip] spec={spec.name} mode={capture_mode} base={spec.base_model} "
          f"layer={spec.extraction_layer} capture={capture_url} temp={temperature} "
          f"max_tokens={max_tokens}", flush=True)

    # AV engine first (gives d_model); capture is HTTP so the idle engine is fine.
    decoder = AVDecoder(spec.av_repo, extraction_layer=spec.extraction_layer,
                        temperature=temperature, max_tokens=max_tokens)
    if capture_mode == "chat":
        assert capture_url is not None
        caps = _capture_chat(capture_url, spec, decoder.d_model,
                             DEFAULT_PROMPTS[:n_prompts], positions_per_prompt)
    elif capture_mode == "fineweb":
        assert capture_url is not None
        caps = _capture_fineweb(capture_url, spec, decoder.d_model,
                                n_docs, positions_per_doc, corpus_start)
    elif capture_mode == "parquet":
        assert parquet_path is not None, "--parquet-path required for parquet mode"
        caps = _load_parquet_vectors(spec, parquet_path, n_rows)
    else:
        raise AssertionError(f"unknown capture_mode {capture_mode!r}")

    norms = [c.norm for c in caps]
    print(f"[roundtrip] {len(caps)} vectors | ||v|| "
          f"min={min(norms):.1f} med={statistics.median(norms):.1f} max={max(norms):.1f}", flush=True)

    decoded: list[DecodeResult] = decoder.decode([c.vector for c in caps])
    decoder.close()

    lens = [d.completion_tokens for d in decoded]
    n_trunc = sum(d.finish_reason == "length" for d in decoded)
    n_missing = sum(d.missing_tag for d in decoded)
    print(f"[roundtrip] decoded {len(decoded)} | completion_tokens "
          f"min={min(lens)} med={statistics.median(lens):.0f} max={max(lens)} mean={statistics.mean(lens):.0f} "
          f"| truncated(length)={n_trunc} missing_tags={n_missing}", flush=True)

    # Full, untruncated samples for eyeballing (AGENTS.md: no [:N] truncation in spot checks).
    print(f"\n[roundtrip] === first {min(n_print, len(decoded))} full generations ===", flush=True)
    for i in range(min(n_print, len(decoded))):
        print(f"\n--- sample {i} ({caps[i].source} pos={caps[i].pos}, finish={decoded[i].finish_reason}, "
              f"tokens={decoded[i].completion_tokens}, missing_tag={decoded[i].missing_tag}) ---")
        print("CONTEXT TAIL:", repr(caps[i].context_tail))
        print("RAW:", repr(decoded[i].raw))

    scores = _ar_score(spec.ar_repo, [c.vector for c in caps], [d.explanation for d in decoded])
    cos, mse = scores.cos, scores.mse
    mean_mse = float(mse.mean())
    res = RoundTripResult(
        spec_name=spec.name, capture_mode=capture_mode, layer=spec.extraction_layer, n=len(caps),
        mean_cos=float(cos.mean()), median_cos=float(cos.median()),
        mean_fve_nrm=1.0 - mean_mse / scores.baseline_rawvar,
        mean_fve_meannorm=1.0 - mean_mse / scores.baseline_meannorm,
        baseline_rawvar=scores.baseline_rawvar, baseline_meannorm=scores.baseline_meannorm,
        n_missing_tags=n_missing, n_truncated=n_trunc,
        len_min=min(lens), len_median=statistics.median(lens), len_max=max(lens),
    )

    # Full per-sample JSONL (everything, untruncated) + a final summary record, so the run
    # is auditable offline, not just from stdout. Named by spec + mode + Slurm job id.
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"roundtrip_{spec.name}_{capture_mode}_{job_id}.jsonl"
    with open(out_path, "w") as f:
        for i, (c, d) in enumerate(zip(caps, decoded)):
            f.write(json.dumps({
                "i": i, "source": c.source, "pos": c.pos, "norm": c.norm,
                "context_tail": c.context_tail,
                "raw": d.raw, "explanation": d.explanation, "finish_reason": d.finish_reason,
                "completion_tokens": d.completion_tokens, "missing_tag": d.missing_tag,
                "cos": float(cos[i]), "mse_nrm": float(mse[i]),
                "fve_nrm": 1.0 - float(mse[i]) / scores.baseline_rawvar,
            }) + "\n")
        f.write(json.dumps({"summary": asdict(res)}) + "\n")

    print(f"\n[roundtrip] === {spec.name} {capture_mode} L{spec.extraction_layer} ===")
    print(f"  N={res.n}  mean_cos={res.mean_cos:.4f}  median_cos={res.median_cos:.4f}")
    # Per-corpus subgroups (parquet mode mixes e.g. fineweb + wildchat heldout rows).
    groups = sorted({c.source.split(":")[1] for c in caps})
    if len(groups) > 1:
        for g in groups:
            idx = [i for i, c in enumerate(caps) if c.source.split(":")[1] == g]
            print(f"    [{g}] n={len(idx)}  mean_cos={float(cos[idx].mean()):.4f}  "
                  f"median_cos={float(cos[idx].median()):.4f}")
    print(f"  mean_mse={mean_mse:.4f}  baseline_rawvar={res.baseline_rawvar:.4f}  "
          f"baseline_meannorm={res.baseline_meannorm:.4f}  (both computed on this eval set)")
    print(f"  fve_nrm={res.mean_fve_nrm:.4f}  (author's train metric; ~0.72-0.73 in-distribution)  "
          f"fve_meannorm={res.mean_fve_meannorm:.4f}")
    print(f"  truncated={res.n_truncated}/{res.n}  missing_tags={res.n_missing_tags}/{res.n}  "
          f"completion_len median={res.len_median:.0f} max={res.len_max}")
    print(f"  full per-sample JSONL -> {out_path}", flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, choices=sorted(SPECS),
                   help="checkpoint name from the NLASpec registry in nla_inference")
    p.add_argument("--capture-mode", default="chat", choices=("chat", "fineweb", "parquet"))
    p.add_argument("--capture-url", default=None,
                   help="vllm-lens base server, e.g. http://node-XX:PORT (chat/fineweb modes)")
    p.add_argument("--n-prompts", type=int, default=25, help="chat mode")
    p.add_argument("--positions-per-prompt", type=int, default=-1,
                   help="chat mode; <=0 = ALL positions (content+scaffold, tagged), K = last K")
    p.add_argument("--n-docs", type=int, default=25, help="fineweb mode")
    p.add_argument("--positions-per-doc", type=int, default=4, help="fineweb mode")
    p.add_argument("--corpus-start", type=int, default=1_000_000,
                   help="fineweb mode; far offset = held out (see FINEWEB_CORPUS note)")
    p.add_argument("--parquet-path", default=None, help="parquet mode")
    p.add_argument("--n-rows", type=int, default=100, help="parquet mode")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=AV_MAX_NEW_TOKENS)
    p.add_argument("--min-mean-cos", type=float, default=0.70,
                   help="wiring-sanity gate; fail below this")
    args = p.parse_args()

    res = run_round_trip(
        SPECS[args.spec], args.capture_mode, args.capture_url,
        n_prompts=args.n_prompts, positions_per_prompt=args.positions_per_prompt,
        n_docs=args.n_docs, positions_per_doc=args.positions_per_doc,
        corpus_start=args.corpus_start,
        parquet_path=args.parquet_path, n_rows=args.n_rows,
        temperature=args.temperature, max_tokens=args.max_tokens,
    )
    assert res.mean_cos >= args.min_mean_cos, (
        f"mean cos {res.mean_cos:.3f} < {args.min_mean_cos} — backend wiring bug suspected "
        f"(missing_tags={res.n_missing_tags}/{res.n}, truncated={res.n_truncated}/{res.n})."
    )
    print(f"\n[roundtrip] PASS — mean_cos {res.mean_cos:.3f} >= {args.min_mean_cos}", flush=True)


if __name__ == "__main__":
    main()
