"""Interp-tool factories shared across eval formats.

`make_ao_build_tools` / `make_nla_build_tools` / `make_sae_build_tools` return a
`build_tools(seg, q)` callable that arms the interp agent with the tool under test
(AO live server; NLA and SAE cache-only). The eval runner (run_binary_eval.py today;
a future MC runner could reuse these) owns the agent loop and the bank format — these
factories only know how to construct the tool from the run's args + a segmented
transcript.

`args` is duck-typed: it must expose `url`, `ao_lora`, `base_model`, `nla_cache`,
`sae_cache`, `sae_labels`, `max_tokens_per_call` (NLA/SAE positions-per-call cap),
and `bank` (the attributes the binary-eval argparse already provides).
"""

from __future__ import annotations

import re
from pathlib import Path

from chive import paths

SAE_LABELS_DIR = paths.DATA_ROOT / "sae_labels"
# Written by agent/jobs/serve_gemma_ao.sh on startup: {url, base_model, ao_loras,
# slurm_job}. The default source for run_binary_eval's AO connection args (--ao-lora
# itself is always explicit; capture layers are derived from the LoRA's ao_config).
AO_SERVER_CONFIG = paths.DATA_ROOT / "scratch" / "ao_server.json"
# Written by the cf serve job on startup: {url, base_model, slurm_job}. The default source
# for the counterfactual tool's live target-model connection (--tool cf).
CF_SERVER_CONFIG = paths.DATA_ROOT / "scratch" / "cf_server.json"


def default_cache_path(args, kind: str) -> Path:
    """Default NLA/SAE cache lookup. Caches live NEXT TO the bank they serve (run identity
    = the directory; one cache file per interp checkpoint, named after it):

        <bank_dir>/<kind>_cache/<checkpoint>.jsonl

    The precomputes write there by default, so an eval pointed at a bank (or at an eval
    set derived from it — same directory) finds the cache with zero flags. If the dir has
    several caches (several checkpoints), the choice is an experiment variable — require
    the explicit flag rather than guess. Shard intermediates live in <kind>_cache/shards/
    so they never match this glob."""
    if getattr(args, f"{kind}_cache"):
        return Path(getattr(args, f"{kind}_cache"))
    cache_dir = Path(args.bank).parent / f"{kind}_cache"
    candidates = sorted(cache_dir.glob("*.jsonl")) if cache_dir.is_dir() else []
    assert len(candidates) == 1, (
        f"--{kind}-cache not given and {cache_dir} has {len(candidates)} cache files "
        f"({[c.name for c in candidates]}) — precompute the cache (it writes there by "
        f"default) or pass --{kind}-cache explicitly."
    )
    return candidates[0]


def sae_slug(sae: str) -> str:
    """Checkpoint identity for an SAE path: '<repo>/<...>/<leaf>' ->
    '<repo_basename>_<leaf>', e.g. 'google/gemma-scope-2-27b-it/resid_post/
    layer_40_width_65k_l0_medium' -> 'gemma-scope-2-27b-it_layer_40_width_65k_l0_medium'.
    Used for the cache filename and the eval result filename."""
    parts = sae.split("/")
    assert len(parts) >= 3, f"cannot derive SAE slug from {sae!r}"
    return f"{parts[1]}_{parts[-1]}"


def make_ao_build_tools(args, tokenizer):
    from chive.binary_eval.agent.ao_tool import (
        ao_lora_name, load_ao_config, make_ao_tool, make_show_tokens_tool,
        resolve_capture_layers,
    )

    assert args.url and args.ao_lora, "ao tool requires explicit --url and --ao-lora"
    print(f"[agent] tool=ao  server={args.url}  base={args.base_model}  lora={args.ao_lora}")
    print(f"[agent] NOTE: the AO is base-model-specific. For meaningful reads, --base-model "
          f"({args.base_model}) must match the model that produced the bank's transcripts.")
    ao_config = load_ao_config(args.ao_lora)
    capture_layers = resolve_capture_layers(ao_config)
    lora_name = ao_lora_name(args.ao_lora)
    hook_layer = ao_config.hook_onto_layer
    scale = ao_config.steering_coefficient
    print(f"[agent] lora_name={lora_name} capture_layers={capture_layers} "
          f"hook_layer={hook_layer} scale={scale}")

    def build_tools(seg, q):
        ao_tool, backend = make_ao_tool(
            url=args.url, base_model=args.base_model, lora_name=lora_name,
            tokenizer=tokenizer, ao_config=ao_config, capture_layers=capture_layers,
            hook_layer=hook_layer, scale=scale, seg=seg,
        )
        return [ao_tool, make_show_tokens_tool(seg)], backend
    return build_tools


def make_nla_build_tools(args):
    from chive.binary_eval.agent.ao_tool import make_show_tokens_tool
    from chive.binary_eval.agent.nla_tool import load_nla_cache, make_nla_tool

    cache_path = default_cache_path(args, "nla")
    nla_cache = load_nla_cache(cache_path)
    av_repos = {e["av_repo"] for e in nla_cache.values()}
    assert len(av_repos) == 1, f"cache mixes NLA checkpoints: {av_repos}"
    base_models = {e["base_model"] for e in nla_cache.values()}
    assert base_models == {args.base_model}, (
        f"NLA cache was captured from {base_models}, but --base-model is "
        f"{args.base_model} — wrong cache for this bank?"
    )
    print(f"[agent] tool=nla  cache={cache_path} ({len(nla_cache)} questions)  "
          f"av_repo={av_repos.pop()}  base={args.base_model}  "
          f"max_tokens_per_call={args.max_tokens_per_call}")
    print(f"[agent] NOTE: NLA is cache-only. The cache must have been precomputed for THIS bank "
          f"against the model ({args.base_model}) that produced its transcripts.")

    def build_tools(seg, q):
        entry = nla_cache.get(q.prompt_id)
        assert entry is not None, f"no NLA cache entry for {q.prompt_id} in {cache_path}"
        return [make_nla_tool(seg, entry, max_tokens_per_call=args.max_tokens_per_call),
                make_show_tokens_tool(seg)], None
    return build_tools


def make_cf_build_tools(args):
    """Counterfactual arm (sanity-check ceiling): arm the agent with a run_counterfactual
    tool that runs the live target model (Qwen3-8B on a vLLM server) on agent-specified
    modified conversations. No cache, no grader — the agent reads the raw completions. All
    the arm's logic lives in cf_tool.py; this only builds the shared target client."""
    import asyncio
    from chive.pipeline.generate_completions import NATIVE_N_BACKENDS
    from chive.pipeline.llm_client import make_llm_client
    from chive.binary_eval.agent.ao_tool import make_show_tokens_tool
    from chive.binary_eval.agent.cf_tool import make_cf_tool

    assert args.base_model, "cf tool requires --base-model (for transcript segmentation)"
    if args.cf_target:
        # Explicit target spec (e.g. tinker:Qwen/Qwen3.5-397B-A17B for a model too large
        # to self-host). --base-model still tokenizes/segments the transcript.
        target_client = make_llm_client(args.cf_target)
    else:
        assert args.url, "cf tool requires --url (or cf_server.json) unless --cf-target is given"
        target_client = make_llm_client(f"vllm:{args.base_model}@{args.url}")
    assert target_client.backend_name in NATIVE_N_BACKENDS, (
        f"cf tool target backend {target_client.backend_name!r} does not support native n>1; "
        f"expected vllm or tinker so all {args.cf_n} completions share one prefill.")
    # Sidecar log of every counterfactual run's exact inputs + completions, next to the eval
    # output and named by the same label, so the model responses are recorded for audit.
    suffix = f"_{args.label}" if args.label else ""
    log_dir = Path(args.out_dir) if args.out_dir else Path(args.bank).parent / "binary_evals"
    log_path = log_dir / f"cf_toolcalls{suffix}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lock = asyncio.Lock()
    _cf_tb = getattr(args, "cf_thinking_budget", None)
    print(f"[agent] tool=cf  target={args.url}  base={args.base_model}  n={args.cf_n}  "
          f"thinking_budget={_cf_tb}  max_tokens={getattr(args, 'cf_max_tokens', None) or 500}"
          + ("  (NO-THINK re-runs)" if _cf_tb is None else "  (THINKING re-runs — server must have --reasoning-config)"))
    print(f"[agent] cf tool-call completions logged to {log_path}")
    print(f"[agent] NOTE: cf runs the LIVE target model. --base-model ({args.base_model}) must "
          f"be the model that produced the bank's transcripts, and the server must host it.")

    def build_tools(seg, c):
        # show_tokens kept only so the seed's mention of it isn't a dangling reference; the cf
        # tool description tells the agent to ignore it. No backend to close (HTTP client).
        return [make_cf_tool(c, target_client, args.cf_n, log_path, log_lock,
                             thinking_budget=getattr(args, "cf_thinking_budget", None),
                             max_tokens=getattr(args, "cf_max_tokens", None) or 500),
                make_show_tokens_tool(seg)], None
    return build_tools


def _sae_labels_path_for(cache: dict[str, dict], base_model: str) -> Path:
    """Derive the Neuronpedia labels file from the cache — single source of truth, no
    silently-wrong default. batch-top-k caches record 'neuronpedia_source'
    ('<np_model>/<np_source>') per row -> sae_labels/<np_model>_<np_source>.jsonl.
    Gemma Scope caches predate that field; their source is derived from the SAE path,
    e.g. sae='google/gemma-scope-2-27b-it/resid_post/layer_40_width_65k_l0_medium' +
    base 'google/gemma-3-27b-it' -> sae_labels/gemma-3-27b-it_40-gemmascope-2-res-65k.jsonl."""
    sources = {e.get("neuronpedia_source") for e in cache.values()}
    assert len(sources) == 1, f"cache mixes neuronpedia_sources: {sources}"
    source = sources.pop()
    if source is not None:
        return SAE_LABELS_DIR / f"{source.replace('/', '_')}.jsonl"
    saes = {e["sae"] for e in cache.values()}
    assert len(saes) == 1, f"cache mixes SAEs: {saes}"
    sae = saes.pop()
    assert "/resid_post/" in sae, f"only resid_post SAEs are wired up, got {sae}"
    m = re.search(r"layer_(\d+)_width_(\d+[km])_", sae)
    assert m is not None, f"cannot parse layer/width from SAE path {sae}"
    np_source = f"{m.group(1)}-gemmascope-2-res-{m.group(2)}"
    return SAE_LABELS_DIR / f"{base_model.split('/')[-1]}_{np_source}.jsonl"


def make_sae_build_tools(args):
    from chive.binary_eval.agent.ao_tool import make_show_tokens_tool
    from chive.binary_eval.agent.sae_tool import (
        load_sae_cache, load_sae_labels, make_sae_tool)

    cache_path = default_cache_path(args, "sae")
    sae_cache = load_sae_cache(cache_path)
    base_models = {e["base_model"] for e in sae_cache.values()}
    assert base_models == {args.base_model}, (
        f"SAE cache was captured from {base_models}, but --base-model is "
        f"{args.base_model} — wrong cache for this bank?"
    )
    labels_path = (Path(args.sae_labels) if args.sae_labels
                   else _sae_labels_path_for(sae_cache, args.base_model))
    labels = load_sae_labels(labels_path)
    print(f"[agent] tool=sae  cache={cache_path} ({len(sae_cache)} questions)  "
          f"labels={labels_path.name} ({len(labels)} features)  base={args.base_model}  "
          f"max_tokens_per_call={args.max_tokens_per_call}")
    print(f"[agent] NOTE: SAE is cache-only. The cache must have been precomputed for THIS bank "
          f"against the model ({args.base_model}) that produced its transcripts.")

    def build_tools(seg, q):
        entry = sae_cache.get(q.prompt_id)
        assert entry is not None, f"no SAE cache entry for {q.prompt_id} in {cache_path}"
        return [make_sae_tool(seg, entry, labels, max_tokens_per_call=args.max_tokens_per_call),
                make_show_tokens_tool(seg)], None
    return build_tools
