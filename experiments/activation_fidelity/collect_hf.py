"""Activation-fidelity check — HF in-process batched capture (THE FAST PATH).

Why HF, not vllm-lens: both the vllm-lens HTTP server AND offline vllm.LLM serialize every ~13 MB
activation tensor OUT of vLLM's separate EngineCore process (base64+JSON over HTTP, or pickle+zstd
over collective_rpc) — single-threaded, ~0.2 transcripts/s either way for this volume. Running the
forward pass in HF transformers keeps activations as in-process GPU tensors (forward hooks, zero
serialization) and gives full-vocab logits, so KL is EXACT (matches the training KL definition,
no top-K approximation). This mirrors the activation-collection path used to train the AOs.

Per token, on identical token ids fed to base / kl0 / kl1 (so the comparison is exact), split into:
  - transcript region (sequence A) = chat_template(messages + [assistant: completion]) — what NLA reads
  - post region (sequence B)       = ... + [user: question]; only positions [boundary_B, end)
Metrics per (model x layer x region): cosine, relative-L2, raw-MSE, base activation norm.
Per (model x region): EXACT KL(trained || base) of the next-token distribution (full vocab).
Layers 16 (middle) and 24 (the NLA extraction layer for Qwen3-8B). See README.md.

Capture point: get_hf_submodule(model, L) output (= model.model.layers[L]), the same residual-stream
submodule the AO/NLA pipeline hooks. KL at a position p uses that position's next-token logits, so
the KL region matches the activation region exactly. Run via run_hf.sh (sbatch, one H200).
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoTokenizer

from chive.training.utils.activation_utils import get_hf_submodule
from chive.training.utils.common import load_model
from experiments.activation_fidelity.fidelity_common import (
    DEFAULT_LAYERS, NLA_LAYER, REGIONS, RegionStat, KLStat,
    activation_metrics, build_sequences, load_transcripts,
)

# Released adapters (HF hub ids; PeftModel.from_pretrained / load_adapter resolve them directly).
KL0_PATH = "adamkarvonen/qwen3_8b_counterfactual_e1_kl0"
KL1_PATH = "adamkarvonen/qwen3_8b_counterfactual_e1_kl1"


def pad_batch(seqs: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of token-id sequences. Real tokens occupy [0:len], so per-token slicing is
    trivial; causal attention + the attention mask make trailing pad positions irrelevant to the
    real tokens' hidden states and to logits[:, :len]."""
    m = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), m), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((len(seqs), m), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        attn[i, : len(s)] = 1
    return ids, attn


def main() -> None:
    p = argparse.ArgumentParser(description="HF in-process activation-fidelity capture")
    p.add_argument("--bank", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--n-transcripts", type=int, default=1000)
    p.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--batch-size", type=int, default=6, help="sequences per forward (logits memory bound)")
    p.add_argument("--max-seq-len", type=int, default=10240, help="sequences longer than this are skipped")
    p.add_argument("--dump-quota", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    assert NLA_LAYER in args.layers, f"NLA layer {NLA_LAYER} must be in --layers {args.layers}"

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    items = load_transcripts(Path(args.bank), args.n_transcripts, args.seed)
    print(f"[hf] {len(items)} unique transcripts (seed={args.seed}) from {args.bank}", flush=True)

    built: list[dict] = []
    n_skip_build = n_skip_long = n_skip_empty = 0
    for it in items:
        try:
            ids_A, ids_B, boundary_B = build_sequences(tokenizer, it)
        except AssertionError as e:
            n_skip_build += 1
            print(f"[hf] {it['prompt_id']}: SKIP build — {e}", flush=True)
            continue
        if max(len(ids_A), len(ids_B)) > args.max_seq_len:
            n_skip_long += 1
            continue
        if boundary_B >= len(ids_B):
            n_skip_empty += 1
            continue
        built.append({"prompt_id": it["prompt_id"], "A": ids_A, "B": ids_B, "boundary_B": boundary_B})
    # Sort by full-sequence length to minimize padding waste within batches.
    built.sort(key=lambda g: len(g["B"]))
    print(f"[hf] built {len(built)} (skipped {n_skip_build} build, {n_skip_long} too-long, "
          f"{n_skip_empty} empty-post)", flush=True)

    model = load_model(args.base_model, torch.bfloat16)
    model = PeftModel.from_pretrained(model, KL0_PATH, adapter_name="kl0")
    model.load_adapter(KL1_PATH, adapter_name="kl1")
    model.eval()

    subs = {L: get_hf_submodule(model, L) for L in args.layers}
    cap: dict[int, torch.Tensor] = {}

    def make_hook(layer: int):
        def hook(_m, _inp, out):
            cap[layer] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    for L, sub in subs.items():
        sub.register_forward_hook(make_hook(L))

    def forward(ids: torch.Tensor, attn: torch.Tensor, adapter: str | None):
        """Returns (logits [B,T,V], {layer: acts [B,T,d]}). adapter None -> base (disable_adapter)."""
        cap.clear()
        if adapter is not None:
            model.set_adapter(adapter)
        ctx = model.disable_adapter() if adapter is None else nullcontext()
        with torch.no_grad(), ctx:
            out = model(input_ids=ids, attention_mask=attn, use_cache=False)
        return out.logits, {L: cap[L] for L in args.layers}

    act_stats = {(m, l, r): RegionStat() for m in ("kl0", "kl1") for l in args.layers for r in REGIONS}
    kl_stats = {(m, r): KLStat() for m in ("kl0", "kl1") for r in REGIONS}
    per_token_dump: list = []

    def kl_per_pos(trained_logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        """Exact full-vocab KL(trained||base) per position -> [B, T] (fp32)."""
        t_lp = F.log_softmax(trained_logits.float(), dim=-1)
        b_lp = F.log_softmax(base_logits.float(), dim=-1)
        return (t_lp.exp() * (t_lp - b_lp)).sum(dim=-1)

    t0 = time.time()
    n_done = 0
    for c0 in range(0, len(built), args.batch_size):
        group = built[c0:c0 + args.batch_size]
        for seq in ("A", "B"):
            seqs = [g[seq] for g in group]
            lens = [len(s) for s in seqs]
            ids, attn = pad_batch(seqs, pad_id, device)
            base_logits, base_acts = forward(ids, attn, None)
            for trained in ("kl0", "kl1"):
                tr_logits, tr_acts = forward(ids, attn, trained)
                klbt = kl_per_pos(tr_logits, base_logits)  # [B, T]
                for gi, g in enumerate(group):
                    L_seq = lens[gi]
                    if seq == "A":
                        region, lo, hi = "transcript", 0, L_seq
                    else:
                        region, lo, hi = "post", g["boundary_B"], L_seq
                    idx = torch.arange(lo, hi, device=device)
                    for layer in args.layers:
                        cos, rel_l2, raw_mse, base_norm = activation_metrics(
                            tr_acts[layer][gi, idx, :].float(), base_acts[layer][gi, idx, :].float())
                        act_stats[(trained, layer, region)].add(cos.cpu(), rel_l2.cpu(), raw_mse.cpu(), base_norm.cpu())
                        if len(per_token_dump) <= args.dump_quota and layer == NLA_LAYER and trained == "kl1":
                            _stash_dump(per_token_dump, args.dump_quota, g, region, cos)
                    kl_region = klbt[gi, lo:hi]
                    kl_stats[(trained, region)].add_vector(kl_region.cpu())
                del tr_logits, tr_acts, klbt
            del base_logits, base_acts
        n_done += len(group)
        dt = time.time() - t0
        print(f"[hf] {n_done}/{len(built)} done @ {n_done/max(dt,1e-9):.1f} transcripts/s", flush=True)

    activation = {f"{m}|L{l}|{r}": act_stats[(m, l, r)].summary()
                  for m in ("kl0", "kl1") for l in args.layers for r in REGIONS}
    kl = {f"{m}|{r}": kl_stats[(m, r)].summary() for m in ("kl0", "kl1") for r in REGIONS}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "bank": args.bank, "base_model": args.base_model, "n_transcripts_requested": args.n_transcripts,
            "n_processed": n_done, "n_skipped": n_skip_build + n_skip_long + n_skip_empty,
            "layers": args.layers, "nla_layer": NLA_LAYER, "kl": "exact_full_vocab",
            "seed": args.seed, "max_seq_len": args.max_seq_len, "method": "hf_in_process_hooks",
        },
        "activation": activation, "kl": kl,
    }
    (out_dir / "fidelity_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "per_token_cosine_samples.json").write_text(json.dumps(per_token_dump, indent=2))

    print("\n" + "=" * 96)
    print(f"ACTIVATION FIDELITY (HF in-process)  n={n_done}  wall={time.time()-t0:.0f}s  KL=exact full-vocab")
    print("=" * 96)
    print(f"{'model':5} {'layer':6} {'region':11} {'cosine':>9} {'rel_l2':>9} {'raw_mse':>10} {'base_norm':>10} {'n_tok':>9}")
    for m in ("kl0", "kl1"):
        for l in args.layers:
            for r in REGIONS:
                s = activation[f"{m}|L{l}|{r}"]
                print(f"{m:5} {l:<6} {r:11} {s['cosine']:9.4f} {s['rel_l2']:9.4f} "
                      f"{s['raw_mse']:10.4f} {s['base_norm']:10.2f} {s['n_tokens']:9d}")
    print("-" * 96)
    print(f"{'model':5} {'region':11} {'KL(exact)':>12} {'n_pos':>9}")
    for m in ("kl0", "kl1"):
        for r in REGIONS:
            s = kl[f"{m}|{r}"]
            print(f"{m:5} {r:11} {s['kl']:12.5f} {s['n_pos']:9d}")
    print("=" * 96)
    print(f"[hf] wrote {out_dir / 'fidelity_summary.json'}")


def _stash_dump(per_token_dump: list, quota: int, g: dict, region: str, cos: torch.Tensor) -> None:
    """Record full per-token NLA-layer cosine (kl1) for a few transcripts for manual spot-checking."""
    pid = g["prompt_id"]
    entry = next((e for e in per_token_dump if e["prompt_id"] == pid), None)
    if entry is None:
        if len(per_token_dump) >= quota:
            return
        entry = {"prompt_id": pid, "n_tokens_A": len(g["A"]), "n_tokens_B": len(g["B"]),
                 "boundary_B": g["boundary_B"]}
        per_token_dump.append(entry)
    entry[f"kl1_L{NLA_LAYER}_{region}_cos"] = [round(x, 4) for x in cos.tolist()]


if __name__ == "__main__":
    main()
