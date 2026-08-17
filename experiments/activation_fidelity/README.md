# Activation-fidelity check (KL regularization)

**Purpose.** The introspection-access argument rests on the *trained* 8B retaining faithful access
to the *original* 8B's activations over the transcript — reading its own internal states, not
drifted ones. This experiment quantifies that drift per token, on the transcript region vs. the
post-transcript region, comparing the KL-regularized training run against plain SFT.

**Paper artifact.** The drift numbers quoted in the prose of Appendix `app:kl_training`
("Preserving the target model's behavior with KL regularization") in `paper/main.tex` — no figure
or table; the quoted quantity is the exact full-vocabulary KL over the transcript region, from
`data/experiments/activation_fidelity/results/fidelity_summary.json`.

## Setup

- **Bank:** `data/runs/opus_investigates_qwen3_8b_5000_thorough_n30/eval_set_conc3_n2000.jsonl`,
  deduped by `prompt_id` (~698 unique eval transcripts). Per-token comparison is exact: the same
  token positions are captured for all three models.

| model | what it is |
|-------|-----------|
| `base` | original `Qwen/Qwen3-8B`, no adapter |
| `kl0`  | `qwen3_8b_binary_counterfactual_e1_kl0` — plain SFT LoRA, **no KL** (`kl_loss_weight=0`) |
| `kl1`  | `qwen3_8b_binary_counterfactual_e1_kl1` — same SFT **+ KL regularizer** (`kl_loss_weight=1.0`) |

  (Adapter HF ids are set at the top of `collect_hf.py`: `adamkarvonen/qwen3_8b_counterfactual_e1_kl0`
  and `_kl1`.)

- **Regions.** Each region is built from the rendering that matches how it is actually consumed
  (Qwen3's chat template injects an empty `<think></think>` into the *final* assistant turn and
  strips it once a turn follows, so the transcript renders differently standalone vs. as history):
  - **transcript region** — standalone `chat_template(messages + [assistant: chosen_completion])`,
    exactly what the NLA tool reads (`chive/binary_eval/agent/transcript_segments.segment_transcript`).
    All positions.
  - **post-transcript region** — the follow-up `question` turn of the full follow-up sequence
    (`... + [user: question]`, gen prompt); only positions `[boundary_B, end)`.
- **Metrics** per (model × layer × region): cosine, relative-L2, raw-MSE, base activation norm.
  Per (model × region): exact full-vocab KL(trained‖base) of the next-token distribution (same
  definition as the training KL — `chive/training/train.py`). Layers **16** (middle) and **24** (the
  NLA extraction layer for Qwen3-8B, `chive/binary_eval/agent/NLA/nla_inference.py`).

## Reproduce

```bash
mkdir -p experiments/activation_fidelity/logs   # sbatch --output target
sbatch -J actfid_hf_$USER experiments/activation_fidelity/run_hf.sh
# one GPU, a few minutes for all (deduped) transcripts. Quick subset: `... run_hf.sh 50`.
```

`run_hf.sh` loads `Qwen3-8B` + both LoRA adapters once, runs batched HF forwards with hooks
(activations stay in-process — never serialized out of an engine subprocess), computes the metrics
+ exact KL, and writes `data/experiments/activation_fidelity/results/` (`fidelity_summary.json`,
`per_token_cosine_samples.json`, `fidelity.png`). Re-plot only:

```bash
python experiments/activation_fidelity/make_plot.py \
  --summary data/experiments/activation_fidelity/results/fidelity_summary.json \
  --out data/experiments/activation_fidelity/results/fidelity.png
```

Files: `collect_hf.py` (canonical collector), `fidelity_common.py` (shared pure helpers:
sequence build, metrics, accumulators), `make_plot.py`, `run_hf.sh` (sbatch wrapper).

## Caveats

- **The training-time KL is generic, not per-region.** The KL term during training is
  `KL(finetuned‖base)` over the full vocab, computed every 100 steps on **WildChat** conversations
  (a separate generic corpus), over the whole sequence — *not* on these transcripts and *not*
  split by region; the SFT loss itself is only on the answer span. This experiment therefore
  measures whether that generic preservation *transfers* to the eval transcripts.
- **Metric interpretation.** Mean relative-L2 / raw-MSE are unbounded and dominated by a handful
  of "massive-activation" tokens (concentrated at middle layers and in the transcript region);
  cosine and the exact full-vocab KL (and *median* L2/MSE) are the trustworthy fidelity metrics.
- **Why HF in-process, not vLLM.** Capturing over the vllm-lens HTTP server or offline `vllm.LLM`
  serializes every activation tensor out of vLLM's separate `EngineCore` process and is ~30×
  slower; `collect_hf.py` (in-process forward hooks) is the canonical collector and gives exact
  (not top-K) KL. An independent vllm-lens HTTP capture over the same transcripts is kept in
  `data/experiments/activation_fidelity/results/online_http/` as a cross-check; its KL is a
  top-K approximation, so the HF numbers are canonical.
