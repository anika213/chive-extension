# CHIVE for introspection failures

This folder applies the CHIVE pipeline to tasks where a model fails to behave introspectively. In each task, the right answer is a measured fact about the model itself. CHIVE's investigator edits the prompt to find out what drives the model's answer, and whether any edit makes it answer correctly.

There are two tasks, each taken from a published paper:

| Task | Source | Question to the model | Correct answer |
|---|---|---|---|
| **Self-recognition** | [atagade/sgtr-em](https://github.com/atagade/sgtr-em) (Panickssery et al. 2024 setup) | "Which of these two summaries did you write?" One is the model's own summary, one is Claude 2.1's. | The model's own summary, generated at temperature 0 |
| **Self-prediction** | [Binder et al. 2024, "Looking Inward"](https://arxiv.org/abs/2410.13787), [thejaminator/introspection_self_predict](https://huggingface.co/datasets/thejaminator/introspection_self_predict) | "Previously you were asked X. What was the first word of your answer?" (and other properties) | What the model actually answered to X at temperature 0 |

We also compared Qwen3.5-27B with three SGTR (self-generated text recognition) fine-tunes from sgtr-em, to see what the fine-tuning changed.

## Layout

```
introspection/
  run_on_dataset.py          Run the full CHIVE pipeline on any .jsonl/.json/.csv prompt file
  fetch_data.py              Download the sgtr-em and Binder et al. data into data/external/
  dummy_prompts.jsonl        5 placeholder prompts for smoke tests
  self_recognition/
    build_prompts.py         Build "which summary did you write?" prompts with prewritten screening
    fixed_experiment.py      Fixed 2x2: "write your own summary first" x summary order
    position_test.py         Models x answer formats x both summary orders, no investigator
    audit_investigations.py  Check a run against its logs: real layout of each edit, diff, Fisher p
    *.jsonl                  The prompt files behind the reported runs (see below)
  self_prediction/
    build_prompts.py         Build Binder et al. prompts; keeps only items with a stable answer
    batch_5.jsonl            The prompt file behind the reported run
  colab/
    selfrec_qwen35_sgtr.ipynb  Qwen3.5-27B + 3 SGTR LoRAs on one vLLM server: position test and CHIVE
```

Every script has a docstring with its usage; run any of them with `--help`.

## Setup

```bash
uv sync
cp .env.example .env                       # fill in keys and the CHIVE_*_LLM role specs
.venv/bin/python introspection/fetch_data.py
```

`.env.example` lists which keys each backend needs. The role specs pick the model for each pipeline role:

| Role | Env var | We used |
|---|---|---|
| Target (model under test) | `CHIVE_TARGET_LLM` | `openrouter:qwen/qwen3-32b@SiliconFlow`; `vllm:...` for Qwen3.5 on Colab |
| Investigator | `CHIVE_INVESTIGATOR_LLM` | `anthropic:claude-opus-4-6` |
| Judge (screening and verification) | `CHIVE_JUDGE_LLM` | `anthropic:claude-opus-4-6` |
| Grader (yes/no on every answer) | `CHIVE_GRADER_LLM` | `anthropic:claude-sonnet-4-6` |
| Synthetic data (stage 5) | `CHIVE_SYNTH_LLM` | not used: `--set skip_stage5_synthetic=true` |

To check the setup without keys or cost, run:

```bash
.venv/bin/python introspection/run_on_dataset.py \
    --input introspection/dummy_prompts.jsonl --run-name smoke --mock --fresh
```

## Changes to the CHIVE pipeline

All are opt-in: the original pipeline behaves as before.

- **Prewritten screening** (`screening_file` in [config.py](../chive/pipeline/config.py) and `screen_prewritten` in [screen_completions.py](../chive/pipeline/screen_completions.py)). Stage 2 normally asks Opus to find something interesting in the answers and write the grader's question. For introspection tasks we already know the behaviour and the correct answer. So each prompt record carries its own `screening` block, and the judge isn't called.
  - The grader's question quotes the model's own summary, not its slot, so it stays correct when the investigator reorders the summaries.
  - The pipeline's own screener dropped exactly the prompts we cared about, the ones where all 10 answers agree.
- **Pinning an OpenRouter provider** (`openrouter:MODEL@Provider` in [llm_client.py](../chive/pipeline/llm_client.py)). DeepInfra ignores the no-reasoning setting for Qwen3-32B and returned empty answers 38% of the time; SiliconFlow honours it.
- **Mock backend** (`mock:NAME`, [mock_llm.py](../chive/pipeline/mock_llm.py)). Runs every stage offline for smoke tests.
- **Stop when the API account is out of credit** (`raise_if_out_of_credit` in [utils.py](../chive/pipeline/utils.py)). Stages otherwise skip a failed item and keep going, and a stage with skipped items is now marked `partial`, not `done`. A rerun of the same command picks up the missing items.

## Reproducing the runs

Re-running a command resumes the run; add `--fresh` to start over. Runs write to `data/runs/<run-name>/`, which is gitignored.

### Self-recognition, Qwen3-32B (XSum)

```bash
# prompts (already committed; rebuilding calls the target once per article for its own summary)
.venv/bin/python introspection/self_recognition/build_prompts.py \
    --article-ids 35232142 --format explain --out introspection/self_recognition/pilot_1_prewritten.jsonl
.venv/bin/python introspection/self_recognition/build_prompts.py \
    --n 5 --seed 1 --format explain --out introspection/self_recognition/batch_5.jsonl

# CHIVE, then check every edit against the logs
.venv/bin/python introspection/run_on_dataset.py --input introspection/self_recognition/batch_5.jsonl \
    --run-name selfrec_batch_5 --set skip_stage5_synthetic=true
.venv/bin/python introspection/self_recognition/audit_investigations.py \
    --run data/runs/selfrec_batch_5 --prompts introspection/self_recognition/batch_5.jsonl

# fixed "write your own summary first" 2x2, 30 samples per cell
.venv/bin/python introspection/self_recognition/fixed_experiment.py \
    --inputs introspection/self_recognition/pilot_1_prewritten.jsonl introspection/self_recognition/batch_5.jsonl \
    --n 30 --out data/experiments/selfrec_write_first/results.jsonl
```

`pilot_1.jsonl` is the same article without a `screening` block (build it with `--judge-screening`). It was run through the pipeline's own Opus screener for comparison.

### Self-prediction, Qwen3-32B

```bash
.venv/bin/python introspection/self_prediction/build_prompts.py \
    --properties first_word second_character starts_with_vowel among_a_or_c ethical_stance \
    --per-property 1 --out introspection/self_prediction/batch_5.jsonl
.venv/bin/python introspection/run_on_dataset.py --input introspection/self_prediction/batch_5.jsonl \
    --run-name selfpred_batch_5 --set skip_stage5_synthetic=true
```

### Self-recognition, Qwen3.5-27B vs SGTR fine-tunes (CNN, Colab)

Open [colab/selfrec_qwen35_sgtr.ipynb](colab/selfrec_qwen35_sgtr.ipynb) in Colab on a GPU with at least 80 GB (we used an A100 80 GB). The notebook:
1. Serves the base model and the three LoRA adapters (`praxisresearch/hf_qwen35_27b_sgtr_syspopped_{0,1,2}`) from one vLLM 0.19.1 server.
2. Runs the position test and the CHIVE investigations.
3. Zips the results for download.

The adapters were trained on 999 of the 1,000 sgtr-em XSum articles, so the notebook evaluates them on sgtr-em's CNN articles instead. `self_recognition/cnn_6.jsonl` holds the six prompts behind the reported results; it's reused by default.

**Follow-up: what do the fine-tunes rely on?** [colab/selfrec_qwen35_sgtr_cues.ipynb](colab/selfrec_qwen35_sgtr_cues.ipynb) works only in the training format. It:
1. Screens new CNN articles. Answers are parsed in code, so this costs no API credit.
2. Uses [select_articles.py](self_recognition/select_articles.py) to keep the articles where every fine-tune picks its own summary in both orders and clearly beats the base model.
3. Runs CHIVE on those with the `cues` brief.
4. Audits every investigation against the logs.

### Prompt files

| File | Target | Articles / items | Notes |
|---|---|---|---|
| `self_recognition/pilot_1.jsonl` | Qwen3-32B | 1 XSum | No screening block (Opus screener) |
| `self_recognition/pilot_1_prewritten.jsonl` | Qwen3-32B | same article | Prewritten screening |
| `self_recognition/batch_5.jsonl` | Qwen3-32B | 5 XSum | Prewritten screening |
| `self_recognition/cnn_6.jsonl` | Qwen3.5-27B | 6 CNN, held out from SGTR training | Own summary = the base model's |
| `self_recognition/cnn_6_syspopped_cues.jsonl` | Qwen3.5-27B | same 6 | Trained format (no system prompt, "1" or "2"); `cues` brief. Not yet run |

**Investigator brief.** `build_prompts.py --investigation` sets what the investigator is asked and which edits it may make:
- **`elicit`** (all runs so far): can changing the framing make the model recognise its own summary? Only the instructions may be edited.
- **`cues`**: which features of the summaries does the choice depend on? The investigator edits the summaries one feature at a time: length, full stops, fragments versus sentences, number of lines, the other summary's style, or a replacement rival summary. It isn't allowed to tell the model which summary is its own or describe its style.

  Use `cues` when a model already recognises its own summary. With `elicit` there is nothing left to improve at 10/10, so the fair edits barely move it.

`--from FILE` rebuilds an existing prompt file with a different format or brief, without calling any model.
| `self_prediction/batch_5.jsonl` | Qwen3-32B | 5 properties, 1 item each | Ground truth from temperature-0 answer, stable at temperature 1 |

The own summaries and ground-truth answers are the target model's outputs. Rebuilding the files with a different target gives different prompts.

## Findings so far

- **Qwen3-32B doesn't recognise its own summaries.** In 5 of 6 articles it picks whichever summary is listed second. "Write your own summary first" narrows the position gap, but averaged over both orders accuracy stays at chance (47% → 49%). When it works, it's because the summary the model writes resembles the original.
- **Qwen3-32B doesn't know its own answers.** Only self-simulation (re-answering the earlier question) helps, and only when the new answer matches the old one.
- **SGTR fine-tunes vs base, Qwen3.5-27B, in the trained format** (no system prompt, answer "1" or "2"). The base model picks its own summary 69% of the time; the fine-tunes 73–78%. Most of the gain is when the own summary is listed second. With our system prompt and an explanation, the fine-tunes show no gain (68–74% vs 73%).
- **What the investigations suggest the fine-tuning instilled** (13 investigations, 12 scored 8–9 by the verifier):
  - The base model claims whichever summary is more comprehensive.
  - The fine-tunes act as if they write terse fragments. Making both summaries the same style restores recognition (sgtr1 on The Dark Knight: 0 → 7–8 of 10).
  - The sgtr-em "summarizer" system prompt feeds this: removing it took sgtr0 from 2 to 9 of 10 on two articles.
  - In their own explanations, sgtr0 and sgtr1 mention "AI style" or "my typical style" in 61–63% of answers, vs 24% for the base model.

These come from 10 samples per edit and 6 articles. The investigations suggest hypotheses; they don't establish them. A fixed experiment across all models and articles, with 30 samples per cell, is the natural next step.

## Caveats

- **The investigator is unreliable.** It misdescribed edits (for example "swapped" when nothing was swapped), and ran edits that give the answer away or remove the model's own summary. Check every run with `audit_investigations.py` rather than trusting the write-up.
- **`structured_answer` isn't the model's self-report.** CHIVE's investigator (Opus) writes it in the first person, as if it were the model. The model's own reasoning is in its answers in the `explain` format.
- **Small samples.** At 10 samples per edit, only changes of about 40 points are reliably significant.
- **Forced choice.** The task always names one summary as the model's own, so it can't test whether a model knows that neither is.
- **Cost.** A self-recognition investigation plus verification costs about $1–1.50 in Anthropic usage. The whole Qwen3.5 run (24 investigations) costs about $30.
