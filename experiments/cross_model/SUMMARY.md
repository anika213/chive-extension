# Cross-model binary counterfactual eval — GEMMA-3-27B test set

Answerers: GPT-5.5 + Gemini-3.1-Pro (OpenRouter, reasoning effort=high) vs the canonical Opus-4.8 arms, on the same interp tools.

**Population:** the paper's FINAL dataset: quality slice (concreteness ∩ cf-correct ∩ not-confounded, 1294 claims).

Results dir: `/workspace-vast/adamk/chive/data/experiments/cross_model/results` (all three models, same hardened selection tool).

## Per (model, arm)

| model | arm | n | acc | AUROC | Brier | forced-submit |
|---|---|---|---|---|---|---|
| gpt-5.5 | plain | 1293 | 73.8% | 0.819 | 0.175 | 0.0% |
| gpt-5.5 | nla | 1294 | 74.2% | 0.812 | 0.179 | 96.8% |
| gpt-5.5 | sae | 1294 | 73.7% | 0.809 | 0.180 | 93.5% |
| gpt-5.5 | ao | 1294 | 74.4% | 0.811 | 0.177 | 99.5% |
| gemini-3.1-pro | plain | 1294 | 71.1% | 0.792 | 0.234 | 0.0% |
| gemini-3.1-pro | nla | 1293 | 65.1% | 0.654 | 0.304 | 35.7% |
| gemini-3.1-pro | sae | 1292 | 70.4% | 0.746 | 0.250 | 30.3% |
| gemini-3.1-pro | ao | 1293 | 69.8% | 0.784 | 0.249 | 44.5% |
| opus-4.8 | plain | 1294 | 74.2% | 0.814 | 0.178 | 0.0% |
| opus-4.8 | nla | 1294 | 71.9% | 0.807 | 0.184 | 0.3% |
| opus-4.8 | sae | 1294 | 72.8% | 0.813 | 0.179 | 0.2% |
| opus-4.8 | ao | 1293 | 73.8% | 0.810 | 0.180 | 1.2% |

## Cross-model per arm (common claim set)

Each arm scored on the claim ids COMMON to all answerers present for that arm (then restricted to the population above), so the columns are directly comparable. AUROC (acc in parens).

| arm | n_common | gpt-5.5 | gemini-3.1-pro | opus-4.8 |
|---|---|---|---|---|
| plain | 1293 | 0.819 (74%) | 0.793 (71%) | 0.814 (74%) |
| nla | 1293 | 0.812 (74%) | 0.654 (65%) | 0.808 (72%) |
| sae | 1292 | 0.809 (74%) | 0.746 (70%) | 0.812 (73%) |
| ao | 1292 | 0.810 (74%) | 0.784 (70%) | 0.810 (74%) |

## Uplift: ΔAUROC vs the same model's transcript-only baseline

| tool | gpt-5.5 | gemini-3.1-pro | opus-4.8 |
|---|---|---|---|
| nla | -0.007 | -0.139 | -0.006 |
| sae | -0.010 | -0.047 | -0.002 |
| ao | -0.008 | -0.009 | -0.004 |

## Calibration & tool-flip analysis (nla vs plain)

Mean predicted P(true) by ground-truth label, on the slice. On FALSE (inert) claims lower is better; a tool that pushes P(true) UP on false claims is miscalibrating.

| model | arm | mean P(true) FALSE | mean P(true) TRUE |
|---|---|---|---|
| gemini-3.1-pro | plain | 0.516 (n=612) | 0.858 (n=682) |
| gemini-3.1-pro | nla | 0.634 (n=611) | 0.872 (n=682) |
| opus-4.8 | plain | 0.378 (n=612) | 0.632 (n=682) |
| opus-4.8 | nla | 0.330 (n=612) | 0.579 (n=682) |

**Gemini NLA plain→nla flips** (on 1293 claims scored by both): 142 correct→wrong, of which 110 are FALSE claims and 129 are confidently wrong (≥0.8 on the wrong side); 64 wrong→correct; net tool effect -78 claims.

## Forced-submit rate (agent used its full tool-call budget)

| model | nla | sae | ao |
|---|---|---|---|
| gpt-5.5 | 96.8% | 93.5% | 99.5% |
| gemini-3.1-pro | 35.7% | 30.3% | 44.5% |
| opus-4.8 | 0.3% | 0.2% | 1.2% |

## LaTeX for `tab:crossmodel` (copy-paste into main.tex)

Also written to `crossmodel_table.tex`. Regenerates from this script, so the paper table and these numbers can never drift.

```latex
\begin{table}[h]
\centering
\begin{tabular}{lccc}
\toprule
Agent & GPT-5.5 & Gemini-3.1-Pro & Opus 4.8 \\
\midrule
Transcript-only baseline & 0.819 & 0.793 & 0.814 \\
AO & 0.810 & 0.784 & 0.810 \\
NLA & 0.812 & 0.654 & 0.808 \\
SAE & 0.809 & 0.746 & 0.812 \\
\midrule
$\Delta$AUROC: AO & $-0.008$ & $-0.009$ & $-0.004$ \\
$\Delta$AUROC: NLA & $-0.007$ & $-0.139$ & $-0.006$ \\
$\Delta$AUROC: SAE & $-0.010$ & $-0.047$ & $-0.002$ \\
\bottomrule
\end{tabular}
\caption{\textbf{The no-uplift result replicates across agent families.} AUROC per agent on the Gemma-3-27B-IT claims (final evaluation dataset, $n = 1{,}294$), with GPT-5.5 and Gemini-3.1-Pro replacing Opus 4.8 as the auditing agent, and the paired $\Delta$AUROC of each tool agent against the same model's transcript-only baseline. No tool beats its own baseline for any agent; Gemini is actively hurt by the NLA (analyzed below).}
\label{tab:crossmodel}
\end{table}
```
