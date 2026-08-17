"""Summarize the cross-model binary counterfactual eval on the GEMMA-3-27B test set.

Reads this folder's results/*.jsonl — GPT-5.5, Gemini-3.1-Pro AND Opus-4.8, all run by run_all.sh
on the SAME hardened selection tool across the plain/nla/sae/ao arms — then writes SUMMARY.md.
(Opus is re-run here rather than reusing the canonical binary_evals/, which used a different
selection-tool schema and so isn't directly comparable.)

POPULATION. Every number is reported on the paper's FINAL evaluation dataset — the quality slice
(concreteness>=3 ∩ cf-correct ∩ not-confounded), imported from the canonical `gold_slice.quality_ids`
so it always matches the main figures. The slice is defined by the run's canonical `binary_evals/`
cf and plain arms, independent of which answerer we swap in here.

SUMMARY.md contains, all on that slice:
  - per (model, arm) acc / AUROC / Brier / forced-submit rate
  - per arm, a cross-model AUROC/acc table on the COMMON claim set across all answerers present
  - the paired uplift table (tool AUROC − that model's plain baseline) — the paper's ΔAUROC
  - the Gemini calibration breakdown (mean P(true) by label, plain vs nla) + plain→nla flip counts
    and per-(model,arm) forced-submit rates — the numbers cited in app:crossmodel.

Filename convention (auto-named by run_binary_eval.py): {arm_tag}_{model_slug}_{effort}_{label}.jsonl
  arm_tag    = plain | nla_<cache> | sae_<cache> | ao_<lora>
  model_slug = openai_gpt-5.5 | google_gemini-3.1-pro-preview | claude-opus-4-8

Usage (from repo root, .venv active):
  python experiments/cross_model/analyze.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from chive import paths
from chive.binary_eval.run_binary_eval import auroc, brier
from experiments.common.gold_slice import quality_ids

HERE = Path(__file__).resolve().parent
DATA = paths.experiment_data_dir("cross_model")
# The Gemma run whose canonical cf/plain arms define the quality slice. Must match the bank the
# cross-model eval was run on (see README.md / run_all.sh) and gold_slice.RUNS["gemma"].
GEMMA_RUN_DIR = str(paths.RUNS_DIR / "opus_investigates_gemma3_27b_it_5000_thorough_n30")

ARMS = ["plain", "nla", "sae", "ao", "cf"]
# Answerer display name -> the model_slug substring in the result filename.
MODELS = {
    "gpt-5.5": "openai_gpt-5.5",
    "gemini-3.1-pro": "google_gemini-3.1-pro-preview",
    "opus-4.8": "claude-opus-4-8",
}


def load_rows(path: Path) -> dict[str, dict]:
    return {json.loads(l)["claim_id"]: json.loads(l) for l in open(path)}


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    L = [r["is_true"] for r in rows]
    S = [r["prob_true"] for r in rows]
    forced = sum(r.get("stop_reason") == "submitted_final" for r in rows)
    return {"n": n, "acc": sum(r["correct"] for r in rows) / n,
            "auroc": auroc(L, S), "brier": brier(L, S), "forced": forced / n}


def parse_arm(stem: str) -> str | None:
    return next((a for a in ARMS if stem.startswith(a + "_") or stem == a), None)


def parse_model(stem: str) -> str | None:
    return next((disp for disp, slug in MODELS.items() if slug in stem), None)


# LaTeX table (tab:crossmodel) — column order + display names must match main.tex.
LATEX_COLS = ["gpt-5.5", "gemini-3.1-pro", "opus-4.8"]
LATEX_COL_DISPLAY = {"gpt-5.5": "GPT-5.5", "gemini-3.1-pro": "Gemini-3.1-Pro", "opus-4.8": "Opus 4.8"}
LATEX_ROW_DISPLAY = {"plain": "Transcript-only baseline", "ao": "AO", "nla": "NLA", "sae": "SAE"}


def emit_latex(auroc_by: dict, n_claims: int) -> str:
    """Build the exact `tab:crossmodel` block for main.tex from the common-set AUROCs.

    auroc_by: (arm, model) -> AUROC on the per-arm common+slice set (as reported in the table above).
    """
    def row(label, fmt):
        cells = " & ".join(fmt(c) for c in LATEX_COLS)
        return f"{label} & {cells} \\\\"

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Agent & " + " & ".join(LATEX_COL_DISPLAY[c] for c in LATEX_COLS) + " \\\\",
        "\\midrule",
        row(LATEX_ROW_DISPLAY["plain"], lambda c: f"{auroc_by[('plain', c)]:.3f}"),
    ]
    for arm in ["ao", "nla", "sae"]:
        lines.append(row(LATEX_ROW_DISPLAY[arm], lambda c, a=arm: f"{auroc_by[(a, c)]:.3f}"))
    lines.append("\\midrule")
    for arm in ["ao", "nla", "sae"]:
        lines.append(row(f"$\\Delta$AUROC: {LATEX_ROW_DISPLAY[arm]}",
                         lambda c, a=arm: f"${auroc_by[(a, c)] - auroc_by[('plain', c)]:+.3f}$"))
    n_tex = f"{n_claims:,}".replace(",", "{,}")   # 1380 -> "1{,}380"
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{\\textbf{The no-uplift result replicates across agent families.} AUROC per agent "
        "on the Gemma-3-27B-IT claims (final evaluation dataset, $n = " + n_tex + "$), with GPT-5.5 "
        "and Gemini-3.1-Pro replacing Opus 4.8 as the auditing agent, and the paired $\\Delta$AUROC "
        "of each tool agent against the same model's transcript-only baseline. No tool beats its own "
        "baseline for any agent; Gemini is actively hurt by the NLA (analyzed below).}",
        "\\label{tab:crossmodel}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=DATA / "results")
    ap.add_argument("--out", type=Path, default=HERE / "SUMMARY.md")
    args = ap.parse_args()

    # The paper's FINAL evaluation dataset: the quality slice, from the canonical definition.
    keep = quality_ids(GEMMA_RUN_DIR)
    slice_desc = (f"the paper's FINAL dataset: quality slice "
                  f"(concreteness ∩ cf-correct ∩ not-confounded, {len(keep)} claims)")

    def restrict(ids):
        return [c for c in ids if c in keep]

    # (arm, model) -> (path, rows). All three models live in results/, on the same hardened tool.
    found: dict[tuple[str, str], tuple[Path, dict[str, dict]]] = {}
    paths = sorted(glob.glob(str(args.results_dir / "*.jsonl")))
    for p in paths:
        stem = Path(p).stem
        arm, model = parse_arm(stem), parse_model(stem)
        if arm is None or model is None:
            continue  # e.g. cf_toolcalls log, or an answerer we don't track
        found[(arm, model)] = (Path(p), load_rows(Path(p)))

    present_arms = [a for a in ARMS if any((a, m) in found for m in MODELS)]
    present_models = [m for m in MODELS if any((a, m) in found for a in ARMS)]

    L = ["# Cross-model binary counterfactual eval — GEMMA-3-27B test set\n",
         "Answerers: GPT-5.5 + Gemini-3.1-Pro (OpenRouter, reasoning effort=high) vs the canonical "
         "Opus-4.8 arms, on the same interp tools.\n",
         f"**Population:** {slice_desc}.\n",
         f"Results dir: `{args.results_dir}` (all three models, same hardened selection tool).\n",
         "## Per (model, arm)", "",
         "| model | arm | n | acc | AUROC | Brier | forced-submit |",
         "|---|---|---|---|---|---|---|"]
    for model in present_models:
        for arm in present_arms:
            if (arm, model) not in found:
                continue
            rows = found[(arm, model)][1]
            m = metrics([rows[c] for c in restrict(sorted(rows))])
            au = "n/a" if m["auroc"] is None else f"{m['auroc']:.3f}"
            L.append(f"| {model} | {arm} | {m['n']} | {m['acc']:.1%} | {au} | "
                     f"{m['brier']:.3f} | {m['forced']:.1%} |")

    # ---- cross-model AUROC on the common claim set per arm, restricted to the slice ----
    L += ["", "## Cross-model per arm (common claim set)", "",
          "Each arm scored on the claim ids COMMON to all answerers present for that arm (then "
          "restricted to the population above), so the columns are directly comparable. "
          "AUROC (acc in parens).", ""]
    header = "| arm | n_common | " + " | ".join(present_models) + " |"
    L += [header, "|" + "---|" * (len(present_models) + 2)]
    auroc_by = {}  # (arm, model) -> auroc on the common+slice set, for the uplift table
    for arm in present_arms:
        models_here = [m for m in present_models if (arm, m) in found]
        common = restrict(sorted(set.intersection(*[set(found[(arm, m)][1]) for m in models_here])))
        cells = []
        for m in present_models:
            if (arm, m) not in found:
                cells.append("—"); continue
            mm = metrics([found[(arm, m)][1][c] for c in common])
            auroc_by[(arm, m)] = mm["auroc"]
            au = "n/a" if mm["auroc"] is None else f"{mm['auroc']:.3f}"
            cells.append(f"{au} ({mm['acc']:.0%})")
        L.append(f"| {arm} | {len(common)} | " + " | ".join(cells) + " |")

    # ---- paired uplift: tool AUROC − that model's plain AUROC (the paper's ΔAUROC) ----
    tool_arms = [a for a in present_arms if a != "plain"]
    if "plain" in present_arms and tool_arms:
        L += ["", "## Uplift: ΔAUROC vs the same model's transcript-only baseline", "",
              "| tool | " + " | ".join(present_models) + " |",
              "|" + "---|" * (len(present_models) + 1)]
        for arm in tool_arms:
            cells = []
            for m in present_models:
                if (arm, m) in auroc_by and ("plain", m) in auroc_by \
                        and auroc_by[(arm, m)] is not None and auroc_by[("plain", m)] is not None:
                    cells.append(f"{auroc_by[(arm, m)] - auroc_by[('plain', m)]:+.3f}")
                else:
                    cells.append("—")
            L.append(f"| {arm} | " + " | ".join(cells) + " |")

    # ---- Gemini/Opus calibration + flips + forced-submit (the app:crossmodel narrative) ----
    L += ["", "## Calibration & tool-flip analysis (nla vs plain)", "",
          "Mean predicted P(true) by ground-truth label, on the slice. On FALSE (inert) claims "
          "lower is better; a tool that pushes P(true) UP on false claims is miscalibrating.", "",
          "| model | arm | mean P(true) FALSE | mean P(true) TRUE |",
          "|---|---|---|---|"]
    for m in ["gemini-3.1-pro", "opus-4.8"]:
        for arm in ["plain", "nla"]:
            if (arm, m) not in found:
                continue
            rows = found[(arm, m)][1]
            ids = restrict(sorted(rows))
            F = [rows[c]["prob_true"] for c in ids if not rows[c]["is_true"]]
            T = [rows[c]["prob_true"] for c in ids if rows[c]["is_true"]]
            L.append(f"| {m} | {arm} | {sum(F)/len(F):.3f} (n={len(F)}) | "
                     f"{sum(T)/len(T):.3f} (n={len(T)}) |")

    if ("plain", "gemini-3.1-pro") in found and ("nla", "gemini-3.1-pro") in found:
        gp = found[("plain", "gemini-3.1-pro")][1]
        gn = found[("nla", "gemini-3.1-pro")][1]
        ids = restrict(sorted(set(gp) & set(gn)))
        hurt = [c for c in ids if gp[c]["correct"] and not gn[c]["correct"]]
        helped = [c for c in ids if not gp[c]["correct"] and gn[c]["correct"]]
        false_hurt = [c for c in hurt if not gn[c]["is_true"]]
        conf_wrong = [c for c in hurt
                      if (not gn[c]["is_true"] and gn[c]["prob_true"] >= 0.8)
                      or (gn[c]["is_true"] and gn[c]["prob_true"] <= 0.2)]
        L += ["", f"**Gemini NLA plain→nla flips** (on {len(ids)} claims scored by both): "
                  f"{len(hurt)} correct→wrong, of which {len(false_hurt)} are FALSE claims and "
                  f"{len(conf_wrong)} are confidently wrong (≥0.8 on the wrong side); "
                  f"{len(helped)} wrong→correct; net tool effect {len(helped) - len(hurt):+d} claims."]

    L += ["", "## Forced-submit rate (agent used its full tool-call budget)", "",
          "| model | " + " | ".join(a for a in tool_arms) + " |",
          "|" + "---|" * (len(tool_arms) + 1)]
    for m in present_models:
        cells = []
        for arm in tool_arms:
            if (arm, m) not in found:
                cells.append("—"); continue
            rows = found[(arm, m)][1]
            ids = restrict(sorted(rows))
            fr = sum(rows[c].get("stop_reason") == "submitted_final" for c in ids) / len(ids)
            cells.append(f"{fr:.1%}")
        L.append(f"| {m} | " + " | ".join(cells) + " |")

    # ---- copy-paste LaTeX for tab:crossmodel (regenerates the paper table) ----
    latex = emit_latex(auroc_by, len(keep))
    tex_path = args.out.parent / "crossmodel_table.tex"
    tex_path.write_text(latex)
    L += ["", "## LaTeX for `tab:crossmodel` (copy-paste into main.tex)", "",
          f"Also written to `{tex_path.name}`. Regenerates from this script, so the paper table "
          "and these numbers can never drift.", "", "```latex", latex.rstrip("\n"), "```"]

    text = "\n".join(L) + "\n"
    args.out.write_text(text)
    print(text)
    print(f"-> {args.out}")
    print(f"-> {tex_path}")


if __name__ == "__main__":
    main()
