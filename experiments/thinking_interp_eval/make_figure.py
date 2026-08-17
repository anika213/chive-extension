"""Appendix figure (app:cot): interpretability tools give no uplift on the reasoning-mode
(thinking-enabled) Qwen3-8B run.

Writes thinking_overall_score.png into this script's plots/ dir (the self-contained
convention used by main_counterfactual_eval/analyze.py); paper/collect_figures.sh copies
it into paper/figures/. Reuses the main interp-eval figure's plotting code so the style,
metrics, and slice are identical to Figure fig:interp_eval_main: the paper quality slice
(concreteness >= 3 AND cf-correct AND not-confounded) and the `cf` arm dropped from the
bars (it is near-ceiling on this slice by construction).

Run from the repo root:
    PYTHONPATH=. python -m experiments.thinking_interp_eval.make_figure
"""
import os

from chive import paths
from experiments.common.gold_slice import quality_funnel
from experiments.main_counterfactual_eval import analyze

RUN = "opus_investigates_qwen3_8b_think1024_5000_thorough_n30"
HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")


def main():
    rd = os.path.join(str(paths.RUNS_DIR), RUN)
    arms = analyze.discover_arms(rd)
    assert arms, f"no scored arms in {rd}"
    common = analyze.common_ids(arms)
    ids, funnel = quality_funnel(rd, restrict_to=common)
    print(f"run: {RUN}")
    print(f"arms discovered: {[a['cat'] for a in arms]}")
    print(f"common claims: {len(common)}  quality slice: {len(ids)}  funnel: {funnel}")
    print("AUROC per arm on the quality slice (cf is near-ceiling by construction):")
    for a in arms:
        y = analyze.arr(a, ids, "is_true").astype(int)
        p = analyze.arr(a, ids, "prob_true")
        print(f"  {a['display']:24s} n={len(ids):5d}  AUROC={analyze.auroc(y, p):.3f}")

    # redirect the shared plotter's output to this script's plots/ dir; drop cf as the main fig does
    analyze.OUT = PLOTS
    os.makedirs(PLOTS, exist_ok=True)
    analyze.plot_overall(arms, ids, "thinking", "qwen3-8b (reasoning)", drop_cats={"cf"})


if __name__ == "__main__":
    main()
