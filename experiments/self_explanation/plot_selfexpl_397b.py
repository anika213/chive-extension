"""Counterfactual-simulatability bar chart for the Qwen3.5-397B-A17B target (appendix).

Companion to plot_selfexpl_by_target.py (the Qwen3-8B figure) and identical in style —
the shared bar_figure() lives there. Single WildChat panel (`q397_5000`); PETRI was only
run for the 397B target, so we keep this figure symmetric with the WildChat-only 8B one.

Run: PYTHONPATH=. python experiments/self_explanation/plot_selfexpl_397b.py
Writes plots/figure_selfexpl_397b.png.
"""
from __future__ import annotations

from pathlib import Path

from experiments.self_explanation.analyze import analyze_run
from experiments.self_explanation.plot_selfexpl_by_target import bar_figure

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots" / "figure_selfexpl_397b.png"


def main() -> None:
    bar_figure(analyze_run("q397_5000"), OUT)


if __name__ == "__main__":
    main()
