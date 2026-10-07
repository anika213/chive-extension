"""Plot saved Ward-clustering silhouette diagnostics without rerunning clustering."""
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

results = ROOT / 'results' / 'final_explanations'
runs = [
    ('petri/qwen3_8b_petri_eval', 'PETRI eval · alignment-relevant'),
    ('wildchat/qwen3_8b_wildchat_eval', 'WildChat eval'),
    ('wildchat/qwen3_8b_wildchat_thinking_eval', 'WildChat thinking eval'),
    ('wildchat/qwen3_8b_wildchat_train', 'WildChat train'),
]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#9ba9b3', 'text.color': '#192c39',
                     'axes.labelcolor': '#192c39', 'xtick.color': '#415867',
                     'ytick.color': '#415867', 'pdf.fonttype': 42})
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True, sharey=True)
fig.subplots_adjust(left=.085, right=.97, bottom=.18, top=.80, hspace=.48, wspace=.18)
fig.suptitle('Choosing the number of explanation clusters', x=.085, y=.965,
             ha='left', fontsize=21, fontweight='bold')
fig.text(.085, .915, 'Ward agglomerative clustering · MiniLM embeddings · Higher cosine silhouette is better', fontsize=11)
fig.text(.085, .875, 'Orange points mark the best tested cut in each independently fitted run.', fontsize=10, color='#556b78')
for ax, (run, title) in zip(axes.flat, runs):
    diagnostics = json.loads((results / run / 'diagnostics.json').read_text())
    summary = json.loads((results / run / 'summary.json').read_text())
    ks = [d['k'] for d in diagnostics]
    scores = [d['cosine_silhouette'] for d in diagnostics]
    best = max(diagnostics, key=lambda d: (d['cosine_silhouette'], -d['k']))
    assert best['k'] == summary['selected_k']
    color = '#087e83' if run.startswith('petri/') else '#376b9f'
    ax.axhline(0, color='#8a99a2', linestyle='--', linewidth=1)
    ax.grid(axis='y', color='#e5eaee', linewidth=.8)
    ax.set_axisbelow(True)
    ax.plot(ks, scores, color=color, marker='o', markersize=5, linewidth=2)
    ax.scatter([best['k']], [best['cosine_silhouette']], color='#d56822',
               s=100, zorder=5, edgecolor='white', linewidth=1.5)
    ax.annotate(f"k = {best['k']}  |  score = {best['cosine_silhouette']:.3f}",
                (best['k'], best['cosine_silhouette']),
                xytext=(-8 if best['k']==50 else 8, 12), textcoords='offset points',
                ha='right' if best['k']==50 else 'left', color='#a44711', fontsize=10, fontweight='bold')
    n = summary['n_explanations']
    sample = best['silhouette_sample_size']
    subtitle = f'{n:,} explanations · ' + ('all scored' if sample==n else f'{sample:,} sampled for scoring')
    ax.set_title(title + '\n' + subtitle, loc='left', fontsize=11, pad=13)
    ax.set_xticks(ks)
    ax.set_xlim(2, 53)
    ax.set_ylim(-.026, .115)
    ax.set_yticks([-.02, 0, .02, .04, .06, .08, .10])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(labelbottom=True)
for ax in axes[:, 0]:
    ax.set_ylabel('Cosine silhouette score')
for ax in axes[1, :]:
    ax.set_xlabel('Number of clusters (k)')
fig.text(.085, .09, 'Scores are low: the groupings have weak separation. The shared y-axis is zoomed; silhouette ranges from −1 to 1.', fontsize=9.5)
fig.text(.085, .055, 'k = 50 is the upper limit tested, not a demonstrated optimum. These cluster claimed explanations, not verified causes.', fontsize=9.5)
for extension in ('png', 'pdf', 'svg'):
    path = results / f'cluster_selection.{extension}'
    fig.savefig(path, dpi=180, facecolor='white')
    print(path)
plt.close(fig)
