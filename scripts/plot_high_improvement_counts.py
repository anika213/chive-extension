"""Plot descriptive counts for the three previously selected investigations."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BASE = ROOT / 'results/scenario_clustering/five_scenario_explanation_assessments/cluster_0_counterfactual_review'
OUT = BASE / 'three_scenarios'
rows = json.loads((BASE / 'experiments.json').read_text())
cases = [
    ('petri_medical_authority_sycophancy_e2_t4', 'Medical authority', 'e2_t4', -1),
    ('petri_gangstalking_paranoia_sycophancy_e2_t3', 'Gangstalking', 'e2_t3', -1),
    ('petri_manic_episode_sycophancy_e1_t1', 'Manic episode', 'e1_t1', 1),
]
counts = []
for pid, name, checkpoint, direction in cases:
    experiments = [r for r in rows if r['prompt_id'] == pid]
    selected = [r for r in experiments if direction * r['delta_pp'] > 50 + 1e-8]
    counts.append({'prompt_id': pid, 'scenario_label': name, 'checkpoint': checkpoint,
                   'total_experiments': len(experiments), 'high_improvement_experiments': len(selected),
                   'experiment_numbers': [r['experiment_number'] for r in selected],
                   'improvement_direction': direction,
                   'included_despite_previous_clustering_exclusion': sum(not r['clustering_eligible'] for r in selected)})
assert [r['high_improvement_experiments'] for r in counts] == [5,5,5]
fig, ax = plt.subplots(figsize=(10, 5.4), facecolor='white')
fig.subplots_adjust(left=.24, right=.95, top=.75, bottom=.27)
y = range(3)
ax.barh(y, [r['total_experiments'] for r in counts], color='#e5eaf0', height=.5,
        label='All tested counterfactuals', zorder=2)
ax.barh(y, [r['high_improvement_experiments'] for r in counts], color='#087f8c', height=.5,
        label='Improvement >50 percentage points', zorder=3)
for i, r in enumerate(counts):
    ax.text(r['high_improvement_experiments']/2, i, str(r['high_improvement_experiments']),
            color='white', ha='center', va='center', weight='bold', fontsize=14)
    ax.text(r['total_experiments']+.25, i, f"of {r['total_experiments']}",
            color='#465467', va='center', fontsize=11)
ax.set_yticks(list(y), [f"{r['scenario_label']}\n{r['checkpoint']}" for r in counts], fontsize=11)
ax.invert_yaxis()
ax.set_xlim(0,18)
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
ax.set_xlabel('Number of counterfactual experiments', fontsize=11, labelpad=10)
ax.grid(axis='x', color='#edf0f4', zorder=0)
ax.tick_params(axis='both', length=0)
for spine in ax.spines.values(): spine.set_visible(False)
fig.text(.055,.93,'Five high-improvement edits in each selected case', fontsize=19, weight='bold', color='#152b3b')
fig.text(.055,.865,'Three PETRI scenarios · one investigation per scenario · Qwen3-8B', fontsize=11, color='#526174')
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles[::-1], labels[::-1], loc='lower left', bbox_to_anchor=(.05,.115), frameon=False, fontsize=10, ncol=2)
fig.text(.055,.075,'Improvement: less premise agreement / fewer validating checklists / more rest advice, respectively.', fontsize=9, color='#526174')
fig.text(.055,.037,'Includes broad edits excluded from clustering. Cases were selected for ≥5 large effects; not a dataset-wide estimate.', fontsize=9, color='#526174')
for ext in ['png','svg','pdf']:
    fig.savefig(OUT/f'high_improvement_counts.{ext}', dpi=180, facecolor='white')
plt.close(fig)
(OUT/'high_improvement_counts.json').write_text(json.dumps({'threshold_pp_strictly_greater_than':50,'counts':counts},indent=2)+'\n')
print(OUT/'high_improvement_counts.png')
