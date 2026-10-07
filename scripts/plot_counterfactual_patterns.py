"""Descriptive outcome plots and explicitly annotated intervention features."""
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

BASE = ROOT / 'results/scenario_clustering/five_scenario_explanation_assessments/cluster_0_counterfactual_review'
OUT = BASE / 'three_scenarios'
rows = json.loads((BASE/'experiments.json').read_text())
cases = [('petri_medical_authority_sycophancy_e2_t4','Medical authority · e2_t4',-1),
         ('petri_gangstalking_paranoia_sycophancy_e2_t3','Gangstalking · e2_t3',-1),
         ('petri_manic_episode_sycophancy_e1_t1','Manic episode · e1_t1',1)]
fig, axes = plt.subplots(1,3,figsize=(14,7),sharex=True)
fig.subplots_adjust(left=.06,right=.98,top=.78,bottom=.19,wspace=.30)
for ax,(pid,title,sign) in zip(axes,cases):
    local = sorted([r for r in rows if r['prompt_id']==pid],key=lambda r:sign*r['delta_pp'],reverse=True)
    for i,r in enumerate(local):
        improvement=sign*r['delta_pp']; color='#087f8c' if improvement>50+1e-8 else '#8393a5'
        ax.plot([0,improvement],[i,i],color=color,alpha=.4,lw=2)
        ax.scatter(improvement,i,s=52,color=color,zorder=4)
    ax.axvline(50,color='#c77518',ls='--',lw=1.5)
    ax.axvline(0,color='#b3beca',lw=1)
    ax.set_yticks(range(len(local)),[f"Exp. {r['experiment_number']}" for r in local],fontsize=9)
    ax.invert_yaxis();ax.set_title(title,fontsize=12,pad=12)
    ax.set_xlim(-25,110);ax.set_xticks([-20,0,25,50,75,100]);ax.set_xlabel('Improvement (percentage points)',fontsize=10)
    ax.grid(axis='x',color='#edf0f4');ax.tick_params(length=0)
    for spine in ax.spines.values():spine.set_visible(False)
fig.text(.06,.935,'How much does each counterfactual help?',fontsize=22,weight='bold',color='#152b3b')
fig.text(.06,.88,'All 44 tests in the three selected investigations, ordered by measured improvement within each case.',fontsize=11,color='#526174')
fig.legend(handles=[Line2D([0],[0],marker='o',color='#087f8c',lw=0,label='Improvement >50 pp'),Line2D([0],[0],marker='o',color='#8393a5',lw=0,label='Other tests'),Line2D([0],[0],color='#c77518',ls='--',label='50 pp threshold')],loc='lower left',bbox_to_anchor=(.055,.07),frameon=False,ncol=3)
fig.text(.06,.04,'Positive = less premise agreement / fewer validating checklists / more rest advice, respectively. Estimates use 30 responses per condition.',fontsize=9,color='#526174')
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'experiment_improvements.{ext}',dpi=180,facecolor='white')
plt.close(fig)

# Human-readable analyst annotations of actual stored edits; these are not learned clusters or causal labels.
features=[
    {'prompt_id':cases[0][0],'title':cases[0][1],
     'columns':['Remove\nhistory','Remove stated\ncredentials','Replace safety\nassertion','Attribute claim\nto colleague'],
     'experiments':[2,5,6,8,12], 'matrix':[[1,1,0,0],[0,0,1,0],[1,0,1,0],[0,0,1,0],[0,0,0,1]],
     'note':'Different edits touch different factors;\nseveral change more than one.'},
    {'prompt_id':cases[1][0],'title':cases[1][1],
     'columns':['Remove\nchecklist request','Explicit psychological /\nclinical framing','Ask about seeing\na psychiatrist','Mention\ntherapist advice'],
     'experiments':[3,7,12,13,15], 'matrix':[[1,0,0,0],[1,1,0,0],[1,1,1,0],[1,1,0,1],[1,0,0,0]],
     'note':'All five remove the checklist request;\nadditional clinical cues vary.'},
    {'prompt_id':cases[2][0],'title':cases[2][1],
     'columns':['Direct sleep /\njudgment question','Ask about\na friend','Increase severity /\nrisky plans','System instruction\nto recommend rest','Add partner’s\nconcern'],
     'experiments':[4,6,8,9,14], 'matrix':[[1,0,0,0,0],[1,0,0,0,0],[0,1,1,0,0],[0,0,0,1,0],[0,0,0,0,1]],
     'note':'Several intervention types elicit rest advice.\nThe friend-framing edit bundles changes.'},
]
fig,axes=plt.subplots(1,3,figsize=(16,7),gridspec_kw={'width_ratios':[4,4,5]})
fig.subplots_adjust(left=.055,right=.98,top=.73,bottom=.33,wspace=.35)
for ax,f in zip(axes,features):
    mat=np.array(f['matrix']);ax.imshow(mat,cmap=ListedColormap(['#edf1f5','#087f8c']),vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(len(f['columns'])),f['columns'],rotation=35,ha='right',fontsize=9)
    ax.set_yticks(range(5),[f'Exp. {n}' for n in f['experiments']],fontsize=10)
    for i in range(5):
        for j in range(mat.shape[1]):
            if mat[i,j]:ax.text(j,i,'●',ha='center',va='center',color='white',fontsize=12)
    ax.set_xticks(np.arange(-.5,mat.shape[1],1),minor=True);ax.set_yticks(np.arange(-.5,5,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=3);ax.tick_params(which='both',length=0)
    ax.set_title(f['title'],fontsize=12,pad=15)
    for spine in ax.spines.values():spine.set_visible(False)
    for n in f['experiments']:
        r=next(r for r in rows if r['prompt_id']==f['prompt_id'] and r['experiment_number']==n)
        assert abs(r['delta_pp'])>50+1e-8
fig.text(.055,.94,'What do the five successful edits have in common?',fontsize=23,weight='bold',color='#152b3b')
fig.text(.055,.885,'Each row is an experiment. Each filled cell marks a change present in that experiment. Columns are specific to each case.',fontsize=11,color='#526174')
fig.text(.055,.84,'Analyst annotations of the actual prompt edits — descriptive features, not inferred causal mechanisms.',fontsize=11,color='#526174')
for ax,f in zip(axes,features):fig.text(ax.get_position().x0,.145,f['note'],fontsize=10,color='#152b3b')
fig.text(.055,.055,'Selected successes only; no claim of necessity or sufficiency. Some broad edits were excluded from clustering. Annotations are illustrative, not exhaustive.',fontsize=9,color='#526174')
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'successful_edit_features.{ext}',dpi=180,facecolor='white')
plt.close(fig)
(OUT/'successful_edit_features.json').write_text(json.dumps({'annotation_method':'Analyst coding of actual prompt diffs, illustrative and non-exhaustive; not cluster labels or causal estimates','cases':features},indent=2)+'\n')
html='''<!doctype html><html><head><meta charset="utf-8"><title>Counterfactual overview plots</title><style>body{font-family:system-ui;max-width:1400px;margin:40px auto;padding:0 24px;color:#152b3b}img{width:100%;border:1px solid #e5eaf0;margin:20px 0}p{line-height:1.6}a{color:#087f8c}</style></head><body><h1>Counterfactual overview: three selected investigations</h1><p><a href="review.html">Review the 15 successful experiments and exact prompt diffs</a>. High improvement means strictly more than 50 percentage points in the intended direction. These cases were selected for having at least five large effects.</p>'''
for name in ['high_improvement_counts','experiment_improvements','successful_edit_features']:
    html+=f'<img src="{name}.png" alt="{name.replace("_"," ")}"><p>Download: <a href="{name}.svg">SVG</a> · <a href="{name}.pdf">PDF</a></p>'
html+='</body></html>'
(OUT/'plots.html').write_text(html)
print(OUT/'plots.html')
