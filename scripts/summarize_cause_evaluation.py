"""Assemble observed/null, stability, and density-clustering evaluation plots."""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=ROOT/'results/cause_evaluation'
w=json.loads((R/'null_stability/summary.json').read_text())
h=json.loads((R/'hdbscan/summary.json').read_text());rob=json.loads((R/'hdbscan/robustness_summary.json').read_text())
nulls=json.loads((R/'hdbscan/null_pipeline_runs.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12,9))
fig.subplots_adjust(left=.1,right=.96,top=.82,bottom=.12,hspace=.6,wspace=.32)
fig.suptitle('PETRI cause facets: structure, stability, and coverage',x=.1,y=.96,ha='left',fontsize=20,fontweight='bold')
fig.text(.1,.91,'Same 801 investigations · verification ≥7 · investigator-supported statements, not verified causes',fontsize=11)
fig.text(.1,.865,'Null tests support embedding structure; stability and reading limit claims about the exact groups.',fontsize=11)
a=axes[0,0]
observed=w['observed']['maximum'];means=[observed];low=[observed];high=[observed]
for key in ['isotropic_unit_sphere','column_permutation_renormalized']:
 d=w['embedding_nulls'][key]['selection_aware_max_over_k2_to_800']['null_distribution'];means.append(d['mean']);low.append(d['min']);high.append(d['max'])
a.bar([0,1,2],means,color=['#087e83','#a5b7c4','#a5b7c4'],width=.55)
a.errorbar([0,1,2],means,yerr=[np.array(means)-low,np.array(high)-means],fmt='none',color='#243f4d',capsize=5)
for i,v in enumerate(means):a.text(i,v+.006,f'{v:.3f}',ha='center')
a.set_xticks([0,1,2],['Observed','Random unit\nvectors','Permuted\ncoordinates']);a.set_ylim(0,.16)
a.set_ylabel('Best cosine silhouette');a.set_title('Ward: repeat the full k = 2–800 search\n20 null replicates each; whiskers show min–max',loc='left',fontsize=11)
a=axes[0,1]
xx=np.arange(3);names=['Bootstrap','80% subsample','Different model']
for offset,key,color in [(-.18,'k50','#729aaf'),(.18,'k336','#087e83')]:
 vals=[w['resampling']['bootstrap_with_replacement'][key]['ari_vs_full']['mean'],w['resampling']['subsample80_fixed_k'][key]['ari_vs_full']['mean'],w['alternate_model']['comparison'][key]['ari_vs_mpnet']]
 a.bar(xx+offset,vals,width=.34,label=key,color=color)
 for pos,v in zip(xx+offset,vals):a.text(pos,v+.025,f'{v:.2f}',ha='center',fontsize=9)
a.set_xticks(xx,names);a.set_ylim(0,1);a.set_ylabel('Adjusted Rand index');a.legend(frameon=False,loc='upper left')
a.set_title('Ward: agreement with the reference partition\nResampling means; model comparison is MPNet vs MiniLM',loc='left',fontsize=11)
a=axes[1,0]
aris=[h['seed_stability']['ari_shared_assigned']['mean'],rob['subsample_vs_reference']['ari_shared_assigned']['mean'],rob['cross_model']['ari_shared_assigned']]
coverage=[h['seed_stability']['shared_assigned_fraction']['mean'],rob['subsample_vs_reference']['shared_assigned_fraction']['mean'],rob['cross_model']['shared_assigned_fraction']]
a.bar(xx-.18,aris,width=.34,label='ARI on shared assigned items',color='#087e83');a.bar(xx+.18,coverage,width=.34,label='Shared assigned fraction',color='#d5a261')
for i,(v,c) in enumerate(zip(aris,coverage)):
 a.text(i-.18,v+.025,f'{v:.2f}',ha='center',fontsize=9);a.text(i+.18,c+.025,f'{c:.0%}',ha='center',fontsize=9)
a.set_xticks(xx,['UMAP seeds','80% subsample','Different model']);a.set_ylim(0,1.15);a.set_ylabel('ARI / fraction');a.legend(frameon=False,fontsize=8,loc='upper right')
a.set_title('UMAP + HDBSCAN: stable core, changing coverage\nSeed pairs; subsamples vs reference; model pair',loc='left',fontsize=11)
a=axes[1,1]
vals=[h['primary']['original_cosine_silhouette_assigned']];mins=[vals[0]];maxs=[vals[0]]
for kind in ['isotropic','column_permutation']:
 scores=[r['original_cosine_silhouette_assigned'] for r in nulls if r['kind']==kind and r['original_cosine_silhouette_assigned'] is not None]
 vals.append(np.mean(scores));mins.append(min(scores));maxs.append(max(scores))
a.bar([0,1,2],vals,color=['#087e83','#a5b7c4','#a5b7c4'],width=.55)
a.errorbar([0,1,2],vals,yerr=[np.array(vals)-mins,np.array(maxs)-vals],fmt='none',color='#243f4d',capsize=5)
for i,v in enumerate(vals):a.text(i,max(v,0)+.005,f'{v:.3f}',ha='center')
a.axhline(0,color='#777',lw=.8);a.set_xticks([0,1,2],['Observed','Random unit\nvectors','Permuted\ncoordinates']);a.set_ylim(-.015,.07);a.set_ylabel('Original-space cosine silhouette')
a.set_title('UMAP + HDBSCAN: assigned records only\n10 null runs each; preset settings; noise excluded',loc='left',fontsize=11)
fig.text(.1,.055,'Preset HDBSCAN: 18 clusters, 563 assigned (70.3%), 238 unclustered. Reading reveals both causal phrasing and topical groups.',fontsize=10)
for ext in ['png','pdf','svg']:fig.savefig(R/f'evaluation_overview.{ext}',dpi=180,facecolor='white')
plt.close(fig)
# Validate generated memberships and review samples.
src=ROOT/'results/cause_facets/petri/qwen3_8b_petri_eval'
records=[json.loads(l) for l in (src/'causes.jsonl').read_text().splitlines()]
labels=np.load(R/'hdbscan/primary_labels.npy')
assert len(labels)==len(records)==801 and int((labels>=0).sum())==h['primary']['assigned']
ids={r['prompt_id']:i for i,r in enumerate(records)}
read=json.loads((R/'reading/umap_hdbscan_preset.json').read_text())
assert len(read['five_members_each'])==10
for c in read['five_members_each']:
 assert len(c['random_members'])==5 and len({m['prompt_id'] for m in c['random_members']})==5
 assert all(labels[ids[m['prompt_id']]]==c['cluster_id'] for m in c['random_members'])
assert len(json.loads((R/'hdbscan/parameter_runs.json').read_text()))==90
assert len(json.loads((R/'hdbscan/null_pipeline_runs.json').read_text()))==20
(R/'validation.json').write_text(json.dumps({'n':801,'hdbscan_memberships_valid':True,'reading_samples_valid':True,'hdbscan_parameter_runs':90,'hdbscan_null_runs':20,'note':'Ward null/stability validation in null_stability/validation.json'},indent=2)+'\n')
print('Overview plots and checks complete')
