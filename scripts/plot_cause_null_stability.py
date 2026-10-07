"""Plot saved null/stability diagnostics without recomputing analysis."""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=ROOT/'results/cause_evaluation/null_stability'
SRC=ROOT/'results/cause_facets/petri/qwen3_8b_petri_eval'
curves=np.load(P/'embedding_null_curves.npz');obs=json.loads((SRC/'diagnostics.json').read_text())
fig,ax=plt.subplots(figsize=(10,5))
ax.plot([r['k'] for r in obs],[r['cosine_silhouette'] for r in obs],color='#a24da4',lw=2,label='Observed cause facets')
for family,color,label in [('isotropic_unit_sphere','#517d9a','Isotropic unit-sphere null'),('column_permutation_renormalized','#cc812e','Permuted-coordinate null')]:
 x=curves['scores'][curves['families']==family];ax.plot(curves['k'],x.mean(axis=0),color=color,label=f'{label} (20 refits)')
 ax.fill_between(curves['k'],np.quantile(x,.025,axis=0),np.quantile(x,.975,axis=0),color=color,alpha=.2)
ax.set(xlabel='Number of clusters (k)',ylabel='Cosine silhouette',title='PETRI cause facets exceed both refitted null baselines')
ax.grid(alpha=.2);ax.legend();fig.text(.5,.01,'Same full k=2–800 search per replicate. Shading is the pointwise 2.5–97.5% range, not a simultaneous confidence band.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.04,1,1))
for ext in ['png','pdf','svg']:fig.savefig(P/f'null_sweep_comparison.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
rep=json.loads((P/'resample_replicates.json').read_text());pairs=json.loads((P/'resample_pairwise.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
kinds=['bootstrap_with_replacement','subsample80_fixed_k','subsample80_scaled_k'];names=['Bootstrap','80% fixed k','80% scaled k']
for ax,rows,key,title in [(axes[0],rep,'ari_vs_full_on_shared_ids','Compared with full-data clustering'),(axes[1],pairs,'ari_between_resamples_on_shared_ids','Between resampled clusterings')]:
 values=[];labels=[]
 for kind,name in zip(kinds,names):
  for k in [50,336]:values.append([r[key] for r in rows if r['kind']==kind and r['reference_k']==k]);labels.append(f'{name}\nk={k}')
 bp=ax.boxplot(values,tick_labels=labels,patch_artist=True,showfliers=False)
 for i,b in enumerate(bp['boxes']):b.set_facecolor('#aad6d6' if i%2==0 else '#dfb3df')
 ax.set_title(title);ax.set_ylim(0,1);ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Adjusted Rand index (ARI)')
fig.suptitle('PETRI cause clustering: resampling stability depends on the perturbation')
fig.text(.5,.01,'20 resamples per condition; 190 dependent pairwise comparisons. ARIs use only shared unique IDs. Scaled k: 40 and 269.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.95))
for ext in ['png','pdf','svg']:fig.savefig(P/f'resampling_stability.{ext}',dpi=180,bbox_inches='tight')
