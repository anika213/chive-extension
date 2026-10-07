"""Evaluate every valid PETRI hierarchy cut, preserving the original analysis."""
import csv,json,os
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import cut_tree
from sklearn.metrics import pairwise_distances,silhouette_samples
ROOT=Path(__file__).resolve().parent.parent
SRC=ROOT/'results/behaviour_facets/petri/qwen3_8b_petri_eval'
OUT=SRC/'extended_sweep';OUT.mkdir(exist_ok=True)
x=np.load(SRC/'embeddings.npy');tree=np.load(SRC/'linkage.npy');n=len(x)
assert n==801 and np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-5)
ks=np.arange(2,n)
labels=cut_tree(tree,n_clusters=ks)
d=pairwise_distances(x,metric='cosine');np.fill_diagonal(d,0)
rows=[]
for j,k in enumerate(ks):
    s=silhouette_samples(d,labels[:,j],metric='precomputed');sizes=np.bincount(labels[:,j])
    rows.append({'k':int(k),'cosine_silhouette':float(s.mean()),'singleton_clusters':int((sizes==1).sum()),
                 'singleton_item_fraction':float((sizes==1).sum()/n),'median_cluster_size':float(np.median(sizes)),
                 'min_cluster_size':int(sizes.min()),'max_cluster_size':int(sizes.max()),
                 'negative_silhouette_fraction':float((s<0).mean())})
with (OUT/'diagnostics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(OUT/'diagnostics.json').write_text(json.dumps(rows,indent=2)+'\n')
best=max(rows,key=lambda r:(r['cosine_silhouette'],-r['k']))
threshold=best['cosine_silhouette']-.005
idx=best['k']-2;lo=hi=idx
while lo>0 and rows[lo-1]['cosine_silhouette']>=threshold:lo-=1
while hi<len(rows)-1 and rows[hi+1]['cosine_silhouette']>=threshold:hi+=1
report={'n':n,'evaluated_k_range':[2,n-1],'n_cuts':len(rows),'best':best,
        'near_peak_definition':'Contiguous range containing peak with silhouette within 0.005 absolute of peak; descriptive tolerance, not statistical confidence.',
        'near_peak_k_range':[rows[lo]['k'],rows[hi]['k']],
        'near_peak_score_floor':threshold,'all_records_scored':True,
        'method':'Saved MPNet L2-normalized embeddings and Ward hierarchy; cosine silhouette; no refitting or text changes.'}
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
records=[json.loads(line) for line in (SRC/'behaviours.jsonl').read_text().splitlines()]
with (SRC/'assignments.csv').open() as f: original=list(csv.DictReader(f))
assert [r['prompt_id'] for r in original]==[r['prompt_id'] for r in records]
with (OUT/'best_assignments.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['prompt_id','cluster_id','k'])
    for r,label in zip(records,labels[:,idx]):w.writerow([r['prompt_id'],int(label),best['k']])
for old in json.loads((SRC/'diagnostics.json').read_text()):
    assert abs(rows[old['k']-2]['cosine_silhouette']-old['cosine_silhouette'])<1e-6
np.savez_compressed(OUT/'all_cuts.npz',k=ks,labels=labels)
print(json.dumps(report,indent=2),flush=True)
print('Selected checkpoints:',json.dumps([r for r in rows if r['k'] in [50,75,100,125,150,175,200,250,300,400,500,600,700,800]],indent=2))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#203541','text.color':'#203541'})
fig,axes=plt.subplots(1,2,figsize=(12,5.8),sharey=True)
fig.subplots_adjust(left=.08,right=.97,top=.75,bottom=.24,wspace=.16)
fig.suptitle('PETRI: silhouette levels off around 200–275 clusters',x=.08,y=.96,ha='left',fontsize=19,fontweight='bold')
fig.text(.08,.89,'801 behaviour summaries · verification ≥ 7 · MPNet + L2 normalization · Ward hierarchy',fontsize=11)
fig.text(.08,.835,'Every cut from k = 2 to 800 scored on all 801 investigations.',fontsize=10,color='#536974')
for ax,limit,title in zip(axes,[350,800],['Detail: approach to the peak','Full sweep: score eventually declines']):
    ax.plot(ks,[r['cosine_silhouette'] for r in rows],color='#087e83',lw=2)
    ax.axvspan(rows[lo]['k'],rows[hi]['k'],color='#087e83',alpha=.12)
    ax.scatter([best['k']],[best['cosine_silhouette']],s=65,color='#d56822',zorder=5)
    ax.annotate('Peak: k = 249, score = 0.204',(best['k'],best['cosine_silhouette']),xytext=(0,15),textcoords='offset points',ha='center',fontsize=10,color='#a44711')
    ax.set_xlim(0,limit);ax.set_ylim(0,.24);ax.set_xlabel('Number of clusters (k)')
    ax.set_title(title,loc='left',fontsize=11,pad=12);ax.grid(axis='y',alpha=.2)
    ax.set_axisbelow(True)
axes[0].set_ylabel('Cosine silhouette (higher is better)')
fig.text(.08,.13,'Shaded band: k = 217–276, within 0.005 of the peak. This is a descriptive tolerance, not a confidence interval.',fontsize=10)
fig.text(.08,.075,'At k = 249: median cluster size = 3; 44 singleton clusters. The score remains modest despite the finer groups.',fontsize=10)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'petri_extended_sweep.{ext}',dpi=180,facecolor='white')
plt.close(fig)
(OUT/'README.md').write_text('''# Extended PETRI behaviour clustering sweep

Same 801 PETRI investigations with verification >=7, saved L2-normalized MPNet
embeddings and Ward hierarchy. No changes to source data or earlier outputs.
Every valid k from 2 to 800 is evaluated using cosine silhouette on all records.
The original candidate scores were reproduced to within 1e-6.

The curve levels off approximately around 200–275 clusters. The maximum is
0.204086 at k=249. A contiguous near-peak range within 0.005 absolute score of
that maximum is k=217–276. That threshold is descriptive, not a statistically
estimated change point or confidence interval. Scores decline after the peak.
At k=249, median cluster size is 3, 44 clusters are singletons, and maximum size
is 16. Higher scores do not demonstrate a broad behavioural taxonomy or
establish verified causes. Singleton silhouettes are zero under sklearn.

Files: diagnostics.json/csv (all cuts), summary.json, all_cuts.npz (k and labels;
row order matches ../behaviours.jsonl), best_assignments.csv (k=249), and plot
PNG/PDF/SVG. Cluster IDs remain numeric and local to this cut.

Reproduce from workspace root:
`clustering/.venv/bin/python clustering/scripts/sweep_petri_clusters.py`
''')
