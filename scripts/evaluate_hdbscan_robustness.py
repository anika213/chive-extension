"""HDBSCAN resampling/model checks, always reporting shared assignment coverage."""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
os.environ.setdefault('NUMBA_NUM_THREADS','4');os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'.cache/numba'))
import numpy as np
import umap,hdbscan
from sklearn.metrics import adjusted_rand_score
SRC=ROOT/'results/cause_facets/petri/qwen3_8b_petri_eval';OUT=ROOT/'results/cause_evaluation/hdbscan'
x=np.load(SRC/'embeddings.npy');reference=np.load(OUT/'primary_labels.npy');n=len(x)
def fit(z):
 y=umap.UMAP(n_neighbors=30,n_components=10,min_dist=0,metric='cosine',random_state=42,n_jobs=1).fit_transform(z)
 return hdbscan.HDBSCAN(min_cluster_size=10,min_samples=5,metric='euclidean',cluster_selection_method='eom',core_dist_n_jobs=1).fit_predict(y)
def compare(a,b):
 shared=(a>=0)&(b>=0);union=(a>=0)|(b>=0)
 valid=shared.sum()>1 and len(np.unique(a[shared]))>1 and len(np.unique(b[shared]))>1
 return {'ari_shared_assigned':float(adjusted_rand_score(a[shared],b[shared])) if valid else None,
         'ari_all_including_noise':float(adjusted_rand_score(a,b)),
         'shared_assigned_count':int(shared.sum()),'shared_assigned_fraction':float(shared.mean()),
         'assignment_jaccard':float(shared.sum()/union.sum()) if union.sum() else None}
alt=np.load(ROOT/'results/cause_evaluation/null_stability/minilm_embeddings.npy');al=fit(alt)
np.save(OUT/'minilm_labels.npy',al)
model={'models':['all-mpnet-base-v2','all-MiniLM-L6-v2'],'same_pipeline_and_seed':True,
       'minilm_clusters':len(set(al)-{-1}),'minilm_assigned':int((al>=0).sum()),**compare(reference,al)}
rows=[]
for rep in range(20):
 ids=np.sort(np.random.default_rng(6000+rep).choice(n,size=round(n*.8),replace=False));labels=fit(x[ids])
 rows.append({'replicate':rep,'n_subsample':len(ids),'n_clusters':len(set(labels)-{-1}),
              'assigned':int((labels>=0).sum()),**compare(reference[ids],labels)})
 print('Subsample',rep,flush=True)
np.save(OUT/'minilm_labels.npy',al)
(OUT/'robustness_runs.json').write_text(json.dumps(rows,indent=2)+'\n')
summary={'cross_model':model,'subsampling':'20 independent 80% subsets, no replacement; fixed UMAP/HDBSCAN settings and seed42; compared with full-data preset restricted to same IDs.',
         'subsample_vs_reference':{}}
for field in ['ari_shared_assigned','ari_all_including_noise','shared_assigned_fraction','assignment_jaccard','n_clusters','assigned']:
 values=[r[field] for r in rows if r[field] is not None]
 summary['subsample_vs_reference'][field]={'mean':float(np.mean(values)),'q05':float(np.quantile(values,.05)),'q95':float(np.quantile(values,.95))}
(OUT/'robustness_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
