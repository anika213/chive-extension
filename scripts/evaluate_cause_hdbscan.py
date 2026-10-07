"""UMAP/HDBSCAN sensitivity and seed stability for the same PETRI cause facets."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
os.environ.setdefault('NUMBA_NUM_THREADS','4')
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'.cache/numba'))
import csv,json,itertools
import numpy as np
import umap,hdbscan
from sklearn.metrics import adjusted_rand_score,silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
SRC=ROOT/'results/cause_facets/petri/qwen3_8b_petri_eval'
OUT=ROOT/'results/cause_evaluation/hdbscan';OUT.mkdir(parents=True,exist_ok=True)
X=np.load(SRC/'embeddings.npy');N=len(X)
RECORDS=[json.loads(l) for l in (SRC/'causes.jsonl').read_text().splitlines()]
D=pairwise_distances(X,metric='cosine');np.fill_diagonal(D,0)
SEEDS=[13,42,77,101,202]
PRIMARY={'neighbors':30,'min_cluster_size':10,'min_samples':5,'seed':42}

def save(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def stats(labels,y,distances):
    mask=labels>=0;unique,sizes=np.unique(labels[mask],return_counts=True);count=int(mask.sum())
    valid=1<len(unique)<count
    return {'n_clusters':len(unique),'assigned':count,'noise':int((~mask).sum()),'coverage':float(mask.mean()),
            'largest_cluster':int(sizes.max()) if len(sizes) else 0,
            'largest_cluster_fraction_of_assigned':float(sizes.max()/count) if count else None,
            'original_cosine_silhouette_assigned':float(silhouette_score(distances[np.ix_(mask,mask)],labels[mask],metric='precomputed')) if valid else None,
            'umap_euclidean_silhouette_assigned':float(silhouette_score(y[mask],labels[mask])) if valid else None}

rows=[];all_labels={};primary_y=None
for neighbors in [15,30,50]:
    for seed in SEEDS:
        yp=OUT/f'umap_neighbors{neighbors}_seed{seed}.npy'
        if yp.exists():y=np.load(yp)
        else:
            y=umap.UMAP(n_neighbors=neighbors,n_components=10,min_dist=0,metric='cosine',random_state=seed,n_jobs=1).fit_transform(X)
            np.save(yp,y)
        print('UMAP',neighbors,seed,flush=True)
        for mcs,ms in itertools.product([10,20,30],[5,10]):
            model=hdbscan.HDBSCAN(min_cluster_size=mcs,min_samples=ms,metric='euclidean',cluster_selection_method='eom',core_dist_n_jobs=1).fit(y)
            labels=model.labels_;key=f'n{neighbors}_mcs{mcs}_ms{ms}_s{seed}'
            all_labels[key]=labels
            row={'neighbors':neighbors,'min_cluster_size':mcs,'min_samples':ms,'seed':seed,**stats(labels,y,D)}
            rows.append(row)
            if all(row[k]==v for k,v in PRIMARY.items()):
                primary_y=y
                np.save(OUT/'primary_labels.npy',labels)
                np.save(OUT/'primary_probabilities.npy',model.probabilities_)
                np.save(OUT/'primary_umap.npy',y)
np.savez_compressed(OUT/'all_labels.npz',**all_labels)
save(OUT/'parameter_runs.json',rows)
with (OUT/'parameter_runs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
stability=[]
for neighbors,mcs,ms in itertools.product([15,30,50],[10,20,30],[5,10]):
    for a,b in itertools.combinations(SEEDS,2):
        la=all_labels[f'n{neighbors}_mcs{mcs}_ms{ms}_s{a}'];lb=all_labels[f'n{neighbors}_mcs{mcs}_ms{ms}_s{b}']
        shared=(la>=0)&(lb>=0);union=(la>=0)|(lb>=0)
        valid=shared.sum()>1 and len(np.unique(la[shared]))>1 and len(np.unique(lb[shared]))>1
        stability.append({'neighbors':neighbors,'min_cluster_size':mcs,'min_samples':ms,'seed_a':a,'seed_b':b,
                          'ari_including_noise_as_one_label':float(adjusted_rand_score(la,lb)),
                          'ari_shared_assigned':float(adjusted_rand_score(la[shared],lb[shared])) if valid else None,
                          'shared_assigned':int(shared.sum()),'shared_assigned_fraction':float(shared.mean()),
                          'assigned_jaccard':float(shared.sum()/union.sum()) if union.sum() else None})
save(OUT/'seed_stability.json',stability)
labels=np.load(OUT/'primary_labels.npy');probs=np.load(OUT/'primary_probabilities.npy')
with (OUT/'primary_assignments.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['prompt_id','cluster_id','membership_strength'])
    for r,label,prob in zip(RECORDS,labels,probs):w.writerow([r['prompt_id'],int(label),float(prob)])
# Apply the same fixed pipeline to unit-normal random embeddings; no parameter tuning.
nulls=[]
for kind in ['isotropic','column_permutation']:
    for rep in range(10):
        rng=np.random.default_rng(5000+rep)
        z=normalize(rng.normal(size=X.shape)).astype(np.float32) if kind=='isotropic' else normalize(np.column_stack([rng.permutation(X[:,j]) for j in range(X.shape[1])]))
        y=umap.UMAP(n_neighbors=30,n_components=10,min_dist=0,metric='cosine',random_state=42,n_jobs=1).fit_transform(z)
        model=hdbscan.HDBSCAN(min_cluster_size=10,min_samples=5,metric='euclidean',cluster_selection_method='eom',core_dist_n_jobs=1).fit(y)
        dz=pairwise_distances(z,metric='cosine');np.fill_diagonal(dz,0)
        nulls.append({'kind':kind,'replicate':rep,**stats(model.labels_,y,dz)})
        print('Null',kind,rep,flush=True)
save(OUT/'null_pipeline_runs.json',nulls)
primary=next(r for r in rows if all(r[k]==v for k,v in PRIMARY.items()))
same=[r for r in stability if all(r[k]==PRIMARY[k] for k in ['neighbors','min_cluster_size','min_samples'])]
def summarize(field):
    values=[r[field] for r in same if r[field] is not None]
    return {'mean':float(np.mean(values)),'min':float(np.min(values)),'max':float(np.max(values))} if values else None
result={'n':N,'scope':'PETRI investigator-supported cause statements, verification >=7; not verified causes',
        'primary_selection':'Preset n_neighbors=30, min_cluster_size=10, min_samples=5, seed=42; not selected by silhouette',
        'umap':{'dimensions':10,'min_dist':0,'metric':'cosine'},'hdbscan':{'metric':'euclidean','selection':'eom'},
        'primary':primary,'seed_stability':{field:summarize(field) for field in ['ari_shared_assigned','ari_including_noise_as_one_label','shared_assigned_fraction','assigned_jaccard']},
        'sensitivity_runs':len(rows),'null_replicates_per_kind':10,
        'interpretation':'Evaluate original-space silhouette with coverage; reduced-space silhouette can be inflated by UMAP. Noise is not a semantic cluster; ARI including noise may be dominated by common rejections.'}
save(OUT/'summary.json',result)
assert len(RECORDS)==len(labels)==801 and np.isfinite(primary_y).all()
print(json.dumps(result,indent=2),flush=True)
