"""Null baselines and stability for PETRI cause facets; no semantic cause labels.
Reproduce: clustering/.venv/bin/python clustering/scripts/evaluate_cause_null_stability.py
"""
import os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import json,hashlib,csv,platform
import numpy as np
import sklearn,torch,sentence_transformers
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances,silhouette_samples,adjusted_rand_score
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import cut_tree
from sentence_transformers import SentenceTransformer
SRC=ROOT/'results/cause_facets/petri/qwen3_8b_petri_eval'
OUT=ROOT/'results/cause_evaluation/null_stability';OUT.mkdir(parents=True,exist_ok=True)
KS=[50,336];SEED=20261001;N_NULL=20;N_PERM=200;N_RESAMPLE=20

def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def stats(values):
 a=np.asarray(values,float)
 return {'n':len(a),'mean':float(a.mean()),'std':float(a.std(ddof=1)) if len(a)>1 else 0.,
         'min':float(a.min()),'q025':float(np.quantile(a,.025)),'median':float(np.median(a)),
         'q975':float(np.quantile(a,.975)),'max':float(a.max())}
def tail(values,observed):return {'observed':float(observed),'null_distribution':stats(values),'empirical_upper_tail_p_plus1':float((1+sum(v>=observed for v in values))/(len(values)+1))}
def ward(x,ks):
 n=len(x);m=AgglomerativeClustering(n_clusters=None,distance_threshold=0,linkage='ward',metric='euclidean',compute_distances=True).fit(x)
 c=np.zeros(n-1)
 for i,ch in enumerate(m.children_):c[i]=sum(1 if q<n else c[q-n] for q in ch)
 tree=np.column_stack([m.children_,m.distances_,c]);return cut_tree(tree,n_clusters=ks)
def distances(x):
 d=pairwise_distances(x,metric='cosine');np.fill_diagonal(d,0);return d
def score(d,l):return float(silhouette_samples(d,l,metric='precomputed').mean())
def label_permutations(d,reference):
 rng=np.random.default_rng(SEED+1);rows=[]
 for rep in range(N_PERM):
  # SAME size distribution for each k; labels assigned to randomly chosen points.
  order=rng.permutation(len(d));row={'replicate':rep}
  for k in KS:row[f'k{k}']=score(d,reference[k][order])
  rows.append(row)
 save('label_permutation_replicates.json',rows)
 report={f'k{k}':tail([r[f'k{k}'] for r in rows],score(d,reference[k])) for k in KS}
 save('label_permutation_summary.json',report);print('label permutations',json.dumps(report),flush=True)
 return report

def embedding_nulls(x,observed):
 n,p=x.shape;allks=np.arange(2,n);rows=[];curves=[]
 for family,offset in [('isotropic_unit_sphere',10),('column_permutation_renormalized',20)]:
  rng=np.random.default_rng(SEED+offset)
  for rep in range(N_NULL):
   start=time.time()
   if family=='isotropic_unit_sphere':z=normalize(rng.normal(size=x.shape).astype(np.float32))
   else:
    # Each embedding coordinate permuted independently over observations, then L2 normalize.
    z=np.stack([rng.permutation(x[:,j]) for j in range(p)],axis=1);z=normalize(z)
   assert np.allclose(np.linalg.norm(z,axis=1),1,atol=1e-6)
   cuts=ward(z,allks);d=distances(z);scores=np.array([score(d,cuts[:,j]) for j in range(len(allks))]);idx=int(np.argmax(scores))
   row={'family':family,'replicate':rep,'best_k':int(allks[idx]),'best_cosine_silhouette':float(scores[idx]),
        **{f'k{k}':float(scores[k-2]) for k in KS},'seconds':round(time.time()-start,3)}
   rows.append(row);curves.append(scores);save('embedding_null_replicates.json',rows)
   np.savez_compressed(OUT/'embedding_null_curves.npz',k=allks,scores=np.asarray(curves),families=np.array([r['family'] for r in rows]),replicates=np.array([r['replicate'] for r in rows]))
   print('embedding null',json.dumps(row),flush=True)
 report={}
 for family in sorted({r['family'] for r in rows}):
  rr=[r for r in rows if r['family']==family]
  report[family]={'fixed_k':{f'k{k}':tail([r[f'k{k}'] for r in rr],observed[f'k{k}']) for k in KS},
      'selection_aware_max_over_k2_to_800':tail([r['best_cosine_silhouette'] for r in rr],observed['maximum']),
      'best_k_distribution':stats([r['best_k'] for r in rr])}
 save('embedding_null_summary.json',report);return report

def resamples(x,reference):
 n=len(x);rows=[];pairwise=[];saved={};rng=np.random.default_rng(SEED+30)
 for kind in ['bootstrap_with_replacement','subsample80_fixed_k','subsample80_scaled_k']:
  # Reuse identical 80% subsets in fixed/scaled variants to isolate resolution choice.
  rng=np.random.default_rng(SEED+30 if kind=='bootstrap_with_replacement' else SEED+31)
  sets=[];maps=[]
  for rep in range(N_RESAMPLE):
   if kind=='bootstrap_with_replacement':sample=rng.choice(n,n,replace=True)
   else:sample=np.sort(rng.choice(n,int(round(.8*n)),replace=False))
   unique=np.unique(sample);actualks=KS if kind!='subsample80_scaled_k' else [round(k*len(sample)/n) for k in KS]
   cuts=ward(x[sample],actualks);mapped=np.full((n,len(KS)),-1,dtype=int)
   for i in unique:
    for j in range(len(KS)):
     ll=cuts[sample==i,j]
     # Duplicate copies must have merged at zero distance before these cuts.
     assert len(np.unique(ll))==1,'Duplicate ID spans clusters; mapping ambiguous.'
     mapped[i,j]=ll[0]
   sets.append(unique);maps.append(mapped)
   for j,k in enumerate(KS):
    sizes=np.bincount(mapped[unique,j]);sizes=sizes[sizes>0]
    rows.append({'kind':kind,'replicate':rep,'reference_k':k,'resample_fit_k':actualks[j],
       'draws':len(sample),'unique_ids':len(unique),'coverage_fraction':len(unique)/n,
       'ari_vs_full_on_shared_ids':float(adjusted_rand_score(reference[k][unique],mapped[unique,j])),
       'clusters_on_unique_ids':len(sizes),'median_unique_cluster_size':float(np.median(sizes)),
       'singleton_clusters_on_unique_ids':int((sizes==1).sum())})
   saved[f'{kind}_rep{rep}_draws']=sample;saved[f'{kind}_rep{rep}_labels']=mapped
  for a in range(N_RESAMPLE):
   for b in range(a+1,N_RESAMPLE):
    common=np.intersect1d(sets[a],sets[b])
    for j,k in enumerate(KS):pairwise.append({'kind':kind,'a':a,'b':b,'reference_k':k,'shared_ids':len(common),
        'ari_between_resamples_on_shared_ids':float(adjusted_rand_score(maps[a][common,j],maps[b][common,j]))})
  print('resamples finished',kind,flush=True)
 save('resample_replicates.json',rows);save('resample_pairwise.json',pairwise);np.savez_compressed(OUT/'resample_memberships.npz',**saved)
 report={}
 for kind in ['bootstrap_with_replacement','subsample80_fixed_k','subsample80_scaled_k']:
  report[kind]={}
  for k in KS:
   rr=[r for r in rows if r['kind']==kind and r['reference_k']==k];pp=[r for r in pairwise if r['kind']==kind and r['reference_k']==k]
   report[kind][f'k{k}']={'ari_vs_full':stats([r['ari_vs_full_on_shared_ids'] for r in rr]),
      'pairwise_ari':stats([r['ari_between_resamples_on_shared_ids'] for r in pp]),
      'unique_ids':stats([r['unique_ids'] for r in rr]),'pairwise_shared_ids':stats([r['shared_ids'] for r in pp]),
      'resample_fit_k':rr[0]['resample_fit_k']}
 save('resample_summary.json',report);print('resampling summaries',json.dumps(report),flush=True);return report

def alternate_model(records,reference):
 torch.set_num_threads(4);modelpath=ROOT/'models/all-MiniLM-L6-v2';model=SentenceTransformer(str(modelpath),device='cpu',local_files_only=True)
 texts=[r['cause_facet'] for r in records];lengths=[len(model.tokenizer.encode(t,add_special_tokens=True,truncation=False,verbose=False)) for t in texts]
 save('minilm_token_lengths.json',{'min':min(lengths),'max':max(lengths),'limit':model.max_seq_length,'lengths':lengths})
 assert max(lengths)<=model.max_seq_length,'Refusing silent MiniLM truncation'
 x=model.encode(texts,normalize_embeddings=True,batch_size=32,show_progress_bar=True);assert np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-6)
 np.save(OUT/'minilm_embeddings.npy',x);cuts=ward(x,KS);np.save(OUT/'minilm_labels.npy',cuts);d=distances(x)
 report={'model':'sentence-transformers/all-MiniLM-L6-v2','model_revision':json.loads((modelpath/'download_provenance.json').read_text())['revision'],
         'text_sha256':hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest(),'normalized':True,'chunking':False,'truncation':False,
         'min_tokens':min(lengths),'max_tokens':max(lengths),'same_ids':True,'comparison':{}}
 for j,k in enumerate(KS):report['comparison'][f'k{k}']={'ari_vs_mpnet':float(adjusted_rand_score(reference[k],cuts[:,j])),
  'minilm_cosine_silhouette':score(d,cuts[:,j])}
 save('alternate_model_summary.json',report);print('MiniLM',json.dumps(report),flush=True);return report

def main():
 records=[json.loads(l) for l in (SRC/'causes.jsonl').open()];x=np.load(SRC/'embeddings.npy');n=len(x)
 assert n==len(records)==801 and np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-6)
 original=np.load(SRC/'all_cuts.npz');reference={k:original['labels'][:,k-2] for k in KS};d=distances(x)
 observed={f'k{k}':score(d,reference[k]) for k in KS};observed['maximum']=json.loads((SRC/'summary.json').read_text())['best']['cosine_silhouette']
 save('design.json',{'n':n,'dimensions':x.shape[1],'seed':SEED,'fixed_k':KS,'label_permutations':N_PERM,'embedding_null_replicates_per_family':N_NULL,
  'resamples_per_kind':N_RESAMPLE,'observed':observed,'selection_aware_null_scan':[2,n-1],
  'source_embedding_sha256':hashlib.sha256((SRC/'embeddings.npy').read_bytes()).hexdigest(),
  'source_causes_sha256':hashlib.sha256((SRC/'causes.jsonl').read_bytes()).hexdigest(),'prompt_ids':[r['prompt_id'] for r in records]})
 print('Starting',observed,flush=True)
 # Save less expensive diagnostics first; selection-aware nulls take longest.
 perm=label_permutations(d,reference);res=resamples(x,reference);alt=alternate_model(records,reference);null=embedding_nulls(x,observed)
 save('summary.json',{'observed':observed,'label_permutations':perm,'embedding_nulls':null,'resampling':res,'alternate_model':alt})
 save('runtime.json',{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__,
  'sentence_transformers':sentence_transformers.__version__,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
