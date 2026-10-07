"""Cluster explicitly supported investigator cause statements; PETRI score >=7.
These are claimed causes, not independently verified causal labels.
"""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import csv,hashlib,html,json,re,platform
from collections import Counter
import numpy as np
import torch,sklearn,sentence_transformers
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances,silhouette_samples
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import cut_tree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'results/cause_facets/petri/qwen3_8b_petri_eval';OUT.mkdir(parents=True,exist_ok=True)
BEH=ROOT/'results/behaviour_facets/petri/qwen3_8b_petri_eval'
DATA=ROOT/'data/petri/qwen3_8b_petri_eval'
MODEL=ROOT/'models/all-mpnet-base-v2'
def save(path,x):path.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def jsonl(path,rows):
 with path.open('w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')

def extract():
 scores={};vhash=hashlib.sha256()
 for number,line in enumerate((DATA/'verification.jsonl').open('rb'),1):
  vhash.update(line);r=json.loads(line);assert r['prompt_id'] not in scores;scores[r['prompt_id']]=(r['score'],number)
 records=[];excluded=[];audit=[];h=hashlib.sha256();seen=set();statuses=Counter()
 for number,line in enumerate((DATA/'investigations.jsonl').open('rb'),1):
  h.update(line);r=json.loads(line);pid=r['prompt_id'];assert pid not in seen;seen.add(pid)
  score,vline=scores.get(pid,(None,None))
  if score is None or score<7:
   excluded.append({'prompt_id':pid,'reason':'missing_verification' if score is None else 'verification_below_7','verification_score':score});continue
  text=(r.get('structured_findings') or {}).get('structured_answer','')
  start=re.search(r'(?m)^Core causes\s*$',text);end=re.search(r'(?m)^(?:Negative evidence|Final answer)\s*$',text[start.end():]) if start else None
  if not start or not end:
   excluded.append({'prompt_id':pid,'reason':'missing_or_unparseable_core_causes'});continue
  offset=start.end();core=text[offset:offset+end.start()];matches=list(re.finditer(r'(?m)^\s*(\d+)[.)]\s+',core));claims=[]
  for i,m in enumerate(matches):
   bstart=m.end();bend=matches[i+1].start() if i+1<len(matches) else len(core);block=core[bstart:bend]
   status=re.search(r'(?m)^Status:\s*(.+)$',block)
   if not status:raise ValueError(f'Unrecognized status in {pid}')
   s=status.group(1).strip();statuses[s]+=1
   raw=block[:status.start()];statement=raw.strip();a=offset+bstart+len(raw)-len(raw.lstrip());b=a+len(statement)
   assert text[a:b]==statement
   claims.append({'number':int(m.group(1)),'status':s,'included':s=='supported','statement':statement,'structured_answer_char_span':[a,b]})
  kept=[c['statement'] for c in claims if c['included']]
  if not kept:excluded.append({'prompt_id':pid,'reason':'no_explicitly_supported_core_cause'});continue
  records.append({'source':'petri','run':'qwen3_8b_petri_eval','prompt_id':pid,'verification_score':score,'source_line':number,
                  'verification_line':vline,'text_field':'structured_findings.structured_answer:Core causes:Status supported:statement',
                  'cause_facet':'\n'.join(kept),'core_cause_entries':claims,'structured_answer_sha256':hashlib.sha256(text.encode()).hexdigest()})
  audit.append({'prompt_id':pid,'source_line':number,'classifier_question':r.get('classifier_question'),
    'baseline_grader_yes':r.get('baseline_grader_yes'),'baseline_grader_verdicts':r.get('baseline_grader_verdicts'),
    'baseline_completion_count':len(r.get('completions',[])),
    'experiments':[{k:e.get(k) for k in ['exp_num','n','temperature','grader_yes','grader_verdicts','edit_record','turns_spec']} for e in r.get('experiment_log',[])]})
 records.sort(key=lambda r:r['prompt_id']);audit.sort(key=lambda r:r['prompt_id'])
 jsonl(OUT/'causes.jsonl',records);jsonl(OUT/'measured_outcomes_audit.jsonl',audit);save(OUT/'excluded.json',excluded)
 behavior=[json.loads(l)['prompt_id'] for l in (BEH/'behaviours.jsonl').open()]
 same=[r['prompt_id'] for r in records]==behavior
 provenance={'investigations_total':len(seen),'retained':len(records),'excluded':dict(Counter(r['reason'] for r in excluded)),
    'core_cause_status_counts':dict(statuses),'supported_claims_per_record':dict(Counter(sum(c['included'] for c in r['core_cause_entries']) for r in records)),
    'same_ids_and_order_as_behaviour':same,'source_file_sha256':{'investigations.jsonl':h.hexdigest(),'verification.jsonl':vhash.hexdigest()},
    'scope':'Only exact statement text before Status: supported within Core causes, joined by newline per investigation. Excludes partial/refuted claims, evidence narratives, observed behaviour, negative evidence and final answer.',
    'causal_evidence_warning':'Supported is the investigator status, not an independent per-cause verification. >=7 filters investigations. Measurements are saved separately for audit and not embedded.'}
 save(OUT/'input_provenance.json',provenance);print(json.dumps(provenance),flush=True)
 return records,provenance

def embed(records):
 torch.set_num_threads(4);model=SentenceTransformer(str(MODEL),device='cpu',local_files_only=True)
 texts=[r['cause_facet'] for r in records];lengths=[len(model.tokenizer.encode(t,add_special_tokens=True,truncation=False,verbose=False)) for t in texts]
 lengths_info={'min':min(lengths),'max':max(lengths),'median':float(np.median(lengths)),'max_sequence_length':model.max_seq_length,
     'over_limit':[{'prompt_id':r['prompt_id'],'tokens':n} for r,n in zip(records,lengths) if n>model.max_seq_length],
     'records':[{'prompt_id':r['prompt_id'],'tokens':n} for r,n in zip(records,lengths)]}
 save(OUT/'token_lengths.json',lengths_info);print('Token range',min(lengths),max(lengths),'limit',model.max_seq_length,flush=True)
 if max(lengths)>model.max_seq_length:raise ValueError('No silent truncation: overlength text requires explicit handling.')
 revision=json.loads((MODEL/'download_provenance.json').read_text())['revision'];fingerprint=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()
 meta={'model':'sentence-transformers/all-mpnet-base-v2','revision':revision,'text_sha256':fingerprint,'device':'cpu','l2_normalized':True,'chunking':False,'truncation':False}
 cache=OUT/'embedding_metadata.json'
 if cache.exists() and all(json.loads(cache.read_text()).get(k)==v for k,v in meta.items()):x=np.load(OUT/'embeddings.npy')
 else:x=model.encode(texts,batch_size=32,normalize_embeddings=True,show_progress_bar=True);np.save(OUT/'embeddings.npy',x)
 norms=np.linalg.norm(x,axis=1);assert np.isfinite(x).all() and np.allclose(norms,1,atol=1e-6)
 meta.update(shape=list(x.shape),min_l2_norm=float(norms.min()),max_l2_norm=float(norms.max()));save(cache,meta);return x

def cluster(records,x):
 n=len(x);tree=AgglomerativeClustering(n_clusters=None,distance_threshold=0,metric='euclidean',linkage='ward',compute_distances=True).fit(x)
 counts=np.zeros(n-1)
 for i,ch in enumerate(tree.children_):counts[i]=sum(1 if c<n else counts[c-n] for c in ch)
 linkage=np.column_stack([tree.children_,tree.distances_,counts]);np.save(OUT/'linkage.npy',linkage)
 ks=np.arange(2,n);cuts=cut_tree(linkage,n_clusters=ks);d=pairwise_distances(x,metric='cosine');np.fill_diagonal(d,0);rows=[]
 for j,k in enumerate(ks):
  labels=cuts[:,j];sizes=np.bincount(labels);assert len(sizes)==k and sizes.sum()==n
  s=silhouette_samples(d,labels,metric='precomputed')
  rows.append({'k':int(k),'cosine_silhouette':float(s.mean()),'singleton_clusters':int((sizes==1).sum()),
    'singleton_item_fraction':float((sizes==1).sum()/n),'median_cluster_size':float(np.median(sizes)),
    'min_cluster_size':int(sizes.min()),'max_cluster_size':int(sizes.max()),'negative_silhouette_fraction':float((s<0).mean())})
 save(OUT/'diagnostics.json',rows)
 with (OUT/'diagnostics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 best=max(rows,key=lambda r:(r['cosine_silhouette'],-r['k']));idx=best['k']-2;lo=hi=idx;floor=best['cosine_silhouette']-.005
 while lo>0 and rows[lo-1]['cosine_silhouette']>=floor:lo-=1
 while hi<len(rows)-1 and rows[hi+1]['cosine_silhouette']>=floor:hi+=1
 summary={'n':n,'facet':'investigator explicitly supported core-cause statements','verification_filter':'>=7','evaluated_k_range':[2,n-1],
  'n_cuts':len(rows),'best':best,'near_peak_k_range':[rows[lo]['k'],rows[hi]['k']],'near_peak_score_floor':floor,
  'near_peak_definition':'Contiguous range containing peak within 0.005 absolute silhouette; descriptive tolerance, not confidence interval.',
  'all_records_scored':True,'semantic_labels_assigned':False,'uses_measured_outcomes_as_clustering_features':False}
 save(OUT/'summary.json',summary);np.savez_compressed(OUT/'all_cuts.npz',k=ks,labels=cuts)
 with (OUT/'best_assignments.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['prompt_id','verification_score','cluster_id','k'])
  for r,label in zip(records,cuts[:,idx]):w.writerow([r['prompt_id'],r['verification_score'],int(label),best['k']])
 examples=[]
 for label in np.unique(cuts[:,idx]):
  ii=np.flatnonzero(cuts[:,idx]==label);center=normalize(x[ii].mean(axis=0,keepdims=True))[0];nearest=ii[np.argsort(-(x[ii]@center),kind='stable')[:3]]
  examples.append({'cluster_id':int(label),'size':len(ii),'examples_nearest_centroid':[records[i] for i in nearest]})
 save(OUT/'cluster_examples.json',examples);print(json.dumps(summary),flush=True);return summary,rows,examples

def report(s,rows,examples,provenance):
 behrows=json.loads((BEH/'extended_sweep/diagnostics.json').read_text());behs=json.loads((BEH/'extended_sweep/summary.json').read_text())
 fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
 for ax,limit,title in zip(axes,[min(len(rows)+1,max(s['near_peak_k_range'][1],behs['near_peak_k_range'][1])+70),len(rows)+1],['Detail around the peaks','Full sweep']):
  for rr,ss,color,label in [(behrows,behs,'#087e83','Behaviour'),(rows,s,'#a24da4','Claimed supported causes')]:
   ax.plot([r['k'] for r in rr],[r['cosine_silhouette'] for r in rr],label=f"{label}: peak k={ss['best']['k']}, {ss['best']['cosine_silhouette']:.3f}",color=color)
   ax.axvspan(*ss['near_peak_k_range'],color=color,alpha=.09);ax.scatter([ss['best']['k']],[ss['best']['cosine_silhouette']],color=color,zorder=5)
  ax.set_xlim(0,limit);ax.set_xlabel('Number of clusters (k)');ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
 axes[0].set_ylabel('Cosine silhouette (higher is better)')
 fig.suptitle(f"PETRI · {'same' if provenance['same_ids_and_order_as_behaviour'] else 'unmatched'} {s['n']} investigations · verification ≥7 · MPNet + L2 + Ward")
 fig.text(.5,.01,'Shading: contiguous range within 0.005 of each peak; descriptive tolerance, not a confidence interval.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.04,1,1))
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'cause_vs_behaviour_sweep.{ext}',dpi=180,bbox_inches='tight')
 parts=['<!doctype html><meta charset="utf-8"><title>PETRI cause facets</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;line-height:1.5;padding:0 20px}summary{cursor:pointer}details{border-top:1px solid #ddd;padding:12px}blockquote{background:#f4f6f8;padding:16px;white-space:pre-wrap}small{color:#555}</style><h1>PETRI cause facet clustering</h1><p>One combined item per investigation: exact Core causes statements explicitly marked supported by the investigator. Excludes partially supported/refuted statements and evidence prose. These remain investigator claims; verification ≥7 is an investigation filter, not independent verification of each cause. Measurements were saved for audit but not embedded.</p>',
  f"<p>{s['n']} items, k={s['best']['k']}, silhouette={s['best']['cosine_silhouette']:.4f}; near-peak range {s['near_peak_k_range']}. Numeric IDs only.</p><img width='100%' src='cause_vs_behaviour_sweep.png' alt='Cause versus behaviour silhouette sweeps'>"]
 for c in examples:
  parts.append(f"<details><summary>Cluster {c['cluster_id']} — {c['size']} investigations</summary>")
  for r in c['examples_nearest_centroid']:parts.append(f"<small>{html.escape(r['prompt_id'])} · verification {r['verification_score']}</small><blockquote>{html.escape(r['cause_facet'])}</blockquote>")
  parts.append('</details>')
 (OUT/'review.html').write_text('\n'.join(parts))
if __name__=='__main__':
 records,provenance=extract();x=embed(records);s,rows,examples=cluster(records,x);report(s,rows,examples,provenance)
 save(OUT/'runtime.json',{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'sklearn':sklearn.__version__,
  'sentence_transformers':sentence_transformers.__version__,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'device':'cpu','threads':4})
