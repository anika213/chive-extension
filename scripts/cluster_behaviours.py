"""Reproduce separate PETRI/WildChat eval behaviour clusters at verification >= 7.
Run: clustering/.venv/bin/python clustering/scripts/cluster_behaviours.py
"""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import csv, hashlib, html, json, platform
from collections import Counter
import numpy as np
import torch, sklearn, sentence_transformers
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import cut_tree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = ROOT / 'results' / 'behaviour_facets'
MODEL = ROOT / 'models' / 'all-mpnet-base-v2'
KS = [5,10,15,20,30,40,50]

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def extract(source, dest):
    run = f'qwen3_8b_{source}_eval'; directory = ROOT/'data'/source/run
    hashes = {}; scores = {}; verification_lines = {}
    for name in ['verification.jsonl']:
        h=hashlib.sha256()
        for number,line in enumerate((directory/name).open('rb'),1):
            h.update(line); r=json.loads(line); pid=r['prompt_id']; assert pid not in scores
            scores[pid]=r.get('score'); verification_lines[pid]=number
        hashes[name]=h.hexdigest()
    records=[]; excluded=[]; ids=set(); h=hashlib.sha256()
    for number,line in enumerate((directory/'investigations.jsonl').open('rb'),1):
        h.update(line); r=json.loads(line); pid=r['prompt_id']; assert pid not in ids; ids.add(pid)
        score=scores.get(pid); text=r.get('behavior_summary'); reason=None
        if score is None: reason='missing_verification'
        elif not isinstance(score,(int,float)) or not np.isfinite(score): reason='invalid_verification'
        elif score<7: reason='verification_below_7'
        elif not isinstance(text,str) or not text.strip(): reason='missing_behavior_summary'
        if reason: excluded.append({'prompt_id':pid,'verification_score':score,'reason':reason});continue
        records.append({'source':source,'run':run,'prompt_id':pid,'verification_score':score,
                        'source_line':number,'verification_line':verification_lines[pid],
                        'text_field':'behavior_summary','behavior_summary':text})
    hashes['investigations.jsonl']=h.hexdigest()
    assert not set(scores)-ids
    records.sort(key=lambda r:r['prompt_id'])
    with (dest/'behaviours.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    save(dest/'excluded.json',excluded)
    provenance={'source_directory':str(directory.relative_to(ROOT)), 'file_sha256':hashes,
                'investigations_total':len(ids),'verification_records':len(scores),'retained':len(records),
                'excluded':dict(Counter(r['reason'] for r in excluded)),
                'verification_score_histogram':dict(sorted(Counter(str(scores[i]) for i in ids if i in scores).items())),
                'one_to_one_prompt_id_join':True, 'orphan_verifications':0,
                'screener_provenance':'chive/chive/pipeline/investigate.py:1718 copies screening_result[behavior_summary] into investigation output',
                'text_transform':'none; preserved verbatim', 'duplicate_summary_texts':len(records)-len(set(r['behavior_summary'] for r in records))}
    save(dest/'input_provenance.json',provenance)
    print(source,'retained',len(records),'exclusions',provenance['excluded'],flush=True)
    return records,provenance

def embed(records,model,dest):
    texts=[r['behavior_summary'] for r in records]
    lengths=[len(model.tokenizer.encode(t,add_special_tokens=True,truncation=False,verbose=False)) for t in texts]
    save(dest/'token_lengths.json',{'max_seq_length':model.max_seq_length,'min':min(lengths),'max':max(lengths),'median':float(np.median(lengths)),
        'over_limit':[{'prompt_id':r['prompt_id'],'tokens':n} for r,n in zip(records,lengths) if n>model.max_seq_length],
        'records':[{'prompt_id':r['prompt_id'],'tokens':n} for r,n in zip(records,lengths)]})
    print('Tokens min/max:',min(lengths),max(lengths),'limit:',model.max_seq_length,flush=True)
    if max(lengths)>model.max_seq_length: raise ValueError('Long summary requires explicit handling; refusing silent truncation.')
    fingerprint=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()
    revision=json.loads((MODEL/'download_provenance.json').read_text())['revision']
    meta={'model':'sentence-transformers/all-mpnet-base-v2','revision':revision,'text_sha256':fingerprint,
          'device':'cpu','normalize_embeddings':True,'chunking':False,'truncation':False,'max_sequence_length':model.max_seq_length}
    cache=dest/'embedding_metadata.json'
    if cache.exists() and all(json.loads(cache.read_text()).get(k)==v for k,v in meta.items()):
        vectors=np.load(dest/'embeddings.npy')
    else:
        vectors=model.encode(texts,batch_size=32,normalize_embeddings=True,show_progress_bar=True,convert_to_numpy=True)
        np.save(dest/'embeddings.npy',vectors)
    norms=np.linalg.norm(vectors,axis=1)
    assert vectors.shape==(len(records),768) and np.isfinite(vectors).all() and np.allclose(norms,1,atol=1e-6)
    meta.update(shape=list(vectors.shape),min_l2_norm=float(norms.min()),max_l2_norm=float(norms.max()))
    save(cache,meta);return vectors

def cluster(records,vectors,dest,provenance):
    n=len(records)
    tree=AgglomerativeClustering(n_clusters=None,distance_threshold=0,linkage='ward',metric='euclidean',compute_distances=True).fit(vectors)
    counts=np.zeros(n-1)
    for i,children in enumerate(tree.children_):counts[i]=sum(1 if c<n else counts[c-n] for c in children)
    linkage=np.column_stack([tree.children_,tree.distances_,counts]);np.save(dest/'linkage.npy',linkage)
    cuts=cut_tree(linkage,n_clusters=KS);diagnostics=[]
    for j,k in enumerate(KS):
        labels=cuts[:,j];sizes=np.bincount(labels);assert len(sizes)==k and sizes.sum()==n
        values=silhouette_samples(vectors,labels,metric='cosine')
        diagnostics.append({'k':k,'cosine_silhouette':float(values.mean()),'silhouette_sample_size':n,
            'negative_silhouette_fraction':float((values<0).mean()),'smallest_cluster':int(sizes.min()),
            'largest_cluster':int(sizes.max()),'largest_cluster_fraction':float(sizes.max()/n),'singleton_clusters':int((sizes==1).sum())})
    best=max(diagnostics,key=lambda d:(d['cosine_silhouette'],-d['k']));labels=cuts[:,KS.index(best['k'])]
    with (dest/'assignments.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['source','run','prompt_id','verification_score','cluster_id']+[f'cluster_k{k}' for k in KS])
        for i,r in enumerate(records):w.writerow([r['source'],r['run'],r['prompt_id'],r['verification_score'],int(labels[i])]+cuts[i].tolist())
    examples=[]
    for label in np.unique(labels):
        indices=np.flatnonzero(labels==label);centroid=normalize(vectors[indices].mean(axis=0,keepdims=True))[0]
        nearest=indices[np.argsort(-(vectors[indices]@centroid),kind='stable')[:3]]
        examples.append({'cluster_id':int(label),'size':len(indices),'examples_nearest_centroid':[records[i] for i in nearest]})
    save(dest/'cluster_examples.json',examples);save(dest/'diagnostics.json',diagnostics)
    summary={'source':records[0]['source'],'run':records[0]['run'],'n_behaviours':n,'investigations_total':provenance['investigations_total'],
        'excluded':provenance['excluded'],'verification_filter':'>= 7','selected_k':best['k'],'selected_cosine_silhouette':best['cosine_silhouette'],
        'selected_at_search_boundary':best['k'] in [min(KS),max(KS)],'candidates':KS,'linkage':'ward','metric':'euclidean',
        'embedding_model':'sentence-transformers/all-mpnet-base-v2','l2_normalized':True,'silhouette_metric':'cosine',
        'silhouette_full_dataset':True,'cluster_names_assigned':False,'scope':'screener behaviour summaries, not causal claims',
        'largest_cluster_fraction':best['largest_cluster_fraction'],'negative_silhouette_fraction':best['negative_silhouette_fraction']}
    save(dest/'summary.json',summary);print(json.dumps(summary),flush=True)
    return summary,diagnostics,examples

def report(results):
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    parts=['<!doctype html><meta charset="utf-8"><title>Behaviour facet clusters</title><style>body{font:16px system-ui;max-width:1050px;margin:40px auto;padding:0 20px;line-height:1.5}summary{cursor:pointer}details{border-top:1px solid #ddd;padding:12px}blockquote{background:#f4f6f8;padding:16px;margin:12px 0}small{color:#555}</style><h1>Qwen3-8B behaviour facet clusters</h1><p>Verification score ≥ 7; PETRI and standard WildChat eval clustered separately. MPNet embeddings, L2 normalized, Ward/Euclidean hierarchy. Numeric cluster IDs only. These are behaviours, not verified causes.</p><p>Selection: highest full-dataset cosine silhouette among k = 5, 10, 15, 20, 30, 40, 50. Boundary selections do not establish the optimum outside that range.</p><img src="cluster_selection.png" width="100%" alt="Cosine silhouette versus cluster count">']
    for ax,(s,ds,ex) in zip(axes,results):
        ax.plot([d['k'] for d in ds],[d['cosine_silhouette'] for d in ds],'-o',color='#2867a0')
        ax.scatter([s['selected_k']],[s['selected_cosine_silhouette']],s=90,color='#df751d',zorder=3)
        ax.set_title(f"{s['source'].upper()} eval · n = {s['n_behaviours']:,}")
        ax.set_xlabel('Number of clusters (k)');ax.set_xticks(KS);ax.grid(alpha=.2)
        ax.annotate(f"Selected k={s['selected_k']}\n{s['selected_cosine_silhouette']:.3f}",(s['selected_k'],s['selected_cosine_silhouette']),xytext=(-8,12),textcoords='offset points',ha='right')
        parts.append(f"<h2>{s['source'].upper()} eval</h2><p>{s['n_behaviours']:,} of {s['investigations_total']:,} investigations retained; selected k={s['selected_k']}, silhouette {s['selected_cosine_silhouette']:.4f}. Examples below are the three summaries closest to each cluster's normalized centroid.</p>")
        for c in ex:
            parts.append(f"<details><summary>Cluster {c['cluster_id']} — {c['size']} investigations</summary>")
            for r in c['examples_nearest_centroid']:parts.append(f"<small>{html.escape(r['prompt_id'])} · verification {r['verification_score']}</small><blockquote>{html.escape(r['behavior_summary'])}</blockquote>")
            parts.append('</details>')
    axes[0].set_ylabel('Cosine silhouette (higher is better)')
    ymin,ymax=axes[0].get_ylim();axes[0].set_ylim(min(0,ymin),ymax+.04)
    fig.suptitle('Behaviour summaries · MPNet · L2 normalization · verification ≥ 7')
    fig.tight_layout()
    for suffix in ['png','pdf','svg']:fig.savefig(OUT/f'cluster_selection.{suffix}',dpi=180,bbox_inches='tight')
    (OUT/'review.html').write_text('\n'.join(parts))

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    model=SentenceTransformer(str(MODEL),device='cpu',local_files_only=True)
    results=[]
    for source in ['petri','wildchat']:
        dest=OUT/source/f'qwen3_8b_{source}_eval';dest.mkdir(parents=True,exist_ok=True)
        records,provenance=extract(source,dest);vectors=embed(records,model,dest)
        results.append(cluster(records,vectors,dest,provenance))
    save(OUT/'summary.json',[r[0] for r in results]);report(results)
    save(OUT/'runtime.json',{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
        'sklearn':sklearn.__version__,'sentence_transformers':sentence_transformers.__version__,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'device':'cpu','threads':4})
