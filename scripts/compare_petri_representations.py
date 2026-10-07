"""Checkpoint 1 only: common metrics, frozen clustering settings, no LLM calls."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('NUMBA_NUM_THREADS','4')
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'.cache/numba'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import json
import hashlib
import sys
from collections import Counter
from html import escape
import importlib.metadata
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
import umap
import hdbscan
from refine_behaviour_clusters import page, table, save, plt

OUT=ROOT/'results/petri_representation_variants/step1'
INPUT=ROOT/'data/checkpoint1_locations/petri/behaviours.parquet'
BASE=ROOT/'results/checkpoint2_locations/petri/behaviours'
SEED=20261004
WARD={'n_clusters':200,'linkage':'ward','metric':'euclidean'}
UMAP={'n_components':10,'n_neighbors':30,'min_dist':0,'metric':'cosine','random_state':SEED,'n_jobs':1}
HDB={'min_cluster_size':10,'min_samples':5,'metric':'euclidean','cluster_selection_method':'eom','core_dist_n_jobs':1}
VARIANTS=['mpnet_baseline','e5_instruct']  # Qwen deferred at user's explicit request.


def cosine_distances(v):
    d=np.clip(1-v@v.T,0,2).astype(np.float32);np.fill_diagonal(d,0);return d


def fit(v):
    w=AgglomerativeClustering(**WARD).fit_predict(v)
    u=umap.UMAP(**UMAP).fit_transform(v)
    model=hdbscan.HDBSCAN(**HDB).fit(u)
    return {'ward':w,'hdbscan':model.labels_},u,model.probabilities_


def seed_metrics(labels, seeds):
    assigned=labels>=0
    groups=[np.flatnonzero(labels==k) for k in sorted(set(labels[assigned]))]
    if not groups:return {'weighted_top_one_share':None,'clusters_majority_one_scenario':0,'median_distinct_scenarios':None}
    tops=[max(Counter(seeds[g]).values()) for g in groups]
    return {'weighted_top_one_share':float(sum(tops)/assigned.sum()),
            'clusters_majority_one_scenario':sum(t/len(g)>.5 for t,g in zip(tops,groups)),
            'median_distinct_scenarios':float(np.median([len(set(seeds[g])) for g in groups]))}


def distribution(values):
    a=np.array([x for x in values if x is not None],dtype=float)
    if not len(a):return {'valid_replicates':0,'mean':None,'q025':None,'q975':None}
    return {'valid_replicates':len(a),'mean':float(a.mean()),'q025':float(np.quantile(a,.025)),'q975':float(np.quantile(a,.975))}


def seed_null(labels,seeds):
    obs=seed_metrics(labels,seeds);rep=[]
    for i in range(200):
        # Shuffle labels over all 801 input rows, retaining the fixed clustering/noise mask.
        rep.append(seed_metrics(labels,np.random.default_rng(SEED+20000+i).permutation(seeds)))
    result={'observed':obs,'permutations':200,'null':{},'excess_over_null':{},'upper_tail_p':{}}
    for k,v in obs.items():
        dist=distribution([r[k] for r in rep]);result['null'][k]=dist
        result['excess_over_null'][k]=v-dist['mean'] if v is not None and dist['mean'] is not None else None
        result['upper_tail_p'][k]=(1+sum(r[k]>=v for r in rep))/201 if v is not None else None
    return result,rep


def metric(labels,v,seeds):
    assigned=labels>=0;ids=sorted(set(labels[assigned]));sizes=[int(sum(labels==k)) for k in ids]
    sil=float(silhouette_score(cosine_distances(v)[np.ix_(assigned,assigned)],labels[assigned],metric='precomputed')) if 2<=len(ids)<sum(assigned) else None
    s,null=seed_null(labels,seeds)
    return {'clusters':len(ids),'assigned':int(sum(assigned)),'noise_fraction':float((~assigned).mean()),
            'cosine_silhouette':sil,'cluster_sizes':[{'cluster_id':int(k),'size':n} for k,n in zip(ids,sizes)],
            'size_distribution':{'min':min(sizes),'q25':float(np.quantile(sizes,.25)),'median':float(np.median(sizes)),
                                 'q75':float(np.quantile(sizes,.75)),'max':max(sizes),'singletons':sum(n==1 for n in sizes)} if sizes else None,
            'seed_concentration':s},null


def ari(a,b):
    mask=(a>=0)&(b>=0)
    return {'jointly_assigned_rows':int(mask.sum()),'jointly_assigned_fraction':float(mask.mean()),
            'jointly_assigned_ari':float(adjusted_rand_score(a[mask],b[mask])) if mask.sum()>=2 else None,
            'all_rows_noise_as_one_label_ari':float(adjusted_rand_score(a,b))}


def verify_seed_metric():
    labels=np.array([0,0,1,1]); seeds=np.array(['a','a','b','b'])
    assert seed_metrics(labels,seeds)=={'weighted_top_one_share':1.,'clusters_majority_one_scenario':2,'median_distinct_scenarios':1.}
    assert seed_metrics(labels,np.array(['a','b','a','b']))['weighted_top_one_share']==.5
    assert seed_metrics(np.array([0,0,-1,-1]),seeds)['weighted_top_one_share']==1.


def run(variant):
    verify_seed_metric();OUT.mkdir(parents=True,exist_ok=True)
    records=pq.read_table(INPUT).to_pylist();old=json.loads((BASE/'records.json').read_text())
    assert len(records)==801 and [(r['prompt_id'],r['behaviour_text']) for r in records]==[(r['prompt_id'],r['behaviour_text']) for r in old]
    assert all(r['source']=='petri' and r['verif_score']>=7 and r['scenario_seed_proxy'] for r in records)
    folder=OUT/variant;folder.mkdir(exist_ok=True)
    if variant=='mpnet_baseline':
        v=np.load(BASE/'embeddings.npy');meta=json.loads((BASE/'embedding_metadata.json').read_text());meta['instruction']='';meta['reused_from']=str(BASE)
        np.save(folder/'embeddings.npy',v);save(folder/'embedding_metadata.json',meta)
    else:
        v=np.load(folder/'embeddings.npy');meta=json.loads((folder/'embedding_metadata.json').read_text())
        assert meta['input_prompt_ids']==[r['prompt_id'] for r in records]
        assert meta['input_sha256']==hashlib.sha256(INPUT.read_bytes()).hexdigest()
    assert np.isfinite(v).all() and np.allclose(np.linalg.norm(v,axis=1),1,atol=1e-5)
    baseline_rows=pq.read_table(BASE/'assignments.parquet').to_pylist()
    baseline={m:np.array([r[f'{m}_cluster_id'] for r in baseline_rows]) for m in ['ward','hdbscan']}
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.iterdir() if p.is_file()}
    with threadpool_limits(limits=4):
        if variant=='mpnet_baseline':
            # Reuse original fitted assignments; reproduce the frozen Ward cut as a check.
            reproduced=AgglomerativeClustering(**WARD).fit_predict(v)
            assert adjusted_rand_score(reproduced,baseline['ward'])==1.
            labels=baseline;reduced=np.load(BASE/'umap10.npy');strengths=np.load(BASE/'hdbscan_probabilities.npy')
        else:labels,reduced,strengths=fit(v)
        np.save(folder/'umap10.npy',reduced);np.save(folder/'hdbscan_probabilities.npy',strengths)
        pq.write_table(pa.Table.from_pylist([{**r,'row_index':i,'ward_cluster_id':int(labels['ward'][i]),'hdbscan_cluster_id':int(labels['hdbscan'][i]),'hdbscan_membership_strength':float(strengths[i])} for i,r in enumerate(records)]),folder/'assignments.parquet')
        seeds=np.array([r['scenario_seed_proxy'] for r in records]);metrics={};permutations={}
        for method in ['ward','hdbscan']:
            metrics[method],permutations[method]=metric(labels[method],v,seeds)
            metrics[method]['ari_against_same_method_mpnet']=ari(labels[method],baseline[method])
        null=[]
        for i in range(10):
            rng=np.random.default_rng(SEED+1000+i)
            shuffled=normalize(np.column_stack([rng.permutation(v[:,j]) for j in range(v.shape[1])])).astype(np.float32)
            nl,_,_=fit(shuffled);d=cosine_distances(shuffled);result={'replicate':i}
            for method in ['ward','hdbscan']:
                l=nl[method];mask=l>=0;n_clusters=len(set(l[mask]))
                result[method]={'clusters':n_clusters,'noise_fraction':float((~mask).mean()),
                  'silhouette':float(silhouette_score(d[np.ix_(mask,mask)],l[mask],metric='precomputed')) if 2<=n_clusters<mask.sum() else None}
            null.append(result);save(folder/'embedding_null_replicates.json',null)
            print(variant,'embedding null',i+1,'/10',flush=True)
        for method in metrics:
            metrics[method]['embedding_shuffle_null']=distribution([r[method]['silhouette'] for r in null])
            metrics[method]['embedding_shuffle_null_noise_fraction']=distribution([r[method]['noise_fraction'] for r in null])
        samples={}
        for method,l in labels.items():
            rng=np.random.default_rng(SEED+30000);samples[method]=[]
            for group in sorted(metrics[method]['cluster_sizes'],key=lambda r:(-r['size'],r['cluster_id']))[:10]:
                ix=np.flatnonzero(l==group['cluster_id']);selected=rng.choice(ix,size=min(5,len(ix)),replace=False)
                samples[method].append({**group,'members':[records[int(i)] for i in selected]})
        report={'variant':variant,'n':len(records),'families':len(set(seeds)),'embedding':meta,'methods':metrics,
                'ari_between_methods':ari(labels['ward'],labels['hdbscan']),
                'hyperparameters':{'ward':WARD,'umap':UMAP,'hdbscan':HDB,'scenario_permutations':200,'embedding_null_replicates':10,'seed':SEED},
                'versions':{n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','umap-learn','hdbscan','torch','sentence-transformers','transformers']}}
        save(folder/'metrics.json',report);save(folder/'scenario_null_replicates.json',permutations);save(folder/'samples.json',samples)
        save(folder/'records.json',records)
    assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.iterdir() if p.is_file()}
    save(folder/'validation.json',{'same_801_rows_and_texts':True,'finite_unit_vectors':True,'baseline_files_unchanged':True,'seed_metric_synthetic_checks':True,'baseline_hashes':hashes})
    print('COMPLETE',variant,flush=True)


def render():
    reports={v:json.loads((OUT/v/'metrics.json').read_text()) for v in VARIANTS}
    fmt=lambda x:'undefined' if x is None else f'{x:.3f}'
    pct=lambda x:'undefined' if x is None else f'{100*x:.1f}%'
    overview=[];seedrows=[];arirows=[];sizerows=[]
    for name,r in reports.items():
        for method,m in r['methods'].items():
            s=m['seed_concentration'];o=s['observed'];null=s['null'];sz=m['size_distribution']
            overview.append([name,method,str(m['clusters']),pct(m['noise_fraction']),fmt(m['cosine_silhouette']),fmt(m['embedding_shuffle_null']['mean']),pct(o['weighted_top_one_share']),pct(null['weighted_top_one_share']['mean']),f"{100*s['excess_over_null']['weighted_top_one_share']:+.1f} pp"])
            seedrows.append([name,method,str(o['clusters_majority_one_scenario']),fmt(null['clusters_majority_one_scenario']['mean']),fmt(o['median_distinct_scenarios']),fmt(null['median_distinct_scenarios']['mean']),f"[{pct(null['weighted_top_one_share']['q025'])}, {pct(null['weighted_top_one_share']['q975'])}]",fmt(s['upper_tail_p']['weighted_top_one_share'])])
            a=m['ari_against_same_method_mpnet'];arirows.append([name,method,fmt(a['jointly_assigned_ari']),str(a['jointly_assigned_rows']),fmt(a['all_rows_noise_as_one_label_ari'])])
            sizerows.append([name,method,str(sz['min']),fmt(sz['q25']),fmt(sz['median']),fmt(sz['q75']),str(sz['max']),str(sz['singletons'])])
    body='<p><strong>CHECKPOINT 1 — stopped for review. Steps 2 and 3 have not started.</strong></p><p>801 PETRI Qwen3-8B investigations, all qualifying checkpoints; verification ≥7; 147 scenario families. Exactly the same behaviour_text strings and row order as the saved baseline. No new filters, paraphrases, generative LLM calls, or semantic cluster labels.</p>'
    body+='<p><a href="protocol.json">Full protocol and hashes</a> · <a href="comparison.json">All metrics</a></p><h2>Representation comparison</h2>'
    body+=table(['Representation','Method','Clusters','Noise','Cosine silhouette','Shuffled-embedding mean','Top-one seed share','Scenario-null mean','Excess over null'],overview)
    conclusions=[]
    for name in VARIANTS[1:]:
        for method in ['ward','hdbscan']:
            base=reports['mpnet_baseline']['methods'][method]['seed_concentration']['excess_over_null']['weighted_top_one_share']
            new=reports[name]['methods'][method]['seed_concentration']['excess_over_null']['weighted_top_one_share']
            conclusions.append(f'{name} / {method}: excess seed concentration changed by {100*(new-base):+.1f} percentage points relative to MPNet (negative is less concentration).')
    body+='<h2>Did concentration drop relative to the null?</h2>'+''.join(f'<p>{escape(c)}</p>' for c in conclusions)
    body+='<p><strong>Coarsening caveat:</strong> E5 HDBSCAN has 18 clusters rather than 22, and its largest cluster has 196 members rather than 74. Its five random examples span direct execution instead of script writing, including details despite an omission instruction, redundant CRM queries, supplier blame despite contrary evidence, and launching only one subagent. This is a broad group; lower scenario concentration does not establish one coherent failure pattern.</p>'
    body+='<p>These are descriptive differences, not paired significance tests of model improvement. Lower concentration alone does not establish more meaningful behaviour clusters; inspect the examples and coverage. Scenario families may share real behaviours, so some concentration is legitimate. Multiple checkpoints from an episode are dependent.</p>'
    body+='<img src="comparison.png" style="max-width:100%" alt="Seed concentration and cosine silhouette compared with nulls">'
    body+='<h2>All requested scenario metrics</h2>'+table(['Representation','Method','Clusters >50% one scenario','Null mean','Median distinct scenarios','Null mean','Top-one share: null 95% interval','Upper-tail permutation p'],seedrows)
    body+='<h2>Cluster size distributions (noise excluded)</h2>'+table(['Representation','Method','Minimum','Q25','Median','Q75','Maximum','Singletons'],sizerows)
    body+='<h2>ARI against the same baseline method</h2>'+table(['Representation','Method','ARI jointly assigned','Rows compared','ARI including noise as one label'],arirows)
    body+='<h2>ARI between Ward and HDBSCAN</h2>'+table(['Representation','ARI jointly assigned','Rows compared','All-row ARI'],[[n,fmt(r['ari_between_methods']['jointly_assigned_ari']),str(r['ari_between_methods']['jointly_assigned_rows']),fmt(r['ari_between_methods']['all_rows_noise_as_one_label_ari'])] for n,r in reports.items()])
    body+='<h2>Interpretation and fixed protocol</h2><ul><li>Ward is fixed at the original baseline-selected k=200 for every representation. The earlier baseline chose k from 10–200; this comparison does not reselect k for each model, so only representation changes. This fine cut includes small clusters and singletons, which inflate raw seed concentration; the permutation null retains all cluster sizes.</li><li>UMAP: 10 dimensions, cosine, 30 neighbours, min_dist=0, seed 20261004. HDBSCAN: min_cluster_size=10, min_samples=5, Euclidean, EOM. Identical settings throughout.</li><li>Seed concentration = sum of per-cluster largest scenario counts divided by assigned rows. Noise is excluded from all cluster summaries. Scenario labels are shuffled over all 801 rows 200 times while assignments/noise masks stay fixed. The null is conditional on cluster sizes and is not a test of independent scenario episodes.</li><li>Embedding null: independently shuffle each coordinate across rows, normalize and refit both pipelines for 10 replicates. Whole-row shuffling would preserve clustering geometry and is not used. Undefined silhouettes (fewer than two clusters) are retained as null failures with valid replicate counts in the JSON, not replaced by zero.</li><li>All silhouettes use cosine distance in the original embedding space, on assigned rows. Null envelopes use their own assigned subsets; HDBSCAN null coverage is saved. ARI primarily excludes noise in either clustering; all-row ARI is secondary because noise is not one coherent cluster.</li><li>E5: CPU float32, mean pooling, exact requested instruction, 1024 dimensions, L2 normalization, and no truncation. MPNet baseline uses its original 768-dimensional normalized vectors. Qwen is deferred at the user’s request: its weights were downloaded, but no Qwen inference or clustering was run. No text-generation inference is performed.</li><li>All instructions, pinned revisions, versions, sizes, null replicates and input hashes are saved. Original baseline files were verified unchanged.</li></ul>'
    body+='<p>Model usage: <a href="https://huggingface.co/intfloat/multilingual-e5-large-instruct">E5 model card</a> · <a href="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B">Qwen embedding model card</a>.</p><h2>Five random members of each of the ten largest clusters</h2><p>Fixed sampling seed; all members shown if a cluster has fewer than five. Samples are selected without semantic curation.</p>'
    for name,r in reports.items():
        body+=f'<h2>{name}</h2><p><a href="{name}/assignments.parquet">Assignments</a> · <a href="{name}/metrics.json">Full metrics</a> · <a href="{name}/samples.json">Samples</a></p>'
        samples=json.loads((OUT/name/'samples.json').read_text())
        for method,groups in samples.items():
            body+=f'<h3>{method}</h3>'
            for group in groups:
                body+=f'<details><summary>Cluster {group["cluster_id"]} · {group["size"]} members</summary>'+table(['Investigation','Scenario','Behaviour text'],[[escape(x['prompt_id']),escape(x['scenario_seed_proxy']),escape(x['behaviour_text'])] for x in group['members']])+'</details>'
    page(OUT/'review.html','PETRI behaviour representations · checkpoint 1',body)
    save(OUT/'comparison.json',reports)
    save(OUT/'protocol.json',{'input':str(INPUT),'input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),'ward':WARD,'umap':UMAP,'hdbscan':HDB,'seed_permutations':200,'embedding_shuffle_replicates':10,'checkpoint':1,'steps_2_and_3_started':False,'generative_llm_calls':0,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    labels=[];observed=[];expected=[];sils=[];sn=[]
    for name,r in reports.items():
        for method,m in r['methods'].items():
            labels.append(name.replace('_',' ')+' / '+method);observed.append(100*m['seed_concentration']['observed']['weighted_top_one_share']);expected.append(100*m['seed_concentration']['null']['weighted_top_one_share']['mean']);sils.append(m['cosine_silhouette']);sn.append(m['embedding_shuffle_null']['mean'])
    fig,axes=plt.subplots(1,2,figsize=(15,5.2));y=np.arange(len(labels))
    for ax,obs,null,title,xlabel in [(axes[0],observed,expected,'Scenario concentration','Weighted top-one scenario share (%)'),(axes[1],sils,sn,'Separation in embedding space','Cosine silhouette')]:
        ax.barh(y-.16,obs,height=.3,label='Observed',color='#087f8c');ax.barh(y+.16,null,height=.3,label='Null mean',color='#dba34a');ax.set_yticks(y,labels,fontsize=9);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel(xlabel);ax.legend(fontsize=9);ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=170);fig.savefig(OUT/'comparison.svg');plt.close(fig)
    lines=['# Checkpoint 1: PETRI representation comparison','','Stopped for review. No Step 2 or 3 runs.','', '|Representation|Method|Clusters|Noise|Silhouette|Embedding-null|Top seed share|Scenario-null|Excess|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    lines+=['|'+'|'.join(row)+'|' for row in overview];lines+=['']+conclusions+['','See review.html for all metrics and 5 random examples per top-10 cluster.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='report':render()
    else:
        assert sys.argv[1] in VARIANTS
        run(sys.argv[1])
