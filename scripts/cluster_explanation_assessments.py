"""Separate exploratory clustering of verifier explanation_assessment prose."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent.parent
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import hashlib
import json
import argparse
import re
from collections import Counter
from html import escape
from html.parser import HTMLParser
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import linkage,cut_tree,dendrogram,leaves_list
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples
from refine_behaviour_clusters import page,table,save,stats,cohesion,signature,plt

SOURCE=ROOT/'results/scenario_clustering/related_health_scenarios'
OUT=ROOT/'results/scenario_clustering/related_health_explanation_assessments'


def render_members(path,title,indices,records,cosine,is_cluster=True):
    body='<p><a href="review.html">Back to groups</a> · <a href="../review.html">All assessment results</a></p>'
    if is_cluster:
        means={i:float((cosine[i,indices].sum()-1)/(len(indices)-1)) for i in indices}
        order=sorted(indices,key=lambda i:(-means[i],records[i]['prompt_id']))
        body+=f'<p><strong>{len(indices)} investigations · mean cosine {cohesion(indices,cosine):.3f}.</strong> Ordered by average similarity to the other members.</p>'
    else:
        means={};order=sorted(indices,key=lambda i:records[i]['prompt_id'])
        body+=f'<p>{len(indices)} unassigned investigations. This residual list is not a cluster.</p>'
    body+='<p>The text below is the <strong>verifier’s assessment of the investigator’s explanation</strong>. These clusters describe similar review prose and claims, not verified causal mechanisms. Original wording is preserved.</p>'
    for i in order:
        r=records[i];metric=f' · mean cosine to others {means[i]:.3f}' if i in means else ''
        body+=f'<article class="card member" id="item-{i}"><h2>{escape(r["prompt_id"])}</h2><p class="muted">Scenario: {escape(r["scenario"])} · verification {r["verif_score"]}/10 · {r["token_count"]} tokens{metric}</p><p><strong>explanation_assessment</strong></p><p class="text assessment">{escape(r["explanation_assessment"])}</p><details><summary>Behaviour summary for context — not embedded in this run</summary><p class="text">{escape(r["behaviour_text"])}</p></details></article>'
    page(path,title,body)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scenarios',nargs='+',help='Scenario names; include all qualifying checkpoints from these scenarios.')
    parser.add_argument('--name',help='Separate output folder name for this selection.')
    args=parser.parse_args()
    if args.scenarios:
        scenario_order=list(dict.fromkeys(args.scenarios))
        assert all(re.fullmatch('[a-z0-9_]+',s) for s in scenario_order)
        output_name=args.name or 'selected_explanation_assessments'
        assert re.fullmatch('[a-z0-9_]+',output_name)
        assert output_name!='related_health_explanation_assessments','Preserve the original three-scenario run.'
        pool=json.loads((ROOT/'results/checkpoint2_locations/petri/behaviours/records.json').read_text())
        base=[]
        for r in pool:
            match=re.fullmatch(r'petri_(.+)_e\d+_t\d+',r['prompt_id'])
            if match and match[1] in scenario_order:base.append({**r,'scenario':match[1]})
        base.sort(key=lambda r:(scenario_order.index(r['scenario']),r['prompt_id']))
        assert set(r['scenario'] for r in base)==set(scenario_order)
    else:
        base=json.loads((SOURCE/'records.json').read_text())
        scenario_order=list(dict.fromkeys(r['scenario'] for r in base))
        output_name=args.name or 'related_health_explanation_assessments'
        assert re.fullmatch('[a-z0-9_]+',output_name)
    OUT=ROOT/'results/scenario_clustering'/output_name
    OUT.mkdir(parents=True,exist_ok=True)
    ids={r['prompt_id'] for r in base}
    prior_path=ROOT/'results/scenario_clustering/related_health_explanation_assessments/summary.json'
    prior_digest=hashlib.sha256(prior_path.read_bytes()).hexdigest() if prior_path.exists() and prior_path.parent!=OUT else None
    verifications={};source_path=ROOT/'data/petri/qwen3_8b_petri_eval/verification.jsonl'
    with source_path.open() as f:
        for number,line in enumerate(f,1):
            r=json.loads(line)
            if r['prompt_id'] in ids:verifications[r['prompt_id']]=(number,r)
    assert set(verifications)==ids
    records=[]
    for b in base:
        number,v=verifications[b['prompt_id']];text=v.get('explanation_assessment')
        assert isinstance(text,str) and text.strip(),b['prompt_id']
        assert v['score']>=7 and v['score']==b['verif_score']
        records.append({**b,'explanation_assessment':text,'verification_source_line':number})
    torch.set_num_threads(2)
    model=SentenceTransformer(str(ROOT/'models/all-mpnet-base-v2'),device='cpu',local_files_only=True)
    texts=[r['explanation_assessment'] for r in records]
    lengths=[len(model.tokenizer(t,truncation=False)['input_ids']) for t in texts]
    assert max(lengths)<=model.max_seq_length,'Stop: assessment exceeds model limit; do not truncate.'
    for r,length in zip(records,lengths):r['token_count']=length
    vectors=model.encode(texts,batch_size=16,device='cpu',normalize_embeddings=True,show_progress_bar=False).astype(float)
    vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
    cosine=np.clip(vectors@vectors.T,-1,1);np.fill_diagonal(cosine,1)
    distance=np.clip(1-cosine,0,2);np.fill_diagonal(distance,0)
    n=len(records);candidates=[];seen=set();trees={};cuts=[]
    for method in ['ward','average','complete']:
        tree=linkage(vectors,method='ward') if method=='ward' else linkage(squareform(distance),method=method)
        trees[method]=tree;np.save(OUT/f'{method}_linkage.npy',tree)
        for k in range(2,n):
            labs=cut_tree(tree,n_clusters=k).ravel();cuts.append({'method':method,'k':k,'labels':labs.tolist()})
            groups=[np.flatnonzero(labs==c).tolist() for c in sorted(set(labs))]
            groups=[g for g in groups if len(g)>=2 and cohesion(g,cosine)>.4]
            while len(groups)>=2:
                sig=signature(groups)
                if sig not in seen:
                    seen.add(sig);candidates.append({'groups':[g[:] for g in groups],'method':method,'original_cut_k':k,**stats(groups,cosine,distance)})
                del groups[int(np.argmin([cohesion(g,cosine) for g in groups]))]
    assert candidates
    best=max(candidates,key=lambda c:(c['min_cluster_mean_cosine'],c['assigned'],c['mean_cluster_cosine']))
    near=[c for c in candidates if c['min_cluster_mean_cosine']>=best['min_cluster_mean_cosine']-.02]
    presets={'fullest_coverage':max(candidates,key=lambda c:(c['assigned'],c['min_cluster_mean_cosine'],c['mean_cluster_cosine'])),
             'cosine_first':max(near,key=lambda c:(c['assigned'],c['min_cluster_mean_cosine'],c['mean_cluster_cosine']))}
    for threshold in [.6,.7]:
        available=[c for c in candidates if c['min_cluster_mean_cosine']>threshold]
        if available:presets[f'cosine_above_{int(threshold*100)}']=max(available,key=lambda c:(c['assigned'],c['min_cluster_mean_cosine'],c['mean_cluster_cosine']))
    for name,c in presets.items():
        folder=OUT/name;folder.mkdir(exist_ok=True);labels=np.full(n,-1,dtype=int);rows=[]
        c['scenario_composition']=[dict(Counter(records[i]['scenario'] for i in g)) for g in c['groups']]
        c['mixed_scenario_clusters']=sum(len(d)>1 for d in c['scenario_composition'])
        for k,g in enumerate(c['groups']):
            labels[g]=k
            render_members(folder/f'cluster_{k}.html',f'Explanation assessments · cluster {k}',g,records,cosine)
            rows.append([f'<a href="cluster_{k}.html">Cluster {k}</a>',str(len(g)),f'{cohesion(g,cosine):.3f}',escape(', '.join(f'{s}: {count}' for s,count in c['scenario_composition'][k].items()))])
        assigned=np.flatnonzero(labels>=0)
        scores=silhouette_samples(distance[np.ix_(assigned,assigned)],labels[assigned],metric='precomputed')
        assert np.isclose(scores.mean(),c['mean_silhouette'])
        score_by_row=dict(zip(map(int,assigned),map(float,scores)))
        pq.write_table(pa.Table.from_pylist([{**r,'cluster_id':int(labels[i]),'cosine_silhouette':score_by_row.get(i)} for i,r in enumerate(records)]),folder/'assignments.parquet')
        render_members(folder/'unassigned.html','Unassigned explanation assessments',np.flatnonzero(labels<0).tolist(),records,cosine,False)
        save(folder/'summary.json',c)
        body=f'<p><a href="../review.html">Compare results</a></p><p><strong>{c["assigned"]}/{n} assessments assigned · {c["clusters"]} groups · {c["unassigned"]} unassigned.</strong> Each item is one top-level verification assessment, with minimum group size two.</p>'
        body+=table(['Read every assessment','Investigations','Mean cosine','Scenario composition'],rows)
        body+='<p><a href="unassigned.html">All unassigned assessments</a> · <a href="assignments.parquet">Assignments and complete texts</a></p><p>Shared review language can influence these similarities. These are interpretive-text clusters; their scores do not validate the claims in the assessments.</p>'
        page(folder/'review.html','Explanation assessments · '+name.replace('_',' '),body)
    order=leaves_list(trees['average']).astype(int);labels=[f'{i+1:02d}' for i in range(n)]
    fig,axes=plt.subplots(1,2,figsize=(16,11) if n>25 else (15,8))
    dendrogram(trees['average'],labels=labels,orientation='right',ax=axes[0],color_threshold=0,above_threshold_color='#086c79');axes[0].invert_yaxis()
    axes[0].set_title('Average-linkage hierarchy');axes[0].set_xlabel('Cosine distance')
    im=axes[1].imshow(cosine[np.ix_(order,order)],vmin=0,vmax=1,cmap='Blues')
    axes[1].set_xticks(range(n),[labels[i] for i in order],rotation=90);axes[1].set_yticks(range(n),[labels[i] for i in order]);axes[1].set_title('Assessment-text cosine similarity')
    fig.colorbar(im,ax=axes[1],fraction=.046,pad=.04);fig.suptitle(f'{n} verifier explanation assessments · {len(scenario_order)} related PETRI scenarios');fig.tight_layout();fig.canvas.draw()
    a,b=axes[0].get_position(),axes[1].get_position();axes[0].set_position([a.x0,b.y0,a.width,b.height])
    fig.savefig(OUT/'similarity_and_hierarchy.png',dpi=160);fig.savefig(OUT/'similarity_and_hierarchy.svg');plt.close(fig)
    metadata={'model':'sentence-transformers/all-mpnet-base-v2','model_revision':'e8c3b32edf5434bc2275fc9bab85f82640a19130','device':'cpu','normalize_embeddings':True,
              'input_field':'verification.jsonl → explanation_assessment','ensemble_assessments_used':False,'max_sequence_length':model.max_seq_length,
              'token_lengths':lengths,'truncation':False,'chunking':False,'paraphrasing':False,'text_sha256':hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()}
    summary={'input_rows':n,'source':'petri','scenarios':dict(Counter(r['scenario'] for r in records)),'verification_min':7,
             'facet':'verifier explanation_assessment prose (interpretation, not measured cause)','input_field':'verification.jsonl → explanation_assessment',
             'candidate_partitions':len(candidates),'minimum_group_size':2,'minimum_mean_cosine':.4,'presets':presets,
             'selection':'Same related-scenario tree-cut search. Fullest coverage maximizes assigned count, then minimum cluster mean cosine. Cosine-first maximizes minimum cluster mean cosine, then coverage within 0.02 of that best. Silhouette diagnostic only.',
             'whole_set_mean_cosine':float(cosine[np.triu_indices(n,1)].mean()),'generative_llm_calls':0,'embedding_calls':1,'semantic_labels_assigned':False}
    save(OUT/'records.json',records);save(OUT/'summary.json',summary);save(OUT/'embedding_metadata.json',metadata);save(OUT/'candidates.json',candidates);save(OUT/'all_hierarchy_cuts.json',cuts)
    pq.write_table(pa.Table.from_pylist(records),OUT/'explanation_assessments.parquet');np.save(OUT/'embeddings.npy',vectors);np.save(OUT/'cosine.npy',cosine)
    body=f'<p>{len(scenario_order)} related PETRI scenarios and all {n} qualifying investigation checkpoints across both runs, with verification ≥7. The input is <code>verification.jsonl → explanation_assessment</code>: the <strong>verifier’s review of the investigator’s explanation</strong>. All qualifying turn checkpoints are included; this is not a latest-turn-only selection.</p>'
    body+=table(['Included scenario','Assessments'],[[s,str(sum(r['scenario']==s for r in records))] for s in scenario_order])
    body+='<p class="notice">This explicitly explores interpretive prose. It is separate from the observation-only behaviour and measured-edit clusterings. Neither the verifier’s endorsement nor similarity between its assessments establishes a shared verified cause. The score ≥7 filter also favors positively reviewed investigations.</p>'
    body+=f'<p>One complete assessment per investigation; {min(lengths)}–{max(lengths)} tokens. CPU MPNet embeddings, L2-normalized; no truncation, chunking, rewriting, or generative-model calls. Rates quoted inside these assessments are preserved as part of their prose, not used as a separate measured feature. The five ensemble reviews are not added as duplicate items.</p>'
    rows=[[f'<a href="{name}/review.html">{name.replace("_"," ")}</a>',str(c['clusters']),f'{c["assigned"]}/{n}',str(c['unassigned']),f'{c["min_cluster_mean_cosine"]:.3f}',str(c['mixed_scenario_clusters'])] for name,c in presets.items()]
    body+=table(['Review every assessment','Groups','Assigned','Unassigned','Lowest mean cosine','Mixed-scenario groups'],rows)
    body+='<p><strong>Cosine-first</strong> prioritizes the weakest group’s mean pairwise similarity, then retains more items within 0.02 of the best. <strong>Fullest coverage</strong> retains as many as possible, then favors similarity. Candidate groups come from Ward, average and complete linkage across all cuts, allowing singletons or weak groups to remain unassigned. Minimum size two, as in the related-scenario analyses. Scores are selected on these same embeddings, not held-out validation.</p>'
    body+='<p>Common expressions such as “well-supported” and “follows cleanly” may increase similarity across otherwise different explanations. Group IDs are local to this run and have no correspondence to the prior behaviour or counterfactual cluster IDs.</p>'
    body+=f'<img src="similarity_and_hierarchy.png" alt="Cosine similarities and hierarchy for {n} verifier assessments"><p><a href="explanation_assessments.parquet">Input table</a> · <a href="embedding_metadata.json">Embedding provenance</a> · <a href="summary.json">Method and results</a> · <a href="../related_health_counterfactuals/review.html">Previous three-scenario measured-edit clustering</a> · <a href="../related_health_scenarios/review.html">Previous three-scenario behaviour clustering</a></p>'
    if len(scenario_order)>3:body+='<p><a href="../related_health_explanation_assessments/review.html">Previous three-scenario explanation-assessment clustering</a></p>'
    broad=presets['fullest_coverage'];lookup={i:k for k,g in enumerate(broad['groups']) for i in g}
    rows=[]
    for i,r in enumerate(records):
        target=f'fullest_coverage/cluster_{lookup[i]}.html' if i in lookup else 'fullest_coverage/unassigned.html'
        rows.append([f'<a href="{target}#item-{i}">{i+1:02d}</a>',escape(r['prompt_id']),str(r['verif_score']),str(r['token_count'])])
    body+='<h2>Every investigation — plot index</h2>'+table(['Open assessment','Investigation','Verification','Tokens'],rows)
    page(OUT/'review.html','Clustering explanation assessments',body)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.targets=[]
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if tag=='a' and a.get('href'):self.targets.append(a['href'])
            if tag=='img' and a.get('src'):self.targets.append(a['src'])
    for path in OUT.rglob('*.html'):
        parser=Links();parser.feed(path.read_text())
        for target in parser.targets:
            name,*anchor=target.split('#');p=path.parent/name;assert p.exists(),(path,target)
            if anchor:assert f'id="{anchor[0]}"' in p.read_text()
    for name,c in presets.items():
        assigned=sum(c['groups'],[]);assert len(assigned)==len(set(assigned))==c['assigned']
        for k,g in enumerate(c['groups']):
            text=(OUT/name/f'cluster_{k}.html').read_text();assert text.count('class="card member"')==len(g)
            assert np.isclose(cohesion(g,cosine),cosine[np.ix_(g,g)][np.triu_indices(len(g),1)].mean())
            for i in g:assert escape(records[i]['explanation_assessment']) in text
    assert [r['explanation_assessment'] for r in records]==[verifications[r['prompt_id']][1]['explanation_assessment'] for r in records]
    if prior_digest:assert hashlib.sha256(prior_path.read_bytes()).hexdigest()==prior_digest
    save(OUT/'validation.json',{'status':'passed','source_texts_preserved':n,'all_tokens_within_model_limit':True,
        'all_member_texts_displayed':True,'all_links_and_anchors_exist':True,'silhouette_checked_against_sklearn':True,'no_existing_result_files_modified':True})
    print(json.dumps({'rows':n,'token_range':[min(lengths),max(lengths)],'candidates':len(candidates),
        'results':{name:{k:c[k] for k in ['clusters','assigned','cluster_sizes','cluster_mean_cosines','mixed_scenario_clusters']} for name,c in presets.items()}},indent=2),flush=True)
    print('REVIEW',OUT/'review.html',flush=True)


if __name__=='__main__':main()
