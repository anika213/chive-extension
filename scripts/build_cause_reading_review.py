"""Random-member audit and c-TF-IDF descriptors; no semantic labels assigned."""
from pathlib import Path
import csv,html,json
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
ROOT=Path(__file__).resolve().parent.parent
SRC=ROOT/'results/cause_facets/petri/qwen3_8b_petri_eval'
OUT=ROOT/'results/cause_evaluation/reading';OUT.mkdir(parents=True,exist_ok=True)
records=[json.loads(l) for l in (SRC/'causes.jsonl').read_text().splitlines()]
behaviors={r['prompt_id']:r['behavior_summary'] for r in map(json.loads,(ROOT/'results/behaviour_facets/petri/qwen3_8b_petri_eval/behaviours.jsonl').read_text().splitlines())}
texts=[r['cause_facet'] for r in records]
cuts=np.load(SRC/'all_cuts.npz')
partitions={'ward_k50':cuts['labels'][:,list(cuts['k']).index(50)],'ward_k336':cuts['labels'][:,list(cuts['k']).index(336)]}
hdb=ROOT/'results/cause_evaluation/hdbscan/primary_labels.npy'
if hdb.exists():partitions['umap_hdbscan_preset']=np.load(hdb)
report={}
for name,labels in partitions.items():
    # Vocabulary from assigned original documents. Per-document ngrams avoid crossing boundaries.
    assigned=np.flatnonzero(labels>=0)
    vectorizer=CountVectorizer(stop_words='english',ngram_range=(1,2),min_df=2)
    doc_counts=vectorizer.fit_transform([texts[i] for i in assigned])
    clusters=np.unique(labels[assigned]);sizes={int(c):int((labels==c).sum()) for c in clusters}
    group=sparse.csr_matrix((np.ones(len(assigned)),(np.searchsorted(clusters,labels[assigned]),np.arange(len(assigned)))),shape=(len(clusters),len(assigned)))
    counts=group@doc_counts
    avg_size=int(np.asarray(counts.sum(axis=1)).mean())
    global_frequency=np.asarray(counts.sum(axis=0)).ravel()
    idf=np.log(1+avg_size/global_frequency)
    scores=normalize(counts,norm='l1',axis=1).multiply(idf).tocsr()
    terms=vectorizer.get_feature_names_out()
    descriptors={}
    for j,c in enumerate(clusters):
        row=scores.getrow(j);order=np.argsort(-row.data,kind='stable')[:12]
        descriptors[int(c)]=[{'term':str(terms[row.indices[i]]),'score':float(row.data[i])} for i in order]
    rng=np.random.default_rng(20261001)
    eligible=[c for c in clusters if sizes[int(c)]>=5]
    chosen=rng.choice(eligible,size=min(10,len(eligible)),replace=False)
    def members(c,limit):
        indices=np.flatnonzero(labels==c)
        sampled=rng.choice(indices,size=min(limit,len(indices)),replace=False)
        return [{'prompt_id':records[i]['prompt_id'],'cause_facet':texts[i],
                 'behavior_summary':behaviors[records[i]['prompt_id']],'verification_score':records[i]['verification_score']} for i in sampled]
    review=[{'cluster_id':int(c),'size':sizes[int(c)],'top_words':descriptors[int(c)],'random_members':members(c,5)} for c in chosen]
    # Additional unconditional cluster sample keeps the tiny clusters visible.
    uniform_chosen=rng.choice(clusters,size=min(10,len(clusters)),replace=False)
    uniform=[{'cluster_id':int(c),'size':sizes[int(c)],'top_words':descriptors[int(c)],'random_members':members(c,5)} for c in uniform_chosen]
    noise=members(-1,5) if (labels==-1).any() else []
    output={'partition':name,'seed':20261001,'n_clusters':len(clusters),'n_clusters_with_at_least_5':len(eligible),
            'reading_selection':'10 clusters sampled uniformly among clusters of size>=5, then 5 members uniformly without replacement. This conditions on size; additional unconditional cluster sample included.',
            'ctfidf':'L1-normalized per-cluster term counts * log(1 + integer mean class token count / total term count). English stopwords, 1-2 grams, min_df=2 on original documents; noise omitted.',
            'all_cluster_descriptors':[{'cluster_id':int(c),'size':sizes[int(c)],'top_words':descriptors[int(c)]} for c in clusters],
            'five_members_each':review,'unconditional_cluster_sample':uniform,'random_noise_examples':noise}
    report[name]=output
    (OUT/f'{name}.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    print(name,'selected',[(r['cluster_id'],r['size']) for r in review],flush=True)
parts=['''<!doctype html><meta charset="utf-8"><title>PETRI cause cluster reading audit</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#203541}details{border:1px solid #ccd7dc;border-radius:8px;padding:14px;margin:14px 0}summary{cursor:pointer;font-weight:bold}blockquote{white-space:pre-wrap;background:#f3f7f8;padding:16px;margin:8px 0}.terms{color:#496573}small{color:#566b75}</style><h1>PETRI cause cluster reading audit</h1><p>Same 801 investigations, verification ≥7. Text consists of investigator-supported statements; those statuses are not independent causal ground truth. c-TF-IDF terms are descriptive words, not human-assigned cluster names.</p><p>For each partition, 10 clusters are sampled uniformly among those with at least 5 members; then 5 random members are shown. This overrepresents larger clusters at k=336. The JSON also contains a separate sample across all cluster sizes. Seed: 20261001.</p>''']
for name,data in report.items():
    parts.append(f'<h2>{name}</h2><p>{data["n_clusters"]} total clusters; {data["n_clusters_with_at_least_5"]} eligible for 5-member reading. <a href="{name}.json">All descriptors and samples</a></p>')
    for cluster in data['five_members_each']:
        terms=', '.join(r['term'] for r in cluster['top_words'])
        parts.append(f'<details><summary>Cluster {cluster["cluster_id"]} · {cluster["size"]} members</summary><p class="terms">c-TF-IDF: {html.escape(terms)}</p>')
        for r in cluster['random_members']:
            parts.append(f'<p><b>{html.escape(r["prompt_id"])}</b> · score {r["verification_score"]}</p><blockquote>{html.escape(r["cause_facet"])}</blockquote><details><summary>Behaviour context (not embedded)</summary>{html.escape(r["behavior_summary"])}</details>')
        parts.append('</details>')
    if data['random_noise_examples']:
        parts.append('<details><summary>5 random unclustered examples</summary>')
        for r in data['random_noise_examples']:parts.append(f'<p>{html.escape(r["prompt_id"])}</p><blockquote>{html.escape(r["cause_facet"])}</blockquote>')
        parts.append('</details>')
(OUT/'review.html').write_text('\n'.join(parts))
(OUT/'README.md').write_text('Random-member reading and c-TF-IDF generated by clustering/scripts/build_cause_reading_review.py. See review.html and partition JSON files for sampling rules, exact text, all cluster terms and additional unconditional samples. Numeric clusters only. Terms describe text and do not verify causes.\n')
