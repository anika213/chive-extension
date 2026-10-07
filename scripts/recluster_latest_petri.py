"""Recluster PETRI using the latest qualifying checkpoint per scenario/run.

Preserves previous outputs and WildChat. Reuses unchanged local MPNet vectors;
refits Ward, UMAP/HDBSCAN, and all null replicates on the selected population.
"""
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re
import statistics

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
from threadpoolctl import threadpool_limits
import torch

import run_checkpoint2 as run
from location_seed_analysis import seed_concentration

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / 'data/checkpoint1_locations/petri'
DATA = ROOT / 'data/petri_latest_qualified'
OUTPUT = ROOT / 'results/petri_latest_qualified'
PRIOR = ROOT / 'results/checkpoint2_locations/petri'


def parse_id(pid):
    match = re.fullmatch(r'petri_(.+)_e(\d+)_t(\d+)', pid)
    if not match: raise ValueError(f'Unexpected PETRI ID: {pid}')
    return (match[1], int(match[2])), int(match[3])


def prepare():
    folder = DATA / 'petri'; folder.mkdir(parents=True, exist_ok=True)
    behaviours = pq.read_table(INPUT / 'behaviours.parquet').to_pylist()
    latest = {}
    for row in behaviours:
        assert row['verif_score'] >= 7
        key, turn = parse_id(row['prompt_id'])
        if key not in latest or turn > parse_id(latest[key]['prompt_id'])[1]: latest[key] = row
    ids = {r['prompt_id'] for r in latest.values()}
    selected_b = [r for r in behaviours if r['prompt_id'] in ids]
    all_causes = pq.read_table(INPUT / 'causes.parquet').to_pylist()
    selected_c = [r for r in all_causes if r['prompt_id'] in ids]
    eligible = [r for r in selected_c if r['clustering_eligible']]
    assert len(selected_b) == 269 and len({k[0] for k in latest}) == 147 and len(eligible) == 1052
    for name, rows in [('behaviours',selected_b),('causes',selected_c),('causes_for_clustering',eligible)]:
        pq.write_table(pa.Table.from_pylist(rows), folder / (name+'.parquet'), compression='zstd')
    counts = Counter(r['prompt_id'] for r in eligible)
    positive = list(counts.values())
    report = {'source':'petri', 'selection':'Apply existing verification >=7 and preprocessing first; choose largest t per (scenario,e). No fallback to earlier checkpoints based on cause availability.',
        'before_behaviours':len(behaviours), 'selected_investigations':len(selected_b), 'scenario_families':len({k[0] for k in latest}),
        'discarded_earlier_checkpoints':len(behaviours)-len(selected_b), 'before_eligible_causes':sum(r['clustering_eligible'] for r in all_causes),
        'retained_cause_rows_including_flagged':len(selected_c), 'eligible_causes':len(eligible),
        'verification_qualified_investigation_ids':sorted(ids),
        'positive_cause_investigations':len(counts), 'zero_eligible_cause_investigations':len(ids)-len(counts),
        'mean_causes_positive_only':float(np.mean(positive)), 'median_causes_positive_only':float(np.median(positive)), 'max_causes':max(positive),
        'selected_turn_distribution':dict(sorted(Counter(parse_id(pid)[1] for pid in ids).items())),
        'wildchat':'unchanged; not reclustered', 'embedding_text':'unchanged; location and seed metadata excluded', 'semantic_labels_assigned':False}
    run.save(folder/'counts.json',report)
    manifest = [{ 'prompt_id':r['prompt_id'], 'scenario':parse_id(r['prompt_id'])[0][0], 'run_number':parse_id(r['prompt_id'])[0][1],
                  'selected_turn':parse_id(r['prompt_id'])[1], 'verif_score':r['verif_score'],
                  'eligible_causes':counts[r['prompt_id']]} for r in selected_b]
    run.save(folder/'selected_investigations.json',manifest)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lines=['# PETRI: latest qualifying checkpoint per run','',report['selection'],'',
           f'- Behaviour rows: {len(behaviours)} → {len(selected_b)} ({report["scenario_families"]} scenario families).',
           f'- Eligible cause experiments: {report["before_eligible_causes"]} → {len(eligible)}.',
           f'- Investigations with eligible causes: {len(counts)}; zero eligible causes: {len(ids)-len(counts)} (excluded from cause-diversity denominators).',
           '- e1 and e2 remain separate. Selected t can be earlier than the final published checkpoint if later checkpoints did not qualify.',
           '- Cause rows retain all existing artifact/size/location-text filters. Flagged rows remain in causes.parquet. Earlier discarded duplicate representatives are not reintroduced.',
           '- Previous results preserved; WildChat unchanged. No semantic cluster labels.','',
           '[Selected investigations](../../data/petri_latest_qualified/petri/selected_investigations.json)',
           '[Behaviour inputs](../../data/petri_latest_qualified/petri/behaviours.parquet)',
           '[Cause inputs](../../data/petri_latest_qualified/petri/causes_for_clustering.parquet)']
    (OUTPUT/'selection.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='verification_qualified_investigation_ids'},indent=2),flush=True)
    return report


def reuse_embeddings(facet):
    name,key=('behaviours','behaviour_text') if facet=='behaviours' else ('causes_for_clustering','cause_text')
    rows=pq.read_table(DATA/'petri'/(name+'.parquet')).to_pylist()
    prior_rows=json.loads((PRIOR/facet/'records.json').read_text())
    identity=lambda r:(r['prompt_id'],r.get('experiment_number'))
    by_id={identity(r):i for i,r in enumerate(prior_rows)}
    indices=[by_id[identity(r)] for r in rows]
    assert all(r[key]==prior_rows[i][key] for r,i in zip(rows,indices))
    vectors=np.load(PRIOR/facet/'embeddings.npy')[indices]
    folder=OUTPUT/'petri'/facet;folder.mkdir(parents=True,exist_ok=True)
    np.save(folder/'embeddings.npy',vectors)
    metadata=json.loads((PRIOR/facet/'embedding_metadata.json').read_text())
    metadata.update(text_sha256=hashlib.sha256(json.dumps([r[key] for r in rows],ensure_ascii=False).encode()).hexdigest(),shape=list(vectors.shape))
    run.save(folder/'embedding_metadata.json',metadata)
    run.save(folder/'embedding_reuse.json',{'source':str(PRIOR/facet), 'source_row_indices':indices,'text_equality_checked':True,
                                         'model':metadata['model'],'model_revision':metadata['model_revision']})


def finish(summaries,selection):
    run.build_report(summaries)
    for name in ['report.md','review.html']:
        path=OUTPUT/name;text=path.read_text()
        text=text.replace('Only PETRI and WildChat eval; separate behaviour and cause embeddings/clusters. Train and thinking deferred.',
            'PETRI only: latest qualifying checkpoint per scenario/run, 269 investigations across 147 scenario families. Behaviours and causes clustered separately. WildChat unchanged.')
        text=text.replace('PETRI and WildChat eval only; behaviours and causes remain separate.',
            'PETRI only: 269 investigations across 147 scenario families, latest qualifying checkpoint per run. Behaviour and cause facets remain separate. WildChat unchanged.')
        text=text.replace('../../data/checkpoint1_approved/review.html','selection.md').replace('approved preprocessing','selection rules and counts').replace('30 eligible original cause summaries per source and revised filters','Selection rules and counts')
        path.write_text(text)
    lines=['## Cluster sizes and cause coverage','',
           '| Facet | Method | Clusters | Minimum size | Median size | Maximum size | Singletons |',
           '|---|---|---:|---:|---:|---:|---:|']
    for result in summaries:
        for method in ['ward','hdbscan']:
            sizes=[g['size'] for g in result[method]['cluster_sizes']]
            lines.append(f'| {result["facet"]} | {method} | {len(sizes)} | {min(sizes) if sizes else "n/a"} | {statistics.median(sizes) if sizes else "n/a"} | {max(sizes) if sizes else "n/a"} | {sizes.count(1)} |')
    lines+=['',f'Cause-diversity counts exclude {selection["zero_eligible_cause_investigations"]} selected investigations with zero eligible causes. HDBSCAN additionally excludes all-noise investigations from assigned-cluster diversity; these counts are in the cause summary JSON.',
            '', 'With only 269 behaviour rows, high k cuts contain many small clusters. Silhouette selection does not establish semantic usefulness; inspect sizes and sampled members. The two runs per scenario remain related observations.',
            '', 'All sample rows below show unchanged embedding text. Cause originals and measured outcomes are supplied for review. No semantic labels were assigned.']
    report=OUTPUT/'report.md';report.write_text(report.read_text()+'\n'+'\n'.join(lines)+'\n')
    page=OUTPUT/'review.html';text=page.read_text()
    text=text.replace('<h1>CHIVE · Checkpoint 2</h1>','<h1>PETRI · latest checkpoint per run</h1>')
    text=text.replace('</pre>','</pre><details><summary>Cluster sizes and interpretation</summary><pre>'+html.escape('\n'.join(lines))+'</pre></details>',1)
    page.write_text(text)
    # The seed diagnostic remains separate from the clustered-samples review.
    assignments=pq.read_table(OUTPUT/'petri/behaviours/assignments.parquet').to_pylist()
    diagnostics={m:seed_concentration(assignments,m) for m in ['ward','hdbscan']}
    run.save(OUTPUT/'behaviour_scenario_concentration.json',diagnostics)


def validate():
    report={'status':'passed','latest_checkpoint_selection':True,'facets':{},'semantic_labels_assigned':False}
    before=pq.read_table(INPUT/'behaviours.parquet').to_pylist()
    selected=pq.read_table(DATA/'petri/behaviours.parquet').to_pylist()
    assert len(selected)==269 and len({parse_id(r['prompt_id'])[0] for r in selected})==269
    for row in selected:
        k,t=parse_id(row['prompt_id'])
        assert t==max(parse_id(r['prompt_id'])[1] for r in before if parse_id(r['prompt_id'])[0]==k)
    ids={r['prompt_id'] for r in selected}
    for facet in ['behaviours','causes']:
        folder=OUTPUT/'petri'/facet
        records=json.loads((folder/'records.json').read_text())
        assert all(r['prompt_id'] in ids for r in records)
        if facet=='causes': assert len(records)==1052 and all(r['clustering_eligible'] for r in records)
        v=np.load(folder/'embeddings.npy');assert np.allclose(np.linalg.norm(v,axis=1),1,atol=1e-6)
        provenance=json.loads((folder/'embedding_reuse.json').read_text())
        assert np.array_equal(v,np.load(PRIOR/facet/'embeddings.npy')[provenance['source_row_indices']])
        cuts=np.load(folder/'ward_all_cuts.npz');summary=json.loads((folder/'summary.json').read_text())
        assert np.array_equal(cuts['k'],np.arange(10,201))
        assert all(len(set(cuts['labels'][:,i]))==k for i,k in enumerate(cuts['k']))
        best=int(np.argmax(cuts['cosine_silhouette']));labels=cuts['labels'][:,best]
        assert summary['ward']['best_k']==cuts['k'][best]
        d=run.distances(v)
        assert abs(silhouette_score(d,labels,metric='precomputed')-summary['ward']['cosine_silhouette'])<2e-6
        density=np.load(folder/'hdbscan_labels.npy');assigned=density>=0
        assert summary['hdbscan']['noise_rows']==int((~assigned).sum())
        if summary['hdbscan']['original_cosine_silhouette_assigned_only'] is not None:
            assert abs(silhouette_score(d[np.ix_(assigned,assigned)],density[assigned],metric='precomputed')-summary['hdbscan']['original_cosine_silhouette_assigned_only'])<2e-6
            assert summary['ari_between_methods']['on_hdbscan_assigned_rows']==float(adjusted_rand_score(labels[assigned],density[assigned]))
        a=pq.read_table(folder/'assignments.parquet').to_pylist()
        assert all(r['prompt_id']==records[i]['prompt_id'] and r['ward_cluster_id']==int(labels[i]) and r['hdbscan_cluster_id']==int(density[i]) for i,r in enumerate(a))
        reading=json.loads((folder/'largest_clusters_reading.json').read_text())
        for method,lab in [('ward',labels),('hdbscan',density)]:
            lookup={(r['prompt_id'],r.get('experiment_number')):int(l) for r,l in zip(records,lab)}
            for g in reading[method]:
                assert len(g['members'])==min(5,g['size'])
                assert all(lookup[r['prompt_id'],r.get('experiment_number')]==g['cluster_id'] for r in g['members'])
            if facet=='causes':assert summary['cause_diversity'][method]==run.cause_diversity(records,lab,sorted(ids))
        nulls=json.loads((folder/'null_replicates.json').read_text())
        assert len(nulls['replicates'])==10 and np.load(folder/'null_curves.npy').shape==(10,191)
        assert nulls['text_sha256']==json.loads((folder/'embedding_metadata.json').read_text())['text_sha256']
        report['facets'][facet]={'rows':len(records),'ward_cuts':191,'null_replicates':10,'reused_vectors_verified':True,'samples_verified':True}
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            a=dict(attrs);target=a.get('href') if tag=='a' else a.get('src') if tag=='img' else None
            if target and not target.startswith('#'):assert (OUTPUT/target).exists(),target
    Links().feed((OUTPUT/'review.html').read_text())
    run.save(OUTPUT/'validation.json',report);print(json.dumps(report,indent=2),flush=True)


def main():
    selection=prepare()
    run.INPUT=DATA;run.OUTPUT=OUTPUT;run.SOURCES=('petri',)
    run.save(OUTPUT/'design.json',{'selection':selection['selection'],'sources':['petri'],'facets':['behaviours','causes'],
        'k':run.KS.tolist(),'null_replicates':run.N_NULL,'seed':run.SEED,'embedding_reuse':'exact subset of unchanged normalized MPNet vectors',
        'ward':'Euclidean Ward on unit vectors','density':'UMAP10 cosine neighbors30 min_dist0; HDBSCAN min_cluster_size10 min_samples5 EOM',
        'semantic_labels':False})
    for facet in run.FACETS:reuse_embeddings(facet)
    torch.set_num_threads(4)
    model=SentenceTransformer(str(run.MODEL),device='cpu',local_files_only=True)
    summaries=[]
    with threadpool_limits(limits=4):
        for facet in run.FACETS:
            summaries.append(run.run_dataset('petri',facet,model))
            run.save(OUTPUT/'summary.json',summaries)
        finish(summaries,selection)
        validate()
    print('COMPLETE — stopped before labelling:',OUTPUT/'review.html',flush=True)


if __name__=='__main__':main()
