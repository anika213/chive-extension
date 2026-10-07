"""Validate location provenance, text-only embeddings, membership and concentration."""
import hashlib
import json
from html.parser import HTMLParser
import runpy

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import silhouette_score

import location_seed_analysis as prep
import run_checkpoint2 as run


def main():
    unit = runpy.run_path(str(prep.ROOT / 'tests/test_location_seed.py'))
    tests = [name for name in unit if name.startswith('test_')]
    for name in tests: unit[name]()
    result = {'status':'passed','unit_tests':tests,'sources':{},'seed_proxy_analysis_completed':False}
    for source in prep.SOURCES:
        original = pq.read_table(prep.INPUT/source/'causes.parquet').to_pylist()
        by_key = {(r['prompt_id'],r['experiment_number']):r for r in original}
        rows = pq.read_table(prep.OUTPUT/source/'causes.parquet').to_pylist()
        eligible = pq.read_table(prep.OUTPUT/source/'causes_for_clustering.parquet').to_pylist()
        assert len(rows)==len(original)
        assert eligible==[r for r in rows if r['clustering_eligible']]
        assert len({r['cause_text'] for r in eligible})==len(eligible)
        for r in rows:
            old=by_key[r['prompt_id'],r['experiment_number']]
            assert r['cause_text_original']==old['cause_text_original']
            assert r['cause_text_with_location']==old['cause_text']
            assert (r['cause_text'],r['location_text_removed'])==prep.strip_location(old['cause_text'])
            assert r['edit_location']=='+'.join(r['edit_locations'])
            for key in ['baseline_rate','counterfactual_rate','delta_pp','is_key','frac_changed','measurement_artifact','context_swap','context_swap_relative']:
                assert r[key]==old[key]
            if r['clustering_eligible']:
                assert old['clustering_eligible'] and r['cause_text'] and not r['location_only_text']
        folder=prep.ROOT/'results/checkpoint2_locations'/source/'causes'
        records=json.loads((folder/'records.json').read_text());assert records==eligible
        metadata=json.loads((folder/'embedding_metadata.json').read_text())
        signature=hashlib.sha256(json.dumps([r['cause_text'] for r in eligible],ensure_ascii=False).encode()).hexdigest()
        assert metadata['text_sha256']==signature
        vectors=np.load(folder/'embeddings.npy');assert vectors.shape==(len(eligible),768)
        assert np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-6)
        cuts=np.load(folder/'ward_all_cuts.npz')
        assert np.array_equal(cuts['k'],np.arange(10,201))
        assert all(len(set(cuts['labels'][:,i]))==k for i,k in enumerate(cuts['k']))
        d=run.distances(vectors)
        for i in [0,95,190]:
            assert abs(silhouette_score(d,cuts['labels'][:,i],metric='precomputed')-cuts['cosine_silhouette'][i])<2e-6
        summary=json.loads((folder/'summary.json').read_text())
        assert summary['ward']['best_k']==int(cuts['k'][np.argmax(cuts['cosine_silhouette'])])
        density=np.load(folder/'hdbscan_labels.npy');assert summary['hdbscan']['noise_rows']==int((density<0).sum())
        assignments=pq.read_table(folder/'assignments.parquet').to_pylist()
        ward=cuts['labels'][:,np.argmax(cuts['cosine_silhouette'])]
        assert all(r['cause_text']==eligible[i]['cause_text'] and r['ward_cluster_id']==ward[i] and r['hdbscan_cluster_id']==density[i] for i,r in enumerate(assignments))
        counts=json.loads((prep.OUTPUT/source/'counts.json').read_text())
        ids=counts['verification_qualified_investigation_ids']
        for method,labels in [('ward',ward),('hdbscan',density)]:
            assert summary['cause_diversity'][method]==run.cause_diversity(records,labels,ids)
        nulls=json.loads((folder/'null_replicates.json').read_text())
        assert nulls['text_sha256']==signature and len(nulls['replicates'])==10
        assert np.load(folder/'null_curves.npy').shape==(10,191)
        oldb=prep.ROOT/'results/checkpoint2'/source/'behaviours'
        newb=prep.ROOT/'results/checkpoint2_locations'/source/'behaviours'
        for name in ['embeddings.npy','ward_all_cuts.npz','hdbscan_labels.npy','assignments.parquet']:
            assert (oldb/name).read_bytes()==(newb/name).read_bytes()
        cross=json.loads((prep.RESULT/source/'location_crosstabs.json').read_text())
        for facet in ['behaviours','causes']:
            for method in ['ward','hdbscan']:
                assert sum(r['cause_rows'] for r in cross if r['facet']==facet and r['method']==method)==len(eligible)
        samples=json.loads((prep.RESULT/source/'text_removal_samples.json').read_text())
        assert len(samples)==30 and all(r in eligible for r in samples)
        if (prep.RESULT/'seed_summary.json').exists():
            result['seed_proxy_analysis_completed']=True
            b=pq.read_table(prep.RESULT/source/'behaviour_assignments.parquet').to_pylist()
            assert len(b)==len(pq.read_table(oldb/'assignments.parquet'))
            for method in ['ward','hdbscan']:
                expected=prep.seed_concentration(b,method)
                assert expected==json.loads((prep.RESULT/source/f'{method}_seed_concentration.json').read_text())
                assert all(r['top1_fraction'] is None and r['distinct_seeds'] is None for r in expected) if source=='wildchat_eval' else all(r['missing_seed_rows']==0 for r in expected)
        result['sources'][source]={'eligible_causes':len(eligible),'retained_causes':len(rows),'unchanged_behaviour_assignments':True,
                                  'text_only_embeddings':True,'cross_tabs_validated':True,'null_replicates':10}
        print('Validated',source,flush=True)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if tag=='a' and a.get('href'):self.links.append(a['href'])
            if tag=='img' and a.get('src'):self.links.append(a['src'])
    for page in [prep.RESULT/'review.html',prep.ROOT/'results/checkpoint2_locations/review.html']:
        parser=Links();parser.feed(page.read_text())
        for path in parser.links:
            if not path.startswith('#'):assert (page.parent/path).exists(),path
    prep.save(prep.RESULT/'validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
