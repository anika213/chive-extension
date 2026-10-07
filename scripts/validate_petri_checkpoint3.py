"""Independent count, residual, probability and artifact checks for checkpoint 3."""
import json
import hashlib
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import silhouette_score,adjusted_rand_score

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'results/petri_representation_variants/checkpoint3'
SOURCE=ROOT/'results/petri_representation_variants/step1'
records=pq.read_table(ROOT/'data/checkpoint1_locations/petri/behaviours.parquet').to_pylist()
seeds=np.array([r['scenario_seed_proxy'] for r in records]);reports=json.loads((OUT/'comparison.json').read_text())
protocol=json.loads((OUT/'protocol.json').read_text());checks={}
assert len(reports)==6
for path,h in protocol['source_hashes'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
all_labels={}
for name,report in reports.items():
    folder=OUT/name;m=report['metrics'];v=np.load(folder/'embeddings.npy');rows=pq.read_table(folder/'assignments.parquet').to_pylist()
    assert [(r['prompt_id'],r['behaviour_text']) for r in rows]==[(r['prompt_id'],r['behaviour_text']) for r in records]
    labels=np.array([r['cluster_id'] for r in rows]);all_labels[name]=labels;mask=labels>=0
    assert np.isfinite(v).all() and np.allclose(np.linalg.norm(v,axis=1),1,atol=1e-5)
    counts=Counter(labels[mask]);assert sum(counts.values())==m['assigned']
    assert len(counts)==m['clusters']
    largest=max(counts.values()) if counts else 0;assert largest==m['largest_cluster']['size']
    if counts:assert abs(largest/mask.sum()-m['largest_cluster']['share_of_assigned'])<1e-12
    big=[k for k,n in counts.items() if n>.15*mask.sum()]
    trim=mask&~np.isin(labels,big);tm=m['seed_concentration_excluding_large_clusters']
    assert trim.sum()==tm['retained_rows'];assert set(big)=={x['cluster_id'] for x in tm['excluded_clusters']}
    for selected,s in [(mask,m['seed_concentration']),(trim,tm['metrics'])]:
        keys=sorted(set(labels[selected]));groups=[seeds[selected&(labels==k)] for k in keys]
        if groups:
            tops=[max(Counter(g).values()) for g in groups]
            assert abs(sum(tops)/selected.sum()-s['observed']['weighted_top_one_share'])<1e-12
            assert sum(t/len(g)>.5 for t,g in zip(tops,groups))==s['observed']['clusters_majority_one_scenario']
            assert np.median([len(set(g)) for g in groups])==s['observed']['median_distinct_scenarios']
        else:assert s['observed']['weighted_top_one_share'] is None
    if m['cosine_silhouette'] is not None:
        assert abs(float(silhouette_score(v[mask],labels[mask],metric='cosine'))-m['cosine_silhouette'])<3e-6
    original=np.load(SOURCE/report['embedding']/'embeddings.npy')
    if report['residualized']:
        for family in set(seeds):
            ix=np.flatnonzero(seeds==family)
            if len(ix)==1:assert np.array_equal(original[ix],v[ix])
            else:
                expected=original[ix].astype(float)-original[ix].astype(float).mean(0)
                expected/=np.linalg.norm(expected,axis=1,keepdims=True)
                assert np.allclose(expected,v[ix],atol=1e-7)
        assert report['residualization']['singleton_rows_unchanged']==12
    for group in json.loads((folder/'samples.json').read_text()):
        assert len(group['members'])==min(5,group['size'])
        assert len({r['prompt_id'] for r in group['members']})==len(group['members'])
        for member in group['members']:
            assert next(r['cluster_id'] for r in rows if r['prompt_id']==member['prompt_id'])==group['cluster_id']
    null=json.loads((folder/'scenario_null_replicates.json').read_text())
    assert len(null['all_clusters'])==len(null['excluding_large_clusters'])==200
    assert len(json.loads((folder/'embedding_null_replicates.json').read_text()))==10
    if report['method']=='gmm':
        probs=np.load(folder/'membership_probabilities.npy');assert probs.shape==(801,20)
        assert np.all(probs>=0) and np.allclose(probs.sum(1),1,atol=1e-5)
        assert np.array_equal(probs.argmax(1),labels)
        assert int((probs.max(1)<.5).sum())==report['method_diagnostics']['ambiguous_max_probability_below_0_5']
        assert np.allclose(np.array([r['membership_probabilities'] for r in rows]),probs)
    if report['method']=='leiden':
        edges=np.load(folder/'graph_edges.npy');weights=np.load(folder/'graph_weights.npy');neighbors=np.load(folder/'knn_indices.npy')
        assert neighbors.shape==(801,15) and (weights>0).all()
        assert len({tuple(e) for e in edges})==len(edges)
        assert np.all(edges[:,0]<edges[:,1])
        assert all(i not in neighbors[i] for i in range(801))
    checks[name]={'input_rows_unchanged':True,'seed_and_size_metrics_checked':True,'silhouette_checked':True,'samples_checked':True,'residuals_checked':bool(report['residualized'])}
for pair in json.loads((OUT/'pairwise_ari.json').read_text()):
    a,b=all_labels[pair['variant_a']],all_labels[pair['variant_b']];mask=(a>=0)&(b>=0)
    if mask.sum()>=2:assert adjusted_rand_score(a[mask],b[mask])==pair['jointly_assigned_ari']
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        for field in ['href','src']:
            if field in a and not urlparse(a[field]).scheme:self.links.append(a[field])
parser=Links();parser.feed((OUT/'review.html').read_text())
for link in parser.links:assert (OUT/link).exists(),link
checks['local_links_checked']=len(parser.links);checks['source_files_unchanged']=True;checks['ari_pairs_checked']=15
(OUT/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
