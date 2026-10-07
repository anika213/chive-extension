"""Validate approved amendments, complete clustering outputs and review artifacts."""
from collections import Counter
from html.parser import HTMLParser
import hashlib
import json
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import adjusted_rand_score, silhouette_score

import prepare_checkpoint2 as prep
import run_checkpoint2 as run


def row_key(row):
    return row['prompt_id'], row.get('experiment_number')


def main():
    report = {'status': 'passed', 'sources': {}, 'runs': {}, 'semantic_labels_assigned': False}
    for source in prep.SOURCES:
        folder = prep.OUTPUT / source
        rows = pq.read_table(folder / 'causes.parquet').to_pylist()
        eligible = pq.read_table(folder / 'causes_for_clustering.parquet').to_pylist()
        assert eligible == [r for r in rows if r['clustering_eligible']]
        old = {row_key(r): r for r in pq.read_table(prep.INPUT / source / 'causes.parquet').to_pylist()}
        for r in rows:
            assert r == prep.update_row(old[row_key(r)])
        assert all(not r['context_swap'] and not r['context_swap_relative'] and not r['measurement_artifact']
                   and r['frac_changed'] <= .5 and not r['clustering_exclusion_reasons'] for r in eligible)
        assert len(rows) == len(old)
        stats = json.loads((folder / 'counts.json').read_text())
        ids = stats['verification_qualified_investigation_ids']
        assert stats['causes_per_investigation'] == prep.positive_stats(eligible, ids)
        n_positive = len({r['prompt_id'] for r in eligible})
        assert stats['causes_per_investigation']['mean'] == len(eligible) / n_positive
        samples = json.loads((folder / 'eligible_cause_originals_30.json').read_text())
        assert len(samples) == 30 and len({row_key(r) for r in samples}) == 30
        eligible_keys = {row_key(r) for r in eligible}
        assert all(row_key(r) in eligible_keys and r['cause_text_original'] == old[row_key(r)]['cause_text_original'] for r in samples)
        report['sources'][source] = {'eligible_causes': len(eligible), 'original_summary_review_rows': len(samples),
                                     'positive_cause_investigation_denominator': n_positive,
                                     'zero_cause_investigations_excluded': len(ids) - n_positive}
        for facet in run.FACETS:
            path = run.OUTPUT / source / facet
            summary = json.loads((path / 'summary.json').read_text())
            records = json.loads((path / 'records.json').read_text())
            name = 'behaviours' if facet == 'behaviours' else 'causes_for_clustering'
            input_path = folder / (name + '.parquet')
            assert records == pq.read_table(input_path).to_pylist()
            assert summary['input_sha256'] == hashlib.sha256(input_path.read_bytes()).hexdigest()
            vectors = np.load(path / 'embeddings.npy')
            assert vectors.shape == (len(records), 768) and np.isfinite(vectors).all()
            assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
            cuts = np.load(path / 'ward_all_cuts.npz')
            assert np.array_equal(cuts['k'], run.KS)
            assert cuts['labels'].shape == (len(records), 191)
            assert all(len(set(cuts['labels'][:,i])) == k for i, k in enumerate(run.KS))
            index = int(np.argmax(cuts['cosine_silhouette']))
            labels = cuts['labels'][:,index]
            assert summary['ward']['best_k'] == int(run.KS[index])
            d = run.distances(vectors)
            assert abs(float(silhouette_score(d, labels, metric='precomputed')) - summary['ward']['cosine_silhouette']) < 2e-6
            density = np.load(path / 'hdbscan_labels.npy')
            assigned = density >= 0
            assert summary['hdbscan']['noise_rows'] == int((~assigned).sum())
            assert summary['hdbscan']['noise_fraction'] == float((~assigned).mean())
            assert summary['ward']['cluster_sizes'] == run.cluster_sizes(labels)
            assert summary['hdbscan']['cluster_sizes'] == run.cluster_sizes(density)
            if summary['hdbscan']['original_cosine_silhouette_assigned_only'] is not None:
                expected = silhouette_score(d[np.ix_(assigned,assigned)], density[assigned], metric='precomputed')
                assert abs(float(expected) - summary['hdbscan']['original_cosine_silhouette_assigned_only']) < 2e-6
                assert summary['ari_between_methods']['on_hdbscan_assigned_rows'] == float(adjusted_rand_score(labels[assigned],density[assigned]))
            nulls = json.loads((path / 'null_replicates.json').read_text())['replicates']
            null_curves = np.load(path / 'null_curves.npy')
            assert len(nulls) == 10 and null_curves.shape == (10, 191) and np.isfinite(null_curves).all()
            assert np.allclose(null_curves.max(axis=1), [r['best_ward_cosine_silhouette'] for r in nulls])
            assert np.allclose(null_curves[:,index], [r['ward_at_observed_best_k'] for r in nulls])
            assignments = pq.read_table(path / 'assignments.parquet').to_pylist()
            assert len(assignments) == len(records)
            assert all(r['ward_cluster_id'] == int(labels[i]) and r['hdbscan_cluster_id'] == int(density[i])
                       and row_key(r) == row_key(records[i]) for i, r in enumerate(assignments))
            reading = json.loads((path / 'largest_clusters_reading.json').read_text())
            for method, lab in [('ward', labels), ('hdbscan', density)]:
                assert len(reading[method]) == min(10, len(set(lab[lab >= 0])))
                by_key = {row_key(r): int(l) for r, l in zip(records, lab)}
                for group in reading[method]:
                    assert len(group['members']) == min(5, group['size'])
                    assert all(by_key[row_key(r)] == group['cluster_id'] for r in group['members'])
            if facet == 'causes':
                assert summary['cause_diversity']['ward'] == run.cause_diversity(records, labels, ids)
                assert summary['cause_diversity']['hdbscan'] == run.cause_diversity(records, density, ids)
            pairs = pq.read_table(path / 'near_duplicates.parquet').to_pylist()
            assert len(pairs) == summary['near_duplicates']['pairs']
            assert all(r['row_i'] < r['row_j'] and r['cosine_similarity'] > .95 for r in pairs)
            report['runs'][source + '/' + facet] = {'rows':len(records), 'ward_cuts_validated':191, 'null_replicates':10,
                                                   'normalized_embeddings':True, 'reading_samples_validated':True,
                                                   'near_duplicate_pairs':len(pairs)}
            print('Validated',source,facet,flush=True)
    class Links(HTMLParser):
        def __init__(self):
            super().__init__();self.targets=[]
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if tag=='a' and a.get('href'):self.targets.append(a['href'])
            if tag=='img' and a.get('src'):self.targets.append(a['src'])
    parser=Links();parser.feed((run.OUTPUT/'review.html').read_text())
    for target in parser.targets:
        if not target.startswith('#'):assert (run.OUTPUT/target).exists(),target
    run.save(run.OUTPUT/'validation.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
