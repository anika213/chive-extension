"""Independent numeric/membership/content checks and optional local Chrome render."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import os
import signal
import subprocess
import tempfile
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import silhouette_samples

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/petri_latest_qualified/cosine_refinement'
INPUT = ROOT / 'results/petri_latest_qualified/petri/behaviours'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--browser', action='store_true'); args = parser.parse_args()
    records = json.loads((INPUT / 'records.json').read_text())
    vectors = np.load(INPUT / 'embeddings.npy').astype(float); vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine = vectors @ vectors.T
    summary = json.loads((OUT / 'summary.json').read_text())
    candidates = json.loads((OUT / 'search_candidates.json').read_text())
    signatures = set()
    for c in candidates:
        flat = sum(c['groups'], [])
        assert len(flat) == len(set(flat)) == c['assigned']
        assert min(map(len, c['groups'])) >= 10
        assert all(0 <= i < len(records) for i in flat)
        sig = tuple(sorted(tuple(sorted(g)) for g in c['groups']))
        assert sig not in signatures; signatures.add(sig)
        means = [float(cosine[np.ix_(g, g)][np.triu_indices(len(g), 1)].mean()) for g in c['groups']]
        assert min(means) > .4
        assert np.allclose(means, c['cluster_mean_cosines'], atol=1e-12)
    best = max(c['min_cluster_mean_cosine'] for c in candidates)
    eligible = [c for c in candidates if c['min_cluster_mean_cosine'] >= best - .02]
    assert summary['presets']['recommended']['assigned'] == max(c['assigned'] for c in eligible)
    checks = 0
    for name, result in summary['presets'].items():
        rows = pq.read_table(OUT / name / 'assignments.parquet').to_pylist()
        assert [r['prompt_id'] for r in rows] == [r['prompt_id'] for r in records]
        ids = np.array([i for i, r in enumerate(rows) if r['cluster_id'] >= 0])
        labels = np.array([rows[i]['cluster_id'] for i in ids])
        distance = np.clip(1 - cosine[np.ix_(ids, ids)], 0, 2); np.fill_diagonal(distance, 0)
        s = silhouette_samples(distance, labels, metric='precomputed')
        assert np.allclose(s, [rows[i]['cosine_silhouette'] for i in ids])
        for k in sorted(set(labels)):
            group = ids[labels == k]
            page = (OUT / name / f'cluster_{k}.html').read_text()
            assert page.count('class="card member"') == len(group)
            for i in group:
                assert html.escape(records[i]['behaviour_text'], quote=True) in page
            checks += len(group)
        page = (OUT / name / 'unassigned.html').read_text()
        assert page.count('class="card member"') == len(rows) - len(ids)
    old = pq.read_table(INPUT / 'assignments.parquet').to_pylist()
    for minimum in [2, 10]:
        folder = OUT / f'within_min{minimum}'
        within = json.loads((folder / 'summary.json').read_text())
        ids = []
        for g in within['groups']:
            members = g['row_indices']; ids.extend(members)
            assert all(old[i]['hdbscan_cluster_id'] == g['parent_cluster'] for i in members)
            assert len(members) >= minimum
            assert len(members) < sum(r['hdbscan_cluster_id'] == g['parent_cluster'] for r in old)
            mean = cosine[np.ix_(members, members)][np.triu_indices(len(members), 1)].mean()
            assert mean > .4 and np.isclose(mean, g['mean_pairwise_cosine'])
            page = (folder / (g['id'] + '.html')).read_text()
            assert page.count('class="card member"') == len(members)
            for i in members: assert html.escape(records[i]['behaviour_text'], quote=True) in page
            checks += len(members)
        assert len(ids) == len(set(ids)) == within['assigned']
        assert len(ids) + within['residual'] == 172
    prior = json.loads((OUT / 'validation.json').read_text())
    for p, digest in prior['input_hashes'].items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
    report = {'status': 'passed', 'unique_candidate_partitions': len(candidates), 'all_candidate_cohesions_recomputed': True,
              'global_minimum_size': 10, 'silhouettes_independently_recomputed': True, 'exact_member_texts_checked': checks,
              'subgroups_within_parent_and_disjoint': True, 'selection_rule_verified': True, 'previous_inputs_unchanged': True}
    (OUT / 'independent_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if not args.browser: return
    profile = tempfile.mkdtemp(prefix='chive-cosine-review-chrome-')
    command = ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '--headless', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
               '--disable-background-networking', '--disable-component-update', '--disable-sync', '--disable-crash-reporter', '--user-data-dir=' + profile,
               '--window-size=1400,1800', '--virtual-time-budget=1000', '--screenshot=' + str(OUT / 'review_preview.png'), '--dump-dom', (OUT / 'review.html').as_uri()]
    with tempfile.TemporaryFile(mode='w+') as stdout, tempfile.TemporaryFile(mode='w+') as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        try: process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM); process.wait(timeout=5)
        stdout.seek(0); dom = stdout.read()
    assert 'Cohesive subclusters and selective reclustering' in dom and '<table>' in dom
    assert (OUT / 'review_preview.png').exists()
    (OUT / 'browser_validation.json').write_text(json.dumps({'status': 'passed', 'page': 'review.html', 'static_dom_rendered': True}) + '\n')
    print('Local HTML rendered in isolated Chrome profile.', flush=True)


if __name__ == '__main__': main()
