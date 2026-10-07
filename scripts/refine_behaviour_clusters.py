"""Explore cohesive PETRI behaviour groups using saved vectors; no model calls.

Preserves every previous assignment. Outputs are exploratory, selected on the
same embeddings on which they are scored, and make no causal or semantic labels.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import hashlib
import html
import json
from collections import Counter
from html.parser import HTMLParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples

RUN = ROOT / 'results/petri_latest_qualified'
INPUT = RUN / 'petri/behaviours'
OUT = RUN / 'cosine_refinement'
MIN_COSINE = .4
QUALITY_TOLERANCE = .02


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def esc(value):
    return html.escape(str(value), quote=True)


def signature(groups):
    return tuple(sorted(tuple(sorted(map(int, g))) for g in groups))


def cohesion(g, cosine):
    block = cosine[np.ix_(g, g)]
    return float((block.sum() - len(g)) / (len(g) * (len(g) - 1)))


def individual_scores(groups, distance):
    """Exact cosine silhouette on assigned rows only, excluding noise."""
    groups = [np.asarray(g, dtype=int) for g in groups]
    indices = np.concatenate(groups)
    means = np.column_stack([distance[np.ix_(indices, g)].mean(axis=1) for g in groups])
    labels = np.concatenate([np.full(len(g), k) for k, g in enumerate(groups)])
    sizes = np.array([len(g) for g in groups])
    a = means[np.arange(len(indices)), labels] * sizes[labels] / (sizes[labels] - 1)
    means[np.arange(len(indices)), labels] = np.inf
    b = means.min(axis=1)
    s = np.divide(b - a, np.maximum(a, b), out=np.zeros_like(a), where=np.maximum(a, b) > 0)
    return indices, labels, s


def stats(groups, cosine, distance):
    indices, labels, values = individual_scores(groups, distance)
    per_cluster = [float(values[labels == k].mean()) for k in range(len(groups))]
    cosines = [cohesion(g, cosine) for g in groups]
    return {
        'clusters': len(groups), 'assigned': len(indices), 'unassigned': len(cosine) - len(indices),
        'coverage': len(indices) / len(cosine), 'min_cluster_size': min(map(len, groups)),
        'min_cluster_mean_cosine': min(cosines), 'mean_cluster_cosine': float(np.mean(cosines)),
        'min_cluster_mean_silhouette': min(per_cluster), 'mean_silhouette': float(values.mean()),
        'cluster_mean_cosines': cosines, 'cluster_mean_silhouettes': per_cluster,
        'cluster_sizes': list(map(len, groups)),
    }


CSS = '''
*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:#193344;font:16px/1.6 system-ui,sans-serif}
main{max-width:1200px;margin:auto;padding:28px 22px 70px}h1{font-size:34px;line-height:1.2}h2{margin-top:32px}
a{color:#086c79}table{width:100%;border-collapse:collapse;background:white;font-size:14px}td,th{padding:11px;border:1px solid #d0dce2;text-align:left}th{background:#eaf1f4}.scroll{overflow:auto}
.card{background:white;padding:22px;border:1px solid #d0dce2;border-radius:10px;margin:18px 0}.text{white-space:pre-wrap;overflow-wrap:anywhere}.muted{font-size:14px;color:#5e707d}code{overflow-wrap:anywhere;font-size:13px}.stats{display:flex;flex-wrap:wrap;gap:14px}.stat{background:white;padding:14px 18px;border:1px solid #d0dce2;border-radius:8px}.stat strong{display:block;font-size:25px}img{max-width:100%}.eyebrow{font-size:12px;color:#086c79;text-transform:uppercase;letter-spacing:.1em}.notice{padding:16px;background:#fff4d9;border-left:4px solid #b58213}
@media(max-width:650px){main{padding:20px 12px}h1{font-size:28px}.card{padding:16px}}
'''


def page(path, title, body):
    path.write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{CSS}</style></head><body><main><div class="eyebrow">PETRI · Qwen3-8B · behaviour summaries</div><h1>{esc(title)}</h1>{body}</main></body></html>')


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def member_page(path, title, group, records, cosine, silhouette_by_row=None, note='', is_cluster=True):
    n = len(group)
    means = {i: float((cosine[i, group].sum() - 1) / (n - 1)) for i in group} if n > 1 and is_cluster else {}
    ordered = sorted(group, key=lambda i: (-means.get(i, 0), records[i]['prompt_id']))
    body = '<p><a href="review.html">Back to groups</a></p>' + note
    if n > 1 and is_cluster:
        block = cosine[np.ix_(group, group)]
        pairs = block[np.triu_indices(n, 1)]
        body += f'<p><strong>{n} members · mean pairwise cosine {pairs.mean():.3f}</strong> · pair range {pairs.min():.3f}–{pairs.max():.3f}</p>'
    if is_cluster:
        body += '<p class="muted">Every entry is shown in full, ordered by mean similarity to the other displayed members. A group mean above 0.4 does not require every pair or member average to exceed 0.4. No semantic labels have been assigned.</p>'
    else:
        body += f'<p>{n} entries, ordered by investigation ID. This is a residual list, not a cluster.</p>'
    for rank, i in enumerate(ordered, 1):
        r = records[i]
        s = '' if silhouette_by_row is None else f' · silhouette {silhouette_by_row[i]:.3f}'
        mean = f' · mean to others {means[i]:.3f}' if i in means else ''
        scenario = f'Scenario: {esc(r["scenario"])} · ' if r.get('scenario') else ''
        body += f'<article class="card member" id="row-{i}"><strong>{rank}. {esc(r["prompt_id"])}</strong><p class="muted">{scenario}Verification {r["verif_score"]}/10{mean}{s}</p><p class="text">{esc(r["behaviour_text"])}</p></article>'
    page(path, title, body)


def within_clusters(records, vectors, cosine, old_labels):
    results = {}
    for minimum in [2, 10]:
        folder = OUT / f'within_min{minimum}'
        folder.mkdir(parents=True, exist_ok=True)
        all_groups, rows, residual = [], [], []
        by_parent = []
        for parent in sorted(set(old_labels) - {-1}):
            ids = np.flatnonzero(old_labels == parent)
            distance = np.clip(1 - cosine[np.ix_(ids, ids)], 0, 2)
            np.fill_diagonal(distance, 0)
            tree = linkage(squareform(distance, checks=False), method='average')
            n = len(ids)
            leaves = {i: [int(ids[i])] for i in range(n)}
            for k, node in enumerate(tree):
                leaves[n + k] = leaves[int(node[0])] + leaves[int(node[1])]
            accepted, rejected = [], []

            def descend(node):
                group = leaves[node]
                if len(group) < minimum:
                    rejected.extend(group)
                elif cohesion(group, cosine) > MIN_COSINE:
                    accepted.append(group)
                elif node >= n:
                    descend(int(tree[node - n, 0])); descend(int(tree[node - n, 1]))
                else:
                    rejected.extend(group)

            # Always inspect proper children: a parent already above 0.4 should
            # not simply be returned unchanged as its own "subcluster".
            descend(int(tree[-1, 0])); descend(int(tree[-1, 1]))
            assert sorted(sum(accepted, []) + rejected) == sorted(ids.tolist())
            accepted.sort(key=lambda g: (-len(g), -cohesion(g, cosine), min(g)))
            for sub, g in enumerate(accepted, 1):
                name = f'parent_{parent}_sub_{sub}'
                mean = cohesion(g, cosine)
                assert len(g) >= minimum and mean > .4 and len(g) < len(ids)
                member_page(folder / f'{name}.html', f'Original cluster {parent} · subcluster {sub}', g, records, cosine)
                all_groups.append({'id': name, 'parent_cluster': int(parent), 'subcluster': sub,
                                   'members': len(g), 'mean_pairwise_cosine': mean, 'row_indices': g})
                rows.append([f'<a href="{name}.html">{parent}.{sub}</a>', str(parent), str(len(g)), f'{mean:.3f}', 'Every entry'])
            by_parent.append({'parent_cluster': int(parent), 'parent_members': len(ids), 'subclusters': len(accepted),
                              'assigned': sum(map(len, accepted)), 'residual': len(rejected)})
            residual.extend(rejected)
        member_page(folder / 'unassigned.html', 'Items outside qualifying subclusters', residual, records, cosine,
                    note='<p>These are residual members of the original six clusters, not a new cluster.</p>', is_cluster=False)
        summary = {'minimum_size': minimum, 'strict_mean_cosine_threshold': .4, 'groups': all_groups,
                   'by_parent': by_parent, 'assigned': sum(g['members'] for g in all_groups),
                   'residual': len(residual), 'original_noise_not_considered': int((old_labels < 0).sum())}
        save(folder / 'summary.json', summary)
        labels = np.full(len(records), -1)
        for k, g in enumerate(all_groups): labels[g['row_indices']] = k
        pq.write_table(pa.Table.from_pylist([{**r, 'original_hdbscan_cluster': int(old_labels[i]),
            'subcluster_id': all_groups[labels[i]]['id'] if labels[i] >= 0 else None} for i, r in enumerate(records)]), folder / 'assignments.parquet')
        body = f'<p><a href="../review.html">All analyses</a> · <a href="../within_min{10 if minimum == 2 else 2}/review.html">Switch to minimum {10 if minimum == 2 else 2} members</a></p><p>Proper subgroups of the six existing HDBSCAN clusters. Average-linkage cosine trees are split at the root, then recursively split until each retained group has mean pairwise cosine <strong>&gt; 0.4</strong> and at least <strong>{minimum}</strong> members. Accepted groups are disjoint. This is a deterministic hierarchy-based search, not an exhaustive search over all possible subsets.</p><p><strong>{len(all_groups)} subclusters · {summary["assigned"]}/172 original cluster members retained · {len(residual)} residual members.</strong> The original 97 HDBSCAN noise items are outside this within-cluster analysis.</p>'
        body += table(['Subcluster', 'Original cluster', 'Members', 'Mean cosine', 'Review'], rows)
        body += '<p><a href="unassigned.html">View every residual member</a> · <a href="assignments.parquet">Assignments</a> · <a href="summary.json">Metrics and membership data</a></p>'
        page(folder / 'review.html', f'Within-cluster subgroups · minimum {minimum} members', body)
        results[str(minimum)] = summary
    return results


def global_search(vectors, cosine, distance, old_labels):
    candidates, seen_bases, seen_results = [], set(), set()
    n = len(vectors)

    def register(groups, provenance):
        sig = signature(groups)
        if len(groups) < 2 or sig in seen_results: return
        seen_results.add(sig)
        groups = [list(g) for g in sig]
        st = stats(groups, cosine, distance)
        assert st['min_cluster_size'] >= 10 and st['min_cluster_mean_cosine'] > .4
        candidates.append({'candidate_id': len(candidates), 'groups': groups, 'provenance': provenance, **st})

    def peel(groups, provenance):
        groups = [list(g) for g in groups]
        # Save the partition and the path obtained by removing its least
        # cohesive cluster. Silhouette never controls selection or trimming.
        while len(groups) >= 2:
            register(groups, provenance)
            if len(groups) == 2: break
            means = [cohesion(g, cosine) for g in groups]
            del groups[int(np.argmin(means))]

    def process(initial, provenance):
        for target in [.400000001, .425, .45, .475, .50, .525, .55, .575, .60, .625, .65, .675, .70, .75, .80, .85, .90, .95]:
            groups = []
            for initial_group in initial:
                g = list(initial_group)
                # Centrality trimming allows peripheral items to stay unassigned.
                while len(g) >= 10 and cohesion(g, cosine) < target:
                    means = (cosine[np.ix_(g, g)].sum(axis=1) - 1) / (len(g) - 1)
                    del g[int(np.argmin(means))]
                if len(g) >= 10: groups.append(g)
            sig = signature(groups)
            if len(groups) < 2 or sig in seen_bases: continue
            seen_bases.add(sig)
            prov = {**provenance, 'cohesion_trim_target': target}
            peel(groups, {**prov, 'member_trimming': 'centrality to target'})
            # Explore tighter cores without imposing inter-cluster separation.
            for maximum_size in [10, 12, 15, 20]:
                trimmed = []
                for initial_group in groups:
                    g = list(initial_group)
                    while len(g) > maximum_size:
                        means = (cosine[np.ix_(g, g)].sum(axis=1) - 1) / (len(g) - 1)
                        del g[int(np.argmin(means))]
                    trimmed.append(g)
                peel(trimmed, {**prov, 'member_trimming': 'centrality to size', 'maximum_size': maximum_size})

    for method in ['ward', 'average', 'complete']:
        tree = linkage(vectors, method='ward') if method == 'ward' else linkage(squareform(distance, checks=False), method=method)
        cuts = cut_tree(tree, n_clusters=np.arange(2, n))
        for column, k in enumerate(range(2, n)):
            labels = cuts[:, column]
            groups = [np.flatnonzero(labels == label).tolist() for label, size in Counter(labels).items() if size >= 10]
            if len(groups) >= 2: process(groups, {'tree_method': method, 'original_cut_k': k})
        print(f'Searched {method}: {len(candidates)} unique eligible partitions', flush=True)
    process([np.flatnonzero(old_labels == k).tolist() for k in sorted(set(old_labels) - {-1})], {'tree_method': 'existing HDBSCAN'})
    assert candidates
    best = max(candidates, key=lambda c: (c['min_cluster_mean_cosine'], c['assigned'], c['mean_cluster_cosine']))
    near = [c for c in candidates if c['min_cluster_mean_cosine'] >= best['min_cluster_mean_cosine'] - QUALITY_TOLERANCE]
    selected = max(near, key=lambda c: (c['assigned'], c['min_cluster_mean_cosine'], c['mean_cluster_cosine']))
    # Two reviewable alternatives explicitly expose the coverage trade-off.
    presets = {'recommended': selected, 'strict_best': best}
    for floor in [.4, .5, .6]:
        feasible = [c for c in candidates if c['min_cluster_mean_cosine'] > floor]
        if feasible:
            presets[f'coverage_cosine_{int(floor * 100):03d}'] = max(feasible, key=lambda c: (c['assigned'], c['min_cluster_mean_cosine'], c['mean_cluster_cosine']))
    return candidates, presets


def render_partition(name, result, records, cosine, distance, old_labels):
    folder = OUT / name; folder.mkdir(exist_ok=True)
    groups = result['groups']
    indices, labels, s = individual_scores(groups, distance)
    reference = silhouette_samples(distance[np.ix_(indices, indices)], labels, metric='precomputed')
    assert np.allclose(s, reference, atol=1e-12)
    silhouette_by_row = dict(zip(map(int, indices), map(float, s)))
    assignments = np.full(len(records), -1)
    rows = []
    for k, g in enumerate(groups):
        assignments[g] = k
        member_page(folder / f'cluster_{k}.html', f'{name.replace("_", " ")} · cluster {k}', g, records, cosine, silhouette_by_row)
        rows.append([f'<a href="cluster_{k}.html">Cluster {k}</a>', str(len(g)),
                     f'{cohesion(g, cosine):.3f}', f'{np.mean([silhouette_by_row[i] for i in g]):.3f}',
                     str(sum(old_labels[g] < 0))])
    unassigned = np.flatnonzero(assignments < 0).tolist()
    member_page(folder / 'unassigned.html', f'{name.replace("_", " ")} · unassigned items', unassigned, records, cosine,
                note='<p>These items are not a cluster. They remain available for review.</p>', is_cluster=False)
    output_rows = [{**r, 'cluster_id': int(assignments[i]), 'original_hdbscan_cluster': int(old_labels[i]),
                    'cosine_silhouette': silhouette_by_row.get(i)} for i, r in enumerate(records)]
    pq.write_table(pa.Table.from_pylist(output_rows), folder / 'assignments.parquet')
    save(folder / 'summary.json', {k: v for k, v in result.items() if k != 'groups'})
    save(folder / 'members.json', output_rows)
    body = '<p><a href="../review.html">Compare all results</a></p>'
    body += f'<p><strong>{len(groups)} clusters · {len(indices)}/269 assigned · {len(unassigned)} unassigned.</strong> Every cluster has at least 10 members and mean pairwise cosine above 0.4.</p>'
    body += f'<p>Lowest cluster mean cosine: <strong>{result["min_cluster_mean_cosine"]:.3f}</strong>. Overall assigned-member silhouette: <strong>{result["mean_silhouette"]:.3f}</strong> (diagnostic only).</p>'
    body += '<p class="notice">Silhouette is computed among assigned items in the original MPNet cosine space. Rejected items do not contribute, so these selected scores are not directly comparable with a partition covering all 269 items. Searching and rejecting on these same vectors makes this an exploratory result, not held-out validation.</p>'
    body += table(['Group', 'Members', 'Mean cosine', 'Mean silhouette', 'Previously HDBSCAN noise'], rows)
    body += '<p><a href="unassigned.html">Read all unassigned entries</a> · <a href="assignments.parquet">Assignments</a> · <a href="summary.json">Metrics and search provenance</a></p>'
    page(folder / 'review.html', name.replace('_', ' ').capitalize(), body)


def validate_html():
    class Parser(HTMLParser):
        def __init__(self): super().__init__(); self.links = []; self.members = 0
        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if tag == 'a' and a.get('href'): self.links.append(a['href'])
            if tag == 'img' and a.get('src'): self.links.append(a['src'])
            if tag == 'article' and 'member' in a.get('class', '').split(): self.members += 1
    total = 0
    for path in OUT.rglob('*.html'):
        parser = Parser(); parser.feed(path.read_text())
        for target in parser.links:
            assert (path.parent / target.split('#')[0]).exists(), (path, target)
        total += 1
    return total


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tracked = [INPUT / n for n in ['records.json', 'embeddings.npy', 'assignments.parquet']]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    records = json.loads((INPUT / 'records.json').read_text())
    old = pq.read_table(INPUT / 'assignments.parquet').to_pylist()
    assert [r['prompt_id'] for r in old] == [r['prompt_id'] for r in records]
    vectors = np.load(INPUT / 'embeddings.npy').astype(np.float64)
    assert len(records) == 269 and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine = np.clip(vectors @ vectors.T, -1, 1); np.fill_diagonal(cosine, 1)
    distance = np.clip(1 - cosine, 0, 2); np.fill_diagonal(distance, 0)
    old_labels = np.array([r['hdbscan_cluster_id'] for r in old])
    within = within_clusters(records, vectors, cosine, old_labels)
    print('Within-cluster groups:', {k: (len(v['groups']), v['assigned'], v['residual']) for k, v in within.items()}, flush=True)
    candidates, presets = global_search(vectors, cosine, distance, old_labels)
    save(OUT / 'search_candidates.json', candidates)
    pq.write_table(pa.Table.from_pylist([{k: v for k, v in c.items() if k not in ['groups', 'provenance']} for c in candidates]), OUT / 'search_metrics.parquet')
    for name, result in presets.items():
        render_partition(name, result, records, cosine, distance, old_labels)
    old_groups = [np.flatnonzero(old_labels == k).tolist() for k in sorted(set(old_labels) - {-1})]
    baseline = stats(old_groups, cosine, distance)
    summary = {'input_rows': len(records), 'model': 'all-mpnet-base-v2, saved normalized vectors', 'minimum_global_cluster_size': 10,
               'required_global_mean_cosine_strictly_above': .4, 'candidate_partitions': len(candidates),
               'selection': 'Maximize the lowest cluster mean pairwise cosine; among candidates within 0.02 of the best, maximize assigned count, then weakest cohesion, then mean cluster cohesion. Silhouette is diagnostic only.',
               'quality_tolerance': QUALITY_TOLERANCE, 'presets': {name: {k: v for k, v in c.items() if k != 'groups'} for name, c in presets.items()},
               'old_hdbscan': baseline, 'within': {k: {x: v for x, v in r.items() if x != 'groups'} for k, r in within.items()},
               'source': 'petri', 'facet': 'behaviours', 'llm_calls': 0, 'new_embedding_calls': 0, 'semantic_labels_assigned': False}
    save(OUT / 'summary.json', summary)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter([c['assigned'] for c in candidates], [c['min_cluster_mean_cosine'] for c in candidates], c='#086c79', s=16, alpha=.25)
    seen = set()
    for name, c in presets.items():
        point = (c['assigned'], c['min_cluster_mean_cosine'])
        if point in seen: continue
        seen.add(point); ax.scatter(*point, marker='*', s=160, edgecolor='black', label=name.replace('_', ' '))
    ax.scatter(baseline['assigned'], baseline['min_cluster_mean_cosine'], marker='X', s=100, c='red', label='Original HDBSCAN')
    ax.set(xlabel='Assigned investigations out of 269', ylabel='Lowest cluster mean pairwise cosine similarity', title='PETRI behaviours: within-cluster similarity versus coverage')
    ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / 'quality_coverage.png', dpi=170); plt.close(fig)
    rows = []
    for name, c in presets.items():
        rows.append([f'<a href="{name}/review.html">{esc(name.replace("_", " "))}</a>', str(c['clusters']), f'{c["assigned"]}/269', str(c['unassigned']),
                     f'{c["min_cluster_mean_cosine"]:.3f}', f'{c["min_cluster_mean_silhouette"]:.3f}', f'{c["mean_silhouette"]:.3f}'])
    rows.append(['Original HDBSCAN', str(baseline['clusters']), f'{baseline["assigned"]}/269', str(baseline['unassigned']),
                 f'{baseline["min_cluster_mean_cosine"]:.3f}', f'{baseline["min_cluster_mean_silhouette"]:.3f}', f'{baseline["mean_silhouette"]:.3f}'])
    body = '<p>Two separate analyses of the same 269 latest-qualifying PETRI investigation behaviour summaries. Existing clusters and vectors are preserved. No causes, other sources, new model calls, or semantic labels are included.</p>'
    body += '<h2>1. Subclusters inside the existing six HDBSCAN clusters</h2>'
    for minimum in ['2', '10']:
        r = within[minimum]
        body += f'<p><a href="within_min{minimum}/review.html"><strong>Open subclusters · minimum {minimum} members</strong></a>: {len(r["groups"])} groups, {r["assigned"]}/172 members retained, {r["residual"]} residual. Every subgroup has mean pairwise cosine &gt; 0.4.</p>'
    body += '<p>Minimum 2 is the unconstrained subgroup view; the additional minimum-10 view avoids tiny pairs. Original noise is not regrouped here. Every subgroup has a separate page showing every entry.</p>'
    body += '<h2>2. Reclustering the entire set, with rejection</h2><p>All 269 items, including the original 97 noise items, are eligible. We search Ward, average-linkage and complete-linkage trees at every cut, plus refinements of the existing HDBSCAN partition. Small groups are rejected; peripheral items and low-cohesion groups may be rejected. Every retained group must have at least 10 members and mean pairwise cosine &gt; 0.4.</p>'
    body += f'<p><strong>Selection rule:</strong> among {len(candidates):,} unique valid candidate partitions, maximize the <em>lowest cluster mean pairwise cosine similarity</em>. Keep candidates within 0.02 of that best score, then take the one retaining the most members. Silhouette does not influence selection or rejection. This is a search over the stated candidates, not a globally optimal partition of every possible subset. The strict best and broader-coverage alternatives are also available.</p>'
    body += table(['Result — click to inspect', 'Clusters', 'Assigned', 'Unassigned', 'Lowest mean cosine', 'Lowest mean silhouette', 'Overall silhouette'], rows)
    body += '<p class="notice">Cosine similarity measures within-group resemblance and is the optimization target. Cosine silhouette measures separation from other retained groups and is diagnostic only. Both use the original vectors, before UMAP. Scores are selected on these same embeddings; rejection and small groups can inflate apparent quality. These are exploratory comparisons, not clustering accuracy or independent validation.</p><img src="quality_coverage.png" alt="Candidate partitions show trade-offs between assigned investigations and weakest cluster mean cosine similarity"><p><a href="search_metrics.parquet">All search metrics</a> · <a href="search_candidates.json">Candidate memberships and provenance</a> · <a href="summary.json">Summary data</a> · <a href="method.md">Full method</a> · <a href="../review.html">Previous clustering review</a></p>'
    page(OUT / 'review.html', 'Cohesive subclusters and selective reclustering', body)
    method = '''# Cosine refinement of current PETRI behaviours

Input: 269 current PETRI behaviour records and existing normalized MPNet embeddings.
Previous records, vectors, assignments, and cause analyses remain unchanged.

## Within existing clusters
Average linkage on original cosine distance. Force a split of each parent at its
root so outputs are proper subgroups. Recursively accept the first descendant
with mean off-diagonal pairwise cosine strictly >0.4 and enough members; otherwise
split again. Groups smaller than the minimum become residual. Two minimum sizes,
2 and 10, are provided. Disjoint groups partition a subset of the original 172
assigned records. Original 97 noise records are excluded only from this part.
A mean threshold does not constrain every pair or every individual member.

## Global candidate search
Fit Ward in normalized Euclidean space and average/complete linkage in cosine
distance. Inspect every cut k=2..268. Discard groups smaller than 10. For each
remaining partition, try mean-cohesion targets just above .4 and .425 through .7
in .025 steps, plus .75, .8, .85, .9, .95: remove the least-central member until
the target is reached, or reject the group if fewer than 10 remain. Also try
centrality-trimmed cores capped at 10, 12, 15 and 20 members. At each variant,
evaluate the path of removing the least-cohesive cluster until two remain. Apply the
same refinements to the original HDBSCAN groups. Deduplicate identical
partitions; retain deterministic first-generation provenance.

At least two clusters are required: silhouette is undefined for one cluster.
All silhouettes use the original cosine distances on assigned rows only, with
noise excluded; individual scores are independently checked against sklearn.
Recommend the maximum-coverage candidate within 0.02 absolute of the best
worst-cluster mean pairwise cosine. Ties prefer worst-cluster mean cosine, then
average cluster cosine. Silhouette never influences selection or rejection.
Also save the strict best and maximum-coverage candidates with every cluster
mean cosine >.4, >.5 and >.6, if feasible. These floors are
explicit operational choices, not universally calibrated quality thresholds.

These are in-sample exploratory selections. They are not a global optimum,
held-out validation, a null test, or evidence for a taxonomy of causes. No
semantic cluster names or generative-model calls are involved. Minimum sizes
apply to investigations, not distinct scenario families. Similarities may reflect
topic, wording or repeated scenario structure. Group IDs are local to each result.
'''
    (OUT / 'method.md').write_text(method)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)] for p in tracked)
    checked = validate_html()
    save(OUT / 'validation.json', {'status': 'passed', 'html_pages_checked': checked,
        'input_hashes': hashes, 'inputs_unchanged': True, 'silhouettes_match_sklearn': True,
        'all_group_means_strictly_above_04': True, 'all_global_group_sizes_at_least_10': True,
        'subgroups_disjoint_and_within_original_parent': True})
    print(json.dumps({'within': {k: (len(v['groups']), v['assigned']) for k, v in within.items()},
        'candidates': len(candidates), 'presets': {k: {x: v[x] for x in ['clusters', 'assigned', 'min_cluster_mean_cosine', 'min_cluster_mean_silhouette', 'mean_silhouette']} for k, v in presets.items()}}, indent=2), flush=True)
    print('REVIEW', OUT / 'review.html', flush=True)


if __name__ == '__main__':
    main()
