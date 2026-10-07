"""Checkpoint 2: independent behaviour/cause clustering for PETRI/WildChat eval.

Local CPU MPNet embeddings; Ward k=10..200; fixed UMAP10/HDBSCAN;
10 independently coordinate-shuffled embedding nulls; no semantic labels.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.cache/numba'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))

from collections import Counter, defaultdict
import hashlib
import html
import json
import platform
import random
import time

import hdbscan
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numba
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.cluster.hierarchy import cut_tree
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
import torch
import umap

INPUT = ROOT / 'data/checkpoint1_approved'
OUTPUT = ROOT / 'results/checkpoint2'
MODEL = ROOT / 'models/all-mpnet-base-v2'
SOURCES = ('petri', 'wildchat_eval')
FACETS = ('behaviours', 'causes')
KS = np.arange(10, 201)
SEED = 20261004
N_NULL = 10


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def distances(vectors):
    d = np.clip(1 - vectors @ vectors.T, 0, 2)
    np.fill_diagonal(d, 0)
    return d.astype(np.float32, copy=False)


def ward(vectors):
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=0, linkage='ward',
                                    metric='euclidean', compute_distances=True).fit(vectors)
    n = len(vectors); counts = np.zeros(n - 1)
    for i, pair in enumerate(model.children_):
        counts[i] = sum(1 if child < n else counts[child - n] for child in pair)
    tree = np.column_stack([model.children_, model.distances_, counts])
    cuts = cut_tree(tree, n_clusters=KS)
    return tree, cuts


@numba.njit(cache=True)
def exact_ward_silhouette_curve(d, children, labels_at_max_k, min_k=10):
    """Exact full-data cosine silhouettes for successive cuts of one hierarchy.

    Aggregate distances once at max k, then update sums at each merge. No sampling
    or approximation; avoids recalculating the same pairwise sums 191 times.
    """
    n = d.shape[0]
    labels = labels_at_max_k.copy()
    max_k = labels.max() + 1
    sizes = np.zeros(max_k, dtype=np.int64)
    for value in labels:
        sizes[value] += 1
    sums = np.zeros((n, max_k), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            sums[i, labels[j]] += d[i, j]
    representatives = np.zeros(2 * n - 1, dtype=np.int64)
    for i in range(n):
        representatives[i] = i
    for i in range(n - 1):
        representatives[n + i] = representatives[children[i, 0]]
    scores = np.zeros(max_k - min_k + 1)
    for k in range(max_k, min_k - 1, -1):
        total = 0.
        for i in range(n):
            own = labels[i]
            if sizes[own] == 1:
                continue
            within = sums[i, own] / (sizes[own] - 1)
            other = np.inf
            for col in range(max_k):
                if col != own and sizes[col] > 0:
                    other = min(other, sums[i, col] / sizes[col])
            denominator = max(within, other)
            if denominator > 0:
                total += (other - within) / denominator
        scores[k - min_k] = total / n
        if k > min_k:
            merge = n - k
            a = labels[representatives[children[merge, 0]]]
            b = labels[representatives[children[merge, 1]]]
            for i in range(n):
                sums[i, a] += sums[i, b]
                if labels[i] == b:
                    labels[i] = a
            sizes[a] += sizes[b]; sizes[b] = 0
    return scores


def curve(d, tree, cuts):
    return exact_ward_silhouette_curve(d, tree[:, :2].astype(np.int64), cuts[:, -1].astype(np.int64), int(KS[0]))


def density_fit(vectors, d, seed):
    y = umap.UMAP(n_components=10, n_neighbors=30, min_dist=0, metric='cosine',
                  random_state=seed, n_jobs=1).fit_transform(vectors)
    model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean',
                            cluster_selection_method='eom', core_dist_n_jobs=1).fit(y)
    labels = model.labels_
    assigned = labels >= 0
    clusters = len(set(labels[assigned]))
    score = float(silhouette_score(d[np.ix_(assigned, assigned)], labels[assigned], metric='precomputed')) if 2 <= clusters < assigned.sum() else None
    reduced_score = float(silhouette_score(y[assigned], labels[assigned], metric='euclidean')) if score is not None else None
    summary = {'n_clusters': clusters, 'assigned_rows': int(assigned.sum()), 'noise_rows': int((~assigned).sum()),
               'noise_fraction': float((~assigned).mean()), 'original_cosine_silhouette_assigned_only': score,
               'umap_euclidean_silhouette_assigned_only': reduced_score}
    return labels, model.probabilities_, y, summary


def cluster_sizes(labels):
    return [{'cluster_id': int(k), 'size': int(n)} for k, n in sorted(Counter(int(x) for x in labels if x >= 0).items(), key=lambda kv: (-kv[1], kv[0]))]


def distribution(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {'valid_replicates': 0, 'mean': None, 'min': None, 'max': None}
    return {'valid_replicates': len(values), 'mean': float(np.mean(values)), 'min': min(values), 'max': max(values),
            'q025': float(np.quantile(values, .025)), 'q975': float(np.quantile(values, .975))}


def cause_diversity(records, labels, all_ids):
    groups = defaultdict(set)
    eligible_ids = {r['prompt_id'] for r in records}
    for r, label in zip(records, labels):
        if label >= 0:
            groups[r['prompt_id']].add(int(label))
    values = [len(v) for v in groups.values() if v]
    return {'denominator': 'investigations with at least one non-noise assigned cause cluster; zero-cluster investigations excluded',
            'investigations_in_denominator': len(values),
            'investigations_with_zero_eligible_causes': len(set(all_ids) - eligible_ids),
            'investigations_with_eligible_causes_but_only_noise': len(eligible_ids - set(groups)),
            'investigations_with_zero_assigned_cause_clusters': len(all_ids) - len(values),
            'mean_distinct_clusters': float(np.mean(values)) if values else None,
            'median_distinct_clusters': float(np.median(values)) if values else None,
            'max_distinct_clusters': max(values) if values else None}


def random_cluster_members(records, labels, seed):
    rng = random.Random(seed)
    result = []
    for cluster in cluster_sizes(labels)[:10]:
        members = [i for i, value in enumerate(labels) if value == cluster['cluster_id']]
        result.append({**cluster, 'members': [records[i] for i in rng.sample(members, min(5, len(members)))]})
    return result


def embed(records, text_key, model, folder):
    texts = [r[text_key] for r in records]
    token_lengths = [len(x) for x in model.tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)['input_ids']]
    if max(token_lengths) > model.max_seq_length:
        raise ValueError(f'Input exceeds tokenizer limit: {folder}; refusing silent truncation')
    signature = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    metadata_path = folder / 'embedding_metadata.json'
    if metadata_path.exists() and (folder / 'embeddings.npy').exists() and json.loads(metadata_path.read_text())['text_sha256'] == signature:
        vectors = np.load(folder / 'embeddings.npy')
    else:
        vectors = model.encode(texts, batch_size=32, device='cpu', normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=True)
        np.save(folder / 'embeddings.npy', vectors)
    assert vectors.shape == (len(records), 768) and np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    save(metadata_path, {'model': 'sentence-transformers/all-mpnet-base-v2', 'device': 'cpu',
                        'model_revision': json.loads((MODEL / 'download_provenance.json').read_text())['revision'],
                        'text_sha256': signature, 'shape': list(vectors.shape), 'normalized': True,
                        'truncation': False, 'chunking': False, 'max_sequence_length': model.max_seq_length,
                        'max_input_tokens': max(token_lengths), 'median_input_tokens': float(np.median(token_lengths))})
    return vectors


def near_duplicates(records, vectors, d, folder):
    pairs = []
    for i in range(len(records)):
        for j in np.flatnonzero(d[i, i+1:] < .05) + i + 1:
            pairs.append({'row_i': i, 'row_j': int(j), 'prompt_id_i': records[i]['prompt_id'], 'prompt_id_j': records[j]['prompt_id'],
                          'experiment_i': records[i].get('experiment_number'), 'experiment_j': records[j].get('experiment_number'),
                          'cosine_similarity': float(1 - d[i,j])})
    schema = pa.schema([('row_i', pa.int64()), ('row_j', pa.int64()), ('prompt_id_i', pa.string()), ('prompt_id_j', pa.string()),
                        ('experiment_i', pa.int64()), ('experiment_j', pa.int64()), ('cosine_similarity', pa.float64())])
    pq.write_table(pa.Table.from_pylist(pairs, schema=schema), folder / 'near_duplicates.parquet', compression='zstd')
    return {'cosine_threshold_strictly_greater_than': .95, 'pairs': len(pairs),
            'rows_in_any_pair': len({v for r in pairs for v in [r['row_i'],r['row_j']]}), 'dropped': 0}


def run_dataset(source, facet, model):
    folder = OUTPUT / source / facet; folder.mkdir(parents=True, exist_ok=True)
    name, key = ('behaviours', 'behaviour_text') if facet == 'behaviours' else ('causes_for_clustering', 'cause_text')
    input_path = INPUT / source / (name + '.parquet')
    records = pq.read_table(input_path).to_pylist()
    save(folder / 'records.json', records)
    print(f'EMBED {source}/{facet}: {len(records):,} rows', flush=True)
    vectors = embed(records, key, model, folder)
    d = distances(vectors)
    duplicates = near_duplicates(records, vectors, d, folder)
    tree, cuts = ward(vectors); scores = curve(d, tree, cuts)
    # Verify optimized aggregation against sklearn on three full-data cuts.
    for index in [0, len(KS)//2, len(KS)-1]:
        reference = silhouette_score(d, cuts[:,index], metric='precomputed')
        assert abs(reference - scores[index]) < 2e-6, (source, facet, int(KS[index]), reference, scores[index])
    best_index = int(np.argmax(scores)); best_k = int(KS[best_index]); labels = cuts[:,best_index]
    np.save(folder / 'ward_linkage.npy', tree)
    np.savez_compressed(folder / 'ward_all_cuts.npz', k=KS, labels=cuts, cosine_silhouette=scores)
    density, probabilities, reduced, density_summary = density_fit(vectors, d, SEED)
    np.save(folder / 'hdbscan_labels.npy', density); np.save(folder / 'umap10.npy', reduced)
    np.save(folder / 'hdbscan_probabilities.npy', probabilities)
    assigned = density >= 0
    ari = float(adjusted_rand_score(labels[assigned], density[assigned])) if assigned.sum() >= 2 and len(set(density[assigned])) >= 2 else None
    null_rows, null_curves = [], []
    checkpoint = folder / 'null_replicates.json'
    if checkpoint.exists():
        cached = json.loads(checkpoint.read_text())
        # Null cache tied to the exact embedding fingerprint and fixed protocol.
        if cached.get('text_sha256') == json.loads((folder / 'embedding_metadata.json').read_text())['text_sha256'] and (folder / 'null_curves.npy').exists():
            null_rows = cached['replicates']; null_curves = list(np.load(folder / 'null_curves.npy'))
    for rep in range(len(null_rows), N_NULL):
        start = time.monotonic(); rng = np.random.default_rng(SEED + 1000 + rep)
        # Row shuffling alone leaves clustering unchanged; independently permute coordinates.
        shuffled = np.column_stack([rng.permutation(vectors[:,col]) for col in range(vectors.shape[1])])
        shuffled = normalize(shuffled).astype(np.float32)
        null_d = distances(shuffled)
        null_tree, null_cuts = ward(shuffled)
        null_scores = curve(null_d, null_tree, null_cuts)
        _, _, _, null_density = density_fit(shuffled, null_d, SEED)
        null_rows.append({'replicate': rep, 'best_k': int(KS[np.argmax(null_scores)]),
                          'best_ward_cosine_silhouette': float(np.max(null_scores)),
                          'ward_at_observed_best_k': float(null_scores[best_index]),
                          'hdbscan': null_density, 'seconds': time.monotonic() - start})
        null_curves.append(null_scores)
        np.save(folder / 'null_curves.npy', np.asarray(null_curves))
        save(checkpoint, {'text_sha256': json.loads((folder / 'embedding_metadata.json').read_text())['text_sha256'], 'replicates': null_rows})
        print(f'NULL {source}/{facet} {rep+1}/{N_NULL}: Ward max {max(null_scores):.4f}; HDBSCAN {null_density["n_clusters"]} clusters, {null_density["noise_fraction"]:.1%} noise; {time.monotonic()-start:.1f}s', flush=True)
    null_array = np.asarray(null_curves)
    sizes = {'ward': cluster_sizes(labels), 'hdbscan': cluster_sizes(density)}
    reading = {'ward': random_cluster_members(records, labels, SEED),
               'hdbscan': random_cluster_members(records, density, SEED + 1)}
    save(folder / 'largest_clusters_reading.json', reading)
    save(folder / 'cluster_sizes.json', sizes)
    assignments = [{**r, 'row_index': i, 'ward_cluster_id': int(labels[i]), 'hdbscan_cluster_id': int(density[i]),
                    'hdbscan_membership_strength': float(probabilities[i])} for i, r in enumerate(records)]
    pq.write_table(pa.Table.from_pylist(assignments), folder / 'assignments.parquet', compression='zstd')
    summary = {
        'source': source, 'facet': facet, 'n': len(records), 'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
        'ward': {'selection': 'highest full-data cosine silhouette among every integer k=10..200; ties choose smaller k',
                 'best_k': best_k, 'cosine_silhouette': float(scores[best_index]), 'noise_fraction': 0,
                 'best_k_at_upper_boundary': best_k == int(KS[-1]), 'cluster_sizes': sizes['ward'],
                 'shuffled_null_best_over_same_k_range': distribution([r['best_ward_cosine_silhouette'] for r in null_rows]),
                 'shuffled_null_at_observed_best_k': distribution([r['ward_at_observed_best_k'] for r in null_rows])},
        'hdbscan': {**density_summary, 'cluster_sizes': sizes['hdbscan'],
                    'shuffled_null_original_cosine_silhouette': distribution([r['hdbscan']['original_cosine_silhouette_assigned_only'] for r in null_rows]),
                    'shuffled_null_noise_fraction': distribution([r['hdbscan']['noise_fraction'] for r in null_rows])},
        'ari_between_methods': {'on_hdbscan_assigned_rows': ari, 'compared_rows': int(assigned.sum()),
                                'coverage_fraction': float(assigned.mean()),
                                'including_noise_as_one_label_secondary': float(adjusted_rand_score(labels, density))},
        'near_duplicates': duplicates, 'null_replicates': N_NULL, 'semantic_labels_assigned': False,
    }
    if facet == 'causes':
        ids = json.loads((INPUT / source / 'counts.json').read_text())['verification_qualified_investigation_ids']
        summary['cause_diversity'] = {'ward': cause_diversity(records, labels, ids), 'hdbscan': cause_diversity(records, density, ids)}
    save(folder / 'summary.json', summary)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(KS, scores, label='Observed Ward', color='#086c79')
    ax.plot(KS, null_array.mean(axis=0), label='Coordinate-shuffled null mean (10 runs)', color='#b06b24')
    ax.fill_between(KS, null_array.min(axis=0), null_array.max(axis=0), color='#b06b24', alpha=.18, label='Null min–max')
    ax.axvline(best_k, linestyle=':', color='#086c79', alpha=.6)
    ax.set(xlabel='Number of clusters (k)', ylabel='Cosine silhouette', title=f'{source} · {facet} · n={len(records):,}')
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(folder / 'silhouette_vs_null.png', dpi=160); plt.close(fig)
    print('DONE', source, facet, 'Ward', best_k, round(scores[best_index], 4), 'HDBSCAN', density_summary, 'ARI', ari, flush=True)
    return summary


def build_report(summaries):
    lines = ['# CHIVE — Checkpoint 2', '', '**Stopped before semantic cluster labelling.**', '',
             'Only PETRI and WildChat eval; separate behaviour and cause embeddings/clusters. Train and thinking deferred. Inputs: [approved preprocessing](../../data/checkpoint1_approved/review.html).', '',
             '## Fixed protocol', '',
             '- all-mpnet-base-v2, local CPU inference, L2 normalization. No truncation or chunking. No generative LLM calls.',
             '- Ward uses Euclidean distance on unit vectors. Every k=10..200 is scored with full-data cosine silhouette; highest silhouette selects the reported cut. The full curve is retained. A peak at k=200 is a search boundary, not an established optimum.',
             '- UMAP: cosine metric, 10 dimensions, n_neighbors=30, min_dist=0, seed 20261004. HDBSCAN: Euclidean in reduced space, min_cluster_size=10, min_samples=5, EOM. Preset, not tuned to silhouette.',
             '- Main HDBSCAN silhouette is cosine in the original embedding space among assigned rows; noise is excluded and coverage reported. UMAP-space silhouette is only a secondary diagnostic.',
             '- Ten null replicates independently shuffle each embedding coordinate across rows, then L2 normalize and refit BOTH pipelines. Simply shuffling whole rows would preserve geometry and is not used. Ward nulls undergo the same k sweep, accounting for cut selection. Ten replicates give a rough reference range, not a precise significance estimate.',
             '- ARI compares Ward’s selected cut with HDBSCAN on non-noise rows. An all-row ARI treating noise as one label is separately provided, not interpreted as a meaningful noise cluster.',
             '- Cosine >0.95 near-duplicate pairs are reported, never dropped. Source and facet cluster IDs are independent. No semantic labels are assigned.',
             '- Cause-diversity means/medians exclude investigations with zero assigned cause clusters; preprocessing zero-eligible-cause counts and HDBSCAN-only-noise investigations are reported separately.', '',
             '## Results', '', '| Source | Facet | N | Ward k | Ward cosine | Null max-over-k mean | HDBSCAN clusters | HDB cosine | HDB noise | ARI assigned |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda v: 'undefined' if v is None else f'{v:.4f}'
    for r in summaries:
        w, h, a = r['ward'], r['hdbscan'], r['ari_between_methods']
        lines.append(f'| {r["source"]} | {r["facet"]} | {r["n"]} | {w["best_k"]} | {w["cosine_silhouette"]:.4f} | {w["shuffled_null_best_over_same_k_range"]["mean"]:.4f} | {h["n_clusters"]} | {fmt(h["original_cosine_silhouette_assigned_only"])} | {h["noise_fraction"]:.1%} | {fmt(a["on_hdbscan_assigned_rows"])} |')
    lines += ['', 'Full cluster sizes, both null summaries, positive-only diversity statistics, membership assignments, all cuts, and reading samples are saved per source/facet. Review [review.html](review.html) for 5 random members from each of the 10 largest clusters for both methods (all clusters if fewer than 10).', '', '**Checkpoint 2: awaiting review; no labelling performed.**', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines))
    esc = lambda value: html.escape(str(value))
    body = ['<h1>CHIVE · Checkpoint 2</h1><p>No semantic cluster labels assigned. PETRI and WildChat eval only; behaviours and causes remain separate.</p><p><a href="report.md">Results and protocol</a> · <a href="../../data/checkpoint1_approved/review.html">30 eligible original cause summaries per source and revised filters</a></p>']
    body.append('<pre>' + esc('\n'.join(lines)) + '</pre>')
    for summary in summaries:
        source, facet = summary['source'], summary['facet']; relative = f'{source}/{facet}'
        folder = OUTPUT / relative
        body.append(f'<section><h2>{source} · {facet}</h2><img src="{relative}/silhouette_vs_null.png" alt="Observed and shuffled-null silhouette curves"><p><a href="{relative}/summary.json">Full metrics and cluster sizes</a> · <a href="{relative}/assignments.parquet">All assignments</a> · <a href="{relative}/near_duplicates.parquet">Near duplicates</a></p>')
        for method, groups in json.loads((folder / 'largest_clusters_reading.json').read_text()).items():
            body.append(f'<h3>{method}: 10 largest clusters, 5 random members each</h3>')
            for group in groups:
                body.append(f'<details><summary>Cluster {group["cluster_id"]} · {group["size"]} members</summary><table><tr><th>Investigation</th><th>Clustering text</th><th>Original cause summary / measurement</th></tr>')
                for r in group['members']:
                    text = r.get('behaviour_text', r.get('cause_text'))
                    extra = r.get('cause_text_original', '')
                    if 'delta_pp' in r:
                        extra += f'\nExperiment {r["experiment_number"]}; Δ {r["delta_pp"]:+.1f} pp; frac_changed {r["frac_changed"]:.4f}'
                    body.append(f'<tr><td>{esc(r["prompt_id"])}</td><td>{esc(text)}</td><td>{esc(extra)}</td></tr>')
                body.append('</table></details>')
        body.append('</section>')
    doc = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CHIVE checkpoint 2</title><style>body{font:15px/1.6 system-ui;background:#f4f7f8;color:#173044;margin:0}main{max-width:1400px;margin:auto;padding:25px}section{background:white;padding:20px;margin:25px 0;border-radius:10px}table{width:100%;border-collapse:collapse;table-layout:fixed}td,th{padding:12px;border:1px solid #cad9df;text-align:left;vertical-align:top;overflow-wrap:anywhere;white-space:pre-wrap}th{background:#e7f0f2}details{margin:12px 0;border:1px solid #cad9df;padding:12px}summary{cursor:pointer;font-weight:600}pre{overflow:auto;font-size:12px}img{max-width:100%}a{color:#086c79}</style><main>' + ''.join(body) + '</main></html>'
    (OUTPUT / 'review.html').write_text(doc)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    save(OUTPUT / 'design.json', {'sources': SOURCES, 'facets': FACETS, 'k': KS.tolist(), 'null_replicates': N_NULL,
                                 'seed': SEED, 'input_directory': str(INPUT), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 'python': platform.python_version(), 'semantic_labels': False})
    model = SentenceTransformer(str(MODEL), device='cpu', local_files_only=True)
    summaries = []
    with threadpool_limits(limits=4):
        for source in SOURCES:
            for facet in FACETS:
                summaries.append(run_dataset(source, facet, model))
                save(OUTPUT / 'summary.json', summaries)
                build_report(summaries)
    print('CHECKPOINT 2 COMPLETE — STOP before labelling:', OUTPUT / 'review.html', flush=True)


if __name__ == '__main__': main()
