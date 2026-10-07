"""Cluster final investigator explanations, separately per source and run."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import csv
import hashlib
import json
import re
from collections import Counter
import numpy as np
from scipy.cluster.hierarchy import cut_tree
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

OUT = ROOT / 'results' / 'final_explanations'
MODEL = ROOT / 'models' / 'all-MiniLM-L6-v2'
CANDIDATES = [5, 10, 15, 20, 30, 40, 50]
FINAL = re.compile(r'(?im)^\s*(?:#+\s*)?(?:\*\*)?Final answer(?:\*\*)?\s*:?\s*$')

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

def extract(path, dest):
    records, excluded = [], []
    scores = {}
    with (path.parent / 'verification.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            scores[r['prompt_id']] = r.get('score')
    with path.open() as f:
        for line_number, line in enumerate(f, 1):
            r = json.loads(line)
            sf = r.get('structured_findings') or {}
            structured = sf.get('structured_answer') or ''
            match = FINAL.search(structured)
            text = structured[match.end():].strip() if match else ''
            field = 'structured_findings.structured_answer:Final answer'
            if not text:
                for key in ('answer', 'first_person_answer'):
                    if isinstance(sf.get(key), str) and sf[key].strip():
                        text, field = sf[key].strip(), 'structured_findings.' + key
                        break
            if not text:
                excluded.append({'prompt_id': r['prompt_id'], 'reason': 'No explicit final explanation found'})
                continue
            records.append({'prompt_id': r['prompt_id'], 'source': path.parts[-3],
                            'run': path.parent.name, 'source_line': line_number,
                            'text_field': field, 'explanation': text,
                            'verification_score': scores.get(r['prompt_id'])})
    records.sort(key=lambda r: r['prompt_id'])
    assert len({r['prompt_id'] for r in records}) == len(records)
    with (dest / 'explanations.jsonl').open('w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    write_json(dest / 'excluded.json', excluded)
    return records, excluded

def embed(records, model, dest):
    fingerprint = hashlib.sha256(json.dumps([r['explanation'] for r in records]).encode()).hexdigest()
    cached = dest / 'embedding_metadata.json'
    if cached.exists() and json.loads(cached.read_text())['text_sha256'] == fingerprint:
        return np.load(dest / 'embeddings.npy')
    chunks, owners, weights = [], [], []
    for i, r in enumerate(records):
        ids = model.tokenizer.encode(r['explanation'], add_special_tokens=False, truncation=False, verbose=False)
        for start in range(0, len(ids), 220):
            piece = ids[start:start + 220]
            chunks.append(model.tokenizer.decode(piece, skip_special_tokens=True))
            owners.append(i)
            weights.append(len(piece))
    print('Embedding', len(records), 'explanations in', len(chunks), 'chunks', flush=True)
    vectors = model.encode(chunks, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    pooled = np.zeros((len(records), vectors.shape[1]), dtype=np.float32)
    np.add.at(pooled, owners, vectors * np.asarray(weights, dtype=np.float32)[:, None])
    pooled = normalize(pooled)
    assert np.isfinite(pooled).all() and np.all(np.linalg.norm(pooled, axis=1) > .99)
    np.save(dest / 'embeddings.npy', pooled)
    write_json(cached, {'model': 'sentence-transformers/all-MiniLM-L6-v2',
                       'model_revision': json.loads((MODEL / 'download_provenance.json').read_text())['revision'],
                       'text_sha256': fingerprint, 'chunks': len(chunks), 'chunk_tokens': 220,
                       'pooling': 'token-count-weighted mean of normalized chunk embeddings, then L2 normalization',
                       'device': str(model.device), 'shape': list(pooled.shape)})
    return pooled

def cluster(records, vectors, dest):
    n = len(records)
    print('Fitting full agglomerative tree:', n, 'records', flush=True)
    estimator = AgglomerativeClustering(n_clusters=None, distance_threshold=0,
                                       metric='euclidean', linkage='ward', compute_distances=True)
    estimator.fit(vectors)
    counts = np.zeros(n - 1)
    for i, children in enumerate(estimator.children_):
        counts[i] = sum(1 if child < n else counts[child - n] for child in children)
    linkage = np.column_stack([estimator.children_, estimator.distances_, counts])
    np.save(dest / 'linkage.npy', linkage)
    candidates = [k for k in CANDIDATES if k < n]
    cuts = cut_tree(linkage, n_clusters=candidates)
    rng = np.random.default_rng(42)
    sample = np.sort(rng.choice(n, min(2000, n), replace=False))
    diagnostics = []
    for j, k in enumerate(candidates):
        labels = cuts[:, j]
        sizes = np.bincount(labels)
        observed = np.unique(labels[sample]).size
        score = float(silhouette_score(vectors[sample], labels[sample], metric='cosine')) if 1 < observed < len(sample) else None
        diagnostics.append({'k': k, 'cosine_silhouette': score,
                            'silhouette_sample_size': len(sample), 'clusters_in_sample': int(observed),
                            'smallest_cluster': int(sizes.min()), 'largest_cluster': int(sizes.max()),
                            'largest_cluster_fraction': float(sizes.max() / n),
                            'singleton_clusters': int((sizes == 1).sum())})
    best = max((d for d in diagnostics if d['cosine_silhouette'] is not None),
               key=lambda d: (d['cosine_silhouette'], -d['k']))
    labels = cuts[:, candidates.index(best['k'])]
    with (dest / 'assignments.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'run', 'prompt_id', 'verification_score', 'cluster_id'] + [f'cluster_k{k}' for k in candidates])
        for i, r in enumerate(records):
            writer.writerow([r['source'], r['run'], r['prompt_id'], r['verification_score'], int(labels[i])] + cuts[i].tolist())
    summaries = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        centroid = normalize(vectors[indices].mean(axis=0, keepdims=True))[0]
        ranked = indices[np.argsort(-(vectors[indices] @ centroid), kind='stable')[:3]]
        summaries.append({'cluster_id': int(label), 'size': int(len(indices)),
                          'examples_nearest_centroid': [records[i] for i in ranked]})
    write_json(dest / 'cluster_examples.json', summaries)
    write_json(dest / 'diagnostics.json', diagnostics)
    summary = {'source': records[0]['source'], 'run': records[0]['run'], 'n_explanations': n,
               'selected_k': best['k'], 'selected_cosine_silhouette': best['cosine_silhouette'],
               'selection': 'Highest silhouette among saved candidate cuts; ties prefer smaller k. Exploratory, not validated taxonomy.',
               'largest_cluster_fraction': best['largest_cluster_fraction'],
               'singleton_clusters': best['singleton_clusters'],
               'method': 'sklearn.cluster.AgglomerativeClustering', 'metric': 'euclidean', 'linkage': 'ward',
               'verification_filter': None, 'scope': 'investigator final explanations; not verified causes',
               'cluster_names_assigned': False}
    assert len(labels) == n and len(np.unique(labels)) == best['k']
    write_json(dest / 'summary.json', summary)
    print(json.dumps(summary), flush=True)
    return summary

def main():
    import torch
    torch.set_num_threads(4)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = SentenceTransformer(str(MODEL), device=device, local_files_only=True)
    summaries = []
    for path in sorted((ROOT / 'data').glob('*/*/investigations.jsonl')):
        dest = OUT / path.parts[-3] / path.parent.name
        dest.mkdir(parents=True, exist_ok=True)
        print('Extracting', path.parent.name, flush=True)
        records, excluded = extract(path, dest)
        vectors = embed(records, model, dest)
        result = cluster(records, vectors, dest)
        result['excluded_count'] = len(excluded)
        summaries.append(result)
    write_json(OUT / 'summary.json', summaries)

if __name__ == '__main__':
    main()
