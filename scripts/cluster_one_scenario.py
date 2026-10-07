"""Cluster all released, qualified checkpoints of one or more PETRI scenarios.

Uses saved MPNet vectors only. Does not select the latest turn per run.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
from collections import Counter
from html import escape
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples
from refine_behaviour_clusters import page, table, member_page, save, cohesion, stats, signature, plt

ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / 'results/checkpoint2_locations/petri/behaviours'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='medical_authority_sycophancy')
    parser.add_argument('--scenarios', nargs='+')
    parser.add_argument('--name', help='Output folder name for a multi-scenario selection')
    args = parser.parse_args()
    scenarios = list(dict.fromkeys(args.scenarios or [args.scenario]))
    scenario = args.name or '__'.join(scenarios)
    if not all(re.fullmatch('[a-z0-9_]+', s) for s in scenarios + [scenario]): raise ValueError('Invalid scenario name')
    def origin(pid):
        match = re.fullmatch(r'petri_(.+)_e\d+_t\d+', pid)
        return match[1] if match else None
    def short_id(pid):
        suffix = pid[len(f'petri_{origin(pid)}_'):]
        return (chr(65 + scenarios.index(origin(pid))) + ' · ' if len(scenarios) > 1 else '') + suffix
    out = ROOT / 'results/scenario_clustering' / scenario
    out.mkdir(parents=True, exist_ok=True)
    source_records = json.loads((PRIOR / 'records.json').read_text())
    indices = sorted([i for i, r in enumerate(source_records) if origin(r['prompt_id']) in scenarios], key=lambda i: (scenarios.index(origin(source_records[i]['prompt_id'])), source_records[i]['prompt_id']))
    records = [{**source_records[i], 'scenario': origin(source_records[i]['prompt_id'])} for i in indices]
    n = len(records)
    if n < 4: raise ValueError(f'Only {n} eligible investigations; cannot form two groups of at least two.')
    raw_ids, raw_texts = [], {}
    with (ROOT / 'data/petri/qwen3_8b_petri_eval/investigations.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            if origin(r['prompt_id']) in scenarios:
                raw_ids.append(r['prompt_id']); raw_texts[r['prompt_id']] = r['behavior_summary']
    prompt_ids = []
    with (ROOT / 'data/seed_provenance/petri_prompts.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            if origin(r['id']) in scenarios: prompt_ids.append(r['id'])
    ids = [r['prompt_id'] for r in records]
    assert set(ids).issubset(raw_ids) and all(r['verif_score'] >= 7 for r in records)
    verification = {}
    with (ROOT / 'data/petri/qwen3_8b_petri_eval/verification.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            if origin(r['prompt_id']) in scenarios: verification[r['prompt_id']] = r['score']
    assert set(ids) == {pid for pid in raw_ids if verification.get(pid, 0) >= 7}, 'Qualified source investigation missing; inspect preprocessing exclusions before proceeding.'
    exclusions = [{'prompt_id': pid, 'verification_score': verification.get(pid),
                   'reason': 'missing verification' if pid not in verification else 'verification below 7'} for pid in sorted(set(raw_ids) - set(ids))]
    tracked = [PRIOR / name for name in ['records.json', 'embeddings.npy', 'assignments.parquet']]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    vectors = np.load(PRIOR / 'embeddings.npy')[indices].astype(float)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine = np.clip(vectors @ vectors.T, -1, 1); np.fill_diagonal(cosine, 1)
    distance = np.clip(1 - cosine, 0, 2); np.fill_diagonal(distance, 0)
    save(out / 'records.json', records)
    save(out / 'original_summaries.json', raw_texts)
    np.save(out / 'embeddings.npy', vectors); np.save(out / 'cosine.npy', cosine)
    pq.write_table(pa.Table.from_pylist(records), out / 'behaviours.parquet')
    candidates, signatures, cuts, trees = [], set(), [], {}
    for method in ['ward', 'average', 'complete']:
        tree = linkage(vectors, method='ward') if method == 'ward' else linkage(squareform(distance), method=method)
        trees[method] = tree; np.save(out / f'{method}_linkage.npy', tree)
        for k in range(2, n):
            labels = cut_tree(tree, n_clusters=k).ravel()
            groups = [np.flatnonzero(labels == label).tolist() for label in sorted(set(labels))]
            cuts.append({'method': method, 'k': k, 'labels': labels.tolist()})
            groups = [g for g in groups if len(g) >= 2 and cohesion(g, cosine) > .4]
            while len(groups) >= 2:
                sig = signature(groups)
                if sig not in signatures:
                    signatures.add(sig)
                    st = stats(groups, cosine, distance)
                    candidates.append({'groups': [g[:] for g in groups], 'method': method, 'original_cut_k': k, **st})
                del groups[int(np.argmin([cohesion(g, cosine) for g in groups]))]
    if not candidates: raise ValueError('No two qualifying groups were found; no selection inferred.')
    best = max(candidates, key=lambda c: (c['min_cluster_mean_cosine'], c['assigned'], c['mean_cluster_cosine']))
    near = [c for c in candidates if c['min_cluster_mean_cosine'] >= best['min_cluster_mean_cosine'] - .02]
    selected = max(near, key=lambda c: (c['assigned'], c['min_cluster_mean_cosine'], c['mean_cluster_cosine']))
    broad = max(candidates, key=lambda c: (c['assigned'], c['min_cluster_mean_cosine'], c['mean_cluster_cosine']))
    presets = {'fullest_coverage': broad, 'cosine_first': selected}
    for k in [2, 3]:
        labs = cut_tree(trees['ward'], n_clusters=k).ravel()
        groups = [np.flatnonzero(labs == c).tolist() for c in range(k)]
        if min(map(len, groups)) >= 2 and min(cohesion(g, cosine) for g in groups) > .4:
            presets[f'ward_k{k}'] = {'groups': groups, 'method': 'ward', 'original_cut_k': k, **stats(groups, cosine, distance)}
    for name, result in presets.items():
        result['scenario_composition'] = [dict(Counter(records[i]['scenario'] for i in g)) for g in result['groups']]
        result['mixed_scenario_clusters'] = sum(len(c) > 1 for c in result['scenario_composition'])
        folder = out / name; folder.mkdir(exist_ok=True)
        labels = np.full(n, -1, dtype=int)
        for k, g in enumerate(result['groups']): labels[g] = k
        assigned = np.flatnonzero(labels >= 0)
        scores = silhouette_samples(distance[np.ix_(assigned, assigned)], labels[assigned], metric='precomputed')
        score_by_row = dict(zip(map(int, assigned), map(float, scores)))
        assert np.isclose(scores.mean(), result['mean_silhouette'])
        table_rows = []
        for k, g in enumerate(result['groups']):
            mean = float(cosine[np.ix_(g, g)][np.triu_indices(len(g), 1)].mean())
            assert len(g) >= 2 and mean > .4
            member_page(folder / f'cluster_{k}.html', f'{scenario} · {name} · cluster {k}', g, records, cosine, score_by_row,
                        note='<p>Saved screener behaviour descriptions, not independent medical assessments or investigator causal explanations.</p>')
            composition = ', '.join(f'{s}: {count}' for s, count in result['scenario_composition'][k].items())
            table_rows.append([f'<a href="cluster_{k}.html">Cluster {k}</a>', str(len(g)), f'{mean:.3f}', escape(composition), ', '.join(short_id(ids[i]) for i in g)])
        residual = np.flatnonzero(labels < 0).tolist()
        member_page(folder / 'unassigned.html', f'{scenario} · unassigned investigations', residual, records, cosine, is_cluster=False)
        rows = [{**r, 'cluster_id': int(labels[i]), 'cosine_silhouette': score_by_row.get(i)} for i, r in enumerate(records)]
        pq.write_table(pa.Table.from_pylist(rows), folder / 'assignments.parquet')
        save(folder / 'summary.json', result)
        body = f'<p><a href="../review.html">All scenario results</a></p><p>{len(assigned)}/{n} assigned · {len(residual)} unassigned. Mean cosine excludes self-similarity. Method: {result["method"]}, starting cut k={result["original_cut_k"]}.</p>'
        body += table(['Group — open every entry', 'Members', 'Mean cosine', 'Scenario composition', 'Run and checkpoint'], table_rows)
        body += '<p><a href="unassigned.html">All unassigned entries</a> · <a href="assignments.parquet">Assignments</a></p><p>These are exploratory numeric groups. Similarity can reflect shared topics and wording; small groups can score highly. No semantic cluster labels.</p>'
        page(folder / 'review.html', scenario + ' · ' + name.replace('_', ' '), body)
    labels_short = [short_id(pid) for pid in ids]
    fig, axes = plt.subplots(1, 2, figsize=(18, 11) if n > 10 else (14, 7), gridspec_kw={'width_ratios': [1.15, 1]})
    im = axes[0].imshow(cosine, vmin=0, vmax=1, cmap='Blues')
    for i in range(n):
        for j in range(n): axes[0].text(j, i, f'{cosine[i,j]:.2f}', ha='center', va='center', color='white' if cosine[i,j] > .65 else '#193344', fontsize=6 if n > 10 else 9)
    axes[0].set_xticks(range(n), labels_short, rotation=75 if n > 10 else 45); axes[0].set_yticks(range(n), labels_short)
    axes[0].set_title('Original MPNet cosine similarity')
    fig.colorbar(im, ax=axes[0], fraction=.046, pad=.04)
    dendrogram(trees['ward'], labels=labels_short, orientation='right', ax=axes[1], color_threshold=0, above_threshold_color='#086c79')
    axes[1].set_title('Ward hierarchy across scenarios and runs'); axes[1].set_xlabel('Ward merge distance (not cosine similarity)')
    legend = '   |   '.join(f'{chr(65+i)}: {s}' for i, s in enumerate(scenarios))
    fig.suptitle(scenario.replace('_', ' ') + f' · {n} investigations', fontsize=16)
    if len(scenarios) > 1: fig.text(.5, .01, legend, ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .04 if len(scenarios) > 1 else 0, 1, 1]); fig.savefig(out / 'similarity_and_hierarchy.png', dpi=160); fig.savefig(out / 'similarity_and_hierarchy.svg'); plt.close(fig)
    all_mean = float(cosine[np.triu_indices(n, 1)].mean())
    scenario_counts = [{ 'scenario': s, 'plot_key': chr(65+i), 'qualified': sum(r['scenario'] == s for r in records),
                        'raw_investigations': sum(origin(pid) == s for pid in raw_ids), 'prompt_pool': sum(origin(pid) == s for pid in prompt_ids)} for i, s in enumerate(scenarios)]
    pair_stats = []
    for a, s in enumerate(scenarios):
        for t in scenarios[a:]:
            pair_values = [cosine[i, j] for i in range(n) for j in range(i+1, n) if (records[i]['scenario'], records[j]['scenario']) in [(s,t),(t,s)]]
            if pair_values: pair_stats.append({'scenario_a': s, 'scenario_b': t, 'pairs': len(pair_values), 'mean_cosine': float(np.mean(pair_values))})
    summary = {'scenario': scenario, 'scenarios': scenarios, 'scenario_counts': scenario_counts, 'scenario_pair_cosines': pair_stats,
               'investigations': n, 'available_raw_investigations': len(raw_ids), 'prompt_pool_entries': len(prompt_ids),
               'prompts_without_investigation': sorted(set(prompt_ids) - set(raw_ids)), 'raw_investigations_excluded': sorted(set(raw_ids) - set(ids)),
               'exclusion_details': exclusions,
               'whole_scenario_mean_cosine': all_mean, 'candidate_partitions': len(candidates), 'presets': presets,
               'selection': 'Across Ward/average/complete tree cuts, discard singletons and groups with mean cosine <=0.4; also consider dropping the weakest-cohesion groups. At least two groups required. Cosine-first optimizes lowest cluster mean cosine, then coverage within 0.02 of best. Fullest-coverage optimizes count, then lowest group mean cosine, then average group mean cosine.',
               'minimum_cluster_size': 2, 'verification_min': 7, 'all_qualifying_checkpoints_included': True, 'source_vector_indices': indices,
               'embedding_source': str(PRIOR), 'llm_calls': 0, 'semantic_labels_assigned': False}
    save(out / 'summary.json', summary); save(out / 'candidates.json', candidates); save(out / 'all_hierarchy_cuts.json', cuts)
    body = '<p>All released, verification-qualified investigations from the selected scenarios, across both runs and all checkpoints, clustered together. This analysis does not use the latest-turn-only selection. Scenario names are metadata, not clustering features.</p>'
    body += f'<p><strong>{n} investigations · overall mean pairwise cosine {all_mean:.3f}.</strong> Minimum cluster size is 2 for this small scenario-focused analysis. All cosine values use saved normalized MPNet behaviour vectors, not UMAP. No new embeddings or LLM calls.</p>'
    body += table(['Plot key', 'Scenario', 'Prompt pool', 'Released investigations', 'Qualified'], [[r['plot_key'],r['scenario'],str(r['prompt_pool']),str(r['raw_investigations']),str(r['qualified'])] for r in scenario_counts])
    if exclusions:
        body += '<details><summary>Excluded investigations and verification scores</summary>' + table(['Investigation', 'Score', 'Reason'], [[r['prompt_id'],str(r['verification_score']),r['reason']] for r in exclusions]) + '</details>'
    body += '<p>For this scenario, e1_t2–e1_t5 and e2_t2–e2_t5 are present. Both t1 prompts exist but have no released investigations. All eight released investigations meet verification ≥7.</p>' if scenario == 'medical_authority_sycophancy' else ''
    rows = [[f'<a href="{name}/review.html">{name.replace("_", " ")}</a>', str(c['clusters']), f'{c["assigned"]}/{n}', ', '.join(map(str, c['cluster_sizes'])), f'{c["min_cluster_mean_cosine"]:.3f}', f'{c["mean_silhouette"]:.3f}',str(c['mixed_scenario_clusters'])] for name, c in presets.items()]
    body += table(['Review — every entry', 'Clusters', 'Assigned', 'Sizes', 'Lowest mean cosine', 'Silhouette (diagnostic)', 'Mixed-scenario groups'], rows)
    body += '<p><strong>Fullest coverage</strong> retains as many investigations as possible, then favors within-group similarity. <strong>Cosine first</strong> prioritizes similarity and can leave other investigations unassigned. High-scoring pairs should be treated as exploratory examples, not evidence of stable categories. Groups are numbered, not semantically labelled.</p>'
    if len(scenarios) > 1:
        body += '<h2>Do similarities cross scenario boundaries?</h2>'
        body += table(['First scenario', 'Second scenario', 'Pairs', 'Mean cosine'], [[r['scenario_a'],r['scenario_b'],str(r['pairs']),f'{r["mean_cosine"]:.3f}'] for r in pair_stats])
    body += '<img src="similarity_and_hierarchy.png" alt="All pairwise similarities and Ward hierarchy for the scenario"><p><a href="similarity_and_hierarchy.svg">Download vector plot</a> · <a href="summary.json">Counts and selection method</a> · <a href="all_hierarchy_cuts.json">Every tested hierarchy cut</a></p>'
    body += '<h2>Every investigation included</h2><p>Verbatim existing preprocessing output from the saved screener summaries. These descriptions are not independent medical assessments.</p>'
    for i, r in enumerate(records):
        body += f'<article class="card member"><strong>{escape(r["prompt_id"])}</strong><p class="muted">Scenario: {escape(r["scenario"])} · Verification {r["verif_score"]}/10</p><p class="text">{escape(r["behaviour_text"])}</p></article>'
    page(out / 'review.html', ('Within one scenario: ' if len(scenarios) == 1 else 'Across related scenarios: ') + scenario.replace('_', ' '), body)
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def __init__(self): super().__init__(); self.links = []
        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if tag == 'a' and a.get('href'): self.links.append(a['href'])
            if tag == 'img' and a.get('src'): self.links.append(a['src'])
    for path in out.rglob('*.html'):
        parser = Links(); parser.feed(path.read_text())
        for target in parser.links: assert (path.parent / target).exists(), (path, target)
    for name, c in presets.items():
        used = sum(c['groups'], []); assert len(used) == len(set(used)) == c['assigned']
        for k, g in enumerate(c['groups']):
            text = (out / name / f'cluster_{k}.html').read_text()
            assert text.count('class="card member"') == len(g)
            for i in g: assert escape(records[i]['behaviour_text'], quote=True) in text
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    save(out / 'validation.json', {'status': 'passed', 'all_html_links_exist': True, 'all_group_members_displayed_verbatim': True,
                                  'silhouettes_checked_against_sklearn': True, 'previous_files_unchanged': True, 'input_hashes': hashes})
    print(json.dumps({k: v for k, v in summary.items() if k not in ['presets', 'source_vector_indices']}, indent=2))
    print('PRESETS', json.dumps({k: {x: v[x] for x in ['clusters', 'assigned', 'cluster_sizes', 'cluster_mean_cosines']} for k, v in presets.items()}, indent=2))
    print('REVIEW', out / 'review.html')


if __name__ == '__main__': main()
