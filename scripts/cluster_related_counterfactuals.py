"""Cluster measured counterfactual edits across the same three PETRI scenarios.

One row per eligible experiment; no investigator causal conclusions are features.
Existing vectors are reused except for an audited deterministic text correction.
"""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import difflib
import hashlib
import json
from collections import Counter
from html import escape
from html.parser import HTMLParser
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples
from refine_behaviour_clusters import page, table, save, cohesion, stats, signature, plt

BEHAVIOURS = ROOT / 'results/scenario_clustering/related_health_scenarios'
PRIOR = ROOT / 'results/checkpoint2_locations/petri/causes'
OUT = ROOT / 'results/scenario_clustering/related_health_counterfactuals'
CORRECTION_KEY = ('petri_medical_authority_sycophancy_e2_t4', 12)
INTERPRETATION = "When the user doesn't personally own the claim, the model freely corrects it."


def context_text(messages):
    return '\n\n'.join(f'[{m["role"]}]\n{m.get("content", "")}' for m in messages)


def group_page(folder, filename, title, group, records, cosine, evidence, is_cluster=True):
    means = {i: float((cosine[i, group].sum() - 1) / (len(group) - 1)) for i in group} if is_cluster else {}
    ordered = sorted(group, key=lambda i: (-means.get(i, 0), records[i]['prompt_id'], records[i]['experiment_number']))
    body = '<p><a href="review.html">Back to groups</a> · <a href="../review.html">All counterfactual results</a></p>'
    if is_cluster:
        body += f'<p><strong>{len(group)} experiments · {len({records[i]["prompt_id"] for i in group})} distinct investigations · mean pairwise cosine {cohesion(group, cosine):.3f}.</strong></p>'
    else: body += f'<p>{len(group)} experiments. This is an unassigned list, not a cluster.</p>'
    body += '<p>Similarity uses only the cleaned edit descriptions. Rates, edit location, scenario, classifier questions and original summaries are review metadata. Each classifier defines what “yes” means; negative change is not universally improvement. Multiple experiments from one investigation are related observations.</p>'
    for i in ordered:
        r = records[i]; e = evidence[(r['prompt_id'], r['experiment_number'])]
        bl, cf = r['baseline_rate'], r['counterfactual_rate']
        br = e['baseline_count']; cr = e['counterfactual_count']
        mean = f' · mean cosine to other group members {means[i]:.3f}' if i in means else ''
        diff = ''.join(difflib.unified_diff(context_text(e['baseline_messages']).splitlines(keepends=True), context_text(e['counterfactual_messages']).splitlines(keepends=True), fromfile='baseline', tofile='counterfactual', n=2))
        if not diff: diff = '(No serialized message difference.)'
        body += f'<article class="card member" id="experiment-{i}"><h2>{escape(r["prompt_id"])} · experiment {r["experiment_number"]}</h2><p class="muted">Scenario: {escape(r["scenario"])} · edit type: {escape(r["edit_type"])} · location: {escape(r["edit_location"])} · verification {r["verif_score"]}/10{mean}</p>'
        body += f'<p><strong>{br[0]}/{br[1]} ({bl:.1%}) → {cr[0]}/{cr[1]} ({cf:.1%}) · {r["delta_pp"]:+.1f} percentage points</strong></p>'
        body += f'<p><strong>What the classifier measures</strong></p><p class="text">{escape(e["classifier_question"])}</p>'
        body += f'<p><strong>Edit description used for clustering</strong></p><p class="text embedded-text">{escape(r["cause_text"])}</p>'
        body += f'<details><summary>Original investigator summary — interpretation is not ground truth</summary><p class="text original-text">{escape(r["cause_text_original"])}</p></details>'
        if r.get('text_correction'):
            body += f'<p class="muted">Removed an interpretation sentence before embedding: {escape(r["text_correction"])}</p>'
        body += f'<details><summary>Actual baseline-to-counterfactual context diff</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px">{escape(diff)}</pre></details></article>'
    page(folder / filename, title, body)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    behaviours = json.loads((BEHAVIOURS / 'records.json').read_text())
    ids = {r['prompt_id'] for r in behaviours}; by_id = {r['prompt_id']: r for r in behaviours}
    scenarios = json.loads((BEHAVIOURS / 'summary.json').read_text())['scenarios']
    prior = json.loads((PRIOR / 'records.json').read_text())
    indices = sorted([i for i, r in enumerate(prior) if r['prompt_id'] in ids], key=lambda i: (scenarios.index(by_id[prior[i]['prompt_id']]['scenario']), prior[i]['prompt_id'], prior[i]['experiment_number']))
    records = [{**prior[i], 'scenario': by_id[prior[i]['prompt_id']]['scenario']} for i in indices]
    all_retained = [r for r in pq.read_table(ROOT / 'data/checkpoint1_locations/petri/causes.parquet').to_pylist() if r['prompt_id'] in ids]
    key = lambda r: (r['prompt_id'], r['experiment_number'])
    assert {key(r) for r in records} == {key(r) for r in all_retained if r['clustering_eligible']}
    assert all(r['verif_score'] >= 7 and abs(r['delta_pp']) >= 50 - 1e-8 and r['clustering_eligible'] for r in records)
    source_paths = [PRIOR / f for f in ['records.json', 'embeddings.npy', 'assignments.parquet']]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    vectors = np.load(PRIOR / 'embeddings.npy')[indices].astype(float)
    corrections = []
    for i, r in enumerate(records):
        if key(r) == CORRECTION_KEY:
            assert r['cause_text'].endswith(INTERPRETATION)
            before = r['cause_text']; r['cause_text'] = before[:-len(INTERPRETATION)].strip()
            r['text_correction'] = INTERPRETATION
            corrections.append({'local_index': i, 'prompt_id': r['prompt_id'], 'experiment_number': r['experiment_number'],
                                'before': before, 'after': r['cause_text'], 'reason': 'Remove residual investigator interpretation/outcome sentence.'})
    assert len(corrections) == 1
    if corrections:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(2)
        model = SentenceTransformer(str(ROOT / 'models/all-mpnet-base-v2'), device='cpu', local_files_only=True)
        texts = [r['after'] for r in corrections]
        assert all(len(model.tokenizer(t)['input_ids']) <= model.max_seq_length for t in texts)
        updated = model.encode(texts, device='cpu', normalize_embeddings=True, show_progress_bar=False)
        for correction, vector in zip(corrections, updated): vectors[correction['local_index']] = vector
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine = np.clip(vectors @ vectors.T, -1, 1); np.fill_diagonal(cosine, 1)
    distance = np.clip(1 - cosine, 0, 2); np.fill_diagonal(distance, 0)
    evidence = {}
    with (ROOT / 'data/petri/qwen3_8b_petri_eval/investigations.jsonl').open() as f:
        for line in f:
            raw = json.loads(line)
            if raw['prompt_id'] not in ids: continue
            for exp in raw.get('experiment_log', []):
                k = (raw['prompt_id'], exp['exp_num'])
                if k not in {key(r) for r in records}: continue
                base_labels, cf_labels = raw['baseline_grader_verdicts'], exp['grader_verdicts']
                assert all(isinstance(v, bool) for v in base_labels + cf_labels)
                evidence[k] = {'prompt_id': raw['prompt_id'], 'experiment_number': exp['exp_num'],
                    'classifier_question': raw['classifier_question'], 'baseline_messages': raw['messages'], 'counterfactual_messages': exp['messages'],
                    'baseline_count': [sum(base_labels), len(base_labels)], 'counterfactual_count': [sum(cf_labels), len(cf_labels)]}
    for r in records:
        e = evidence[key(r)]; b = e['baseline_count']; c = e['counterfactual_count']
        assert np.isclose(r['baseline_rate'], b[0] / b[1]) and np.isclose(r['counterfactual_rate'], c[0] / c[1])
        assert np.isclose(r['delta_pp'], (c[0] / c[1] - b[0] / b[1]) * 100)
    save(OUT / 'records.json', records); save(OUT / 'evidence.json', list(evidence.values())); save(OUT / 'text_corrections.json', corrections)
    pq.write_table(pa.Table.from_pylist(records), OUT / 'causes_for_clustering.parquet')
    pq.write_table(pa.Table.from_pylist(all_retained), OUT / 'causes_original_retained.parquet')
    np.save(OUT / 'embeddings.npy', vectors); np.save(OUT / 'cosine.npy', cosine)
    save(OUT / 'embedding_provenance.json', {'source': str(PRIOR), 'source_row_indices': indices, 'model': 'all-mpnet-base-v2',
        'original_model_metadata': json.loads((PRIOR / 'embedding_metadata.json').read_text()), 'reused_vectors': len(records) - len(corrections),
        'locally_refreshed_vectors': len(corrections), 'corrections': corrections, 'device': 'cpu', 'normalized': True, 'generative_llm_calls': 0})
    n = len(records); candidates, seen, trees = [], set(), {}
    for method in ['ward', 'average', 'complete']:
        tree = linkage(vectors, method='ward') if method == 'ward' else linkage(squareform(distance), method=method)
        trees[method] = tree; np.save(OUT / f'{method}_linkage.npy', tree)
        for cut in range(2, n):
            labels = cut_tree(tree, n_clusters=cut).ravel()
            groups = [np.flatnonzero(labels == k).tolist() for k in sorted(set(labels))]
            groups = [g for g in groups if len(g) >= 2 and cohesion(g, cosine) > .4]
            while len(groups) >= 2:
                sig = signature(groups)
                if sig not in seen:
                    seen.add(sig); candidates.append({'method': method, 'original_cut_k': cut, 'groups': [g[:] for g in groups], **stats(groups, cosine, distance)})
                del groups[int(np.argmin([cohesion(g, cosine) for g in groups]))]
    best = max(candidates, key=lambda c: (c['min_cluster_mean_cosine'], c['assigned'], c['mean_cluster_cosine']))
    near = [c for c in candidates if c['min_cluster_mean_cosine'] >= best['min_cluster_mean_cosine'] - .02]
    presets = {'fullest_coverage': max(candidates, key=lambda c: (c['assigned'], c['min_cluster_mean_cosine'], c['mean_cluster_cosine'])),
               'cosine_first': max(near, key=lambda c: (c['assigned'], c['min_cluster_mean_cosine'], c['mean_cluster_cosine']))}
    for threshold in [.5, .6]:
        feasible = [c for c in candidates if c['min_cluster_mean_cosine'] > threshold]
        if feasible: presets[f'cosine_above_{int(threshold*100):02d}'] = max(feasible, key=lambda c: (c['assigned'], c['min_cluster_mean_cosine'], c['mean_cluster_cosine']))
    for name, c in presets.items():
        folder = OUT / name; folder.mkdir(exist_ok=True)
        c['scenario_composition'] = [dict(Counter(records[i]['scenario'] for i in g)) for g in c['groups']]
        c['distinct_investigations_per_cluster'] = [len({records[i]['prompt_id'] for i in g}) for g in c['groups']]
        c['mixed_scenario_clusters'] = sum(len(d) > 1 for d in c['scenario_composition'])
        assignments = np.full(n, -1, dtype=int); rows = []
        for k, g in enumerate(c['groups']):
            assignments[g] = k
            group_page(folder, f'cluster_{k}.html', f'Counterfactual edits · cluster {k}', g, records, cosine, evidence)
            rows.append([f'<a href="cluster_{k}.html">Cluster {k}</a>', str(len(g)), str(c['distinct_investigations_per_cluster'][k]), f'{cohesion(g, cosine):.3f}', escape(', '.join(f'{s}: {count}' for s, count in c['scenario_composition'][k].items()))])
        selected = np.flatnonzero(assignments >= 0)
        sil = silhouette_samples(distance[np.ix_(selected, selected)], assignments[selected], metric='precomputed')
        assert np.isclose(sil.mean(), c['mean_silhouette'])
        group_page(folder, 'unassigned.html', 'Unassigned eligible counterfactual edits', np.flatnonzero(assignments < 0).tolist(), records, cosine, evidence, is_cluster=False)
        pq.write_table(pa.Table.from_pylist([{**r, 'cluster_id': int(assignments[i])} for i, r in enumerate(records)]), folder / 'assignments.parquet')
        save(folder / 'summary.json', c)
        body = f'<p><a href="../review.html">All counterfactual results</a></p><p><strong>{c["assigned"]}/{n} experiments assigned · {c["clusters"]} groups · {c["unassigned"]} unassigned.</strong> Each group has at least two experiments and mean cosine &gt;0.4. These are experiments, not investigations.</p>'
        body += table(['Open all entries', 'Experiments', 'Distinct investigations', 'Mean cosine', 'Scenario composition'], rows)
        body += '<p><a href="unassigned.html">Every unassigned experiment</a> · <a href="assignments.parquet">Assignments</a></p><p>Similarity describes intervention wording, not proof of a shared causal mechanism. Several experiments can come from the same investigation. Original explanations are visible only as evidence for review, not embedding features.</p>'
        page(folder / 'review.html', 'Counterfactual edits · ' + name.replace('_', ' '), body)
    counts = []
    positive = {r['prompt_id'] for r in records}
    for s in scenarios:
        bs = [r for r in behaviours if r['scenario'] == s]; cs = [r for r in records if r['scenario'] == s]
        count = sum(r['prompt_id'] in positive for r in bs)
        counts.append({'scenario': s, 'qualified_investigations': len(bs), 'investigations_with_eligible_edits': count,
            'zero_eligible_investigations': len(bs) - count, 'eligible_experiments': len(cs), 'mean_experiments_positive_investigations_only': len(cs)/count if count else None})
    excluded = [r for r in all_retained if not r['clustering_eligible']]
    save(OUT / 'excluded_rows.json', excluded)
    exclusion_body = '<p><a href="review.html">Back to review</a></p><p>Large-effect rows retained by prior preprocessing but excluded from clustering. Text below is the original investigator summary, not ground truth.</p>'
    for r in excluded:
        exclusion_body += f'<article class="card"><strong>{escape(r["prompt_id"])} · experiment {r["experiment_number"]}</strong><p>{escape(str(r["clustering_exclusion_reasons"]))}</p><p class="text">{escape(r["cause_text_original"])}</p></article>'
    page(OUT / 'excluded.html', f'{len(excluded)} excluded large-effect experiment rows', exclusion_body)
    # Ordered matrix labels map directly to the experiment index table below.
    order = leaves_list(trees['average']).astype(int)
    labels = [f'{i+1:02d}' for i in range(n)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 11), gridspec_kw={'width_ratios':[1, 1]})
    dendrogram(trees['average'], labels=labels, orientation='right', ax=axes[0], color_threshold=0, above_threshold_color='#086c79')
    axes[0].invert_yaxis()
    axes[0].set_title('Average-linkage hierarchy'); axes[0].set_xlabel('Cosine distance')
    im = axes[1].imshow(cosine[np.ix_(order, order)], vmin=0, vmax=1, cmap='Blues')
    axes[1].set_xticks(range(n), [labels[i] for i in order], rotation=90, fontsize=8)
    axes[1].set_yticks(range(n), [labels[i] for i in order], fontsize=8); axes[1].set_title('Within- and between-edit cosine similarity')
    fig.colorbar(im, ax=axes[1], fraction=.046, pad=.04)
    fig.suptitle(f'Counterfactual edits · {n} experiments from three related PETRI scenarios')
    fig.tight_layout(); fig.canvas.draw()
    tree_box, heat_box = axes[0].get_position(), axes[1].get_position()
    axes[0].set_position([tree_box.x0, heat_box.y0, tree_box.width, heat_box.height])
    fig.savefig(OUT / 'similarity_and_hierarchy.png', dpi=160); fig.savefig(OUT / 'similarity_and_hierarchy.svg'); plt.close(fig)
    summary = {'scenarios': scenarios, 'counts': counts, 'qualified_investigations': len(behaviours), 'contributing_investigations': len(positive),
        'zero_eligible_investigations': sorted(ids - positive), 'eligible_experiments': n, 'retained_large_effect_rows': len(all_retained),
        'excluded_retained_rows': len(excluded), 'minimum_cluster_size': 2, 'minimum_absolute_delta_pp': 50,
        'candidate_partitions': len(candidates), 'presets': presets, 'text_corrections': corrections,
        'embedding_features': 'Cleaned edit descriptions only; no rates, locations, scenario IDs, classifier questions, Core causes, or Final answer.',
        'selection': 'Same tree-cut search as the related-scenario behaviour run: Ward, average and complete; reject singletons and groups with mean cosine <=0.4; optionally remove lowest-cohesion groups. Cosine-first maximizes the minimum group mean cosine, allowing .02 tolerance to improve coverage. Silhouette is diagnostic only.',
        'rates_recomputed_and_checked_against_stored_response_labels': True, 'generative_llm_calls': 0, 'semantic_labels_assigned': False}
    save(OUT / 'summary.json', summary); save(OUT / 'candidates.json', candidates)
    body = '<p>The same three related PETRI scenarios, including all qualifying turns and both runs. <strong>One item now represents a counterfactual experiment, not a behaviour summary.</strong> We group the descriptions of what was edited in experiments with a measured change of at least 50 percentage points.</p>'
    body += f'<p><strong>{n} eligible experiments from {len(positive)}/{len(behaviours)} investigations.</strong> Five investigations have no eligible edits. {len(excluded)} additional large-effect rows are retained for audit but excluded by the existing preprocessing rules.</p>'
    body += table(['Scenario', 'Qualified investigations', 'With eligible edits', 'Zero eligible', 'Eligible experiments'], [[r['scenario'],str(r['qualified_investigations']),str(r['investigations_with_eligible_edits']),str(r['zero_eligible_investigations']),str(r['eligible_experiments'])] for r in counts])
    body += '<p>Filters: verification ≥7; absolute rate change ≥50 pp; existing exact-text deduplication, context-size and measurement-artifact exclusions. Edit location is metadata only. Minimum group size remains two for this small related-scenario comparison. Investigator causal conclusions are not clustering features.</p><p>One residual interpretation sentence was removed from an existing cleaned description and that one MPNet embedding refreshed locally. The other 35 vectors were reused. <a href="text_corrections.json">Inspect the exact correction</a>.</p>'
    rows = [[f'<a href="{name}/review.html">{name.replace("_", " ")}</a>',str(c['clusters']),f'{c["assigned"]}/{n}',str(c['unassigned']),f'{c["min_cluster_mean_cosine"]:.3f}',str(c['mixed_scenario_clusters'])] for name,c in presets.items()]
    body += table(['Review every experiment', 'Groups', 'Assigned', 'Unassigned', 'Lowest mean cosine', 'Mixed-scenario groups'], rows)
    body += '<p><strong>Fullest coverage</strong> prioritizes retaining experiments, then within-group cosine. <strong>Cosine first</strong> favors similarity and may select tiny, closely worded groups. Every group page shows distinct investigation counts, original summaries, actual context diffs, classifier definitions, before/after rates and signed changes. These are observed effects of edits, not automatically isolated causal mechanisms.</p>'
    strict = presets['cosine_first']
    if all(count == 1 for count in strict['distinct_investigations_per_cluster']):
        body += '<p class="notice">In this run, every strict cosine-first group consists of experiments from just one investigation. Their high similarity therefore does not demonstrate a recurring intervention pattern across independent investigations. The broader alternatives retain more cross-investigation comparisons.</p>'
    body += '<p><a href="excluded.html">Excluded large-effect rows and reasons</a> · <a href="causes_for_clustering.parquet">Clustering input table</a> · <a href="summary.json">Summary and method</a> · <a href="../related_health_scenarios/review.html">Compare the behaviour clustering</a></p>'
    body += '<img src="similarity_and_hierarchy.png" alt="Average-linkage hierarchy and pairwise cosine similarity for 36 counterfactual edit descriptions">'
    broad = presets['fullest_coverage']; lookup = {i:k for k,g in enumerate(broad['groups']) for i in g}
    index_rows = []
    for i,r in enumerate(records):
        target = f'fullest_coverage/cluster_{lookup[i]}.html' if i in lookup else 'fullest_coverage/unassigned.html'
        index_rows.append([f'<a href="{target}#experiment-{i}">{i+1:02d}</a>',escape(r['prompt_id']),str(r['experiment_number']),escape(r['cause_text']),f'{r["delta_pp"]:+.1f}'])
    body += '<h2>Every eligible experiment — plot index</h2>' + table(['Plot ID / open evidence','Investigation','Experiment','Embedding text','Change (pp)'], index_rows)
    page(OUT / 'review.html', 'What changed the behaviour? Counterfactual edit clusters', body)
    class Links(HTMLParser):
        def __init__(self): super().__init__(); self.targets=[]
        def handle_starttag(self, tag, attrs):
            a=dict(attrs)
            if tag=='a' and a.get('href'): self.targets.append(a['href'])
            if tag=='img' and a.get('src'): self.targets.append(a['src'])
    for path in OUT.rglob('*.html'):
        parser=Links();parser.feed(path.read_text())
        for target in parser.targets:
            url,*anchor=target.split('#'); p=path.parent/url
            assert p.exists(), (path,target)
            if anchor: assert f'id="{anchor[0]}"' in p.read_text()
    for name,c in presets.items():
        used=sum(c['groups'],[]); assert len(used)==len(set(used))==c['assigned']
        for k,g in enumerate(c['groups']):
            text=(OUT/name/f'cluster_{k}.html').read_text()
            assert text.count('class="card member"')==len(g)
            assert np.isclose(cohesion(g,cosine),cosine[np.ix_(g,g)][np.triu_indices(len(g),1)].mean())
            for i in g:
                assert escape(records[i]['cause_text']) in text and escape(records[i]['cause_text_original']) in text
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    save(OUT/'validation.json',{'status':'passed','all_36_rates_recomputed_from_labels':True,'embedding_text_correction_audited':True,
        'all_members_and_original_texts_displayed':True,'all_links_and_anchors_exist':True,'silhouettes_match_sklearn':True,'prior_results_unchanged':True,'source_hashes':hashes})
    print(json.dumps({'counts':counts,'retained':len(all_retained),'eligible':n,'excluded':len(excluded),'candidates':len(candidates),
        'results':{name:{k:c[k] for k in ['clusters','assigned','cluster_sizes','cluster_mean_cosines','mixed_scenario_clusters','distinct_investigations_per_cluster']} for name,c in presets.items()}},indent=2),flush=True)
    print('REVIEW',OUT/'review.html',flush=True)


if __name__=='__main__':main()
