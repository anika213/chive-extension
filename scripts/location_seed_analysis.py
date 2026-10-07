"""Separate observed edit locations from text; audit behaviour seed concentration.

No generative model calls. PETRI scenario IDs are explicitly identified as proxies,
not an independently recorded seed variable. WildChat seed IDs remain null.
"""
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import preprocessing as base
from preprocessing_context import experiment_messages, context_text

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / 'data/checkpoint1_approved'
OUTPUT = ROOT / 'data/checkpoint1_locations'
RESULT = ROOT / 'results/location_seed'
SOURCES = ('petri', 'wildchat_eval')


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def message_kind(message):
    role, text = message['role'], message['content']
    if role == 'system': return ['system_prompt']
    if role == 'assistant': return ['prior_assistant_turn']
    if role == 'tool': return ['tool_output']
    if role == 'user':
        # Keep the actual transport role separately; explicit wrappers are observable.
        if re.fullmatch(r'\s*<tool_response>.*?</tool_response>\s*', text, re.S):
            return ['tool_output']
        if '<tool_response>' in text:
            return ['user_message', 'tool_output']
        return ['user_message']
    raise ValueError('Unsupported message role: ' + role)


def edit_location(before, after):
    # Exact-message matching anchors prevent an inserted/deleted turn shifting every index.
    keys = lambda messages: [json.dumps(m, sort_keys=True, ensure_ascii=False) for m in messages]
    diff = SequenceMatcher(None, keys(before), keys(after), autojunk=False)
    removed, added = [], []
    for tag, a, b, c, d in diff.get_opcodes():
        if tag != 'equal':
            removed.extend(range(a, b)); added.extend(range(c, d))
    old_roles = sorted({before[i]['role'] for i in removed})
    new_roles = sorted({after[i]['role'] for i in added})
    kinds = sorted({kind for messages, indices in [(before, removed), (after, added)]
                    for i in indices for kind in message_kind(messages[i])})
    mixed = any(len(message_kind(messages[i])) > 1 for messages, indices in [(before, removed), (after, added)] for i in indices)
    return {'edit_location': '+'.join(kinds) if kinds else 'no_change', 'edit_locations': kinds,
            'changed_message_roles': sorted(set(old_roles + new_roles)),
            'removed_or_modified_roles': old_roles, 'added_or_modified_roles': new_roles,
            'changed_baseline_message_indices': removed, 'changed_experiment_message_indices': added,
            'mixed_tool_user_message': mixed,
            'edit_location_method': 'exact_message_sequence_diff_with_explicit_tool_wrapper'}


# Deterministic deletion only: strip explicit role/turn locator spans, never paraphrase.
# Preserve quoted edit payloads: role words inside them can be the actual intervention.
LOCATOR = re.compile(
    r'\b(?:(?:the|a|an|all|both|entire|full|original|final|last|first|initial|prior|previous|earlier|current|new|real|synthetic|multi-turn)\s+)*'
    r'(?:system[ -](?:prompts?|messages?)|(?:user|assistant)(?:[’\']s)?\s+(?:(?:original|first|final|last|prior|previous|earlier|revised)\s+)?(?:messages?|turns?|prompts?|responses?|replies|reply|questions?|requests?|summaries|summary)'
    r'|tool[ _-](?:responses?|outputs?|results?)|conversation\s+(?:history|context)|chat\s+history'
    r'|(?:prior|previous|original|full|all)\s+history|(?:full|original)\s+conversation'
    r'|(?:final|last|first|initial|prior|previous|earlier)\s+(?:messages?|turns?|questions?)'
    r'|turns?\s+\d+(?:\s*[-–,]\s*\d+)*)\b', re.I)
QUOTE = re.compile(r'"[^"\n]*"|(?<!\w)\x27[^\x27\n]*\x27(?!\w)|“[^”\n]*”|‘[^’\n]*’')


def strip_location(text):
    protected = [(m.start(), m.end()) for m in QUOTE.finditer(text)]
    spans = []
    for match in LOCATOR.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in protected):
            continue
        start, end = match.span()
        # Remove an immediately preceding locative preposition, not an edit verb.
        prefix = re.search(r'\b(?:from|in|into|to|within|of|on|with)\s+$', text[:start], re.I)
        if prefix: start = prefix.start()
        spans.append((start, end))
    clean = text
    for start, end in reversed(spans): clean = clean[:start] + clean[end:]
    clean = re.sub(r'\(\s*\)', '', clean)
    # These are empty retention scaffolds left by deleting locator spans.
    clean = re.sub(r'\bwhile\s+keeping\s*(?=[.,;]|$)', '', clean, flags=re.I)
    clean = re.sub(r'^Kept\s+(?:but\s+|[,;]\s*)', '', clean, flags=re.I)
    clean = re.sub(r'\band\s*(?=[,;]|$)', '', clean, flags=re.I)
    clean = re.sub(r'\s+([.,;:])', r'\1', clean)
    clean = re.sub(r'\s+', ' ', clean).strip(' ,;')
    return clean, [text[a:b] for a,b in spans]


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True); RESULT.mkdir(parents=True, exist_ok=True)
    summaries, manifest = {}, []
    for source in SOURCES:
        folder = OUTPUT / source; folder.mkdir(exist_ok=True)
        rows = pq.read_table(INPUT / source / 'causes.parquet').to_pylist()
        needed = {(r['prompt_id'], r['experiment_number']) for r in rows}
        diffs = {(r['prompt_id'], r['experiment_number']): r for r in pq.read_table(ROOT / 'data/checkpoint1_v2' / source / 'context_diffs.parquet').to_pylist()}
        features, seed_metadata = {}, {}
        prompt_path = ROOT / 'data/seed_provenance' / (source + '_prompts.jsonl')
        prompts = {r['id']: r for r in map(json.loads, prompt_path.open())}
        manifest.append({'source': source, 'remote_path': f'datasets/qwen3_8b_{source if source == "wildchat_eval" else "petri_eval"}/prompts.jsonl',
                         'revision': '0283d021bd373d361aadd46a9cf1f9520399d50a', 'bytes': prompt_path.stat().st_size,
                         'sha256': hashlib.sha256(prompt_path.read_bytes()).hexdigest(), 'rows': len(prompts)})
        for line in (ROOT / 'data' / base.SOURCES[source] / 'investigations.jsonl').open():
            record = json.loads(line); pid = record['prompt_id']; before = record['messages']
            prompt = prompts.get(pid)
            if prompt is None: raise ValueError(f'No prompt metadata for {source}/{pid}')
            assert prompt['messages'] == before, (source, pid, 'prompt context mismatch')
            match = re.fullmatch(r'petri_(.+)_e(\d+)_t(\d+)', pid) if source == 'petri' else None
            if source == 'petri' and match is None: raise ValueError('Unexpected PETRI ID: ' + pid)
            seed_metadata[pid] = {'scenario_seed_proxy': match[1] if match else None,
                'seed_provenance': 'PETRI prompt_id scenario prefix; inferred proxy, no explicit seed field' if match else 'unavailable: no scenario seed in published records',
                'episode_id_proxy': match[1] + '_e' + match[2] if match else None,
                'prompt_source_detail': prompt['source'], 'conversation_hash': prompt['conversation_hash']}
            for experiment in record['experiment_log']:
                key = (pid, experiment['exp_num'])
                if key not in needed: continue
                after, _ = experiment_messages(experiment)
                cached = diffs[key]
                assert hashlib.sha256(context_text(before).encode()).hexdigest() == cached['baseline_context_sha256']
                assert hashlib.sha256(context_text(after).encode()).hexdigest() == cached['experiment_context_sha256']
                features[key] = edit_location(before, after)
        assert set(features) == needed
        for row in rows:
            row.update(features[(row['prompt_id'], row['experiment_number'])])
            row.update(seed_metadata[row['prompt_id']])
            old = row['cause_text']; clean, spans = strip_location(old)
            row.update(cause_text_with_location=old, cause_text=clean, location_text_removed=spans)
            location_only = bool(spans) and bool(re.fullmatch(r'(?:(?:removed|deleted|replaced|rewrote|modified|changed|kept|entirely|only|completely)\s*)*[.!]?', clean, re.I))
            row['location_only_text'] = location_only
            if not clean or location_only:
                row['clustering_exclusion_reasons'].append('no_edit_content_after_location_removal')
                row['clustering_eligible'] = False
        # Text normalization can create additional exact duplicates. Keep an audit trail.
        seen = {}; extra_duplicates = 0
        for row in rows:
            row['location_normalization_duplicate_of'] = None
            if not row['clustering_eligible']: continue
            key = row['cause_text']
            if key in seen:
                row['clustering_eligible'] = False
                row['clustering_exclusion_reasons'].append('exact_duplicate_after_location_removal')
                row['location_normalization_duplicate_of'] = seen[key]
                extra_duplicates += 1
            else: seen[key] = f'{row["prompt_id"]}/{row["experiment_number"]}'
        eligible = [r for r in rows if r['clustering_eligible']]
        for name, data in [('causes', rows), ('causes_for_clustering', eligible)]:
            pq.write_table(pa.Table.from_pylist(data), folder / (name + '.parquet'), compression='zstd')
        for name in ['behaviours', 'behaviours_unfiltered']:
            data = pq.read_table(INPUT / source / (name + '.parquet')).to_pylist()
            for row in data: row.update(seed_metadata[row['prompt_id']])
            pq.write_table(pa.Table.from_pylist(data), folder / (name + '.parquet'), compression='zstd')
        old_counts = json.loads((INPUT / source / 'counts.json').read_text())
        counts = Counter(r['prompt_id'] for r in eligible)
        ids = old_counts['verification_qualified_investigation_ids']
        values = [counts[i] for i in ids if counts[i] > 0]
        summary = {'retained_causes': len(rows), 'eligible_causes': len(eligible), 'changed_texts_retained': sum(bool(r['location_text_removed']) for r in rows),
                   'changed_texts_eligible': sum(bool(r['location_text_removed']) for r in eligible),
                   'new_exact_duplicates': extra_duplicates, 'empty_after_removal': sum(not r['cause_text'] for r in rows),
                   'location_only_text': sum(r['location_only_text'] for r in rows),
                   'edit_location_counts_eligible': dict(Counter(r['edit_location'] for r in eligible)),
                   'mixed_tool_user_messages_eligible': sum(r['mixed_tool_user_message'] for r in eligible),
                   'causes_per_investigation': {'positive_denominator': len(values), 'zero_eligible_excluded': len(ids)-len(values),
                                              'mean': float(np.mean(values)), 'median': float(np.median(values)), 'max': max(values)}}
        save(folder / 'counts.json', {**old_counts, **summary}); summaries[source] = summary
        save(folder / 'seed_metadata.json', seed_metadata)
        print(source, summary, flush=True)
    save(OUTPUT / 'summary.json', summaries)
    save(ROOT / 'data/seed_provenance/manifest.json', manifest)
    return summaries


def seed_concentration(assignments, method, field='scenario_seed_proxy'):
    groups = {}
    for r in assignments:
        label = r[method + '_cluster_id']
        if label >= 0: groups.setdefault(label, []).append(r)
    result = []
    for label, rows in sorted(groups.items()):
        counts = Counter(r[field] for r in rows if r.get(field) is not None)
        ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        known = sum(counts.values()); n = len(rows)
        result.append({'cluster_id': label, 'size': n, 'known_seed_rows': known, 'missing_seed_rows': n-known,
            'distinct_seeds': len(counts) if known else None,
            'dominant_seed': ranked[0][0] if ranked else None,
            'dominant_seed_count': ranked[0][1] if ranked else None,
            'top1_fraction': ranked[0][1]/n if known == n else None,
            'top2_fraction': sum(v for _,v in ranked[:2])/n if known == n else None,
            'seed_counts': dict(ranked)})
    return result


def seed_audit():
    overall = {}
    for source in SOURCES:
        dest = RESULT / source; dest.mkdir(parents=True, exist_ok=True)
        seeds = json.loads((OUTPUT / source / 'seed_metadata.json').read_text())
        old = ROOT / 'results/checkpoint2' / source / 'behaviours'
        assignments = pq.read_table(old / 'assignments.parquet').to_pylist()
        for row in assignments: row.update(seeds[row['prompt_id']])
        pq.write_table(pa.Table.from_pylist(assignments), dest / 'behaviour_assignments.parquet')
        summary = {'rows': len(assignments), 'known_scenario_seed_proxy_rows': sum(r['scenario_seed_proxy'] is not None for r in assignments), 'methods': {}}
        for method in ['ward','hdbscan']:
            stats = seed_concentration(assignments, method)
            save(dest / f'{method}_seed_concentration.json', stats)
            pq.write_table(pa.Table.from_pylist([{k:v for k,v in r.items() if k != 'seed_counts'} for r in stats]), dest / f'{method}_seed_concentration.parquet')
            known = [r for r in stats if r['top1_fraction'] is not None]
            n = len(known); assigned = sum(r['size'] for r in known)
            # Conditional permutation baseline: preserve source seed frequencies and cluster sizes.
            baseline = []
            if n:
                assigned_rows = [r for r in assignments if r[method+'_cluster_id'] >= 0]
                labels = np.array([r[method+'_cluster_id'] for r in assigned_rows])
                seed_values = np.array([r['scenario_seed_proxy'] for r in assigned_rows])
                rng = np.random.default_rng(20261004)
                for _ in range(500):
                    shuffled = rng.permutation(seed_values)
                    maxima = [max(Counter(shuffled[labels == k]).values()) for k in sorted(set(labels))]
                    baseline.append(sum(maxima)/len(labels))
            summary['methods'][method] = {'clusters': len(stats), 'clusters_with_known_seeds': n,
                'noise_rows': sum(r[method+'_cluster_id'] < 0 for r in assignments),
                'singleton_clusters': sum(r['size'] == 1 for r in stats),
                'clusters_top1_over_half': sum(r['top1_fraction'] > .5 for r in known) if n else None,
                'clusters_top2_over_half': sum(r['top2_fraction'] > .5 for r in known) if n else None,
                'median_distinct_seeds': float(np.median([r['distinct_seeds'] for r in known])) if n else None,
                'median_top1_fraction': float(np.median([r['top1_fraction'] for r in known])) if n else None,
                'row_weighted_top1_fraction': sum(r['dominant_seed_count'] for r in known)/assigned if n else None,
                'row_weighted_top2_fraction': sum(r['top2_fraction']*r['size'] for r in known)/assigned if n else None,
                'permuted_seed_weighted_top1_mean': float(np.mean(baseline)) if n else None,
                'permuted_seed_weighted_top1_p025_p975': np.quantile(baseline,[.025,.975]).tolist() if n else None}
        overall[source] = summary
    save(RESULT / 'seed_summary.json', overall)
    return overall


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--scenario-prefix-proxy', action='store_true')
    args = parser.parse_args()
    prepare()
    if args.scenario_prefix_proxy: seed_audit()
