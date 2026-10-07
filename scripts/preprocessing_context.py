"""Checkpoint 1 revision 2: measured context diffs and clustering eligibility.

No model inference: only a local tokenizer and exact LCS sequence comparisons.
Revision 1 outputs are immutable inputs; revised outputs go to checkpoint1_v2.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import random
import re
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz.distance import LCSseq
from tokenizers import Tokenizer

import preprocessing as base

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_TERMS = ('N/A', 'not applicable', 'not reach', 'as expected', 'trivially', 'cannot be graded')
ARTIFACT_PATTERNS = {
    'N/A': re.compile(r'(?<!\w)n/a(?!\w)', re.I),
    'not applicable': re.compile(r'\bnot applicable\b', re.I),
    'not reach': re.compile(r'\bnot reach\w*\b', re.I),
    'as expected': re.compile(r'\bas expected\b', re.I),
    'trivially': re.compile(r'\btrivially\b', re.I),
    'cannot be graded': re.compile(r'\bcannot be graded\b', re.I),
}
COUNT = r'(?:\d+\s*/\s*\d+|\d+\s+(?:out\s+)?of\s+\d+|\d+(?:\.\d+)?\s*%)'
COUNT_PREFIX = re.compile(
    r'^(?:(?:in|across|for)\s+)?(?:(?:approximately|about|around|roughly|nearly|almost|over|more than|only)\s+)?'
    r'(?:' + COUNT + r'(?=$|[\s,;:—–-])|all(?:\s+\d+)?(?=\s+(?:(?:sampled|baseline|generated)\s+)?(?:completions?|responses?|samples?|outputs?)\b))'
    r'(?:(?:\s+of)?(?:\s+(?:the|sampled|baseline|generated))?'
    r'\s+(?:completions?|responses?|samples?|outputs?|cases?|trials?))?'
    r'(?:\s*\((?:[^)]*(?:\d+\s*/\s*\d+|%)[^)]*|[#\d\s,;–-]+|\d+\s+(?:prompts?|responses?|completions?|outputs?|samples?)\s+total)\))?'
    r'\s*[,;:—–-]?\s*', re.I)


def behaviour_text(text):
    cleaned = base.clean(text)
    match = COUNT_PREFIX.match(cleaned)
    if match:
        # Never remove a numeric prefix glued to an identifier, date, or version.
        rest = cleaned[match.end():].strip()
        if rest and (match.end() == len(cleaned) or not rest.startswith(('/', '%'))):
            return rest, cleaned[:match.end()].strip()
    return cleaned, None


def artifact_matches(original):
    folded = re.sub(r'\s+', ' ', original or '').casefold()
    return [term for term in ARTIFACT_TERMS if ARTIFACT_PATTERNS[term].search(folded)]


def context_text(messages):
    if not isinstance(messages, list):
        raise ValueError('missing_messages')
    blocks = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get('role'), str) or not isinstance(message.get('content'), str):
            raise ValueError('unsupported_message_schema')
        extra = {k: v for k, v in message.items() if k not in ('role', 'content')}
        block = f'⟪ROLE:{message["role"]}⟫\n' + message['content'] + '\n⟪END MESSAGE⟫\n'
        if extra:
            block += '⟪METADATA⟫' + json.dumps(extra, sort_keys=True, ensure_ascii=False) + '\n'
        blocks.append(block)
    return ''.join(blocks)


def experiment_messages(entry):
    """Exact legacy mapping from chive/chive/pipeline/utils.py:676–699.

    Stored messages take priority, even if malformed. Only absent messages use
    the documented legacy full prompt + optional system_prompt representation.
    """
    if 'messages' in entry:
        return entry['messages'], 'messages'
    if not isinstance(entry.get('prompt'), str):
        raise ValueError('missing_messages_and_legacy_prompt')
    messages = []
    if entry.get('system_prompt'):
        messages.append({'role': 'system', 'content': entry['system_prompt']})
    messages.append({'role': 'user', 'content': entry['prompt']})
    return messages, 'legacy_prompt_and_system_prompt'


def change_counts(before, after):
    common = LCSseq.similarity(before, after)
    removed, added = len(before) - common, len(after) - common
    return removed, added


def diff_type(removed, added):
    if removed and added:
        return 'modify'
    if removed:
        return 'remove'
    if added:
        return 'add'
    return 'no_change'


def context_diff(before, after, tokenizer, before_tokens=None):
    if before_tokens is None:
        before_tokens = tokenizer.encode(before, add_special_tokens=False).ids
    after_tokens = tokenizer.encode(after, add_special_tokens=False).ids
    deleted_chars, added_chars = change_counts(before, after)
    deleted_tokens, added_tokens = change_counts(before_tokens, after_tokens)
    return {
        'context_diff_available': True, 'context_diff_error': None,
        'baseline_context_chars': len(before), 'experiment_context_chars': len(after),
        'baseline_context_tokens': len(before_tokens), 'experiment_context_tokens': len(after_tokens),
        'chars_deleted': deleted_chars, 'chars_added': added_chars, 'chars_changed': deleted_chars + added_chars,
        'tokens_deleted': deleted_tokens, 'tokens_added': added_tokens, 'tokens_changed': deleted_tokens + added_tokens,
        'edit_type': diff_type(deleted_chars, added_chars),
        'baseline_context_sha256': hashlib.sha256(before.encode()).hexdigest(),
        'experiment_context_sha256': hashlib.sha256(after.encode()).hexdigest(),
    }


SIZE_FIELDS = ('baseline_context_chars', 'experiment_context_chars', 'baseline_context_tokens', 'experiment_context_tokens',
               'chars_deleted', 'chars_added', 'chars_changed', 'tokens_deleted', 'tokens_added', 'tokens_changed')


def feature_schema():
    return pa.schema([('source', pa.string()), ('prompt_id', pa.string()), ('experiment_number', pa.int64()),
                      ('verif_score', pa.float64()), ('experiment_context_format', pa.string()), ('context_diff_available', pa.bool_()), ('context_diff_error', pa.string()),
                      *[(k, pa.int64()) for k in SIZE_FIELDS], ('edit_type', pa.string()),
                      ('baseline_context_sha256', pa.string()), ('experiment_context_sha256', pa.string())])


def extract_diffs(data_root, out, tokenizer_path):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    progress = time.monotonic()
    for source, relative in base.SOURCES.items():
        folder = data_root / relative
        dest = out / source
        dest.mkdir(parents=True, exist_ok=True)
        scores = {}
        with (folder / 'verification.jsonl').open() as f:
            for line in f:
                r = json.loads(line); scores[r['prompt_id']] = r['score']
        rows = []
        with (folder / 'investigations.jsonl').open() as f:
            for line in f:
                r = json.loads(line)
                try:
                    before = context_text(r.get('messages'))
                    before_tokens = tokenizer.encode(before, add_special_tokens=False).ids
                    before_error = None
                except ValueError as error:
                    before, before_tokens, before_error = None, None, str(error)
                for e in r.get('experiment_log', []):
                    common = {'source': source, 'prompt_id': r['prompt_id'], 'experiment_number': e['exp_num'],
                              'verif_score': scores.get(r['prompt_id']), 'experiment_context_format': None}
                    try:
                        if before_error:
                            raise ValueError('baseline_' + before_error)
                        messages, context_format = experiment_messages(e)
                        common['experiment_context_format'] = context_format
                        after = context_text(messages)
                        diff = context_diff(before, after, tokenizer, before_tokens)
                    except ValueError as error:
                        diff = {k: None for k in SIZE_FIELDS}
                        diff.update(context_diff_available=False, context_diff_error=str(error), edit_type='unknown',
                                    baseline_context_sha256=None, experiment_context_sha256=None)
                    rows.append({**common, **diff})
                if time.monotonic() - progress > 20:
                    print(f'Context diffs: {source}, {len(rows):,} experiments', flush=True)
                    progress = time.monotonic()
        base.write_parquet(dest / 'context_diffs.parquet', rows, feature_schema())
        print(f'Context diffs complete: {source}, {len(rows):,} experiments', flush=True)
    provenance = {
        'tokenizer': 'sentence-transformers/all-mpnet-base-v2 (local tokenizer only; no model inference)',
        'tokenizer_path': str(tokenizer_path), 'tokenizer_sha256': hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        'algorithm': 'Exact longest-common-subsequence insert/delete distance, independently on Unicode code points and full-context token ID sequences; changed = deleted + added.',
        'context_serialization': 'For each stored message: ⟪ROLE:{role}⟫ newline + verbatim content + newline ⟪END MESSAGE⟫ newline. Extra message fields, if any, follow as sorted JSON metadata.',
        'tokenization': 'No truncation, padding, special tokens, or model forward pass.',
        'percentile_population': 'All context-comparable logged experiments with investigation verification >=7 within each source, before effect filtering and exact text deduplication.',
        'context_swap_rule': 'chars_changed > source p90 OR tokens_changed > source p90; strict comparison, linear percentile interpolation. Heuristic size flag, not a claim of a literal context replacement.',
        'model_inference_calls': 0,
        'context_reader': 'Stored messages, else legacy prompt + optional system_prompt, exactly as chive/chive/pipeline/utils.py:experiment_to_messages documents. No context inferred from prose.',
    }
    (out / 'diff_provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + '\n')


def distribution(values):
    if not values:
        return {'n': 0}
    a = np.asarray(values, dtype=np.float64)
    return {'n': len(values), 'mean': float(a.mean()), 'min': float(a.min()),
            **{f'p{q}': float(np.percentile(a, q, method='linear')) for q in [25, 50, 75, 90, 95, 99]},
            'max': float(a.max())}


def causes_per_investigation(causes, ids):
    by_id = Counter(c['prompt_id'] for c in causes)
    values = [by_id[pid] for pid in ids]
    positive = [n for n in values if n]
    def stats(xs):
        return {'n_investigations': len(xs), 'mean': float(np.mean(xs)) if xs else None,
                'median': float(np.median(xs)) if xs else None, 'max': max(xs) if xs else None}
    return {'all_verification_qualified_investigations': stats(values),
            'investigations_with_at_least_one_cause': stats(positive), 'zero_cause_investigations': values.count(0)}


def extended_cause_schema():
    return pa.schema(list(base.schemas()[1]) + [
        pa.field('verif_score', pa.float64()),
        *[pa.field(k, pa.int64()) for k in SIZE_FIELDS],
        pa.field('context_diff_available', pa.bool_()), pa.field('context_diff_error', pa.string()),
        pa.field('context_swap', pa.bool_()), pa.field('context_swap_char_p90', pa.float64()), pa.field('context_swap_token_p90', pa.float64()),
        pa.field('measurement_artifact', pa.bool_()), pa.field('measurement_artifact_matches', pa.list_(pa.string())),
        pa.field('clustering_eligible', pa.bool_()), pa.field('clustering_exclusion_reasons', pa.list_(pa.string())),
    ])


def process_source(source, baseline, out):
    folder = out / source
    feature_rows = pq.read_table(folder / 'context_diffs.parquet').to_pylist()
    features = {(r['prompt_id'], r['experiment_number']): r for r in feature_rows}
    assert len(features) == len(feature_rows)
    pool = [r for r in feature_rows if r['verif_score'] is not None and r['verif_score'] >= 7 and r['context_diff_available']]
    sizes = {field: distribution([r[field] for r in pool]) for field in SIZE_FIELDS}
    char_cut, token_cut = sizes['chars_changed']['p90'], sizes['tokens_changed']['p90']
    old_causes = pq.read_table(baseline / source / 'causes.parquet').to_pylist()
    old_cause_keys = {(r['prompt_id'], r['experiment_number']) for r in old_causes}
    all_causes = pq.read_table(baseline / source / 'causes_unfiltered.parquet').to_pylist()
    causes, all_updated = [], []
    exclusion_counts = Counter()
    for c in all_causes:
        key = (c['prompt_id'], c['experiment_number'])
        f = features[key]
        swap = bool(f['context_diff_available'] and (f['chars_changed'] > char_cut or f['tokens_changed'] > token_cut))
        matches = artifact_matches(c['cause_text_original'])
        reasons = []
        if key not in old_cause_keys:
            reasons.append('not_in_revision1_filtered_cause_table')
        if not f['context_diff_available']:
            reasons.append('missing_context_diff')
        if swap:
            reasons.append('context_swap')
        if matches:
            reasons.append('measurement_artifact')
        updated = {**c, 'edit_type': f['edit_type'], 'verif_score': f['verif_score'],
                   **{k: f[k] for k in SIZE_FIELDS},
                   'context_diff_available': f['context_diff_available'], 'context_diff_error': f['context_diff_error'],
                   'context_swap': swap, 'context_swap_char_p90': char_cut, 'context_swap_token_p90': token_cut,
                   'measurement_artifact': bool(matches), 'measurement_artifact_matches': matches,
                   'clustering_eligible': not reasons, 'clustering_exclusion_reasons': reasons}
        all_updated.append(updated)
        if key in old_cause_keys:
            causes.append(updated)
            exclusion_counts.update(reasons)
    eligible = [r for r in causes if r['clustering_eligible']]
    schema = extended_cause_schema()
    for name, rows in [('causes', causes), ('causes_unfiltered', all_updated), ('causes_for_clustering', eligible)]:
        base.write_parquet(folder / (name + '.parquet'), rows, schema)
    all_behaviours = pq.read_table(baseline / source / 'behaviours_unfiltered.parquet').to_pylist()
    behaviours, before_after, b_exclusions, seen = [], [], [], {}
    qualified_ids = [b['prompt_id'] for b in all_behaviours if b['verif_score'] is not None and b['verif_score'] >= 7]
    for b in all_behaviours:
        original = b['behaviour_text']
        cleaned, removed = behaviour_text(original) if original else (original, None)
        b['behaviour_text'] = cleaned
        if removed:
            before_after.append({'prompt_id': b['prompt_id'], 'source': source, 'verif_score': b['verif_score'],
                                 'before': original, 'after': cleaned, 'removed_prefix': removed})
        if b['verif_score'] is None or b['verif_score'] < 7 or not cleaned:
            b_exclusions.append({**b, 'reason': 'missing_text_or_score_below7'})
        elif cleaned in seen:
            b_exclusions.append({**b, 'reason': 'exact_duplicate_after_count_stripping', 'retained_prompt_id': seen[cleaned]})
        else:
            seen[cleaned] = b['prompt_id']
            behaviours.append(b)
    for name, rows in [('behaviours', behaviours), ('behaviours_unfiltered', all_behaviours)]:
        base.write_parquet(folder / (name + '.parquet'), rows, base.schemas()[0])
    for name, rows in [('behaviour_prefix_edits', before_after), ('behaviour_exclusions', b_exclusions),
                       ('clustering_exclusions', [r for r in causes if not r['clustering_eligible']])]:
        with (folder / (name + '.jsonl')).open('w') as file:
            for r in rows:
                base.emit(file, r)
    rng = random.Random(base.SEED + list(base.SOURCES).index(source))
    flagged = [r for r in causes if r['measurement_artifact']]
    sample = {'behaviours': rng.sample(behaviours, min(15, len(behaviours))),
              'causes': rng.sample(eligible, min(15, len(eligible))),
              'measurement_artifacts': rng.sample(flagged, min(10, len(flagged))),
              'behaviour_before_after': rng.sample([r for r in before_after if r['verif_score'] is not None and r['verif_score'] >= 7],
                                                  min(10, sum(r['verif_score'] is not None and r['verif_score'] >= 7 for r in before_after)))}
    report = {
        'source': source, 'size_distributions': sizes,
        'row_counts': {'investigations_unfiltered': len(all_behaviours), 'investigations_score_ge7': len(qualified_ids),
                       'behaviours_after_count_cleanup_and_dedup': len(behaviours),
                       'behaviours_count_prefix_removed_all': len(before_after),
                       'behaviours_count_prefix_removed_score_ge7': sum(r['verif_score'] is not None and r['verif_score'] >= 7 for r in before_after),
                       'experiments_all': len(feature_rows), 'size_reference_experiments': len(pool),
                       'missing_context_diff_all': sum(not r['context_diff_available'] for r in feature_rows),
                       'causes_in_parquet': len(causes), 'causes_context_swap': sum(r['context_swap'] for r in causes),
                       'causes_measurement_artifact': len(flagged),
                       'causes_both_swap_and_artifact': sum(r['context_swap'] and r['measurement_artifact'] for r in causes),
                       'causes_missing_context_diff': sum(not r['context_diff_available'] for r in causes),
                       'causes_after_context_swap_exclusion': sum(not r['context_swap'] and r['context_diff_available'] for r in causes),
                       'causes_eligible_for_clustering': len(eligible)},
        'artifact_term_counts': dict(Counter(t for r in flagged for t in r['measurement_artifact_matches'])),
        'edit_type_distributions': {'all_size_reference_experiments': dict(Counter(r['edit_type'] for r in pool)),
                                    'retained_cause_rows': dict(Counter(r['edit_type'] for r in causes)),
                                    'eligible_cause_rows': dict(Counter(r['edit_type'] for r in eligible))},
        'context_formats': dict(Counter(r['experiment_context_format'] for r in feature_rows)),
        'causes_per_investigation_before_new_flags': causes_per_investigation(causes, qualified_ids),
        'causes_per_investigation_after_new_flags': causes_per_investigation(eligible, qualified_ids),
        'new_exclusion_reason_counts': dict(exclusion_counts),
    }
    (folder / 'review_samples.json').write_text(json.dumps(sample, indent=2, ensure_ascii=False) + '\n')
    (folder / 'counts.json').write_text(json.dumps(report, indent=2) + '\n')
    print(source, json.dumps(report['row_counts']), flush=True)
    return report, sample


def write_report(out, reports, samples, baseline):
    lines = ['# CHIVE — Checkpoint 1, revision 2', '', '**STOPPED before embedding or clustering. No LLM calls or model inference.**', '',
             '## Methods and schema changes', '',
             '- Original schema mapping, verified rate calculations, effect threshold (absolute delta_pp >=50), within-source exact cause-text deduplication, and safe edit-text extraction are inherited from [revision 1](../checkpoint1/preprocessing_log.md). Revision 1 is preserved.',
             '- Cause rows retained at revision 1 stay in causes.parquet, including newly flagged rows. New flags are evaluated AFTER the existing exact cause-text deduplication. causes_for_clustering.parquet is the exact clustering_eligible subset; no embeddings have been computed.',
             '- Contexts come directly from investigations.jsonl: messages versus experiment_log[].messages, joined by source, prompt_id, and exp_num. For legacy experiments with no messages field, use the stored prompt as the full user message and prepend system_prompt when present, exactly as CHIVE’s documented experiment_to_messages reader specifies (chive/chive/pipeline/utils.py:676–699). No context is inferred from investigator prose. Missing/unsupported contexts are flagged and excluded from clustering, not guessed.',
             '- Each context is rendered in stored message order, with explicit role/end-message headers and verbatim content. Character counts are Unicode code points, including these structural markers; token counts use the local all-mpnet-base-v2 tokenizer with no truncation, padding or special tokens. Token counts are NOT Qwen token counts. No model weights or inference are used.',
             '- Exact LCS diffs are computed independently for characters and token IDs. Added = edited length minus LCS length; deleted = baseline length minus LCS length; changed = added + deleted. Replacements contribute both deleted and added units. Sizes are not merely differences in total context length.',
             '- context_swap = characters changed > source p90 OR tokens changed > source p90 (strictly greater). Percentiles use linear interpolation across ALL context-comparable experiments with verification >=7 in that source, before the effect threshold and cause-text deduplication. This is a size proxy, not a semantic claim that the context was replaced.',
             '- edit_type is now based only on the character diff: delete-only → remove, add-only → add, both → modify; identical contexts → no_change; unavailable context → unknown. Tokenizer normalization cannot hide character-only edits. no_change is reported but is not an additional exclusion requested by this revision.',
             '- measurement_artifact checks the ORIGINAL summary, case-insensitively, for N/A, not applicable, not reach, as expected, trivially, cannot be graded. Whitespace is normalized; phrases/markers have word boundaries (so region/add and medication/anxiety do not match N/A), and not reach includes reached/reaching. Quoted occurrences count. This is a requested lexical flag, not proof that the measurement is invalid.',
             '- Behaviour cleanup removes leading numeric count/percentage phrases and all-N-completions phrases only. It preserves the remaining description verbatim apart from existing light Markdown/whitespace cleanup; it does not delete the observed behaviour sentence. Exact behaviour-text deduplication is re-run after this cleanup.',
             '- New cause columns: verif_score; baseline/experiment context chars/tokens; chars_deleted/added/changed; tokens_deleted/added/changed; context_diff_available/error; context_swap and its source char/token p90 cutoffs; measurement_artifact and matched terms; clustering_eligible and clustering_exclusion_reasons. edit_type replaces the old verb-based value.',
             '- Raw/unfiltered copies and the diff measurements for every experiment remain available per source. Character/token counts and hashes are in context_diffs.parquet. Previous rate disagreement flags remain in revision 1; their measurements are unchanged.', '',
             '## Updated row counts', '',
             '| Source | Behaviours | Causes kept in Parquet | Context swap | Artifact | Both flags | Missing diff | Clustering-eligible causes |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s, r in reports.items():
        c = r['row_counts']
        lines.append(f'| {s} | {c["behaviours_after_count_cleanup_and_dedup"]:,} | {c["causes_in_parquet"]:,} | {c["causes_context_swap"]:,} | {c["causes_measurement_artifact"]:,} | {c["causes_both_swap_and_artifact"]:,} | {c["causes_missing_context_diff"]:,} | {c["causes_eligible_for_clustering"]:,} |')
    lines += ['', '## Edit-size distributions and source-specific p90 cutoffs', '',
              'Population: all verification-qualified experiments with available contexts in each source. Units are deleted + added. The two p90 columns independently define the OR exclusion rule.', '',
              '| Source | Unit | N | Mean | Min | P25 | Median | P75 | P90 cutoff | P95 | P99 | Max |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s, r in reports.items():
        for field in ('chars_changed', 'tokens_changed'):
            d = r['size_distributions'][field]
            lines.append(f'| {s} | {field} | {d["n"]:,} | ' + ' | '.join(f'{d[k]:,.2f}' for k in ['mean','min','p25','p50','p75','p90','p95','p99','max']) + ' |')
    lines += ['', '## Diff-derived edit types', '', '| Source | Population | Remove | Add | Modify | No change | Unknown |', '|---|---|---:|---:|---:|---:|---:|']
    for s, r in reports.items():
        for pop, d in r['edit_type_distributions'].items():
            lines.append(f'| {s} | {pop} | ' + ' | '.join(f'{d.get(k,0):,}' for k in ['remove','add','modify','no_change','unknown']) + ' |')
    lines += ['', '## Causes per investigation', '',
              'Main denominator: every verification-qualified investigation in the source, INCLUDING those with zero eligible causes. Statistics are after existing exact cause-text deduplication; the conditional positive-only statistics are also saved in counts.json.', '',
              '| Source | Population | Investigations | Mean | Median | Max | Zero-cause investigations |', '|---|---|---:|---:|---:|---:|---:|']
    for s, r in reports.items():
        for key, label in [('causes_per_investigation_before_new_flags','Before new exclusions'), ('causes_per_investigation_after_new_flags','Eligible after exclusions')]:
            d = r[key]; a = d['all_verification_qualified_investigations']
            lines.append(f'| {s} | {label} | {a["n_investigations"]:,} | {a["mean"]:.3f} | {a["median"]:g} | {a["max"]} | {d["zero_cause_investigations"]:,} |')
    lines += ['', '## Examples and files', '',
              '[Open review.html](review.html) for 10 flagged artifact examples and 10 behaviour before/after examples per source where available, plus 15 random rows of each eligible facet per source. Random seed 20261004 + source index; no similarity-based selection.', '',
              'Per source: behaviours.parquet, behaviours_unfiltered.parquet, causes.parquet (flagged rows retained), causes_unfiltered.parquet, causes_for_clustering.parquet, context_diffs.parquet, counts.json, review_samples.json, behaviour_prefix_edits.jsonl, behaviour_exclusions.jsonl, clustering_exclusions.jsonl.', '',
              'Reproduce: `clustering/.venv/bin/python clustering/scripts/preprocessing.py`. Reuse completed exact context diffs with `--reuse-context-diffs`. Original checkpoint at ' + str(baseline) + '.', '',
              '**Awaiting Checkpoint 1 approval. Near-duplicate embedding checks, embeddings, and clustering have not run.**', '']
    (out / 'preprocessing_log.md').write_text('\n'.join(lines))
    esc = lambda x: html.escape(str(x))
    sections = []
    for source, sample in samples.items():
        c = reports[source]['row_counts']
        section = [f'<section id="{source}"><h2>{source}</h2><p>{c["behaviours_after_count_cleanup_and_dedup"]:,} behaviours; {c["causes_in_parquet"]:,} cause rows retained; <strong>{c["causes_eligible_for_clustering"]:,} eligible causes</strong>.</p>']
        section += ['<p>' + ' · '.join(f'<a href="{source}/{f}.parquet">{f}</a>' for f in ['behaviours','causes','causes_for_clustering','context_diffs','causes_unfiltered']) + '</p>']
        section += ['<h3>Measurement artifact examples: original summaries</h3><table><tr><th>Investigation / experiment</th><th>Matched phrases</th><th>Original summary</th></tr>']
        for r in sample['measurement_artifacts']:
            section += [f'<tr><td>{esc(r["prompt_id"])}<br>Experiment {r["experiment_number"]}<br>Δ {r["delta_pp"]:+.1f} pp</td><td>{esc(", ".join(r["measurement_artifact_matches"]))}</td><td>{esc(r["cause_text_original"])}</td></tr>']
        if not sample['measurement_artifacts']:
            section += ['<tr><td colspan="3">No flagged cause rows in this source.</td></tr>']
        section += ['</table><h3>Leading-count cleanup: before / after</h3><table><tr><th>Investigation</th><th>Before</th><th>After</th></tr>']
        for r in sample['behaviour_before_after']:
            section += [f'<tr><td>{esc(r["prompt_id"])}</td><td>{esc(r["before"])}</td><td>{esc(r["after"])}</td></tr>']
        section += ['</table><details><summary>15 random eligible behaviour rows</summary><table><tr><th>Investigation</th><th>Score</th><th>Behaviour</th></tr>']
        for r in sample['behaviours']:
            section += [f'<tr><td>{esc(r["prompt_id"])}</td><td>{r["verif_score"]}</td><td>{esc(r["behaviour_text"])}</td></tr>']
        section += ['</table></details><details><summary>15 random eligible cause rows</summary><table><tr><th>ID / diff / measurement</th><th>Original</th><th>Stripped edit text</th></tr>']
        for r in sample['causes']:
            section += [f'<tr><td>{esc(r["prompt_id"])}<br>Experiment {r["experiment_number"]}<br>{r["edit_type"]}; {r["chars_changed"]:,} chars / {r["tokens_changed"]:,} tokens<br>Δ {r["delta_pp"]:+.1f} pp</td><td>{esc(r["cause_text_original"])}</td><td>{esc(r["cause_text"])}</td></tr>']
        section += ['</table></details></section>']
        sections.extend(section)
    # Render statistical tables as HTML, and also retain the complete methods text.
    stats_tables = []
    i = 0
    title = ''
    while i < len(lines):
        if lines[i].startswith('## '):
            title = lines[i][3:]
        if lines[i].startswith('|') and i + 1 < len(lines) and lines[i + 1].startswith('|---'):
            headers = [c.strip() for c in lines[i].strip('|').split('|')]
            stats_tables.append('<h3>' + esc(title) + '</h3><div style="overflow-x:auto"><table style="min-width:1100px"><tr>' + ''.join('<th>' + esc(c) + '</th>' for c in headers) + '</tr>')
            i += 2
            while i < len(lines) and lines[i].startswith('|'):
                cells = [c.strip() for c in lines[i].strip('|').split('|')]
                stats_tables.append('<tr>' + ''.join('<td>' + esc(c) + '</td>' for c in cells) + '</tr>')
                i += 1
            stats_tables.append('</table></div>')
            continue
        i += 1
    # Embed the full report verbatim so the methods are also available offline.
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CHIVE checkpoint 1 — revision 2</title><style>body{margin:0;background:#f3f6f8;color:#172d3c;font:15px/1.6 system-ui}main{max-width:1500px;margin:auto;padding:24px}section{background:white;padding:20px;border-radius:10px;margin:24px 0}table{width:100%;border-collapse:collapse;table-layout:fixed}td,th{border:1px solid #d4dfe5;padding:12px;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#e4eeef}th:first-child{width:24%}pre{white-space:pre;overflow:auto;font:13px/1.6 monospace}a{color:#086d78}summary{cursor:pointer;font-weight:600;padding:14px}.notice{background:#fff1d8;padding:16px;border-left:4px solid #a87520}@media(max-width:700px){main,section{padding:8px}td,th{padding:6px;font-size:12px}}</style><main><h1>CHIVE · Checkpoint 1 · Revision 2</h1><p class="notice">Stopped before embeddings. Flagged causes remain in Parquet with clustering_eligible=false. No LLM calls or model inference. Token counts use a local tokenizer only.</p><p><a href="preprocessing_log.md">Full methods, schema, distributions and count log</a></p><nav>' + ' · '.join(f'<a href="#{s}">{s}</a>' for s in base.SOURCES) + '</nav><details><summary>All statistics and methods (complete report)</summary><pre>' + esc('\n'.join(lines)) + '</pre></details>' + ''.join(sections) + '</main></html>'
    page = page.replace('<nav>', ''.join(stats_tables) + '<nav>', 1)
    (out / 'review.html').write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--base-dir', type=Path, default=ROOT / 'data/checkpoint1')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/checkpoint1_v2')
    parser.add_argument('--reuse-context-diffs', action='store_true')
    args = parser.parse_args()
    if args.base_dir.resolve() == args.output_dir.resolve():
        raise ValueError('Revision 1 and revision 2 output directories must differ.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.reuse_context_diffs:
        extract_diffs(args.data_root, args.output_dir, ROOT / 'models/all-mpnet-base-v2/tokenizer.json')
    else:
        provenance = json.loads((args.output_dir / 'diff_provenance.json').read_text())
        tokenizer_path = ROOT / 'models/all-mpnet-base-v2/tokenizer.json'
        if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != provenance['tokenizer_sha256']:
            raise ValueError('Local tokenizer changed; recompute context diffs.')
        for source in base.SOURCES:
            if pq.read_schema(args.output_dir / source / 'context_diffs.parquet') != feature_schema():
                raise ValueError('Context diff schema changed; recompute context diffs.')
    reports, samples = {}, {}
    for source in base.SOURCES:
        reports[source], samples[source] = process_source(source, args.base_dir, args.output_dir)
    (args.output_dir / 'counts.json').write_text(json.dumps(reports, indent=2) + '\n')
    write_report(args.output_dir, reports, samples, args.base_dir)
    used_files = []
    for source in base.SOURCES:
        for name in ['causes.parquet', 'causes_unfiltered.parquet', 'behaviours_unfiltered.parquet']:
            path = args.base_dir / source / name
            used_files.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (args.output_dir / 'revision_provenance.json').write_text(json.dumps({
        'revision': 2, 'base_checkpoint_inputs': used_files,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'context_diff_provenance': 'diff_provenance.json', 'random_seed_base': base.SEED,
        'model_inference_calls': 0, 'embeddings_computed': False, 'clusters_computed': False,
    }, indent=2) + '\n')
    print(f'CHECKPOINT 1: {args.output_dir / "review.html"}. STOP.', flush=True)


if __name__ == '__main__':
    main()
