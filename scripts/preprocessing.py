"""Observation-only CHIVE preprocessing. STOP at checkpoint 1; no ML/LLM calls.

Run: clustering/.venv/bin/python clustering/scripts/preprocessing.py (current revision 2).
Use --base-only to reproduce revision 1. Context controls live in
preprocessing_context.py and use a local tokenizer, never model inference.
The prior interpretation pipeline is archived in preprocessing_legacy.py and is
never imported or executed here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import html
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    'petri': 'petri/qwen3_8b_petri_eval',
    'wildchat_eval': 'wildchat/qwen3_8b_wildchat_eval',
    'wildchat_thinking': 'wildchat/qwen3_8b_wildchat_thinking_eval',
    'wildchat_train': 'wildchat/qwen3_8b_wildchat_train',
}
SEED = 20261004


def clean(text):
    """Remove Markdown presentation, preserve words/case/negation/identifiers."""
    text = re.sub(r'(?m)^\s*```[^\n]*\n|^\s*```\s*$', '', text)
    text = re.sub(r'!?\[([^\]\n]+)\]\(([^)\n]+)\)', r'\1 (\2)', text)
    text = re.sub(r'(?m)^\s{0,3}(?:#{1,6}\s+|>\s*|[-*+]\s+)', '', text)
    text = re.sub(r'(\*\*|__)(?=\S)(.+?)(?<=\S)\1', r'\2', text, flags=re.S)
    text = re.sub(r'(?<!\w)(\*|_)(?=\S)(.+?)(?<=\S)\1(?!\w)', r'\2', text)
    text = re.sub(r'`+([^`]+)`+', r'\1', text)
    return re.sub(r'\s+', ' ', text).strip()


# Quoted/inline-code spans are protected: numeric literals may be the actual edit.
QUOTES = re.compile(r'`+[^`]*`+|"(?:\\.|[^"\\])*"|(?<!\w)\'(?:[^\']|\'(?=\w))*?\'(?!\w)|“[^”]*”|‘[^’]*’')


def mask_quotes(text):
    return QUOTES.sub(lambda m: 'Q' + ' ' * (len(m.group()) - 1), text)


EDIT_WORD = r'(?:remov\w*|delet\w*|add(?:s|ed|ing)?|insert\w*|chang\w*|replac\w*|rewr\w*|reword\w*|rephras\w*|renam\w*|reorder\w*|modif\w*|edit\w*|swap\w*|substitut\w*|omit\w*|strip\w*|append\w*|prepend\w*|shorten\w*|simplif\w*|convert\w*|alter\w*|introduc\w*|specif\w*|clarif\w*|explicitly|kept|keep\w*|retain\w*|preserv\w*|ask(?:s|ed|ing)?|instruct\w*|told|used|using|sent|said|set|gave|give|made|mak\w*|refram\w*|reset|switch\w*|isolat\w*|reduc\w*|increas\w*|extend\w*|expand\w*|repeat\w*|rais\w*|directly|test(?:ed|ing)?|provid\w*|present\w*|translat\w*|correct\w*|restructur\w*|condens\w*)'
EDIT_WORD = EDIT_WORD.replace('specif\\w*', 'specif(?:y|ies|ied|ying)')
EDIT_WORD = EDIT_WORD.replace('instruct\\w*', 'instruct(?:s|ed|ing)?')
EDIT_WORD = EDIT_WORD.replace('|explicitly', '').replace('|directly', '')
EDIT = re.compile(r'\b' + EDIT_WORD + r'\b', re.I)
CONTEXT = re.compile(r'^(?:(?:a |the )?(?:standalone|single[- ]turn|minimal|fresh|original|baseline|exact rerun|no (?:prior |conversation |system )|only |everything else|all other|otherwise unchanged)|(?:all|both|full|same|prior|earlier|remaining|rest of|without)\b.{0,65}\b(?:kept|unchanged|history|context|removed|preserved))', re.I)
RESULT = re.compile(r'\b\d+\s*/\s*\d+\b|%|\b(?:rates?|percentage(?:s)?|pp|dropped|increased|decreased|remained|rose|fell|plummeted|persisted|disappeared|vanished|eliminated|suppressed|triggered|induced|yielded|resulted|outcomes?|results?|confirm\w*|suggest\w*|demonstrat\w*|prov(?:e[sd]?|ing)|indicat\w*|show(?:s|ed|ing)|impli\w*)\b', re.I)
OBSERVATION = re.compile(r'^(?:the )?(?:(?:model|grader|assistant|responses?|completions?|outputs?|behavior|behaviour|it|this|that|both updates|both knowledge updates|all \d+ responses|zero|got|result|rate|in all|across all|even with|even without|qualitatively|quantitatively)\b)', re.I)
INTERPRET = re.compile(r'\b(?:suggesting|confirming|showing|indicating|demonstrating|proving|implying|establishing)\b', re.I)


def strip_outcomes(original):
    """Conservative, deterministic extraction; never paraphrases or uses an LLM.

    Keep edit-description sentences/clauses and explicit context-preservation
    clauses. Drop result/interpretation clauses, protecting quoted edit payloads.
    Ambiguous sentences are removed and logged, not rewritten into hypotheses.
    """
    text = re.sub(r'\s+', ' ', original).strip()
    masked = mask_quotes(text)
    sentence_ends = [m.end() for m in re.finditer(r'[.!?](?:\s+|$)', masked)
                     if not re.search(r'\b(?:vs|etc|e\.g|i\.e|Dr|Mr|Ms|Mrs|Prof|St)\.\s*$', masked[:m.end()])]
    # Sentence punctuation may be inside the closing quotation mark.
    for quote in QUOTES.finditer(text):
        if re.search(r'[.!?][\'"”’]$', quote.group()) and re.match(r'\s+[A-Z0-9]', text[quote.end():]):
            sentence_ends.append(quote.end())
    bounds = [0] + sorted(set(sentence_ends))
    if bounds[-1] != len(text):
        bounds.append(len(text))
    kept, removed = [], []
    for start, end in zip(bounds, bounds[1:]):
        sentence = text[start:end].strip()
        if not sentence:
            continue
        view = mask_quotes(sentence)
        # A result sentence can contain edit-like words in its causal explanation
        # ("... confirming the instruction matters"). Those words do not make
        # it an edit description. Detect the result before searching for edits.
        result_lead = re.search(r'\b(?:confirming|suggesting|proving|indicating|demonstrating|showing that|dropped|decreased|disappeared|vanished|plummeted|eliminated)\b', view, re.I)
        if result_lead and not EDIT.search(view[:result_lead.start()]) and not CONTEXT.search(view[:result_lead.start()]):
            removed.append({'text': sentence, 'reason': 'qualitative_result_or_inference'})
            continue
        # Drop parenthetical numeric outcomes, not quoted numeric edit payloads.
        spans = [m.span() for m in re.finditer(r'\([^()]*\)', view)
                 if re.match(r'\(\s*(?:\d+\s*/\s*\d+|\d+(?:\.\d+)?\s*%|(?:rate|result|outcome)\s*[:=])', m.group(), re.I)]
        for a, b in reversed(spans):
            removed.append({'text': sentence[a:b], 'reason': 'outcome_parenthesis'})
            sentence = sentence[:a] + sentence[b:]
        view = mask_quotes(sentence)
        # Cut off result or inference clauses after an edit; preserve the exact prefix.
        cut = None
        for m in re.finditer(r',\s*|;\s*|\s+[—–]\s+|:\s*|\s+(?:and|but|which|so|yielding|resulting in|leading to)\s+', view, re.I):
            # Commas inside an edit's parenthesis/list do not start outcome clauses.
            prefix_view = view[:m.start()]
            if prefix_view.count('(') > prefix_view.count(')') or prefix_view.count('[') > prefix_view.count(']'):
                continue
            suffix = view[m.end():]
            prefix = view[:m.start()]
            if (EDIT.search(prefix) or CONTEXT.search(prefix)) and (
                INTERPRET.match(suffix)
                or re.match(r'testing whether\b', suffix, re.I)
                or (RESULT.search(suffix) and not EDIT.match(suffix)
                    and not re.search(r'\b(?:remained|remains|kept|held)\s+(?:the same|identical|unchanged|constant)\b', suffix, re.I))
                or OBSERVATION.match(suffix)
            ):
                cut = m.start()
                break
        if cut is not None:
            removed.append({'text': sentence[cut:], 'reason': 'outcome_or_interpretation_tail'})
            sentence = sentence[:cut].rstrip(' ,;:—–')
            view = mask_quotes(sentence)
        # Leading edit verbs like "Increased timeout" are edits, not outcomes.
        inspected = re.sub(r'^\s*(?:Increased|Reduced|Removed|Changed)\b', 'EDIT', view, flags=re.I)
        # Generic words like "result", "rate", "outcome", or a name "Rose"
        # may describe the prompt itself; they are not sufficient to delete an edit.
        result_match = re.search(r'\b\d+\s*/\s*\d+\b|\d\s*%|\b(?:rate|frequency)\s+(?:was|is|dropped|rose|fell|increased|decreased|remained)\b', inspected, re.I)
        if (result_match and (EDIT.match(view) or re.match(r'^The edit\b', view, re.I))
                and not re.search(r'\b(?:dropped|decreased|rose|fell|plummeted|persisted|yielded)\b', view, re.I)
                and re.search(r'\b(?:from|to|threshold|percentages|percentage|fraction)\b', inspected[:result_match.start()], re.I)):
            # An unquoted numeric literal explicitly specified as an edit is kept.
            result_match = None
        has_edit = bool(EDIT.search(view) or CONTEXT.search(view))
        # Summaries also use verbless edit descriptions: "English question with
        # a hint", "System prompt + last turn", etc. Keep a first descriptive
        # sentence if it contains neither outcome nor causal-inference cues.
        if start == 0 and not RESULT.search(view) and not re.search(r'\b(?:cause|driver|mechanism|explains|because|depends on)\b', view, re.I):
            has_edit = True
        observation_start = bool(OBSERVATION.match(view))
        # A passive description of editing an assistant/model message is valid.
        if observation_start and re.search(r'\b(?:was|were|is)\s+(?:changed|replaced|removed|rewritten|modified|added|deleted|rephrased)\b', view, re.I):
            observation_start = False
        if not has_edit or result_match or observation_start:
            removed.append({'text': sentence, 'reason': 'result_or_interpretation' if (result_match or observation_start) else 'unrecognized_description'})
        else:
            kept.append(sentence)
    return clean(' '.join(kept)), removed


def edit_type(original):
    beginning = clean(original)
    if re.match(r'^(?:Removed|Deleted)\b', beginning, re.I):
        return 'remove'
    if re.match(r'^(?:Added|Inserted)\b', beginning, re.I):
        return 'add'
    if re.match(r'^(?:Changed|Replaced|Rewrote)\b', beginning, re.I) or (
        re.match(r'^Kept\b', beginning, re.I) and re.search(r'\bbut\b', beginning, re.I)
    ):
        return 'modify'
    return 'unknown'


def parse_rate(value):
    """Parse the leading measured x/n only; leave trailing reliability prose flagged."""
    if not isinstance(value, str):
        return None, 'missing_or_nonstring_rate'
    match = re.match(r'^\s*(\d+)\s*/\s*(\d+)(.*)$', value, re.S)
    if not match:
        return None, 'unparseable_rate'
    k, n = int(match[1]), int(match[2])
    if n <= 0 or not 0 <= k <= n:
        return None, 'invalid_rate_counts'
    return Fraction(k, n), 'annotated_rate' if match[3].strip() else None


def label_rate(labels):
    if labels is None or labels == []:
        return None, 'missing_labels'
    if not isinstance(labels, list) or any(type(v) is not bool for v in labels):
        return None, 'invalid_labels'
    return Fraction(sum(labels), len(labels)), None


def emit(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def extract_inputs(data_root, out):
    """Save only allowed fields; never read explanatory sections into this cache."""
    manifest = json.loads((data_root / 'download_manifest.json').read_text())
    with (out / 'input_observations.jsonl').open('w') as dest:
        for source, relative in SOURCES.items():
            folder = data_root / relative
            scores = {}
            for line in (folder / 'verification.jsonl').open():
                verification = json.loads(line)
                pid = verification['prompt_id']
                if pid in scores:
                    raise ValueError(f'Duplicate verification ID: {source}/{pid}')
                score = verification.get('score')
                if score is not None and (type(score) not in (int, float) or not 1 <= score <= 10):
                    raise ValueError(f'Unexpected score: {source}/{pid}: {score}')
                scores[pid] = score
            seen = set()
            for line_num, line in enumerate((folder / 'investigations.jsonl').open(), 1):
                r = json.loads(line)
                pid = r['prompt_id']
                if pid in seen:
                    raise ValueError(f'Duplicate investigation ID: {source}/{pid}')
                seen.add(pid)
                sf = r.get('structured_findings') or {}
                small = {k: r.get(k) for k in ['prompt_id', 'behavior_summary', 'baseline_grader_verdicts', 'baseline_grader_yes']}
                small.update(source=source, source_file=str(folder / 'investigations.jsonl'), source_line=line_num,
                             verif_score=scores.get(pid), baseline_behavior=sf.get('baseline_behavior'),
                             experiment_summaries=sf.get('experiment_summaries'),
                             experiment_log=[{k: e.get(k) for k in ['exp_num', 'grader_verdicts', 'grader_yes', 'n']}
                                             for e in r.get('experiment_log', [])])
                emit(dest, small)
            print(f'Extracted permitted fields: {source} ({len(seen):,} investigations)', flush=True)
    (out / 'input_provenance.json').write_text(json.dumps({
        'repository': manifest['repository'], 'revision': manifest['revision'],
        'source_mapping': SOURCES, 'files': [f for f in manifest['files']
            if f['local_path'].endswith(('/investigations.jsonl', '/verification.jsonl'))],
        'cache_sha256': hashlib.sha256((out / 'input_observations.jsonl').read_bytes()).hexdigest(),
    }, indent=2) + '\n')


def schemas():
    import pyarrow as pa
    behaviours = pa.schema([('prompt_id', pa.string()), ('source', pa.string()),
                            ('verif_score', pa.float64()), ('behaviour_text', pa.string())])
    causes = pa.schema([('prompt_id', pa.string()), ('source', pa.string()), ('experiment_number', pa.int64()),
                        ('is_key', pa.bool_()), ('edit_type', pa.string()), ('baseline_rate', pa.float64()),
                        ('counterfactual_rate', pa.float64()), ('delta_pp', pa.float64()),
                        ('cause_text', pa.string()), ('cause_text_original', pa.string())])
    return behaviours, causes


def write_parquet(path, rows, schema):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression='zstd')


def process_source(source, records, directory):
    directory.mkdir(exist_ok=True)
    counts = Counter()
    behaviours, causes, all_behaviours, all_causes = [], [], [], []
    seen_b, seen_c, originals, score_lookup = {}, {}, {}, {}
    flag_counts = Counter()
    with (directory / 'flags.jsonl').open('w') as flags, (directory / 'exclusions.jsonl').open('w') as exclusions, (directory / 'text_edits.jsonl').open('w') as text_edits:
        def flag(common, reason, **extra):
            flag_counts[reason] += 1
            emit(flags, {**common, 'reason': reason, **extra})

        def exclude(common, facet, reason, **extra):
            counts[f'excluded_{facet}_{reason}'] += 1
            emit(exclusions, {**common, 'facet': facet, 'reason': reason, **extra})

        for r in records:
            pid, score = r['prompt_id'], r['verif_score']
            common = {'source': source, 'prompt_id': pid}
            counts['investigations_input'] += 1
            score_lookup[pid] = score
            passed = score is not None and score >= 7
            counts['investigations_score_ge7'] += int(passed)
            counts['investigations_missing_score'] += int(score is None)
            counts['investigations_score_below7'] += int(score is not None and score < 7)
            original = r.get('behavior_summary')
            field = 'behavior_summary'
            if not isinstance(original, str) or not original.strip():
                original, field = r.get('baseline_behavior'), 'structured_findings.baseline_behavior'
            text = clean(original) if isinstance(original, str) else None
            b = {**common, 'verif_score': score, 'behaviour_text': text}
            all_behaviours.append(b)
            if not passed:
                exclude(common, 'behaviour', 'missing_score' if score is None else 'score_below7')
            elif not text:
                exclude(common, 'behaviour', 'missing_text')
            else:
                counts['behaviours_after_text_cleanup'] += 1
                if text in seen_b:
                    exclude(common, 'behaviour', 'exact_duplicate', retained_prompt_id=seen_b[text])
                else:
                    seen_b[text] = pid
                    behaviours.append(b)
                    originals[pid] = {'original_text': original, 'original_field': field}
                    counts['behaviours_after_dedup'] += 1
            summaries = r.get('experiment_summaries') or []
            summary_map = {}
            for s in summaries:
                required = ['experiment_number', 'summary', 'baseline_rate', 'counterfactual_rate', 'is_key']
                if any(k not in s for k in required) or type(s['experiment_number']) is not int or type(s['is_key']) is not bool:
                    raise ValueError(f'Unexpected summary schema: {source}/{pid}')
                if s['experiment_number'] in summary_map:
                    raise ValueError(f'Duplicate experiment summary: {source}/{pid}')
                summary_map[s['experiment_number']] = s
            base, base_error = label_rate(r.get('baseline_grader_verdicts'))
            if base is not None and sum(r['baseline_grader_verdicts']) != r.get('baseline_grader_yes'):
                flag(common, 'baseline_reported_yes_disagrees_with_labels')
            log_ids = set()
            for e in r.get('experiment_log', []):
                num = e['exp_num']
                if type(num) is not int or num in log_ids:
                    raise ValueError(f'Invalid/duplicate logged experiment: {source}/{pid}/{num}')
                log_ids.add(num)
                key = {**common, 'experiment_number': num}
                counts['experiments_input'] += 1
                counts['experiments_after_score_filter'] += int(passed)
                s = summary_map.get(num)
                counter, counter_error = label_rate(e.get('grader_verdicts'))
                if counter is not None:
                    if sum(e['grader_verdicts']) != e.get('grader_yes'):
                        flag(key, 'counterfactual_reported_yes_disagrees_with_labels')
                    if len(e['grader_verdicts']) != e.get('n'):
                        flag(key, 'experiment_n_disagrees_with_label_count')
                pb, pe = (None, None)
                chosen = []
                discrepancy = False
                for kind, measured, error, raw in [
                    ('baseline', base, base_error, s.get('baseline_rate') if s else None),
                    ('counterfactual', counter, counter_error, s.get('counterfactual_rate') if s else None),
                ]:
                    parsed, parse_note = parse_rate(raw)
                    if parse_note and s:
                        flag(key, f'{kind}_{parse_note}', reported=raw)
                    if measured is not None and parsed is not None and measured != parsed:
                        discrepancy = True
                        flag(key, f'{kind}_parsed_vs_labels_disagree', reported=raw,
                             parsed_rate=float(parsed), recomputed_rate=float(measured))
                    # Invalid labels are not treated as absent; exclude instead of guessing.
                    value = measured if measured is not None else parsed if error == 'missing_labels' else None
                    if measured is None and value is not None:
                        flag(key, f'{kind}_rate_string_fallback', reported=raw)
                    chosen.append(value)
                if discrepancy:
                    counts['experiments_with_rate_disagreement'] += 1
                b_rate, c_rate = chosen
                delta = (c_rate - b_rate) * 100 if b_rate is not None and c_rate is not None else None
                raw_text = s.get('summary') if s else None
                valid_description = isinstance(raw_text, str) and bool(raw_text.strip())
                stripped, removals = strip_outcomes(raw_text) if valid_description else (None, [])
                row = {**key, 'is_key': s['is_key'] if s else None,
                       'edit_type': edit_type(raw_text) if valid_description else 'unknown',
                       'baseline_rate': float(b_rate) if b_rate is not None else None,
                       'counterfactual_rate': float(c_rate) if c_rate is not None else None,
                       'delta_pp': float(delta) if delta is not None else None,
                       'cause_text': stripped, 'cause_text_original': raw_text}
                all_causes.append(row)
                if removals:
                    emit(text_edits, {**key, 'removed_spans': removals, 'empty_after_stripping': not bool(stripped)})
                if not valid_description:
                    flag(key, 'missing_edit_summary')
                if delta is None:
                    flag(key, 'unusable_rates')
                if not passed:
                    exclude(key, 'cause', 'missing_score' if score is None else 'score_below7')
                elif not valid_description or delta is None:
                    exclude(key, 'cause', 'incomplete', missing_summary=not valid_description, missing_rates=delta is None)
                else:
                    counts['causes_after_completeness_filter'] += 1
                    if abs(delta) < 50:
                        exclude(key, 'cause', 'effect_below50pp')
                    else:
                        counts['causes_after_effect_filter'] += 1
                        if not stripped:
                            exclude(key, 'cause', 'no_safe_edit_text')
                            flag(key, 'no_safe_edit_text', original=raw_text)
                        else:
                            counts['causes_after_text_cleanup'] += 1
                            if stripped in seen_c:
                                exclude(key, 'cause', 'exact_duplicate', retained=seen_c[stripped])
                            else:
                                seen_c[stripped] = key
                                causes.append(row)
                                counts['causes_after_dedup'] += 1
            if set(summary_map) - log_ids:
                raise ValueError(f'Summaries without measured experiments: {source}/{pid}')
    b_schema, c_schema = schemas()
    for name, rows, schema in [('behaviours', behaviours, b_schema), ('causes', causes, c_schema),
                               ('behaviours_unfiltered', all_behaviours, b_schema), ('causes_unfiltered', all_causes, c_schema)]:
        write_parquet(directory / f'{name}.parquet', rows, schema)
    rng = random.Random(SEED + list(SOURCES).index(source))
    review_b = [{**b, **originals[b['prompt_id']]} for b in rng.sample(behaviours, min(15, len(behaviours)))]
    review_c = [{**c, 'verif_score': score_lookup[c['prompt_id']]} for c in rng.sample(causes, min(15, len(causes)))]
    sample_keys = {(c['prompt_id'], c['experiment_number']) for c in review_c}
    review_flags = {k: [] for k in sample_keys}
    with (directory / 'flags.jsonl').open() as sample_flag_file:
        for line in sample_flag_file:
            f = json.loads(line)
            key = (f['prompt_id'], f.get('experiment_number'))
            if key in review_flags:
                review_flags[key].append(f['reason'])
    for c in review_c:
        c['review_flags'] = review_flags[(c['prompt_id'], c['experiment_number'])]
    samples = {'source': source, 'seed': SEED + list(SOURCES).index(source), 'behaviours': review_b, 'causes': review_c}
    (directory / 'review_samples.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2) + '\n')
    for k in ['investigations_input', 'investigations_score_ge7', 'behaviours_after_text_cleanup', 'behaviours_after_dedup',
              'experiments_input', 'experiments_after_score_filter', 'causes_after_completeness_filter',
              'causes_after_effect_filter', 'causes_after_text_cleanup', 'causes_after_dedup']:
        counts[k] += 0
    result = {'counts': dict(counts), 'flags': dict(flag_counts),
              'edit_types_final': dict(Counter(c['edit_type'] for c in causes)),
              'samples': samples}
    (directory / 'counts.json').write_text(json.dumps({k: v for k, v in result.items() if k != 'samples'}, indent=2) + '\n')
    return result


def render_review(out, results):
    esc = lambda v: html.escape(str(v))
    parts = []
    for source, result in results.items():
        samples = result['samples']
        counts = result['counts']
        parts.append(f'<section id="{source}"><h2>{esc(source)}</h2><p>{counts["behaviours_after_dedup"]:,} retained behaviour rows; {counts["causes_after_dedup"]:,} retained cause rows.</p><p>Parquet: <a href="{source}/behaviours.parquet">behaviours</a> · <a href="{source}/causes.parquet">causes</a> · <a href="{source}/behaviours_unfiltered.parquet">unfiltered behaviours</a> · <a href="{source}/causes_unfiltered.parquet">unfiltered causes</a> · <a href="{source}/flags.jsonl">flags</a> · <a href="{source}/exclusions.jsonl">exclusions</a></p><p>Seed {samples["seed"]}. Random samples drawn after filters and within-source exact deduplication.</p><h3>15 behaviour rows</h3><table><tr><th>ID / score</th><th>Original text</th><th>Cleaned behaviour text</th></tr>')
        for b in samples['behaviours']:
            parts.append(f'<tr><td>{esc(b["prompt_id"])}<br>Score: {b["verif_score"]}<br>{esc(b["original_field"])}</td><td>{esc(b["original_text"])}</td><td>{esc(b["behaviour_text"])}</td></tr>')
        parts.append('</table><h3>15 cause rows: measured edits, not causal interpretations</h3><table><tr><th>ID / measurement</th><th>Original summary</th><th>Stripped edit description</th></tr>')
        for c in samples['causes']:
            warnings = '<br>Flags: ' + esc(', '.join(c['review_flags'])) if c['review_flags'] else ''
            parts.append(f'<tr><td>{esc(c["prompt_id"])}<br>Experiment {c["experiment_number"]}<br>Score {c["verif_score"]}; is_key={c["is_key"]}<br>Type: {esc(c["edit_type"])}<br>{c["baseline_rate"]:.1%} → {c["counterfactual_rate"]:.1%}<br>Δ {c["delta_pp"]:+.1f} pp{warnings}</td><td>{esc(c["cause_text_original"])}</td><td>{esc(c["cause_text"])}</td></tr>')
        parts.append('</table></section>')
    nav = ' · '.join(f'<a href="#{s}">{s}</a>' for s in SOURCES)
    doc = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>CHIVE checkpoint 1</title><style>body{font:15px/1.6 system-ui;color:#172d3c;background:#f3f6f8;margin:0}main{max-width:1500px;margin:auto;padding:28px}section{background:white;padding:22px;margin:24px 0;border-radius:10px}table{border-collapse:collapse;width:100%;table-layout:fixed}th,td{vertical-align:top;text-align:left;border:1px solid #d4dfe5;padding:12px;overflow-wrap:anywhere;white-space:pre-wrap}th{background:#e4eeef}th:first-child{width:22%}a{color:#086d78}.notice{background:#fff1d8;padding:16px;border-left:4px solid #a87520}@media(max-width:650px){main{padding:8px}section{padding:8px}td,th{padding:6px;font-size:12px}}</style><main><h1>CHIVE · Checkpoint 1</h1><p class="notice">Review before embedding. No LLM, embeddings, similarity computation, or clustering has been run. Behaviour and cause facets are separate, as are all four sources. “Cause” means an edit with a measured ≥50 pp change, not a proven causal explanation.</p><p><a href="preprocessing_log.md">Schema, filter counts, rules and flags</a></p><nav>' + nav + '</nav>' + ''.join(parts) + '</main></html>'
    (out / 'review.html').write_text(doc)


def write_log(out, results):
    lines = ['# CHIVE observation preprocessing — CHECKPOINT 1', '',
             '**STOPPED for user review. No LLM calls, embeddings, clustering, or cluster labels.**', '',
             '## Schema mapping', '',
             '| Requested | Stored field |', '|---|---|',
             '| prompt_id | investigations.jsonl: prompt_id |',
             '| source | Source directory mapping below; no cross-source joins |',
             '| behaviour_text | behavior_summary; fallback structured_findings.baseline_behavior |',
             '| verif_score | verification.jsonl: score, joined by source and prompt_id |',
             '| cause_text_original | structured_findings.experiment_summaries[].summary |',
             '| Experiment metadata | experiment_number, baseline_rate, counterfactual_rate, is_key in the same summary |',
             '| Baseline labels | baseline_grader_verdicts[] |',
             '| Counterfactual labels | experiment_log[].grader_verdicts[] |',
             '| Experiment join | summary experiment_number = log exp_num within source/prompt_id |', '',
             '## Rules applied', '',
             '- Qwen3-8B only, selected from the four explicitly mapped downloaded runs. Raw inputs stay intact.',
             '- Verification >=7. Missing-score investigations are excluded from filtered tables.',
             '- Baseline and counterfactual rates use stored boolean labels. If labels are absent, use the leading x/n in the summary; never substitute qualitative estimates. Annotated rate strings are flagged. Invalid labels are not guessed.',
             '- Parsed versus recomputed rate disagreements are flagged; measured labels take priority. Grader reliability notes remain flags, not corrected observations.',
             '- Signed delta_pp = 100 × (counterfactual − baseline). The absolute >=50 threshold uses exact rational arithmetic, avoiding floating-point boundary errors.',
             '- Missing descriptions/rates are excluded, as approved. Incomplete rows remain in unfiltered audit tables, with null fields where unavailable.',
             '- Cause text uses only experiment summary edit descriptions. No Core causes, Final answer, structured_answer, or investigator report is read into the preprocessing cache.',
             '- Deterministic quote-aware sentence/clause rules remove result and interpretation clauses. Quotes and inline code protect literal edited strings (including fractions and percentages). Unrecognized/ambiguous sentences are removed rather than paraphrased. Empty results are excluded and flagged.',
             '- Removed spans are recorded in text_edits.jsonl. This is heuristic extraction; review the side-by-side samples for over-stripping or missed outcomes before approving.',
             '- Edit type follows the requested leading verbs exactly: Removed/Deleted → remove; Added/Inserted → add; Changed/Replaced/Rewrote and Kept … but → modify; otherwise unknown. Unknown does not imply missing data.',
             '- Both facets get whitespace and Markdown-formatting cleanup; original wording, capitalization, names, negation and identifiers are preserved. Link labels and targets are retained.',
             '- Exact duplicate cleaned texts are removed within each facet/source, keeping the first eligible source-file occurrence. Exclusions log retained IDs. No across-source deduplication.',
             '- Unfiltered Parquet copies are before score/effect/completeness/dedup filtering. Unfiltered cause rows include every logged experiment, with nulls for missing fields.',
             '- Near-duplicate cosine >0.95 checks are deferred to the embedding phase after approval, and will not delete rows.', '',
             '## Source mapping', '']
    lines += [f'- `{s}`: `{p}`' for s, p in SOURCES.items()]
    lines += ['', '## Filter-count log', '', '| Step | ' + ' | '.join(SOURCES) + ' |', '|---|' + '---:|' * len(SOURCES)]
    steps = [('Qwen3-8B investigations / unfiltered behaviour rows', 'investigations_input'),
             ('Missing verification score', 'investigations_missing_score'), ('Score below 7', 'investigations_score_below7'),
             ('Investigations after score >=7', 'investigations_score_ge7'),
             ('Behaviour rows after nonempty text + cleanup', 'behaviours_after_text_cleanup'),
             ('Behaviour rows after exact dedup (FINAL)', 'behaviours_after_dedup'),
             ('Logged experiments / unfiltered cause rows', 'experiments_input'),
             ('Experiments after score >=7', 'experiments_after_score_filter'),
             ('Cause rows after complete summary + usable rates', 'causes_after_completeness_filter'),
             ('Cause rows after absolute delta_pp >=50', 'causes_after_effect_filter'),
             ('Cause rows after nonempty stripped edit text', 'causes_after_text_cleanup'),
             ('Cause rows after exact dedup (FINAL)', 'causes_after_dedup')]
    for title, key in steps:
        lines.append('| ' + title + ' | ' + ' | '.join(f'{r["counts"].get(key, 0):,}' for r in results.values()) + ' |')
    lines += ['', '## Flags and exclusions', '', 'All flags include source, prompt_id, and experiment_number where applicable. Flags are counted as events; one row can carry multiple flags. Disagreement flags do not automatically exclude a row because labels take priority. Full counts and per-row logs are stored separately for each source.', '']
    for source, result in results.items():
        lines += [f'### {source}', '', '| Flag | Count |', '|---|---:|']
        lines += [f'| {k} | {v:,} |' for k, v in sorted(result['flags'].items())]
        lines += ['', '| Exclusion reason | Count |', '|---|---:|']
        lines += [f'| {k} | {v:,} |' for k, v in sorted(result['counts'].items()) if k.startswith('excluded_')]
        lines += ['', f'Final edit types: `{json.dumps(result["edit_types_final"], sort_keys=True)}`.', '']
    lines += ['## Files and review', '',
              'Each source directory contains behaviours.parquet, causes.parquet, behaviours_unfiltered.parquet, causes_unfiltered.parquet, exclusions.jsonl, flags.jsonl, text_edits.jsonl, counts.json, and review_samples.json.', '',
              'Filtered behaviours.parquet columns: prompt_id, source, verif_score, behaviour_text.', '',
              'Filtered causes.parquet columns: prompt_id, source, experiment_number, is_key, edit_type, baseline_rate, counterfactual_rate, delta_pp, cause_text, cause_text_original.', '',
              'Rates are proportions in [0,1]; delta_pp is signed percentage points. is_key is investigator metadata, not an inclusion filter or proof.', '',
              '[Open review.html](review.html): 15 random behaviour rows and 15 random cause rows **per source**, with original versus cleaned/stripped texts side by side. Seeds start at 20261004, increasing by source index; no similarity-based selection or rerolling.', '',
              'Reproduce: `clustering/.venv/bin/python clustering/scripts/preprocessing.py`. Requires PyArrow 25.0.1. Source provenance and hashes are in input_provenance.json. Old schema_audit.json and schema_missing_experiments.jsonl document the earlier preflight; counts.json describes this completed preprocessing run.', '',
              '**Checkpoint 1: awaiting user review. Do not embed or cluster until approved.**', '']
    (out / 'preprocessing_log.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/checkpoint1')
    parser.add_argument('--reuse-extracted-inputs', action='store_true', help='Reuse this run’s allowed-field cache for deterministic text-rule revisions.')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if not args.reuse_extracted_inputs:
        extract_inputs(args.data_root, out)
    grouped = {s: [] for s in SOURCES}
    with (out / 'input_observations.jsonl').open() as f:
        for line in f:
            record = json.loads(line)
            grouped[record['source']].append(record)
    results = {}
    for source in SOURCES:
        results[source] = process_source(source, grouped.pop(source), out / source)
        print(source, json.dumps(results[source]['counts']), flush=True)
    write_log(out, results)
    render_review(out, results)
    (out / 'counts.json').write_text(json.dumps({s: {k: v for k, v in r.items() if k != 'samples'} for s, r in results.items()}, indent=2) + '\n')
    print(f'CHECKPOINT 1: {out / "review.html"}. STOP — no embeddings or clustering.', flush=True)


if __name__ == '__main__':
    import sys
    if '--base-only' in sys.argv:
        sys.argv.remove('--base-only')
        main()
    else:
        from preprocessing_context import main as revised_checkpoint
        revised_checkpoint()
