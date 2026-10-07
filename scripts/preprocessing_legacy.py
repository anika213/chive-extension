"""Save reviewable cause-focused texts BEFORE embedding; standard library only.

Default: PETRI Qwen3-8B eval, investigation verification >=7.
No LLM calls, embeddings, clustering, inferred mechanisms, or automatic entity
redaction. Intervention prose is descriptive metadata; measured rates come only
from investigation grader logs. Missing descriptions are queued for review.

python clustering/scripts/preprocessing.py
python clustering/scripts/preprocessing.py --sources petri wildchat --replacement-map aliases.json
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
import math
from pathlib import Path
import re
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parent.parent
VERSION = '1.0'


def normalize_text(text: str, replacements: dict[str, str] | None = None) -> tuple[str, list[dict]]:
    """Whitespace cleanup + explicit, simultaneous whole-token substitutions.

    No stopword removal, stemming, lowercasing, number stripping, or generic
    regex name guessing: those can destroy negation and intervention specificity.
    Longest aliases match first; replacement values are never processed again.
    """
    changes = []
    replacements = replacements or {}
    if replacements:
        pattern = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(k) for k in sorted(replacements, key=lambda k: (-len(k), k))) + r')(?!\w)')
        def replace(match):
            old = match.group()
            changes.append({'original': old, 'replacement': replacements[old], 'original_span': [match.start(), match.end()]})
            return replacements[old]
        text = pattern.sub(replace, text)
    return re.sub(r'\s+', ' ', text).strip(), changes


def load_replacements(path: Path | None) -> dict:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip() for k, v in data.items()):
        raise ValueError('Replacement map must be an object of nonempty exact strings to nonempty strings.')
    return data


def counts(verdicts, reported_yes) -> tuple[int, int]:
    if not isinstance(verdicts, list) or not verdicts:
        raise ValueError('missing_or_empty_grader_verdicts')
    if any(type(v) not in (bool, int) or v not in (0, 1) for v in verdicts):
        raise ValueError('non_boolean_grader_verdict')
    k, n = sum(verdicts), len(verdicts)
    if type(reported_yes) not in (int, float) or not math.isfinite(reported_yes) or reported_yes != k:
        raise ValueError('grader_yes_disagrees_with_verdicts')
    return int(k), n


def measured_outcome(k0: int, n0: int, k1: int, n1: int) -> dict:
    # Integer comparison avoids spurious direction due to floating-point rounding.
    difference = k1*n0 - k0*n1
    direction = 'increased' if difference > 0 else 'decreased' if difference < 0 else 'unchanged'
    return {'k_baseline': k0, 'n_baseline': n0, 'k_counterfactual': k1, 'n_counterfactual': n1,
            'baseline_rate': k0/n0, 'counterfactual_rate': k1/n1,
            'signed_effect_pp': 100*(k1/n1-k0/n0), 'observed_direction': direction,
            'interpretation': 'Descriptive sample-rate change, not a significance test or proof of a single causal factor.'}


def extract_claims(text: str) -> list[dict]:
    """Separate report claims by their own status; do not infer experiment links."""
    header = re.compile(r'(?m)^\s*(?:#+\s*)?(?:\*\*)?(Observed behavior|Core causes|Negative evidence|Final answer)(?:\*\*)?\s*:?[ \t]*$')
    sections = list(header.finditer(text))
    claims = []
    for i, h in enumerate(sections):
        if h.group(1) not in ('Core causes', 'Negative evidence'):
            continue
        offset = h.end()
        body = text[offset:sections[i+1].start() if i+1 < len(sections) else len(text)]
        items = list(re.finditer(r'(?m)^\s*([0-9]+|[A-Z])[.)]\s+', body))
        for j, item in enumerate(items):
            block = body[item.end():items[j+1].start() if j+1 < len(items) else len(body)]
            status = re.search(r'(?mi)^\s*(?:\*\*)?Status:(?:\*\*)?\s*(.+)$', block)
            if not status:
                continue
            raw = block[:status.start()]
            statement = raw.strip()
            a = offset + item.end() + len(raw)-len(raw.lstrip())
            if not statement:
                continue
            reported = status.group(1).strip().strip('*').strip().lower()
            normalized = reported if reported in ('supported', 'partially supported', 'refuted', 'uncertain') else 'uncertain'
            claims.append({'section': h.group(1), 'number': item.group(1), 'original_text': statement,
                           'reported_status': reported, 'status': normalized, 'source_span': [a, a+len(statement)]})
            assert text[a:a+len(statement)] == statement
    return claims


def emit(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


def read_hashed(path: Path, hashes: dict):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for line_number, line in enumerate(f, 1):
            digest.update(line)
            yield line_number, json.loads(line)
    hashes[path.name] = digest.hexdigest()


def process(source: str, data_root: Path, out_root: Path, min_score: float, aliases: dict) -> dict:
    run = f'qwen3_8b_{source}_eval'
    directory, out = data_root/source/run, out_root/source/run
    inputs = [directory/name for name in ('investigations.jsonl', 'verification.jsonl', 'binary_counterfactual_questions.jsonl')]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    out.mkdir(parents=True, exist_ok=True)
    hashes, scores, descriptions = {}, {}, defaultdict(list)
    for line, r in read_hashed(inputs[1], hashes):
        if r['prompt_id'] in scores:
            raise ValueError('Duplicate verification prompt ID')
        scores[r['prompt_id']] = (r.get('score'), line)
    for line, r in read_hashed(inputs[2], hashes):
        nums = r.get('experiment_numbers')
        if isinstance(nums, list) and len(nums) == 1:
            descriptions[(r['prompt_id'], nums[0])].append((line, r))
    totals, reasons, statuses = Counter(), Counter(), Counter()
    seen, used, preview = set(), set(), []
    filenames = ['claims.jsonl', 'experiments.jsonl', 'embedding_inputs.jsonl', 'review_required.jsonl', 'excluded_investigations.jsonl']
    with ExitStack() as stack:
        outputs = {name: stack.enter_context((out/name).open('w')) for name in filenames}
        for line_number, r in read_hashed(inputs[0], hashes):
            pid = r['prompt_id']
            if pid in seen:
                raise ValueError('Duplicate investigation prompt ID')
            seen.add(pid); totals['investigations_total'] += 1
            score, vline = scores.get(pid, (None, None))
            if type(score) not in (float, int) or not math.isfinite(score) or score < min_score:
                emit(outputs['excluded_investigations.jsonl'], {'prompt_id': pid, 'verification_score': score, 'reason': 'missing_invalid_or_below_threshold_verification'})
                totals['excluded_investigations'] += 1
                continue
            totals['retained_investigations'] += 1
            common = {'source': source, 'run': run, 'prompt_id': pid, 'verification_score': score,
                      'investigation_line': line_number, 'verification_line': vline}
            report = (r.get('structured_findings') or {}).get('structured_answer') or ''
            claims = extract_claims(report)
            for claim in claims:
                text, edits = normalize_text(claim['original_text'], aliases)
                emit(outputs['claims.jsonl'], {**common, **claim, 'normalized_text': text, 'replacements': edits,
                      'evidence_status': 'Investigator claim only; no automatic claim-to-experiment alignment.',
                      'structured_answer_sha256': hashlib.sha256(report.encode()).hexdigest()})
                statuses[claim['status']] += 1; totals['report_claims'] += 1
            if not claims:
                emit(outputs['review_required.jsonl'], {**common, 'reason': 'no_parseable_report_claims'})
                reasons['no_parseable_report_claims'] += 1
            try:
                k0, n0 = counts(r.get('baseline_grader_verdicts'), r.get('baseline_grader_yes'))
                baseline_error = None
            except ValueError as e:
                baseline_error = str(e)
            summaries = {e.get('experiment_number'): e for e in (r.get('structured_findings') or {}).get('experiment_summaries', [])}
            nums = set()
            for experiment in r.get('experiment_log', []):
                num = experiment.get('exp_num');totals['experiments_total'] += 1
                if type(num) is not int or num in nums:
                    raise ValueError(f'{pid}: duplicate/invalid experiment number')
                nums.add(num)
                record = {**common, 'item_id': f'{run}:{pid}:experiment_{num}', 'experiment_number': num,
                          'edit_record': experiment.get('edit_record'), 'temperature': experiment.get('temperature'),
                          'investigator_experiment_summary_unverified': summaries.get(num)}
                reason = baseline_error
                try:
                    k1, n1 = counts(experiment.get('grader_verdicts'), experiment.get('grader_yes'))
                    if baseline_error is None:
                        record['measured_outcome'] = measured_outcome(k0, n0, k1, n1)
                except ValueError as e:
                    reason = str(e)
                candidates = descriptions.get((pid, num), [])
                # Do not silently choose among conflicting intervention descriptions.
                unique = {c.get('intervention', '').strip() for _, c in candidates if isinstance(c.get('intervention'), str) and c['intervention'].strip()}
                if not reason and len(unique) != 1:
                    reason = 'missing_published_intervention_description' if not unique else 'conflicting_published_intervention_descriptions'
                if not reason:
                    claim_line, c = next((line, c) for line, c in candidates if isinstance(c.get('intervention'), str) and c['intervention'].strip())
                    originals = {'intervention': c['intervention'], 'behavior_question': r.get('classifier_question') or c.get('behavior_question')}
                    if not isinstance(originals['behavior_question'], str) or not originals['behavior_question'].strip():
                        reason = 'missing_classifier_question'
                    else:
                        normalized = {}; replacements = {}
                        for key, text in originals.items():
                            normalized[key], replacements[key] = normalize_text(text, aliases)
                        outcome = record['measured_outcome']
                        measured = f"Measured classifier-YES frequency {outcome['observed_direction']}: {k0}/{n0} → {k1}/{n1} ({outcome['signed_effect_pp']:+.1f} percentage points)."
                        text = f"Intervention: {normalized['intervention']}\nObserved outcome: classifier-YES frequency {outcome['observed_direction']}.\nClassifier-YES means: {normalized['behavior_question']}"
                        record.update(original_text=originals, normalized_text=normalized, replacements=replacements,
                                      embedding_text=text, measured_text=measured,
                                      claim_bank_line=claim_line, claim_ids=[cc.get('claim_id') for _, cc in candidates],
                                      reported_factor_status=c.get('factor_status'), edit_cleanliness=c.get('edit_cleanliness'),
                                      description_status='Published prose, not independently validated here; consult the actual prompt edit.',
                                      causal_isolation='not_established' if c.get('edit_cleanliness') in ('mixed','confounded') else 'publisher_assessment_only',
                                      interpretation='One tested intervention per item; an edit can change multiple factors. Direction is descriptive, not supported/refuted hypothesis status.')
                        record['claim_bank_count_mismatches'] = {key: {'reported': c.get(key), 'log': value} for key,value in [('k_baseline', k0),('n_baseline', n0),('k_counterfactual', k1),('n_counterfactual', n1)] if c.get(key) != value}
                        totals['claim_bank_count_mismatches'] += bool(record['claim_bank_count_mismatches'])
                record['ready_for_embedding'] = reason is None
                if reason:
                    record['review_reason'] = reason; reasons[reason] += 1
                    emit(outputs['review_required.jsonl'], record)
                else:
                    emit(outputs['embedding_inputs.jsonl'], record)
                    totals['embedding_inputs'] += 1; used.add(pid)
                    if len(preview) < 30: preview.append(record)
                emit(outputs['experiments.jsonl'], record)
    summary = {'preprocessing_version': VERSION, 'source': source, 'run': run, 'minimum_verification_score': min_score,
               'counts': dict(totals), 'investigations_with_embedding_inputs': len(used), 'review_reasons': dict(reasons),
               'reported_claim_status_counts': dict(statuses), 'input_sha256': hashes,
               'replacement_map': aliases, 'entity_normalization': 'Explicit reviewed aliases only; default keeps all names, filenames, dates and exact quoted cues.',
               'embedding_text_field': 'embedding_text', 'embeddings_computed': False,
               'prose_is_not_ground_truth': True, 'measured_rate_source': 'investigations.jsonl baseline_grader_verdicts / experiment_log.grader_verdicts, checked against grader_yes',
               'files': filenames}
    (out/'manifest.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    parts = ['<!doctype html><meta charset="utf-8"><title>Preprocessing preview</title><style>body{max-width:1050px;margin:40px auto;font:16px/1.6 system-ui}pre{white-space:pre-wrap;background:#f3f6f7;padding:16px}section{margin:30px 0}small{color:#526570}</style><h1>Preprocessed intervention → outcome text</h1><p>First 30 ready items. No embeddings computed. Prose describes interventions; outcomes below are calculated from grader logs.</p>']
    for record in preview:
        parts.append(f'<section><h3>{html.escape(record["item_id"])}</h3><small>Reported factor status: {html.escape(str(record["reported_factor_status"]))}; edit cleanliness: {html.escape(str(record["edit_cleanliness"]))}</small><h4>Before: published intervention description</h4><pre>{html.escape(record["original_text"]["intervention"])}</pre><h4>Saved embedding input</h4><pre>{html.escape(record["embedding_text"])}</pre><p>{html.escape(record["measured_text"])}</p></section>')
    (out/'preview.html').write_text('\n'.join(parts))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sources', choices=['petri','wildchat'], nargs='+', default=['petri'])
    ap.add_argument('--data-root', type=Path, default=ROOT/'data')
    ap.add_argument('--output-dir', type=Path, default=ROOT/'data/preprocessed')
    ap.add_argument('--min-verification-score', type=float, default=7)
    ap.add_argument('--replacement-map', type=Path, help='Reviewed exact aliases → role placeholders JSON. Nothing is removed by default.')
    args = ap.parse_args()
    if not math.isfinite(args.min_verification_score): ap.error('Verification threshold must be finite')
    aliases = load_replacements(args.replacement_map)
    for source in dict.fromkeys(args.sources):
        result = process(source, args.data_root, args.output_dir, args.min_verification_score, aliases)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
