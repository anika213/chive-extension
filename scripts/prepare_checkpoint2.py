"""Apply approved checkpoint-1 amendments to PETRI and WildChat eval only."""
from collections import Counter
import html
import json
from pathlib import Path
import random
import re
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import preprocessing_context as previous

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / 'data/checkpoint1_v2'
OUTPUT = ROOT / 'data/checkpoint1_approved'
SOURCES = ('petri', 'wildchat_eval')
EXTRA_TERMS = ('control', 'sanity check', 'qualitatively', 'grader')


def artifact_matches(original):
    text = re.sub(r'\s+', ' ', original or '')
    return previous.artifact_matches(original) + [t for t in EXTRA_TERMS if re.search(r'\b' + re.escape(t) + r'\b', text, re.I)]


def positive_stats(causes, investigation_ids):
    counts = Counter(c['prompt_id'] for c in causes)
    values = [counts[pid] for pid in investigation_ids if counts[pid] > 0]
    return {'denominator': 'investigations with at least one eligible cause; zero-cause investigations excluded',
            'investigations_with_causes': len(values), 'investigations_with_zero_causes': len(investigation_ids) - len(values),
            'mean': float(np.mean(values)) if values else None, 'median': float(np.median(values)) if values else None,
            'max': max(values) if values else None}


def update_row(row):
    original_artifact = row['measurement_artifact']
    matches = artifact_matches(row['cause_text_original'])
    denominator = row['baseline_context_chars']
    fraction = row['chars_changed'] / denominator if denominator and row['chars_changed'] is not None else None
    relative = fraction is not None and fraction > .5
    reasons = [reason for reason in row['clustering_exclusion_reasons'] if reason != 'measurement_artifact']
    if matches:
        reasons.append('measurement_artifact')
    if relative:
        reasons.append('context_swap_relative')
    if fraction is None:
        reasons.append('undefined_frac_changed')
    return {**row, 'frac_changed': fraction, 'context_swap_relative': relative,
            'measurement_artifact_before_amendment': original_artifact,
            'measurement_artifact': bool(matches), 'measurement_artifact_matches': matches,
            'clustering_exclusion_reasons': reasons, 'clustering_eligible': not reasons}


def main():
    OUTPUT.mkdir(exist_ok=True)
    reports, examples = {}, {}
    for index, source in enumerate(SOURCES):
        folder = OUTPUT / source; folder.mkdir(exist_ok=True)
        table = pq.read_table(INPUT / source / 'causes.parquet')
        rows = [update_row(row) for row in table.to_pylist()]
        schema = pa.schema(list(table.schema) + [pa.field('frac_changed', pa.float64()),
                            pa.field('context_swap_relative', pa.bool_()), pa.field('measurement_artifact_before_amendment', pa.bool_())])
        eligible = [r for r in rows if r['clustering_eligible']]
        for name, items in [('causes', rows), ('causes_for_clustering', eligible)]:
            previous.base.write_parquet(folder / (name + '.parquet'), items, schema)
        for name in ['behaviours.parquet', 'behaviours_unfiltered.parquet']:
            shutil.copy2(INPUT / source / name, folder / name)
        behaviours = pq.read_table(folder / 'behaviours.parquet').to_pylist()
        ids = [r['prompt_id'] for r in pq.read_table(folder / 'behaviours_unfiltered.parquet').to_pylist()
               if r['verif_score'] is not None and r['verif_score'] >= 7]
        features = pq.read_table(INPUT / source / 'context_diffs.parquet').to_pylist()
        qualified = [r for r in features if r['verif_score'] is not None and r['verif_score'] >= 7 and r['context_diff_available'] and r['baseline_context_chars'] > 0]
        all_fractions = [r['chars_changed'] / r['baseline_context_chars'] for r in qualified]
        report = {
            'source': source, 'behaviours': len(behaviours), 'cause_rows_retained_in_parquet': len(rows),
            'eligible_causes_before_amendment': pq.read_table(INPUT / source / 'causes_for_clustering.parquet').num_rows,
            'eligible_causes': len(eligible),
            'measurement_artifact_old': sum(r['measurement_artifact_before_amendment'] for r in rows),
            'measurement_artifact_new': sum(r['measurement_artifact'] for r in rows),
            'newly_flagged_artifacts': sum(r['measurement_artifact'] and not r['measurement_artifact_before_amendment'] for r in rows),
            'artifact_term_counts': dict(Counter(t for r in rows for t in r['measurement_artifact_matches'])),
            'frac_changed_distribution_all_verification_qualified_experiments': previous.distribution(all_fractions),
            'all_verification_qualified_experiments_above_half': sum(v > .5 for v in all_fractions),
            'frac_changed_distribution_retained_causes': previous.distribution([r['frac_changed'] for r in rows if r['frac_changed'] is not None]),
            'cause_rows_p90_flag': sum(r['context_swap'] for r in rows),
            'cause_rows_relative_flag': sum(r['context_swap_relative'] for r in rows),
            'cause_rows_both_size_flags': sum(r['context_swap'] and r['context_swap_relative'] for r in rows),
            'cause_rows_p90_only': sum(r['context_swap'] and not r['context_swap_relative'] for r in rows),
            'cause_rows_relative_only': sum(r['context_swap_relative'] and not r['context_swap'] for r in rows),
            'cause_rows_either_size_flag': sum(r['context_swap'] or r['context_swap_relative'] for r in rows),
            'cause_rows_undefined_frac_changed': sum(r['frac_changed'] is None for r in rows),
            'causes_per_investigation': positive_stats(eligible, ids),
            'verification_qualified_investigation_ids': ids,
        }
        reports[source] = report
        sample = random.Random(20261004 + index).sample(eligible, min(30, len(eligible)))
        examples[source] = sample
        (folder / 'eligible_cause_originals_30.json').write_text(json.dumps(sample, indent=2, ensure_ascii=False) + '\n')
        md = [f'# {source}: 30 random clustering-eligible cause rows', '', 'Original summaries, verbatim. Seed: ' + str(20261004 + index), '']
        for i, row in enumerate(sample, 1):
            md += [f'## {i}. {row["prompt_id"]} / experiment {row["experiment_number"]}', '',
                   f'Edit type: {row["edit_type"]}; delta: {row["delta_pp"]:+.1f} pp; frac_changed: {row["frac_changed"]:.4f}.', '',
                   row['cause_text_original'], '']
        (folder / 'eligible_cause_originals_30.md').write_text('\n'.join(md))
        (folder / 'counts.json').write_text(json.dumps(report, indent=2) + '\n')
        print(source, json.dumps({k: v for k, v in report.items() if not isinstance(v, (dict, list))}), flush=True)
    (OUTPUT / 'counts.json').write_text(json.dumps(reports, indent=2) + '\n')
    lines = ['# Approved checkpoint 1 amendments', '',
             'Only PETRI and WildChat eval are included. WildChat train and thinking are deferred.', '',
             'Added artifact markers: control, sanity check, qualitatively, grader. Matching is case-insensitive, against original summaries, with the same word/marker boundaries used at revision 2. This is a lexical screen, not a determination of validity.', '',
             'frac_changed = (characters deleted + characters added) / baseline_context_chars. It can exceed 1 when the edit adds substantial text. The new exclusion is frac_changed > 0.5 OR the existing source-specific p90 flag. Flagged rows remain in causes.parquet; only causes_for_clustering.parquet is embedded.', '',
             'All cause-diversity statistics use positive-only investigation denominators; zero-cause investigation counts are always reported separately. No semantic cluster labels are assigned.', '',
             '| Source | Artifact old | Artifact new | P90 | Relative >0.5 | Both | Either size rule | Eligible causes |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s, r in reports.items():
        lines.append(f'| {s} | {r["measurement_artifact_old"]} | {r["measurement_artifact_new"]} | {r["cause_rows_p90_flag"]} | {r["cause_rows_relative_flag"]} | {r["cause_rows_both_size_flags"]} | {r["cause_rows_either_size_flag"]} | {r["eligible_causes"]} |')
    lines += ['', '| Source | Fraction population | N | Mean | P25 | Median | P75 | P90 | P95 | P99 | Max | Above 0.5 |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s, r in reports.items():
        for key, label, above in [('frac_changed_distribution_all_verification_qualified_experiments', 'all score-qualified experiments', 'all_verification_qualified_experiments_above_half'),
                                  ('frac_changed_distribution_retained_causes', 'retained cause rows', 'cause_rows_relative_flag')]:
            d = r[key]
            lines.append(f'| {s} | {label} | {d["n"]} | ' + ' | '.join(f'{d[k]:.4f}' for k in ['mean','p25','p50','p75','p90','p95','p99','max']) + f' | {r[above]} |')
    lines += ['', '| Source | Investigations with causes (denominator) | Zero-cause investigations (excluded) | Mean | Median | Max |', '|---|---:|---:|---:|---:|---:|']
    for s, r in reports.items():
        d = r['causes_per_investigation']
        lines.append(f'| {s} | {d["investigations_with_causes"]} | {d["investigations_with_zero_causes"]} | {d["mean"]:.3f} | {d["median"]:g} | {d["max"]} |')
    (OUTPUT / 'preprocessing_log.md').write_text('\n'.join(lines) + '\n')
    esc = lambda s: html.escape(str(s))
    sections = []
    for source, sample in examples.items():
        sections.append(f'<h2 id="{source}">{source}: 30 original summaries</h2><p><a href="{source}/eligible_cause_originals_30.md">Printable Markdown</a> · <a href="{source}/causes_for_clustering.parquet">Eligible causes</a></p><table><tr><th>ID / measured edit</th><th>Original summary (verbatim)</th></tr>')
        for i, r in enumerate(sample, 1):
            sections.append(f'<tr><td>{i}. {esc(r["prompt_id"])}<br>Experiment {r["experiment_number"]}<br>Δ {r["delta_pp"]:+.1f} pp<br>frac_changed {r["frac_changed"]:.4f}<br>{r["edit_type"]}</td><td>{esc(r["cause_text_original"])}</td></tr>')
        sections.append('</table>')
    doc = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Approved inputs — 30 eligible causes per source</title><style>body{font:16px/1.6 system-ui;color:#173044;max-width:1300px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd9e0;text-align:left;vertical-align:top;padding:14px;overflow-wrap:anywhere}th{background:#eaf1f4}td:first-child{width:27%}pre{overflow:auto;font-size:13px}a{color:#086c79}</style><h1>Approved inputs for Checkpoint 2</h1><p>Original investigator summaries below are review material; only stripped edit descriptions are embedded. No semantic labels are assigned.</p><p><a href="preprocessing_log.md">Updated flags, fraction distributions and positive-only statistics</a></p><details><summary>Full preprocessing report</summary><pre>' + esc('\n'.join(lines)) + '</pre></details>' + ''.join(sections) + '</html>'
    (OUTPUT / 'review.html').write_text(doc)
    print('Saved 30 original-summary samples per source:', OUTPUT / 'review.html', flush=True)


if __name__ == '__main__': main()
