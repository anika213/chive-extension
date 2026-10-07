"""Validate checkpoint-1 artifacts against every extracted input measurement."""
from collections import Counter
from fractions import Fraction
from html.parser import HTMLParser
import json
from pathlib import Path

import pyarrow.parquet as pq
import preprocessing as p


def main():
    root = Path(__file__).resolve().parent.parent / 'data/checkpoint1'
    stats = json.loads((root / 'counts.json').read_text())
    retained, original_causes = {}, {}
    report = {}
    for source in p.SOURCES:
        folder = root / source
        b = pq.read_table(folder / 'behaviours.parquet')
        c = pq.read_table(folder / 'causes.parquet')
        bu = pq.read_table(folder / 'behaviours_unfiltered.parquet')
        cu = pq.read_table(folder / 'causes_unfiltered.parquet')
        bs, cs = p.schemas()
        assert b.schema == bu.schema == bs and c.schema == cu.schema == cs
        counts = stats[source]['counts']
        assert b.num_rows == counts['behaviours_after_dedup']
        assert c.num_rows == counts['causes_after_dedup']
        assert bu.num_rows == counts['investigations_input']
        assert cu.num_rows == counts['experiments_input']
        brows, crows = b.to_pylist(), c.to_pylist()
        assert len({r['behaviour_text'] for r in brows}) == len(brows)
        assert len({r['cause_text'] for r in crows}) == len(crows)
        assert all(r['source'] == source and r['verif_score'] >= 7 and r['behaviour_text'] for r in brows)
        assert all(r['source'] == source and r['cause_text'] and r['cause_text_original'] and abs(r['delta_pp']) >= 50
                   and 0 <= r['baseline_rate'] <= 1 and 0 <= r['counterfactual_rate'] <= 1 for r in crows)
        retained[source] = {'behaviours': {r['prompt_id']: r for r in brows},
                            'causes': {(r['prompt_id'], r['experiment_number']): r for r in crows}}
        original_causes[source] = {(r['prompt_id'], r['experiment_number']): r for r in cu.to_pylist()}
        assert len(original_causes[source]) == cu.num_rows
        exclusions = [json.loads(l) for l in (folder / 'exclusions.jsonl').open()]
        ec = Counter(e['facet'] for e in exclusions)
        assert ec['behaviour'] + b.num_rows == bu.num_rows
        assert ec['cause'] + c.num_rows == cu.num_rows
        sample = json.loads((folder / 'review_samples.json').read_text())
        assert len(sample['behaviours']) == min(15, b.num_rows)
        assert len(sample['causes']) == min(15, c.num_rows)
        assert all(r['prompt_id'] in retained[source]['behaviours'] for r in sample['behaviours'])
        assert all((r['prompt_id'], r['experiment_number']) in retained[source]['causes'] for r in sample['causes'])
        flags = [json.loads(l) for l in (folder / 'flags.jsonl').open()]
        final_disagreements = {(r['prompt_id'], r.get('experiment_number')) for r in flags
                               if 'parsed_vs_labels_disagree' in r['reason']
                               and (r['prompt_id'], r.get('experiment_number')) in retained[source]['causes']}
        report[source] = {'behaviours': b.num_rows, 'causes': c.num_rows,
                          'unfiltered_behaviours': bu.num_rows, 'unfiltered_causes': cu.num_rows,
                          'filtered_cause_rows_with_rate_disagreement': len(final_disagreements),
                          'verified_input_experiments': 0}
    for line in (root / 'input_observations.jsonl').open():
        r = json.loads(line)
        assert not {'structured_answer', 'investigation_report', 'Core causes', 'Final answer'} & r.keys()
        source, pid = r['source'], r['prompt_id']
        b = retained[source]['behaviours'].get(pid)
        if b:
            assert b['verif_score'] == r['verif_score']
            assert b['behaviour_text'] == p.clean(r['behavior_summary'] or r['baseline_behavior'])
        summaries = {s['experiment_number']: s for s in r.get('experiment_summaries') or []}
        for e in r['experiment_log']:
            key = (pid, e['exp_num'])
            c = original_causes[source][key]
            summary = summaries.get(e['exp_num'])
            assert c['cause_text_original'] == (summary['summary'] if summary else None)
            for output_key, labels, reported in [('baseline_rate', r['baseline_grader_verdicts'], summary.get('baseline_rate') if summary else None),
                                                ('counterfactual_rate', e['grader_verdicts'], summary.get('counterfactual_rate') if summary else None)]:
                expected = Fraction(sum(labels), len(labels)) if labels else p.parse_rate(reported)[0]
                assert c[output_key] == (float(expected) if expected is not None else None)
            if key in retained[source]['causes']:
                assert retained[source]['causes'][key] == c
                assert r['verif_score'] >= 7
                # A measured zero is valid and must not be replaced by a prose rate.
                base, _ = p.label_rate(r['baseline_grader_verdicts'])
                counter, _ = p.label_rate(e['grader_verdicts'])
                if base is None:
                    base = p.parse_rate(summary['baseline_rate'])[0]
                if counter is None:
                    counter = p.parse_rate(summary['counterfactual_rate'])[0]
                delta = 100 * (counter - base)
                assert abs(delta) >= 50 and c['delta_pp'] == float(delta)
                assert c['edit_type'] == p.edit_type(c['cause_text_original'])
            report[source]['verified_input_experiments'] += 1
    class Page(HTMLParser):
        def __init__(self):
            super().__init__(); self.rows = 0
        def handle_starttag(self, tag, attrs):
            if tag == 'tr':
                self.rows += 1
    page = Page(); page.feed((root / 'review.html').read_text())
    assert page.rows == 128  # 120 sample rows + 8 table headers.
    result = {'status': 'passed', 'sources': report, 'review_sample_rows': 120,
              'checks': ['exact output schemas', 'source isolation', 'score and effect thresholds',
                         'no exact duplicate texts within source/facet', 'all original summaries preserved',
                         'all unfiltered rates verified against inputs', 'all final signed deltas verified',
                         'exclusion accounting', 'review samples belong to final tables'],
              'llm_calls': 0, 'embeddings_computed': False, 'clusters_computed': False}
    (root / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
