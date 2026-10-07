"""Validate revision 2 against preserved revision 1 rows and raw context samples."""
from collections import Counter
from html.parser import HTMLParser
import json
from pathlib import Path
import random

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

import preprocessing_context as p


def main():
    root = p.ROOT / 'data/checkpoint1_v2'
    old = p.ROOT / 'data/checkpoint1'
    reports = json.loads((root / 'counts.json').read_text())
    tokenizer = Tokenizer.from_file(str(p.ROOT / 'models/all-mpnet-base-v2/tokenizer.json'))
    tokenizer.no_truncation(); tokenizer.no_padding()
    results = {}
    for source, relative in p.base.SOURCES.items():
        folder = root / source
        causes = pq.read_table(folder / 'causes.parquet')
        eligible = pq.read_table(folder / 'causes_for_clustering.parquet')
        assert causes.schema == eligible.schema == p.extended_cause_schema()
        crows, erows = causes.to_pylist(), eligible.to_pylist()
        assert erows == [r for r in crows if r['clustering_eligible']]
        features = pq.read_table(folder / 'context_diffs.parquet').to_pylist()
        feature_map = {(r['prompt_id'], r['experiment_number']): r for r in features}
        pool = [f for f in features if f['verif_score'] is not None and f['verif_score'] >= 7 and f['context_diff_available']]
        cc = float(np.percentile([r['chars_changed'] for r in pool], 90))
        tc = float(np.percentile([r['tokens_changed'] for r in pool], 90))
        original = {(r['prompt_id'], r['experiment_number']): r for r in pq.read_table(old / source / 'causes.parquet').to_pylist()}
        assert len(crows) == len(original)  # Flagged rows have not disappeared.
        for c in crows:
            key = c['prompt_id'], c['experiment_number']
            o, f = original[key], feature_map[key]
            assert c['source'] == source
            assert all(c[k] == v for k, v in o.items() if k != 'edit_type')
            assert c['edit_type'] == f['edit_type']
            assert c['context_swap_char_p90'] == cc and c['context_swap_token_p90'] == tc
            assert c['context_swap'] == bool(f['context_diff_available'] and (f['chars_changed'] > cc or f['tokens_changed'] > tc))
            assert c['measurement_artifact_matches'] == p.artifact_matches(o['cause_text_original'])
            assert c['measurement_artifact'] == bool(c['measurement_artifact_matches'])
            assert c['clustering_eligible'] == (not c['context_swap'] and not c['measurement_artifact'] and c['context_diff_available'])
            if f['context_diff_available']:
                assert c['chars_changed'] == c['chars_deleted'] + c['chars_added']
                assert c['tokens_changed'] == c['tokens_deleted'] + c['tokens_added']
                assert c['experiment_context_chars'] - c['baseline_context_chars'] == c['chars_added'] - c['chars_deleted']
                assert c['experiment_context_tokens'] - c['baseline_context_tokens'] == c['tokens_added'] - c['tokens_deleted']
                assert c['edit_type'] == p.diff_type(c['chars_deleted'], c['chars_added'])
        assert len({r['cause_text'] for r in crows}) == len(crows)
        old_b = pq.read_table(old / source / 'behaviours_unfiltered.parquet').to_pylist()
        new_b = pq.read_table(folder / 'behaviours_unfiltered.parquet').to_pylist()
        kept_b = pq.read_table(folder / 'behaviours.parquet').to_pylist()
        assert len(old_b) == len(new_b)
        for a, b in zip(old_b, new_b):
            assert a['prompt_id'] == b['prompt_id'] and a['source'] == b['source'] == source
            assert b['behaviour_text'] == (p.behaviour_text(a['behaviour_text'])[0] if a['behaviour_text'] else a['behaviour_text'])
        assert len({r['behaviour_text'] for r in kept_b}) == len(kept_b)
        assert all(r['source'] == source and r['verif_score'] >= 7 for r in kept_b)
        ids = [r['prompt_id'] for r in old_b if r['verif_score'] is not None and r['verif_score'] >= 7]
        assert reports[source]['causes_per_investigation_after_new_flags'] == p.causes_per_investigation(erows, ids)
        counts = reports[source]['row_counts']
        assert counts['causes_in_parquet'] == len(crows) and counts['causes_eligible_for_clustering'] == len(erows)
        assert counts['behaviours_after_count_cleanup_and_dedup'] == len(kept_b)
        # Independently re-read ten actual baseline/experiment pairs per source.
        rng = random.Random(240104 + list(p.base.SOURCES).index(source))
        chosen = {(r['prompt_id'], r['experiment_number']): r for r in rng.sample(pool, 10)}
        verified = 0
        with (p.ROOT / 'data' / relative / 'investigations.jsonl').open() as stream:
            target_ids = {key[0] for key in chosen}
            for line in stream:
                r = json.loads(line)
                if r['prompt_id'] not in target_ids:
                    continue
                before = p.context_text(r['messages'])
                for e in r['experiment_log']:
                    key = r['prompt_id'], e['exp_num']
                    if key in chosen:
                        actual = p.context_diff(before, p.context_text(p.experiment_messages(e)[0]), tokenizer)
                        assert all(chosen[key][k] == v for k, v in actual.items())
                        verified += 1
                target_ids.remove(r['prompt_id'])
                if not target_ids:
                    break
        assert verified == 10
        sample = json.loads((folder / 'review_samples.json').read_text())
        assert len(sample['measurement_artifacts']) == min(10, counts['causes_measurement_artifact'])
        assert all(r['measurement_artifact'] and not r['clustering_eligible'] for r in sample['measurement_artifacts'])
        assert all(r['clustering_eligible'] for r in sample['causes'])
        results[source] = {'all_cause_rows_validated': len(crows), 'eligible_causes': len(erows),
                           'all_behaviour_rows_validated': len(new_b), 'raw_context_diffs_recomputed': verified}
        print(f'Validated {source}', flush=True)
    class Page(HTMLParser):
        def __init__(self):
            super().__init__(); self.links = []; self.ids = set()
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if 'id' in attrs: self.ids.add(attrs['id'])
            if tag == 'a': self.links.append(attrs['href'])
    page = Page(); page.feed((root / 'review.html').read_text())
    for link in page.links:
        assert (link[1:] in page.ids) if link.startswith('#') else (root / link).exists()
    result = {'status':'passed', 'sources':results, 'all_new_flags_validated':True,
              'all_original_cause_rows_preserved':True, 'all_original_rates_unchanged':True,
              'model_inference_calls':0, 'embeddings_computed':False, 'clusters_computed':False}
    (root / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
