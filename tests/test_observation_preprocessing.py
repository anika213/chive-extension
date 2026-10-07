import importlib.util
import json
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('observation_preprocessing', Path(__file__).resolve().parents[1] / 'scripts' / 'preprocessing.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class ObservationPreprocessingTests(unittest.TestCase):
    def test_edit_payload_is_protected_and_outcomes_removed(self):
        text, removed = p.strip_outcomes('Added "Use 50% of the data, not 1/10" to the prompt. Rate dropped to 0/30, confirming a driver.')
        self.assertEqual(text, 'Added "Use 50% of the data, not 1/10" to the prompt.')
        self.assertTrue(removed)
        text, _ = p.strip_outcomes('Removed earlier failures, yielding 0/30 retries and confirming history is the cause.')
        self.assertEqual(text, 'Removed earlier failures')
        text, _ = p.strip_outcomes('Replaced `retry_rate` with `max_rate`. Both updates correctly assigned to agent 7.')
        self.assertEqual(text, 'Replaced retry_rate with max_rate.')

    def test_negation_and_literal_punctuation_survive(self):
        text, _ = p.strip_outcomes("Changed the instruction to 'Do not retry. Keep 50% untouched.' The model retried in 0/30 cases.")
        self.assertEqual(text, "Changed the instruction to 'Do not retry. Keep 50% untouched.'")
        self.assertEqual(p.clean('**Do not** alter `user_id`.\n\n# Context'), 'Do not alter user_id. Context')
        self.assertEqual(p.strip_outcomes('Increased timeout from 2 to 5 seconds. Rate dropped to 0/30.')[0], 'Increased timeout from 2 to 5 seconds.')
        self.assertEqual(p.strip_outcomes("Added 'Keep the defaults.' 0/30 changed defaults.")[0], "Added 'Keep the defaults.'")
        self.assertEqual(p.strip_outcomes('English question with a hint. Error rate dropped to 0/30.')[0], 'English question with a hint.')
        self.assertEqual(p.strip_outcomes('Removed names (Jane, Molly-Rose Ling). Rate dropped to 0/30.')[0], 'Removed names (Jane, Molly-Rose Ling).')
        self.assertEqual(p.strip_outcomes('Replaced user question with supervised vs. unsupervised learning. Rate was 0/30.')[0], 'Replaced user question with supervised vs. unsupervised learning.')
        self.assertEqual(p.strip_outcomes("Changed final user message to 'Describe the scene again.' Dark mood dropped to 1/30, proving that the instruction to reinterpret is the primary driver.")[0], "Changed final user message to 'Describe the scene again.'")
        self.assertEqual(p.strip_outcomes("Renamed to processCall. Qualitatively, completions used mixed language ('calls/executes/adds').")[0], 'Renamed to processCall.')
        self.assertEqual(p.strip_outcomes("Added 'Avoid emotional language.' The contradiction was completely eliminated, confirming the model can comply when directly instructed.")[0], "Added 'Avoid emotional language.'")
        self.assertEqual(p.strip_outcomes('Removed feedback. Reward mentions dropped to zero; the model used the tool to check responses instead.')[0], 'Removed feedback.')
        self.assertEqual(p.strip_outcomes('David asks about a generic family dinner (no specific honoree). Without a specific subject, defaults to caller.')[0], 'David asks about a generic family dinner (no specific honoree).')

    def test_edit_types_are_literal_leading_verb_tags(self):
        for text, expected in [('Removed X', 'remove'), ('Deleted X', 'remove'), ('Added X', 'add'),
                               ('Inserted X', 'add'), ('Rewrote X', 'modify'), ('Kept X but removed Y', 'modify'),
                               ('The edit removed X', 'unknown'), ('Renamed X', 'unknown')]:
            self.assertEqual(p.edit_type(text), expected)

    def test_fraction_rates_and_invalid_data(self):
        self.assertEqual(p.parse_rate('10/30'), (Fraction(1, 3), None))
        self.assertEqual(p.parse_rate('0/30 (grader unreliable; qualitative 30/30)'), (Fraction(0), 'annotated_rate'))
        self.assertIsNone(p.parse_rate('4/0')[0])
        self.assertIsNone(p.parse_rate('31/30')[0])
        self.assertIsNone(p.label_rate(['yes', 'no'])[0])
        self.assertEqual(abs((Fraction(1, 10) - Fraction(6, 10)) * 100), 50)

    def test_pipeline_labels_override_prose_and_dedup_is_auditable(self):
        def record(pid, score=8, labels=None):
            return {'source': 'petri', 'prompt_id': pid, 'verif_score': score,
                    'behavior_summary': '**Does not** retry.', 'baseline_behavior': None,
                    'baseline_grader_verdicts': [True] * 6 + [False] * 4, 'baseline_grader_yes': 6,
                    'experiment_summaries': [{'experiment_number': 1, 'summary': 'Removed prior failures. Rate dropped to 0/10.',
                                             'baseline_rate': '6/10', 'counterfactual_rate': '0/10', 'is_key': False}],
                    'experiment_log': [{'exp_num': 1, 'grader_verdicts': labels if labels is not None else [True] + [False] * 9,
                                        'grader_yes': 1, 'n': 10},
                                       {'exp_num': 2, 'grader_verdicts': None, 'grader_yes': None, 'n': 10}]}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = p.process_source('petri', [record('a'), record('b'), record('c', 6)], root)
            import pyarrow.parquet as pq
            rows = pq.read_table(root / 'causes.parquet').to_pylist()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['delta_pp'], -50)
            self.assertEqual(rows[0]['counterfactual_rate'], .1)
            self.assertFalse(rows[0]['is_key'])  # is_key is not a filter.
            self.assertEqual(result['flags']['counterfactual_parsed_vs_labels_disagree'], 3)
            self.assertEqual(pq.read_table(root / 'causes_unfiltered.parquet').num_rows, 6)
            self.assertEqual(pq.read_table(root / 'behaviours.parquet').num_rows, 1)
            excluded = [json.loads(l) for l in (root / 'exclusions.jsonl').read_text().splitlines()]
            self.assertTrue(any(e['reason'] == 'incomplete' for e in excluded))
            self.assertTrue(any(e['reason'] == 'exact_duplicate' for e in excluded))


if __name__ == '__main__':
    unittest.main()
