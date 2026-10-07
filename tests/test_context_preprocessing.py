import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import preprocessing_context as p
from tokenizers import Tokenizer


class ContextPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(p.ROOT / 'models/all-mpnet-base-v2/tokenizer.json'))
        cls.tokenizer.no_truncation(); cls.tokenizer.no_padding()

    def test_true_edit_size_not_net_length(self):
        d = p.context_diff('abc', 'axc', self.tokenizer)
        self.assertEqual((d['chars_deleted'], d['chars_added'], d['chars_changed']), (1, 1, 2))
        self.assertEqual(d['edit_type'], 'modify')
        self.assertEqual(p.context_diff('abc', 'abxc', self.tokenizer)['edit_type'], 'add')
        self.assertEqual(p.context_diff('abxc', 'abc', self.tokenizer)['edit_type'], 'remove')
        self.assertEqual(p.context_diff('abc', 'abc', self.tokenizer)['edit_type'], 'no_change')

    def test_case_changes_and_long_inputs(self):
        d = p.context_diff('Keep it', 'keep it', self.tokenizer)
        self.assertEqual(d['edit_type'], 'modify')
        self.assertEqual(d['chars_changed'], 2)
        long_text = 'hello ' * 1000
        d = p.context_diff(long_text, long_text + ' added', self.tokenizer)
        self.assertGreater(d['baseline_context_tokens'], 384)
        self.assertGreater(d['tokens_changed'], 0)

    def test_roles_are_part_of_context_and_missing_is_not_empty(self):
        self.assertNotEqual(p.context_text([{'role':'user','content':'hello'}]),
                            p.context_text([{'role':'system','content':'hello'}]))
        self.assertEqual(p.context_text([]), '')
        with self.assertRaises(ValueError): p.context_text(None)
        messages, kind = p.experiment_messages({'prompt':'full prompt', 'system_prompt':'system instruction'})
        self.assertEqual(messages, [{'role':'system','content':'system instruction'}, {'role':'user','content':'full prompt'}])
        self.assertEqual(kind, 'legacy_prompt_and_system_prompt')
        self.assertEqual(p.experiment_messages({'messages': [], 'prompt':'ignore me'})[0], [])

    def test_count_prefixes_keep_observation(self):
        for prefix in ['In 13/30 completions, ', 'In all 30 completions, ', 'approximately 17/30 ',
                       'Across 13 out of 30 responses, ', 'In 13/30 completions (43%), ',
                       'In 3/10 completions (5, 6, 10), ', 'Across all 10 completions (40 prompts total), ']:
            self.assertEqual(p.behaviour_text(prefix + 'the model does not retry.')[0], 'the model does not retry.')
        self.assertEqual(p.behaviour_text('The model uses 13/30 as an example.')[0], 'The model uses 13/30 as an example.')
        self.assertEqual(p.behaviour_text('13/30foo is a filename.')[0], '13/30foo is a filename.')
        self.assertEqual(p.behaviour_text('In 2026 the model retries.')[0], 'In 2026 the model retries.')

    def test_artifact_matching_uses_original_text(self):
        self.assertEqual(p.artifact_matches('Removed topic. Trivially this was N/A, as expected.'), ['N/A', 'as expected', 'trivially'])
        self.assertEqual(p.artifact_matches('This did not reach the target.'), ['not reach'])
        self.assertEqual(p.artifact_matches('cannot\nbe graded'), ['cannot be graded'])
        self.assertEqual(p.artifact_matches('Removed topic. Rate dropped to zero.'), [])
        self.assertEqual(p.artifact_matches('Changed region/add and medication/anxiety.'), [])

    def test_cause_denominator_includes_zero_investigations(self):
        stats = p.causes_per_investigation([{'prompt_id':'a'}, {'prompt_id':'a'}], ['a','b'])
        self.assertEqual(stats['all_verification_qualified_investigations']['mean'], 1)
        self.assertEqual(stats['zero_cause_investigations'], 1)


if __name__ == '__main__': unittest.main()
