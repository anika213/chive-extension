import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import prepare_checkpoint2 as prep
import run_checkpoint2 as run


class Checkpoint2Tests(unittest.TestCase):
    def test_relative_flag_is_strict_and_artifacts_use_original(self):
        row = {'measurement_artifact':False, 'cause_text_original':'Control experiment; the grader was wrong.',
               'baseline_context_chars':100, 'chars_changed':50, 'clustering_exclusion_reasons':[]}
        result = prep.update_row(row)
        self.assertEqual(result['frac_changed'], .5)
        self.assertFalse(result['context_swap_relative'])
        self.assertTrue(result['measurement_artifact'])
        self.assertFalse(result['clustering_eligible'])
        row['chars_changed']=51;row['cause_text_original']='Removed an instruction.'
        self.assertTrue(prep.update_row(row)['context_swap_relative'])

    def test_positive_only_denominator(self):
        stats=prep.positive_stats([{'prompt_id':'a'},{'prompt_id':'a'}], ['a','b'])
        self.assertEqual(stats['mean'],2)
        self.assertEqual(stats['investigations_with_zero_causes'],1)
        diversity=run.cause_diversity([{'prompt_id':'a'},{'prompt_id':'b'}], np.array([0,-1]), ['a','b','c'])
        self.assertEqual(diversity['mean_distinct_clusters'],1)
        self.assertEqual(diversity['investigations_with_zero_eligible_causes'],1)
        self.assertEqual(diversity['investigations_with_eligible_causes_but_only_noise'],1)

    def test_incremental_silhouette_equals_sklearn(self):
        x=normalize(np.random.default_rng(17).normal(size=(230,12)).astype(np.float32))
        d=run.distances(x);tree,cuts=run.ward(x);scores=run.curve(d,tree,cuts)
        for index in [0,19,90,150,190]:
            self.assertAlmostEqual(scores[index], silhouette_score(d,cuts[:,index],metric='precomputed'), places=6)


if __name__=='__main__':unittest.main()
