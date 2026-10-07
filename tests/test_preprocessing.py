import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('preprocessing_legacy', Path(__file__).resolve().parents[1]/'scripts' / 'preprocessing_legacy.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

class PreprocessingTests(unittest.TestCase):
    def test_aliases_preserve_negation_and_distinct_roles(self):
        text, edits = p.normalize_text('Ravi Kapoor did NOT tell Ingrid; Ravioli is unchanged.',
                                        {'Ravi Kapoor':'PERSON_A','Ravi':'PERSON_A','Ingrid':'PERSON_B'})
        self.assertEqual(text,'PERSON_A did NOT tell PERSON_B; Ravioli is unchanged.')
        self.assertEqual(len(edits),2)
        # Replacements are simultaneous, not recursively applied.
        self.assertEqual(p.normalize_text('Alice', {'Alice':'Bob','Bob':'PERSON_B'})[0], 'Bob')

    def test_measurement_validation_and_direction(self):
        self.assertEqual(p.counts([True,False,1],2),(2,3))
        with self.assertRaises(ValueError):p.counts([True,False],2)
        with self.assertRaises(ValueError):p.counts(['yes'],1)
        self.assertEqual(p.measured_outcome(1,3,2,6)['observed_direction'],'unchanged')
        self.assertEqual(p.measured_outcome(30,30,0,30)['observed_direction'],'decreased')

    def test_claim_statuses_are_not_merged(self):
        text='Observed behavior\nSomething\n\nCore causes\n\n1. History matters\nStatus: supported\nEvidence:\n- prose\n\n2. Identity matters\nStatus: partially supported\n\nNegative evidence\n\nA. Length matters\nStatus: refuted\nEvidence:\n- prose\n\nFinal answer\nConclusion'
        claims=p.extract_claims(text)
        self.assertEqual([r['status'] for r in claims],['supported','partially supported','refuted'])
        self.assertEqual([r['original_text'] for r in claims],['History matters','Identity matters','Length matters'])
        for r in claims:self.assertEqual(text[slice(*r['source_span'])],r['original_text'])

    def test_logs_override_claim_prose_and_missing_descriptions_are_queued(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=root/'data/petri/qwen3_8b_petri_eval';run.mkdir(parents=True)
            def save(name,rows):(run/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            save('verification.jsonl',[{'prompt_id':'p','score':8}])
            report='Core causes\n\n1. History matters\nStatus: supported\n\nFinal answer\nExplanation'
            save('investigations.jsonl',[{'prompt_id':'p','classifier_question':'Does it NOT retry?',
                'baseline_grader_verdicts':[True,True],'baseline_grader_yes':2,
                'structured_findings':{'structured_answer':report},
                'experiment_log':[{'exp_num':1,'grader_verdicts':[False,False],'grader_yes':0},
                                  {'exp_num':2,'grader_verdicts':[True,False],'grader_yes':1}]}])
            save('binary_counterfactual_questions.jsonl',[{'prompt_id':'p','experiment_numbers':[1],
                 'intervention':'Remove earlier failed attempts.','factor_status':'supported',
                 'claim_id':'p_c0','k_baseline':0,'n_baseline':2,'k_counterfactual':2,'n_counterfactual':2}])
            result=p.process('petri',root/'data',root/'out',7,{})
            out=root/'out/petri/qwen3_8b_petri_eval'
            rows=[json.loads(l) for l in (out/'embedding_inputs.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['measured_outcome']['signed_effect_pp'],-100)
            self.assertIn('Does it NOT retry?',rows[0]['embedding_text'])
            self.assertEqual(len(rows[0]['claim_bank_count_mismatches']),2)
            self.assertEqual(result['review_reasons']['missing_published_intervention_description'],1)
            self.assertFalse(result['embeddings_computed'])

if __name__=='__main__':unittest.main()
