"""Build a single-investigation field guide. No embedding or clustering runs."""
from pathlib import Path
import difflib
import json
import hashlib
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
PID = 'petri_medical_authority_sycophancy_e2_t4'


def find_record(path):
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            if record['prompt_id'] == PID: return record
    raise ValueError(f'Missing {PID} in {path}')


def main():
    raw_path = ROOT / 'data/petri/qwen3_8b_petri_eval/investigations.jsonl'
    raw = find_record(raw_path)
    verification = find_record(ROOT / 'data/petri/qwen3_8b_petri_eval/verification.jsonl')
    behaviour = next(r for r in pq.read_table(ROOT / 'data/checkpoint1_locations/petri/behaviours.parquet').to_pylist() if r['prompt_id'] == PID)
    retained = [r for r in pq.read_table(ROOT / 'data/checkpoint1_locations/petri/causes.parquet').to_pylist() if r['prompt_id'] == PID]
    cleaned = {r['experiment_number']: r for r in retained}
    corrected = json.loads((ROOT / 'results/scenario_clustering/related_health_counterfactuals/records.json').read_text())
    for r in corrected:
        if r['prompt_id'] == PID: cleaned[r['experiment_number']] = r
    summaries = {r['experiment_number']: r for r in raw['structured_findings']['experiment_summaries']}
    experiments = []
    for e in raw['experiment_log']:
        labels = e['grader_verdicts']
        assert len(labels) == len(e['completions']) and all(isinstance(v, bool) for v in labels)
        diff = []
        before, after = raw['messages'][-1]['content'], e['messages'][-1]['content']
        for tag, a, b, c, d in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
            diff.append({'operation': tag, 'before': before[a:b], 'after': after[c:d]})
        experiments.append({**e, 'summary': summaries.get(e['exp_num']), 'preprocessed': cleaned.get(e['exp_num']),
                            'yes_count_checked': sum(labels), 'final_message_diff': diff})
    labels = raw['baseline_grader_verdicts']
    assert len(labels) == len(raw['completions']) and sum(labels) == raw['baseline_grader_yes']
    payload = {'prompt_id': PID, 'source': 'petri', 'scenario': 'medical_authority_sycophancy',
        'verification': {k: verification[k] for k in ['score', 'experiment_assessments', 'explanation_assessment', 'issues']},
        'behavior_summary': raw['behavior_summary'], 'behaviour_text_cleaned': behaviour['behaviour_text'],
        'baseline_behavior': raw['structured_findings']['baseline_behavior'], 'interest_score': raw['interest_score'],
        'question': raw['question'], 'classifier_question': raw['classifier_question'], 'suggested_counterfactual': raw['suggested_counterfactual'],
        'messages': raw['messages'], 'completions': raw['completions'], 'baseline_labels': labels,
        'baseline_yes': sum(labels), 'experiments': experiments, 'investigation_report': raw['investigation_report'],
        'structured_findings': raw['structured_findings'], 'n_agent_turns': raw['n_agent_turns'],
        'source_path': str(raw_path), 'eligible_experiment_numbers': sorted(r['experiment_number'] for r in cleaned.values() if r['clustering_eligible']),
        'view_only': True, 'embedding_calls': 0, 'clustering_calls': 0, 'generative_llm_calls': 0}
    data_path = ROOT / 'examples/investigation_parts_example.json'
    data_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    safe = json.dumps(payload, ensure_ascii=False).replace('<', r'\u003c')
    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>CHIVE example — parts of one investigation</title>
<style>
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}body{margin:0;background:#f3f6f8;color:#193344;font:16px/1.6 system-ui,sans-serif}main{max-width:1320px;margin:auto;padding:32px 24px 90px}h1{font-size:38px;line-height:1.15;max-width:850px}h2{font-size:25px;line-height:1.3}h3{font-size:17px}a{color:#086c79}code{font-size:12px;overflow-wrap:anywhere}.eyebrow{font-size:12px;letter-spacing:.12em;color:#086c79;text-transform:uppercase}.panel,.card{background:white;border:1px solid #d0dce2;border-radius:12px;padding:22px;margin:22px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.grid>.card{margin:0;min-width:0}.text,pre{white-space:pre-wrap;overflow-wrap:anywhere}pre{font:13px/1.6 ui-monospace,monospace}.small,.note{font-size:13px;color:#536b78}.badge{display:inline-block;font-size:12px;background:#e2f1ed;color:#1a6253;border-radius:20px;padding:3px 10px}.excluded{background:#fff0d4;color:#7c591a}.stats{display:flex;gap:12px;flex-wrap:wrap}.stat{background:white;border:1px solid #d0dce2;border-radius:8px;padding:12px 18px}.stat strong{font-size:24px;display:block}.nav{display:flex;gap:10px;flex-wrap:wrap;margin:24px 0}.nav a{background:white;padding:8px 12px;border:1px solid #d0dce2;border-radius:8px;text-decoration:none;font-size:14px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d0dce2;padding:10px;text-align:left;vertical-align:top}th{background:#eaf1f4}.option{font-weight:650}.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center}select,button{font:inherit;padding:7px 10px;border:1px solid #bdcdd6;border-radius:6px;background:white;color:#193344;max-width:100%}button{cursor:pointer;color:#086c79}summary{cursor:pointer;font-weight:650;color:#086c79;padding:6px 0}details{margin:15px 0}.role{font-size:12px;text-transform:uppercase;color:#086c79;letter-spacing:.08em}.message{padding:15px;border-left:3px solid #9fbac6;background:#f6f9fa;margin:12px 0}.response{max-height:460px;overflow:auto;padding:16px;background:#f6f9fa;border-radius:8px}.callout{padding:16px;background:#fff4d9;border-left:4px solid #ba8b28}.metric{font-weight:700;font-size:24px;color:#086c79}.deleted{background:#fbe2df;text-decoration:line-through;color:#8b2824}.added{background:#dff2e6;color:#175e34}.bar{height:9px;background:#dce8ed;width:120px;border-radius:5px;overflow:hidden}.bar span{display:block;height:100%;background:#086c79}.report{max-height:600px;overflow:auto}.field{font:12px ui-monospace,monospace;color:#536b78}.answer{padding:6px 10px;border-radius:6px;background:#eaf1f4;display:inline-block;font-size:13px}
@media(max-width:800px){main{padding:22px 12px}.grid{grid-template-columns:1fr}h1{font-size:30px}.panel,.card{padding:16px}}
</style></head><body><main>
<div class="eyebrow">A field guide · one scenario · one investigation · no new clustering</div>
<h1>What different parts of a CHIVE investigation could we cluster?</h1>
<p>Explore the actual stored data for <code id="pid"></code>. This keeps the example fixed while changing <em>what one clustering item would represent</em>. Nothing on this page runs an embedding model or a clustering algorithm.</p>
<div class="stats"><div class="stat"><strong id="score"></strong>verification score</div><div class="stat"><strong id="base-count"></strong>baseline “yes” responses</div><div class="stat"><strong id="exp-count"></strong>counterfactual experiments</div><div class="stat"><strong id="agent-count"></strong>investigator agent turns</div></div>
<nav class="nav"><a href="#options">Compare inputs</a><a href="#behaviour">Behaviour</a><a href="#prompt">Prompt</a><a href="#classifier">Classifier</a><a href="#edit">Edits</a><a href="#responses">Responses</a><a href="#effects">Measured effects</a><a href="#report">Investigator report</a></nav>
<p class="note">Medical claims below are verbatim dataset content and may be incorrect. They are shown to inspect model behaviour, not as medical guidance. Investigator interpretations are distinguished from measured outcomes.</p>

<section class="panel" id="options"><h2>The available choices</h2><div class="scroll"><table><thead><tr><th>Possible input</th><th>One item would be…</th><th>What similarity would emphasize</th><th>Important limit</th></tr></thead><tbody>
<tr><td class="option"><a href="#behaviour">1. Behaviour description</a> <span class="badge">Used already</span></td><td>One investigation</td><td>The response pattern described by the screener</td><td>Topic and scenario wording can dominate; the summary is descriptive prose.</td></tr>
<tr><td class="option"><a href="#edit">2. Cleaned edit description</a> <span class="badge">Used already</span></td><td>One measured large-effect experiment</td><td>The kind of prompt intervention that changed the rate</td><td>Can be incomplete after stripping; inspect the actual diff and rates.</td></tr>
<tr><td class="option"><a href="#edit">3. Exact deleted/added text</a></td><td>One experiment, represented by its before/after change</td><td>The literal change in framing, instructions, or content</td><td>Short spans need context. Keep deletion and addition separate and ordered.</td></tr>
<tr><td class="option"><a href="#responses">4. Actual Qwen response</a></td><td>One sampled response</td><td>Observable output wording, stance, format, or content</td><td>Much longer texts; each investigation contributes many correlated samples. Does not directly identify a cause.</td></tr>
<tr><td class="option"><a href="#classifier">5. Classifier question</a></td><td>One investigation’s measurement definition</td><td>Which behavioural property is being measured</td><td>The question is a definition, not evidence the behaviour happened.</td></tr>
<tr><td class="option"><a href="#effects">6. Measured effect profile</a></td><td>An experiment’s rates, or an investigation’s responses to comparable edits</td><td>Which edits increase, decrease, or leave a measured behaviour unchanged</td><td>Numeric features, not sentence embeddings. Across investigations, align edit families and classifier meaning first; experiment numbers are not shared axes.</td></tr>
<tr><td class="option"><a href="#prompt">7. Original prompt/context</a></td><td>One investigated conversation context</td><td>Scenario, request, topic, and conversation structure</td><td>Useful for grouping contexts, but outside the current observation-only behaviour/cause clustering objective.</td></tr>
<tr><td class="option"><a href="#report">Investigator explanations</a> <span class="badge excluded">Excluded by your rule</span></td><td>Interpretive prose</td><td>The investigator’s proposed causal story</td><td>Not measured ground truth. “Core causes” and “Final answer” stay excluded from clustering.</td></tr>
</tbody></table></div><p class="note">Edit role/location, edit size, verification score and scenario ID can be kept as structured metadata and used for filtering or cross-tabs, without appending them to embedding text.</p></section>

<section class="panel" id="behaviour"><h2>1. What behaviour was noticed?</h2><div class="grid"><div class="card"><h3>Original screener summary</h3><p class="field">behavior_summary</p><p class="text" id="behaviour-original"></p></div><div class="card"><h3>Existing cleaned behaviour input</h3><p class="field">behaviours.parquet → behaviour_text</p><p class="text" id="behaviour-cleaned"></p></div></div><details><summary>Other screening fields</summary><h3>Investigation question</h3><p class="text" id="investigation-question"></p><h3>Suggested counterfactual — a proposal, not proof it was tested</h3><p class="text" id="suggestion"></p><h3>Structured baseline description</h3><p class="text" id="baseline-description"></p></details></section>

<section class="panel" id="prompt"><h2>2. What did Qwen see?</h2><p class="field">messages[] → role + content</p><p>These are the input conversation messages. Previous assistant turns are part of the context; the fresh sampled responses appear later on this page.</p><div id="last-message"></div><details><summary>Read the full baseline context, including previous turns</summary><div id="baseline-context"></div></details></section>

<section class="panel" id="classifier"><h2>3. How was a response classified?</h2><p class="field">classifier_question + baseline_grader_verdicts[]</p><p class="text" id="classifier-text"></p><p id="labels-explanation"></p><details><summary>All 30 baseline response labels</summary><pre id="baseline-labels"></pre></details><p class="note">A label records the classifier’s judgement under this question. Verification score, behaviour rate, and classifier label are different quantities. A “yes” is not universally bad across investigations.</p></section>

<section class="panel" id="edit"><h2>4. What changed in the counterfactual?</h2><div class="controls"><label for="experiment">Experiment</label><select id="experiment"></select><span class="small">Start with experiment 12: a small change in who asserts the claim.</span></div><p class="metric" id="effect-headline"></p><p class="small" id="eligible-note"></p><div class="grid"><div class="card"><h3>Original experiment summary</h3><p class="field">structured_findings.experiment_summaries[].summary</p><p class="text" id="summary-original"></p><p class="note">May mix an edit description, an outcome, and interpretation. Do not embed it unchanged as causal evidence.</p></div><div class="card"><h3>Cleaned edit description</h3><p class="field">causes.parquet → cause_text</p><p class="text" id="summary-cleaned"></p><p class="small" id="correction-note"></p></div></div>
<h3>Exact change to the final message</h3><p class="note">Red marks removed text; green marks added text. This shows the final message only—other context messages may also change in other experiments. Expand the full edited context to inspect them.</p><div class="grid"><div class="card"><h3>Before</h3><p class="text" id="diff-before"></p></div><div class="card"><h3>After</h3><p class="text" id="diff-after"></p></div></div>
<details><summary>Literal changed spans, kept separate from unchanged text</summary><div id="spans"></div></details><details><summary>Structured edit metadata</summary><pre id="edit-metadata"></pre></details><details><summary>Full edited context</summary><div id="edited-context"></div></details></section>

<section class="panel" id="responses"><h2>5. What did Qwen actually output?</h2><p>Choose any saved response. The initial view deliberately selects a baseline “yes” and an edited-context “no” to illustrate the contrast. These are independent samples, not a matched before/after response pair.</p><div class="grid"><div class="card"><h3>Baseline sample</h3><div class="controls"><select id="baseline-response" aria-label="Baseline response"></select><span class="answer" id="baseline-label"></span></div><pre class="response" id="baseline-output"></pre></div><div class="card"><h3>Selected experiment sample</h3><div class="controls"><select id="counter-response" aria-label="Counterfactual response"></select><span class="answer" id="counter-label"></span></div><pre class="response" id="counter-output"></pre></div></div><p class="note">Embedding these would produce one item per completion, rather than one item per investigation or experiment. Individual response clusters would answer a different question from clustering the interventions.</p></section>

<section class="panel" id="effects"><h2>6. What do the measured outcomes look like?</h2><p>This is the full measured profile for all 13 experiments in this investigation, including small-effect experiments that our current cause clustering excludes. Rates are recomputed from saved response-level labels.</p><div class="scroll"><table><thead><tr><th>Experiment</th><th>Baseline yes</th><th>Edited yes</th><th>Change (pp)</th><th>Current cause-clustering status</th></tr></thead><tbody id="effects-table"></tbody></table></div><p class="callout">Across investigations, “experiment 5” does not refer to the same intervention. To cluster whole effect profiles, first align genuinely comparable edit families and account for classifier direction and missing experiments. Never fill an untested edit with zero effect.</p></section>

<section class="panel" id="report"><h2>7. Where is the investigator’s explanation?</h2><span class="badge excluded">Available for inspection · excluded from clustering</span><p>The report records the investigator’s narration of hypotheses, tests, and interpretations. It can help you navigate the experiments, but its explanation is not ground truth. “Core causes” and “Final answer” are not proposed as clustering features here.</p><details><summary>Read the saved investigator report</summary><pre class="report" id="investigator-report"></pre></details><details><summary>Verification assessment and issues</summary><pre class="report" id="verification"></pre></details></section>
<section class="panel"><h2>Files and scope</h2><p>One investigation from the medical-authority scenario. This page is a field-by-field example, not a new clustering result. All displayed text and labels are stored data; the diff is computed mechanically. The existing cleaned edit for experiment 12 omits one residual interpretation sentence, with that correction shown above.</p><p><a href="examples/investigation_parts_example.json">Download the example data</a> · <a href="results/scenario_clustering/related_health_counterfactuals/review.html">Previous counterfactual clustering</a> · <a href="examples/petri_medical_authority_sycophancy_e2_t1_to_t5.html">Conversation checkpoints t1–t5</a></p></section>
<script type="application/json" id="data">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const write=(id,text)=>$(id).textContent=text;
function node(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e}
const current=()=>D.experiments.find(e=>e.exp_num===Number($('experiment').value));
function message(m){const e=node('div',undefined,'message');e.append(node('div',m.role,'role'),node('p',m.content,'text'));return e}
function selectResponses(id,labels,preferred){const s=$(id);s.replaceChildren();labels.forEach((label,i)=>{const o=node('option',`Response ${i+1} · ${label?'YES':'NO'}`);o.value=i;s.append(o)});const first=labels.indexOf(preferred);s.value=String(first<0?0:first)}
function baselineResponse(){const i=Number($('baseline-response').value);write('baseline-output',D.completions[i]);write('baseline-label',D.baseline_labels[i]?'Classifier: YES':'Classifier: NO')}
function counterResponse(){const e=current(),i=Number($('counter-response').value);write('counter-output',e.completions[i]);write('counter-label',e.grader_verdicts[i]?'Classifier: YES':'Classifier: NO')}
function metadata(e){const p=e.preprocessed;return {experiment_number:e.exp_num,is_key:e.summary?.is_key,edit_record:e.edit_record,...(p?{edit_type:p.edit_type,edit_location:p.edit_location,chars_changed:p.chars_changed,tokens_changed:p.tokens_changed,frac_changed:p.frac_changed,context_swap:p.context_swap,context_swap_relative:p.context_swap_relative,measurement_artifact:p.measurement_artifact,clustering_eligible:p.clustering_eligible,clustering_exclusion_reasons:p.clustering_exclusion_reasons}:{preprocessing:'No retained large-effect cause row; inspect the raw experiment and effect.'})}}
function renderExperiment(){const e=current(),p=e.preprocessed,b=D.baseline_yes/D.baseline_labels.length,c=e.yes_count_checked/e.grader_verdicts.length,delta=(c-b)*100;write('effect-headline',`${D.baseline_yes}/${D.baseline_labels.length} → ${e.yes_count_checked}/${e.grader_verdicts.length} yes · ${delta>=0?'+':''}${delta.toFixed(1)} percentage points`);write('eligible-note',p?.clustering_eligible?'Eligible in the current counterfactual clustering.':p?'Retained for audit, excluded from current clustering: '+p.clustering_exclusion_reasons.join(', '):'Not a retained large-effect cause row in the current preprocessing.');write('summary-original',e.summary?.summary??'No stored summary.');write('summary-cleaned',p?.cause_text??'No cleaned cause row retained for this experiment. No text has been invented.');write('correction-note',p?.text_correction?'Removed interpretation sentence: '+p.text_correction:'Cleaned text is existing preprocessing output; compare it with the exact edited context.');$('diff-before').replaceChildren();$('diff-after').replaceChildren();const spans=[];for(const part of e.final_message_diff){if(part.before)$('diff-before').append(node('span',part.before,part.operation==='equal'?'':'deleted'));if(part.after)$('diff-after').append(node('span',part.after,part.operation==='equal'?'':'added'));if(part.operation!=='equal')spans.push(part)}$('spans').replaceChildren();for(const p of spans){const row=node('div',undefined,'card');row.append(node('div',p.operation,'role'),node('p','Removed: '+JSON.stringify(p.before),'text'),node('p','Added: '+JSON.stringify(p.after),'text'));$('spans').append(row)}write('edit-metadata',JSON.stringify(metadata(e),null,2));$('edited-context').replaceChildren(...e.messages.map(message));selectResponses('counter-response',e.grader_verdicts,false);counterResponse()}
write('pid',D.prompt_id);write('score',`${D.verification.score}/10`);write('base-count',`${D.baseline_yes}/${D.baseline_labels.length}`);write('exp-count',D.experiments.length);write('agent-count',D.n_agent_turns);write('behaviour-original',D.behavior_summary);write('behaviour-cleaned',D.behaviour_text_cleaned);write('investigation-question',D.question);write('suggestion',D.suggested_counterfactual);write('baseline-description',D.baseline_behavior);write('classifier-text',D.classifier_question);write('labels-explanation',`${D.baseline_yes} of ${D.baseline_labels.length} baseline responses were labelled YES (${(100*D.baseline_yes/D.baseline_labels.length).toFixed(1)}%).`);write('baseline-labels',D.baseline_labels.map((v,i)=>`${i+1}: ${v?'YES':'NO'}`).join('\n'));write('investigator-report',D.investigation_report);write('verification',JSON.stringify(D.verification,null,2));$('last-message').append(message(D.messages[D.messages.length-1]));$('baseline-context').append(...D.messages.map(message));
for(const e of D.experiments){const o=node('option',`Experiment ${e.exp_num}`);o.value=e.exp_num;$('experiment').append(o);const tr=node('tr'),td=node('td'),button=node('button',String(e.exp_num));button.addEventListener('click',()=>{$('experiment').value=String(e.exp_num);renderExperiment();$('edit').scrollIntoView()});td.append(button);tr.append(td);const b=D.baseline_yes/D.baseline_labels.length,c=e.yes_count_checked/e.grader_verdicts.length;tr.append(node('td',`${D.baseline_yes}/${D.baseline_labels.length}`),node('td',`${e.yes_count_checked}/${e.grader_verdicts.length}`),node('td',`${c-b>=0?'+':''}${((c-b)*100).toFixed(1)}`),node('td',e.preprocessed?.clustering_eligible?'Eligible':e.preprocessed?'Excluded by flags':'Not retained'));$('effects-table').append(tr)}
$('experiment').value='12';selectResponses('baseline-response',D.baseline_labels,true);baselineResponse();renderExperiment();$('experiment').addEventListener('change',renderExperiment);$('baseline-response').addEventListener('change',baselineResponse);$('counter-response').addEventListener('change',counterResponse);
</script></main></body></html>'''
    doc = template.replace('__DATA__', safe)
    path = ROOT / 'example.html'; path.write_text(doc)
    js = doc.split('</script><script>')[1].split('</script>')[0]
    (ROOT / 'examples/investigation_parts_example.js').write_text(js)
    assert json.loads(safe) == payload and len(experiments) == 13
    assert sum(labels) == 21 and next(e for e in experiments if e['exp_num'] == 12)['yes_count_checked'] == 2
    for e in experiments:
        if e['preprocessed']:
            p = e['preprocessed']; assert abs(p['delta_pp'] - (e['yes_count_checked']/len(e['grader_verdicts']) - sum(labels)/len(labels))*100) < 1e-8
    validation = {'status':'passed','investigation':PID,'experiments':len(experiments),'baseline_samples':len(labels),
                  'experiment_samples':sum(len(e['completions']) for e in experiments),'rates_checked_against_labels':True,
                  'eligible_experiments':payload['eligible_experiment_numbers'],'embeddings_run':False,'clustering_run':False,
                  'html_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (ROOT / 'examples/investigation_parts_example_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation,indent=2)); print(path)


if __name__=='__main__': main()
