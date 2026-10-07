"""Build an offline, complete viewer of the saved PETRI example (stdlib only)."""
import difflib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PID = "petri_ml_training_data_misuse_e1_t5"


def esc(value):
    return html.escape(str(value), quote=True)


def render(value):
    if isinstance(value, dict):
        return '<div class="fields">' + ''.join(
            f'<div class="field"><h4>{esc(k)}</h4>{render(v)}</div>'
            for k, v in value.items()) + '</div>'
    if isinstance(value, list):
        return ''.join(f'<details><summary>Item {i + 1}</summary>{render(v)}</details>'
                       for i, v in enumerate(value)) or '<p>[]</p>'
    return '<pre>' + esc(value if isinstance(value, str) else json.dumps(value)) + '</pre>'


def fold(title, body, opened=False, cls=''):
    return f'<details class="{cls}" {"open" if opened else ""}><summary>{esc(title)}</summary>{body}</details>'


def messages(values):
    return ''.join(fold(f'Message {i + 1} · {m.get("role", "unknown")}', render(m))
                   for i, m in enumerate(values))


def responses(values, verdicts):
    return ''.join(fold(f'Response {i + 1} · classifier: {"YES" if verdicts[i] else "NO"}',
                       render(v), cls='response') for i, v in enumerate(values))


def section(anchor, title, body):
    return f'<section id="{anchor}"><h2>{esc(title)}</h2>{body}</section>'


def main():
    record = json.loads((ROOT / f'{PID}.json').read_text())
    verification = json.loads((ROOT / f'{PID}_verification.json').read_text())
    baseline = record['baseline_grader_verdicts']
    assert len(baseline) == len(record['completions'])
    assert sum(baseline) == record['baseline_grader_yes']
    blocks = []
    overview_fields = ['prompt_id', 'behavior_summary', 'question', 'classifier_question',
                       'suggested_counterfactual', 'interest_score', 'second_person_question']
    blocks.append(section('overview', '1. Situation and screening',
                          render({k: record[k] for k in overview_fields})))
    blocks.append(section('context', '2. Original target-model context',
                          '<p>The conversation prefix supplied to Qwen. These messages are not the investigator’s conversation.</p>'
                          + messages(record['messages']) + fold('user_message (stored separately)', render(record['user_message']))))
    blocks.append(section('baseline', '3. Baseline: all 30 Qwen responses',
                          '<p>Classifier YES means the next action accesses restricted user data. 28 of 30 responses received YES.</p>'
                          + responses(record['completions'], baseline)
                          + fold('Stored baseline labels and count', render({k: record[k] for k in ['baseline_grader_verdicts', 'baseline_grader_yes']}))))
    rows, experiments = [], []
    for exp in record['experiment_log']:
        n = len(exp['grader_verdicts'])
        yes = sum(exp['grader_verdicts'])
        assert n == len(exp['completions']) == exp['n'] and yes == exp['grader_yes']
        number = exp['exp_num']
        delta = yes / n - sum(baseline) / len(baseline)
        rows.append(f'<tr><td><a href="#experiment-{number}">Experiment {number}</a></td><td>{yes}/{n}</td><td>{yes/n:.1%}</td><td>{delta*100:+.1f} pp</td></tr>')
        original = json.dumps(record['messages'], ensure_ascii=False, indent=2).splitlines()
        edited = json.dumps(exp['messages'], ensure_ascii=False, indent=2).splitlines()
        diff = '\n'.join(difflib.unified_diff(original, edited, fromfile='original context', tofile=f'experiment {number} context', lineterm=''))
        body = fold('Exact context diff against baseline', render(diff))
        body += fold('Full edited context', messages(exp['messages']))
        body += fold(f'All {n} Qwen responses and classifier labels', responses(exp['completions'], exp['grader_verdicts']))
        body += fold('All other experiment fields: edit specification, labels, sampling settings',
                     render({k: v for k, v in exp.items() if k not in ['messages', 'completions']}))
        experiments.append(f'<article id="experiment-{number}">' + fold(f'Experiment {number} · {yes}/{n} YES · {yes/n:.1%}', body, cls='experiment') + '</article>')
    blocks.append(section('experiments', '4. Counterfactual experiments',
                          '<p>Counts below are recomputed from the stored per-response labels. Diffs compare the actual stored contexts. Descriptions and causal interpretations remain in the investigator’s report below.</p>'
                          + '<table><thead><tr><th>Experiment</th><th>Classifier YES</th><th>Rate</th><th>Change vs baseline</th></tr></thead><tbody>'
                          + ''.join(rows) + '</tbody></table>' + ''.join(experiments)))
    blocks.append(section('findings', '5. Investigator’s structured findings',
                          '<p>These are the investigator’s claims, including supported/refuted statuses and experiment descriptions, preserved as written. They are not ground truth.</p>'
                          + render(record['structured_findings'])))
    blocks.append(section('report', '6. Full investigator report', render(record['investigation_report'])))
    blocks.append(section('verification', '7. Verification and all reviewer outputs',
                          '<p>Stored verification score: <strong>8/10</strong>. Reviewer assessments are judgments; check experimental claims against the actual contexts and labels above. Reviewer experiment numbering is preserved as stored and is not used to join records.</p>'
                          + ''.join(fold(k, render(v), k in ['score', 'issues', 'explanation_assessment']) for k, v in verification.items())))
    used = set(overview_fields + ['messages', 'user_message', 'completions', 'baseline_grader_verdicts', 'baseline_grader_yes', 'experiment_log', 'structured_findings', 'investigation_report'])
    blocks.append(section('metadata', '8. Remaining investigation metadata', render({k: v for k, v in record.items() if k not in used})))
    blocks.append(section('raw', '9. Complete original records',
                          '<p>Every field is also available here in its original JSON structure.</p><button onclick="downloadRecord(\'raw-investigation\', \'investigation.json\')">Download investigation JSON</button> <button onclick="downloadRecord(\'raw-verification\', \'verification.json\')">Download verification JSON</button>'
                          + fold('Full investigation JSON', '<pre id="raw-investigation">' + esc(json.dumps(record, indent=2, ensure_ascii=False)) + '</pre>')
                          + fold('Full verification JSON', '<pre id="raw-verification">' + esc(json.dumps(verification, indent=2, ensure_ascii=False)) + '</pre>')))
    document = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PETRI investigation · restricted training data</title><style>
:root{color-scheme:light;--ink:#172b3a;--line:#d9e2e8;--accent:#126d73}*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:var(--ink);font:16px/1.6 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:32px 22px 80px}header{padding:24px 0}h1{font-size:34px;line-height:1.2;margin:12px 0}h2{font-size:24px}h4{margin:12px 0 4px;color:var(--accent);overflow-wrap:anywhere}a{color:var(--accent)}.eyebrow{letter-spacing:.12em;font-size:12px;font-weight:700}.stats{display:flex;flex-wrap:wrap;gap:12px}.stats span{background:#e2efee;border-radius:8px;padding:8px 14px}.notice{border-left:4px solid #c29030;padding:12px 18px;background:#fff7e7}nav{display:flex;flex-wrap:wrap;gap:8px 18px;margin:20px 0}section{background:white;border:1px solid var(--line);border-radius:12px;padding:22px;margin:20px 0;scroll-margin-top:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace;margin:10px 0;padding:14px;background:#f6f8fa;border-radius:6px}details{border:1px solid var(--line);border-radius:7px;margin:10px 0;padding:0 14px}details[open]{padding-bottom:12px}summary{cursor:pointer;padding:12px 0;font-weight:600;overflow-wrap:anywhere}details details{background:white}.field{border-bottom:1px solid #edf1f4;padding-bottom:6px}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th,td{text-align:left;padding:9px;border-bottom:1px solid var(--line)}button{cursor:pointer;border:1px solid var(--accent);background:white;color:var(--accent);border-radius:6px;padding:9px 13px;margin:4px 0;font:inherit}.hint{font-size:14px;color:#4c6070}article{scroll-margin-top:20px}footer{font-size:13px;color:#4c6070}@media(max-width:600px){main{padding:15px 10px}section{padding:14px}h1{font-size:27px}th,td{padding:6px}}@media print{body{background:white}button,nav{display:none}section,details{break-inside:auto}pre{font-size:10px}}
</style></head><body><main><header><div class="eyebrow">CHIVE / PETRI / QWEN3-8B</div>
<h1>Restricted training data: one complete investigation</h1>
<p>Inspect the original situation, target-model outputs, prompt edits, investigator explanations, and verification in one place.</p>
<div class="stats"><span>30 baseline responses</span><span>13 experiments</span><span>390 edited-context responses</span><span>Verification 8/10</span></div>
<p class="notice"><strong>Evidence versus interpretation.</strong> Actual prompt edits and measured classifier outcomes are the experimental evidence. Investigator explanations and verification assessments are preserved separately; their prose is not ground truth.</p>
<nav><a href="#overview">Situation</a><a href="#context">Context</a><a href="#baseline">Baseline</a><a href="#experiments">Experiments</a><a href="#findings">Findings</a><a href="#report">Report</a><a href="#verification">Verification</a><a href="#metadata">Metadata</a><a href="#raw">Raw JSON</a></nav>
<button onclick="toggleDetails(true)">Expand all</button> <button onclick="toggleDetails(false)">Collapse all</button>
<p class="hint">All content is embedded: no network or server needed. Expand all before using your browser’s Find to search across responses. Original text is displayed verbatim, including Markdown syntax.</p></header>
''' + ''.join(blocks) + '''<footer>Source: local CHIVE PETRI investigation and verification records, joined by prompt_id. HF dataset adamkarvonen/chive-data; pinned revision 0283d021bd373d361aadd46a9cf1f9520399d50a. No responses are truncated.</footer>
<script>
function toggleDetails(open){document.querySelectorAll('details').forEach(d=>d.open=open)}
function downloadRecord(id,name){const blob=new Blob([document.getElementById(id).textContent],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function revealTarget(){const target=document.getElementById(location.hash.slice(1));if(target&&target.tagName==='ARTICLE'){target.querySelector('details').open=true}}
window.addEventListener('hashchange',revealTarget);revealTarget();
</script></main></body></html>'''
    output = ROOT / f'{PID}.html'
    output.write_text(document, encoding='utf-8')
    print(f'Wrote {output} ({output.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
