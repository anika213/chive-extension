"""Render the five stored prefixes of one PETRI audit conversation verbatim."""
import hashlib
import html
import json
import difflib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/seed_provenance/petri_prompts.jsonl'
PREFIX = 'petri_medical_authority_sycophancy_e2'
OUT = Path(__file__).resolve().parent


def measured(labels):
    assert isinstance(labels, list) and labels and all(isinstance(v, bool) for v in labels)
    return f'{sum(labels)}/{len(labels)}', sum(labels) / len(labels)


def investigation_panel(record, score, turn):
    if record is None:
        return '<div class="investigation"><h3>CHIVE investigation at this checkpoint</h3><p>No matching investigation is present in the released investigation file for t1. The prompt exists; the available records do not establish why an investigation is absent.</p></div>'
    baseline, baseline_rate = measured(record['baseline_grader_verdicts'])
    summaries = {r['experiment_number']:r for r in record['structured_findings']['experiment_summaries']}
    experiments = record['experiment_log']
    cards = []
    for experiment in experiments:
        number = experiment['exp_num']; rate, fraction = measured(experiment['grader_verdicts'])
        before = json.dumps(record['messages'], indent=2, ensure_ascii=False).splitlines()
        after = json.dumps(experiment['messages'], indent=2, ensure_ascii=False).splitlines()
        diff = '\n'.join(difflib.unified_diff(before, after, fromfile='Baseline context', tofile=f'Experiment {number} context', lineterm=''))
        summary = summaries.get(number, {}).get('summary', 'No structured experiment summary stored.')
        sample = experiment['completions'][0] if experiment['completions'] else None
        if sample is not None: assert isinstance(sample, str)
        cards.append(f'<details class="experiment"><summary>Experiment {number} · {baseline} → {rate} · {(fraction-baseline_rate)*100:+.1f} percentage points</summary>'
                     f'<p class="note">Investigator-authored summary, verbatim; causal conclusions here are interpretations.</p><div class="verbatim">{esc(summary)}</div>'
                     f'<details><summary>Exact stored context changes</summary><pre class="diff">{esc(diff)}</pre></details>'
                     + (f'<details><summary>First sampled target response from this experiment · classifier label {experiment["grader_verdicts"][0]}</summary><div class="verbatim">{esc(sample)}</div></details>' if sample is not None else '') + '</details>')
    return (f'<div class="investigation" id="investigation-t{turn}"><h3>CHIVE’s agentic investigation of t{turn}</h3>'
            f'<p><strong>{record["n_agent_turns"]} recorded investigator turns · {len(experiments)} counterfactual experiments · verification score {score}/10</strong></p>'
            '<p class="note">These investigator turns are separate from the PETRI t1–t5 checkpoints. An investigator turn can launch multiple experiments.</p>'
            '<h4>Behaviour selected for investigation</h4><div class="verbatim">'+esc(record['behavior_summary'])+'</div>'
            '<h4>The fixed classifier question</h4><div class="verbatim">'+esc(record['classifier_question'])+'</div>'
            f'<p><strong>Baseline: {baseline} responses labelled YES.</strong> The YES definition is the question above and differs across checkpoints. Rates below are recomputed from stored per-response labels.</p>'
            '<details class="agent-report"><summary>Read the investigator’s saved report: hypotheses, tests, and interpretation</summary>'
            '<p class="note">Verbatim investigator prose, including its conclusions. This is a saved report, not a reconstructed API/tool-call transcript; the experiment log below supplies the stored test inputs, outputs, and classifier labels. Prose is not treated as ground truth.</p>'
            '<div class="verbatim">'+esc(record['investigation_report'])+'</div></details>'
            '<h4>Counterfactual experiment log</h4><p>Edit the context → sample Qwen3-8B again → apply the same classifier → compare with baseline.</p>'
            + ''.join(cards) + '</div>')


def esc(value):
    return html.escape(str(value), quote=True)


def message_card(message, index, note=''):
    role = message['role']
    assert role in ('system', 'user', 'assistant')
    return (f'<article class="message {role}"><header><span class="role">{esc(role)}</span>'
            f'<span>Message {index + 1}{esc(" · " + note) if note else ""}</span></header>'
            f'<div class="verbatim">{esc(message["content"])}</div></article>')


def main():
    rows = []
    for line in SOURCE.open():
        row = json.loads(line)
        if row['id'].startswith(PREFIX + '_t'):
            rows.append(row)
    rows.sort(key=lambda r: r['n_turns'])
    assert [r['id'] for r in rows] == [PREFIX + f'_t{i}' for i in range(1, 6)]
    for i, row in enumerate(rows):
        assert row['messages'][-1]['role'] == 'user'
        assert len(row['messages']) == 2 * (i + 1)
        if i:
            previous = rows[i - 1]['messages']
            assert row['messages'][:len(previous)] == previous
            assert [m['role'] for m in row['messages'][len(previous):]] == ['assistant', 'user']
    investigations = {}; scores = {}
    source_dir = ROOT / 'data/petri/qwen3_8b_petri_eval'
    for line in (source_dir / 'investigations.jsonl').open():
        record = json.loads(line)
        if record['prompt_id'].startswith(PREFIX + '_t'):
            investigations[record['prompt_id']] = record
    for line in (source_dir / 'verification.jsonl').open():
        record = json.loads(line)
        if record['prompt_id'] in investigations: scores[record['prompt_id']] = record['score']
    for row in rows:
        if row['id'] in investigations: assert row['messages'] == investigations[row['id']]['messages']
    (OUT / (PREFIX + '_investigations.json')).write_text(json.dumps({'investigations': investigations, 'verification_scores': scores}, indent=2, ensure_ascii=False) + '\n')
    payload = {'scenario_id': 'medical_authority_sycophancy', 'run_id': 'e2',
               'source_file': str(SOURCE.relative_to(ROOT)),
               'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               'dataset_revision': '0283d021bd373d361aadd46a9cf1f9520399d50a', 'checkpoints': rows}
    (OUT / (PREFIX + '_t1_to_t5.json')).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    nav = ''.join(f'<a href="#t{i}" aria-label="Checkpoint t{i}">t{i}</a>' for i in range(1, 6))
    sections = []
    for i, row in enumerate(rows, 1):
        messages = row['messages']
        inherited = len(rows[i - 2]['messages']) if i > 1 else 0
        if i == 1:
            new = '<details class="prior"><summary>System prompt · present at every checkpoint</summary>' + message_card(messages[0], 0) + '</details>'
            intro = 'The conversation starts here: system instructions and the first user question.'
        else:
            new = (f'<details class="prior"><summary>New since t{i-1}: the stored assistant answer to question {i-1}</summary>'
                   + message_card(messages[-2], len(messages)-2, 'historical answer in the PETRI conversation') + '</details>')
            intro = (f'All {inherited} messages from t{i-1} are preserved unchanged. '
                     f'The conversation adds the assistant’s answer to question {i-1}, then user question {i}.')
        new += '<h3>Latest user message · the question CHIVE samples an answer to</h3>' + message_card(messages[-1], len(messages)-1)
        full = ''.join(message_card(message, index, 'inherited' if index < inherited else 'added at this checkpoint') for index, message in enumerate(messages))
        sequence = ' → '.join(['System'] + [label for j in range(1, i+1) for label in ([f'User {j}', f'Assistant {j}'] if j < i else [f'User {j}'])] + [f'Next: assistant {i}'])
        sections.append(f'<section id="t{i}" class="checkpoint"><div class="section-head"><h2>t{i}</h2><span class="pill">{len(messages)} context messages · {i-1} prior answers</span></div>'
                        f'<code class="identifier">{esc(row["id"])}</code><p>{esc(intro)}</p><p class="sequence">{esc(sequence)}</p>'
                        + new + f'<div class="next">The snapshot stops here, before assistant response {i}. CHIVE generates fresh responses to this complete context.</div>'
                        + f'<details class="full"><summary>Read the full t{i} context · all {len(messages)} messages, verbatim</summary>{full}</details>'
                        + investigation_panel(investigations.get(row['id']), scores.get(row['id']), i) + '</section>')
    style = '''
    :root{color-scheme:light;--ink:#173044;--muted:#536574;--line:#d3dfe5;--teal:#086c79}
    *{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:86px}
    body{margin:0;background:#f3f6f8;color:var(--ink);font:16px/1.65 system-ui,-apple-system,sans-serif}
    main{max-width:1040px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(26px,4vw,40px);line-height:1.2;margin:8px 0 16px}
    h2{font-size:30px;margin:0}h3{font-size:16px;margin:22px 0 10px}p{margin:12px 0}
    .eyebrow{text-transform:uppercase;letter-spacing:.12em;color:var(--teal);font-size:12px;font-weight:700}
    .lead{font-size:19px;max-width:850px}.note{color:var(--muted);font-size:14px}.intro{padding-bottom:18px}
    nav{position:sticky;top:0;z-index:5;background:#f3f6f8f5;backdrop-filter:blur(8px);padding:13px 0;display:flex;align-items:center;gap:10px;border-bottom:1px solid var(--line)}
    nav span{margin-right:auto;font-size:14px}nav a{display:inline-block;padding:7px 16px;border:1px solid var(--line);border-radius:8px;background:white;text-decoration:none;font-weight:650;color:var(--teal)}
    nav a:hover,nav a:focus-visible{background:#e1f0f2;border-color:var(--teal)}
    .checkpoint{background:white;border:1px solid var(--line);border-radius:14px;padding:26px;margin-top:28px}
    .section-head{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
    .pill{font-size:13px;background:#eaf1f4;padding:5px 12px;border-radius:20px}.identifier{display:block;margin-top:10px;overflow-wrap:anywhere;font-size:13px;color:var(--muted)}
    .sequence{font-size:13px;padding:12px 14px;background:#f3f6f8;border-radius:6px;overflow-wrap:anywhere}
    .message{border:1px solid var(--line);border-left:4px solid #8798a4;border-radius:8px;overflow:hidden;margin:12px 0;background:#fff}
    .message.user{border-left-color:#087c78}.message.assistant{border-left-color:#9564ae}
    .message header{display:flex;gap:16px;justify-content:space-between;align-items:center;padding:9px 16px;background:#f5f8fa;color:var(--muted);font-size:12px}
    .role{text-transform:uppercase;font-size:11px;letter-spacing:.1em;font-weight:800;color:var(--ink)}
    .verbatim{padding:18px;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.65}
    details{margin:14px 0}summary{cursor:pointer;color:var(--teal);font-weight:600;padding:10px 0}summary:focus-visible{outline:2px solid var(--teal)}
    .prior{background:#f7f9fa;border:1px solid var(--line);padding:0 14px;border-radius:8px}
    .full{border-top:1px solid var(--line);padding-top:10px;margin-top:22px}.next{margin-top:18px;padding:12px 15px;border-radius:6px;background:#eaf5f3;color:#225b53;font-size:14px}
    a{color:var(--teal)}footer{margin-top:28px;color:var(--muted);font-size:13px}
    .investigation{border-top:3px solid #086c79;margin-top:28px;padding-top:15px}.investigation>h3{font-size:22px;margin-top:4px}
    .investigation h4{margin:20px 0 4px}.investigation>.verbatim{background:#f5f8fa;border-radius:6px;padding:14px}
    .experiment,.agent-report{border:1px solid var(--line);border-radius:8px;padding:0 14px}.agent-report{background:#f0f6f7}
    .diff{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.6 ui-monospace,monospace;background:#f5f8fa;padding:14px}
    @media(max-width:600px){main{padding:20px 12px}.checkpoint{padding:18px 14px}nav a{padding:6px 11px}nav span{font-size:12px}.message header{align-items:flex-start}.verbatim{padding:14px}}
    @media print{nav{position:static}body{background:white}.checkpoint{break-before:page;box-shadow:none}.checkpoint:first-of-type{break-before:auto}details:not([open]){display:none}}
    '''
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>Medical authority sycophancy · e2 · t1–t5</title><style>'+style+'</style></head><body><main>'
                '<div class="intro"><div class="eyebrow">PETRI · one conversation · run e2</div><h1>What changes from t1 to t5?</h1>'
                '<p class="lead">Each checkpoint preserves the previous conversation and adds one assistant answer plus one new user question. These are five snapshots of the same conversation.</p>'
                '<p><strong>Two stages:</strong> PETRI generates the conversation. CHIVE then takes a checkpoint and runs a separate agentic investigation: propose edits, test them on Qwen3-8B, classify responses, and refine its hypotheses.</p>'
                '<p>Below each checkpoint, the teal divider starts its CHIVE investigation. Open the investigator report to see its written process, or expand experiments to inspect actual edits and measured results. Released investigations exist for <a href="#investigation-t2">t2</a>, <a href="#investigation-t3">t3</a>, <a href="#investigation-t4">t4</a>, and <a href="#investigation-t5">t5</a>.</p>'
                '<p><strong>Scenario:</strong> <code>medical_authority_sycophancy</code><br><strong>Run:</strong> <code>e2</code> · the user introduces themselves as Dr. Fischler.</p>'
                '<p class="note">The messages below are reproduced verbatim from the experimental dataset, including Markdown, misleading medical claims, and model errors. Prior assistant answers are the stored PETRI conversation history; they are not the 30 newly sampled CHIVE responses.</p></div>'
                '<nav aria-label="Jump to checkpoint"><span>Jump to checkpoint</span>'+nav+'</nav>'
                +''.join(sections)+f'<footer>Source: pinned CHIVE Qwen3-8B PETRI prompt records, revision <code>{payload["dataset_revision"]}</code>. '
                f'<a href="{PREFIX}_t1_to_t5.json">Download all five original prompt records (JSON)</a>. '
                f'<a href="{PREFIX}_investigations.json">Download the four full investigation records, including all sampled responses and labels (JSON)</a>. '
                'The four transitions were checked: every earlier message is identical in the next checkpoint.</footer></main></body></html>')
    path = OUT / (PREFIX + '_t1_to_t5.html')
    path.write_text(document)
    # Ensure every stored message has an exact, escaped representation on the page.
    for row in rows:
        for message in row['messages']:
            assert esc(message['content']) in document
    from html.parser import HTMLParser
    class Check(HTMLParser):
        def __init__(self): super().__init__(); self.sections=[]; self.targets=[];self.ids=[]
        def handle_starttag(self, tag, attrs):
            a=dict(attrs)
            if a.get('id'):self.ids.append(a['id'])
            if tag=='section':self.sections.append(a.get('id'))
            if tag=='a':self.targets.append(a['href'])
    check=Check();check.feed(document)
    assert check.sections == ['t1','t2','t3','t4','t5']
    assert all(target[1:] in check.ids if target.startswith('#') else (OUT/target).exists() for target in check.targets)
    assert len(investigations)==4
    for record in investigations.values():assert esc(record['investigation_report']) in document
    print(f'Saved {path}\nValidated five checkpoints, four identical-prefix transitions, four matched investigations, verbatim message/report content, stored-label rates, and links.')


if __name__ == '__main__': main()
