"""Every item in a selected latest-run PETRI behaviour cluster and its cause rows."""
from collections import Counter
import html
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/petri_latest_qualified'
OUT = RUN / 'cluster_5_behaviours_ward'


def esc(value): return html.escape(str(value), quote=True)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=['ward','hdbscan'],default='hdbscan')
    parser.add_argument('--cluster-id',type=int,default=5)
    args=parser.parse_args();method=args.method;cluster_id=args.cluster_id
    global OUT
    OUT=RUN/f'cluster_{cluster_id}_behaviours_{method}'
    title_method='Ward' if method=='ward' else 'HDBSCAN'
    OUT.mkdir(exist_ok=True)
    assignments = pq.read_table(RUN / 'petri/behaviours/assignments.parquet')
    members = assignments.filter(pc.equal(assignments[method+'_cluster_id'], cluster_id))
    ids = members['prompt_id'].to_pylist()
    sizes=json.loads((RUN/'petri/behaviours/cluster_sizes.json').read_text())[method]
    expected=next(r['size'] for r in sizes if r['cluster_id']==cluster_id)
    assert len(ids)==expected and len(set(ids))==len(ids)
    table = pq.read_table(ROOT / 'data/petri_latest_qualified/petri/causes.parquet')
    causes = table.filter(pc.is_in(table['prompt_id'], value_set=pa.array(ids)))
    rows = causes.to_pylist()
    n_eligible=sum(r['clustering_eligible'] for r in rows)
    pq.write_table(causes, OUT / 'causes.parquet', compression='zstd')
    pq.write_table(members, OUT / 'behaviours.parquet', compression='zstd')
    # The question makes clear what a positive rate means for each investigation.
    questions = {}
    raw = ROOT / 'data/petri/qwen3_8b_petri_eval/investigations.jsonl'
    for line in raw.open():
        record = json.loads(line)
        if record['prompt_id'] in ids:
            questions[record['prompt_id']] = record['classifier_question']
    assert set(questions) == set(ids)
    groups = []
    for member in members.to_pylist():
        pid = member['prompt_id']
        matched = sorted([r for r in rows if r['prompt_id'] == pid], key=lambda r:r['experiment_number'])
        groups.append({'behaviour':member,'classifier_question':questions[pid],'causes':matched})
    (OUT / 'all_items.json').write_text(json.dumps(groups, indent=2, ensure_ascii=False) + '\n')
    sections = []
    navigation = []
    for number, group in enumerate(groups, 1):
        member, matched = group['behaviour'], group['causes']
        pid = member['prompt_id']; eligible = sum(r['clustering_eligible'] for r in matched)
        navigation.append(f'<a href="#item-{number}">Item {number} <span>{len(matched)} causes</span></a>')
        cards = []
        for cause in matched:
            status = 'Eligible' if cause['clustering_eligible'] else 'Excluded'
            reasons = ', '.join(cause['clustering_exclusion_reasons'])
            cards.append(f'<article class="cause {status.lower()}" data-eligible="{str(cause["clustering_eligible"]).lower()}">'
                f'<div class="cause-heading"><h4>Experiment {cause["experiment_number"]}</h4><span class="badge {status.lower()}">{status}</span></div>'
                f'<div class="metrics"><span>Baseline <strong>{cause["baseline_rate"]:.1%}</strong></span><span>Counterfactual <strong>{cause["counterfactual_rate"]:.1%}</strong></span>'
                f'<span>Change <strong>{cause["delta_pp"]:+.1f} pp</strong></span><span>Edit: {esc(cause["edit_type"])}</span><span>is_key: {str(cause["is_key"]).lower()}</span></div>'
                + (f'<p class="exclusion">Exclusion: <code>{esc(reasons)}</code></p>' if reasons else '')
                + '<p class="field">cause_text_original</p><div class="original">'+esc(cause['cause_text_original'])+'</div>'
                + '<details><summary>Preprocessed cause text</summary><div class="original">'+esc(cause['cause_text'])+'</div></details>'
                + '<details><summary>All fields in this parquet row</summary><pre>'+esc(json.dumps(cause,indent=2,ensure_ascii=False))+'</pre></details></article>')
        sections.append(f'<section id="item-{number}" class="item"><header class="item-heading"><span class="number">{number}</span><div><h2>Investigation {number} of {len(ids)}</h2><code>{esc(pid)}</code></div></header>'
            '<div class="columns"><div class="behaviour"><h3>Behaviour cluster member</h3><p class="field">behaviour_text · exact clustering input</p>'
            '<div class="original">'+esc(member['behaviour_text'])+'</div>'
            f'<p class="score">Verification score: <strong>{member["verif_score"]}/10</strong></p>'
            '<details><summary>Classifier question · what the rates count</summary><div class="original">'+esc(group['classifier_question'])+'</div></details>'
            '<p class="note">The rates on the right count YES answers to this question. Their direction can differ across investigations.</p></div>'
            f'<div class="causes"><h3>Matching causes.parquet rows</h3><p class="note">{len(matched)} rows · {eligible} eligible · {len(matched)-eligible} excluded. Every matching row is shown, ordered by experiment number.</p>'
            + (''.join(cards) if cards else '<p>No matching row remains in causes.parquet after its effect filtering and exact-text deduplication. This investigation is still a member of the behaviour cluster.</p>') + '</div></div></section>')
    style = '''
    :root{--ink:#1d3444;--muted:#5d6e7a;--line:#cfdae0;--teal:#086c79;--bg:#f3f6f8}
    *{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:85px}body{margin:0;color:var(--ink);background:var(--bg);font:15px/1.65 system-ui,sans-serif}
    main{max-width:1540px;margin:auto;padding:30px 24px 65px}h1{font-size:clamp(26px,3vw,38px);line-height:1.2;margin:8px 0 14px}h2{font-size:21px;margin:0}h3{font-size:17px;margin:0 0 12px}h4{font-size:16px;margin:0}
    .eyebrow,.field{text-transform:uppercase;font-size:11px;letter-spacing:.09em;font-weight:700;color:var(--muted)}.eyebrow{color:var(--teal)}.intro{max-width:1100px}.intro p{margin:12px 0}
    nav{position:sticky;top:0;z-index:3;background:#f3f6f8f5;backdrop-filter:blur(10px);display:flex;flex-wrap:nowrap;overflow-x:auto;gap:10px;padding:14px 0;border-bottom:1px solid var(--line)}nav a{white-space:nowrap}
    nav a{background:white;border:1px solid var(--line);border-radius:7px;padding:7px 14px;text-decoration:none;font-weight:650}nav span{font-size:12px;font-weight:400;color:var(--muted);margin-left:10px}
    a{color:var(--teal)}a:focus-visible,summary:focus-visible{outline:2px solid var(--teal)}
    .item{margin-top:28px;background:white;border:1px solid var(--line);border-radius:12px;overflow:hidden}.item-heading{display:flex;align-items:center;gap:15px;padding:20px 24px;background:#eaf1f4;border-bottom:1px solid var(--line)}
    .item-heading code{font-size:12px;overflow-wrap:anywhere}.number{background:var(--teal);color:white;border-radius:50%;width:36px;height:36px;display:grid;place-items:center;flex-shrink:0;font-weight:700}
    .columns{display:grid;grid-template-columns:minmax(280px, .85fr) minmax(450px,1.6fr);gap:28px;padding:24px}.behaviour{align-self:start;position:sticky;top:90px}.causes{min-width:0}
    .original{white-space:pre-wrap;overflow-wrap:anywhere}.behaviour>.original{font-size:16px}.field{margin:8px 0}.note{font-size:13px;color:var(--muted)}.score{border-top:1px solid var(--line);padding-top:14px}
    .cause{border:1px solid var(--line);border-left:4px solid #368779;border-radius:8px;margin:15px 0;padding:16px}.cause.excluded{border-left-color:#b27c30;background:#fffcf7}
    .cause-heading{display:flex;align-items:center;justify-content:space-between;gap:10px}.badge{font-size:11px;font-weight:700;padding:3px 9px;border-radius:14px}.badge.eligible{background:#e2f3ec;color:#24594a}.badge.excluded{background:#f9ecd7;color:#80520e}
    .metrics{display:flex;flex-wrap:wrap;gap:4px 18px;font-size:12px;color:var(--muted);margin:12px 0}.metrics strong{color:var(--ink)}.exclusion{font-size:12px;overflow-wrap:anywhere;color:#80520e}
    details{margin:12px 0 0}summary{cursor:pointer;color:var(--teal);font-size:13px;font-weight:650}details>.original{padding-top:12px}pre{font-size:11px;white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;padding:12px;border-radius:6px}
    footer{margin-top:26px;font-size:13px;color:var(--muted)}
    @media(max-width:850px){main{padding:20px 12px}.columns{grid-template-columns:1fr;gap:22px;padding:18px}.behaviour{position:static}.item-heading{padding:16px}nav a{padding:6px 10px}nav span{display:none}}
    @media print{nav{position:static}.behaviour{position:static}.columns{display:block}.item{break-before:page}.cause{break-inside:avoid}body{background:white}}
    '''
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>PETRI {title_method} behaviour cluster {cluster_id} · all members and causes</title><style>'+style+'</style></head><body><main>'
        f'<div class="intro"><div class="eyebrow">PETRI · behaviours · {title_method} · cluster {cluster_id}</div><h1>All cluster members, alongside their cause rows</h1>'
        f'<p><strong>{len(ids)} investigations · {len(rows)} matching cause rows · {n_eligible} eligible, {len(rows)-n_eligible} excluded.</strong> This is {title_method} behaviour cluster {cluster_id} from the latest qualifying-checkpoint run of 269 investigations. Cluster IDs are specific to the method and facet.</p>'
        '<p>The left column shows each behaviour exactly as clustered. The right column shows the original investigator-authored edit summaries from <code>causes.parquet</code>, measured rates, and filter status. These summaries may include causal interpretations; the measured outcomes are the evidence.</p>'
        '<p>This view uses <code>causes.parquet</code> only for causes. No “Core causes” or “Final answer” narrative has been added, and no cluster label has been assigned.</p>'
        f'<p><a href="../review.html">Back to clustered samples</a> · <a href="causes.parquet">Download these {len(rows)} cause rows</a> · <a href="behaviours.parquet">Download all {len(ids)} behaviour members</a> · <a href="all_items.json">Combined JSON</a></p></div>'
        '<nav aria-label="Cluster members">'+''.join(navigation)+'</nav>'+''.join(sections)
        + '<footer>Joined by prompt_id within PETRI. Experiment numbers need not be consecutive: causes.parquet retains substantial measured changes after earlier preprocessing and deduplication. Excluded rows remain visible here. Cause-clustering eligibility is separate from membership in this behaviour cluster.</footer></main></body></html>')
    path=OUT/'review.html';path.write_text(document)
    # Check exact join coverage, preserved schema/values, verbatim content, and navigation.
    assert pq.read_table(OUT/'causes.parquet').equals(causes)
    assert Counter(r['prompt_id'] for g in groups for r in g['causes'])==Counter(rows[i]['prompt_id'] for i in range(len(rows)))
    for g in groups:
        assert esc(g['behaviour']['behaviour_text']) in document
        for r in g['causes']:assert esc(r['cause_text_original']) in document
    from html.parser import HTMLParser
    class Check(HTMLParser):
        def __init__(self):super().__init__();self.ids=set();self.links=[]
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if a.get('id'):self.ids.add(a['id'])
            if tag=='a':self.links.append(a['href'])
    check=Check();check.feed(document)
    for link in check.links:assert link[1:] in check.ids if link.startswith('#') else (OUT/link).exists()
    if method=='hdbscan':
        review=RUN/'review.html';page=review.read_text()
        marker=f'<summary>Cluster {cluster_id} · {len(ids)} members</summary>'
        relative=f'cluster_{cluster_id}_behaviours_hdbscan/review.html'
        assert page.count(marker)==1
        if relative not in page:
            page=page.replace(marker,marker+f'<p><a href="{relative}">View all {len(ids)} members alongside their causes.parquet rows</a></p>')
            review.write_text(page)
    (OUT/'validation.json').write_text(json.dumps({'status':'passed','run':'petri_latest_qualified','facet':'behaviours','method':method,'cluster_id':cluster_id,
        'members':len(ids),'cause_rows':len(rows),'eligible_cause_rows':n_eligible,'excluded_cause_rows':len(rows)-n_eligible,'exact_parquet_subset':True,'semantic_label_assigned':False},indent=2)+'\n')
    print(path)
    print(f'Validated all {len(ids)} members, all {len(rows)} matching cause rows, original texts, parquet round-trip, and links.')


if __name__ == '__main__':main()
