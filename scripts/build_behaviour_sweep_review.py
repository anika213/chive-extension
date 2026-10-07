"""Build a standalone explorer from saved behaviour Ward cuts; no fitting or inference."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'results/behaviour_facets'
OUT = BASE / 'agglomerative_sweep.html'

def load_runs():
    runs = {}
    for source in ['petri', 'wildchat']:
        folder = BASE / source / f'qwen3_8b_{source}_eval'
        records = [json.loads(line) for line in (folder / 'behaviours.jsonl').read_text().splitlines()]
        if source == 'petri':
            saved = np.load(folder / 'extended_sweep/all_cuts.npz')
            ks = saved['k'].tolist()
            labels = saved['labels'].T.tolist()
            metrics = json.loads((folder / 'extended_sweep/diagnostics.json').read_text())
        else:
            with (folder / 'assignments.csv').open() as f:
                assignments = list(csv.DictReader(f))
            assert [a['prompt_id'] for a in assignments] == [r['prompt_id'] for r in records]
            ks = [int(key.removeprefix('cluster_k')) for key in assignments[0] if key.startswith('cluster_k')]
            labels = [[int(a[f'cluster_k{k}']) for a in assignments] for k in ks]
            metrics = json.loads((folder / 'diagnostics.json').read_text())
        assert len(labels) == len(ks) == len(metrics)
        for k, cut, metric in zip(ks, labels, metrics):
            assert len(cut) == len(records) and len(set(cut)) == k and metric['k'] == k
        runs[source] = {'records': [{'id': r['prompt_id'], 'text': r['behavior_summary'], 'score': r['verification_score']} for r in records], 'ks': ks, 'labels': labels, 'metrics': metrics}
    return runs

HTML = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>All behaviour texts · agglomerative clustering</title>
<style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#20303b;background:#fafbf9}a{color:#176459}h1{font-size:34px;line-height:1.2}nav,.controls{background:#eaf2ed;padding:18px;border-radius:10px}.controls{position:sticky;top:0;z-index:1;display:flex;flex-wrap:wrap;gap:14px;align-items:end}label{display:grid;gap:3px}select,input,button{font:inherit;padding:7px;border:1px solid #abc1b4;border-radius:5px;background:white}input[type=range]{width:230px}#search{width:250px}#stats{padding:16px 0;font-weight:600}details{background:white;border:1px solid #dbe4de;border-radius:8px;padding:12px 18px;margin:12px 0}summary{cursor:pointer;font-weight:650}article{border-top:1px solid #e1e7e2;padding:14px 0}article p{white-space:pre-wrap;margin:8px 0}small,.muted{color:#596a60}svg{width:100%;height:auto;background:white;border:1px solid #dbe4de;border-radius:8px}button{cursor:pointer}#cut{min-width:80px}</style></head><body>
<nav><a href="../../reviews/index.html">All reviews</a> · <a href="README.md">Behaviour method and inputs</a> · <a href="petri/qwen3_8b_petri_eval/extended_sweep/README.md">Full PETRI sweep notes</a></nav>
<h1>All behaviour texts<br>Agglomerative clustering across cuts</h1>
<p>The broad behaviour-level analysis: <strong>Ward agglomerative clustering on normalized MPNet embeddings</strong>, spanning all scenarios in each source. Browse every member at each saved hierarchy cut. No new embeddings or clustering were run for this page.</p>
<p>PETRI: all <strong>801 qualifying summaries</strong>, every cut <strong>k = 2–800</strong>. WildChat eval: all <strong>1,641 qualifying summaries</strong>, saved cuts <strong>k = 5, 10, 15, 20, 30, 40, 50</strong>. Sources are fitted separately. Verification ≥7; train/thinking and lower-scoring or unverified records are excluded.</p>
<div class="controls"><label>Source<select id="source"><option value="petri">PETRI · 801 summaries</option><option value="wildchat">WildChat eval · 1,641 summaries</option></select></label><label>Hierarchy cut (number of clusters)<select id="cut"></select></label><label>Coarse → fine<input id="slider" type="range" min="0" max="798" value="8"></label><label>Find text or investigation<input id="search" type="search" placeholder="Search behaviour text or ID"></label><button id="best">Jump to best saved silhouette</button></div>
<div id="stats" role="status" aria-live="polite"></div><svg id="chart" viewBox="0 0 1000 180" role="img" aria-label="Cosine silhouette versus cluster count; selected cut marked"></svg>
<p class="muted">Cuts are specified by cluster count k, not a cosine threshold. Silhouette is measured in the original embedding space. The PETRI peak is k=249 (0.204), with median size 3 and 44 singleton groups—higher score does not establish a broad behavioural taxonomy. Numeric IDs are local to each cut. Search filters displayed members without changing the saved assignments.</p>
<div id="groups"></div><noscript>Enable JavaScript to browse saved cuts. Static summary: <a href="review.html">original behaviour review</a>.</noscript>
<script id="saved-data" type="application/json">__DATA__</script>
<script>
const runs=JSON.parse(document.getElementById('saved-data').textContent);
const source=document.getElementById('source'),cut=document.getElementById('cut'),slider=document.getElementById('slider'),search=document.getElementById('search');
function node(tag,text){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e;}
function chart(run,index){const scores=run.metrics.map(m=>m.cosine_silhouette),max=Math.max(...scores,.001),min=Math.min(...scores,0),range=max-min||1;const x=k=>45+910*(k-run.ks[0])/(run.ks.at(-1)-run.ks[0]||1),y=v=>135-105*(v-min)/range;const points=run.ks.map((k,i)=>`${x(k)},${y(scores[i])}`).join(' ');document.getElementById('chart').innerHTML=`<polyline fill="none" stroke="#176459" stroke-width="2" points="${points}"/><circle cx="${x(run.ks[index])}" cy="${y(scores[index])}" r="5" fill="#c85a26"/><text x="45" y="20" fill="#20303b">Cosine silhouette · selected k=${run.ks[index]}</text><text x="45" y="165">${run.ks[0]} clusters</text><text x="855" y="165">${run.ks.at(-1)} clusters</text><text x="45" y="40">max ${max.toFixed(4)}</text>`;}
function render(){const run=runs[source.value],index=Number(cut.value),k=run.ks[index],labels=run.labels[index],query=search.value.toLowerCase().trim(),groups=new Map();slider.value=index;for(let i=0;i<labels.length;i++){const id=labels[i];if(!groups.has(id))groups.set(id,[]);groups.get(id).push(i);}const target=document.getElementById('groups');target.replaceChildren();let shown=0,visible=0;for(const [id,members] of [...groups].sort((a,b)=>b[1].length-a[1].length||a[0]-b[0])){const filtered=members.filter(i=>!query||(run.records[i].id+' '+run.records[i].text).toLowerCase().includes(query));if(!filtered.length)continue;shown+=filtered.length;visible++;const detail=node('details');detail.dataset.cluster=id;detail.open=!!query||k<=5;detail.append(node('summary',`Cluster ${id} · ${members.length} members${query?' · '+filtered.length+' matching':''}`));const fill=()=>{if(detail.dataset.loaded)return;detail.dataset.loaded='true';for(const i of filtered){const r=run.records[i],article=node('article');article.append(node('strong',r.id),node('small',` · verification ${r.score}/10`),node('p',r.text));detail.append(article);}};detail.addEventListener('toggle',()=>{if(detail.open)fill();});if(detail.open)fill();target.append(detail);}document.getElementById('stats').textContent=`${run.records.length} summaries · ${k} clusters · cosine silhouette ${run.metrics[index].cosine_silhouette.toFixed(4)} · ${shown} members in ${visible} groups displayed`;if(!visible)target.append(node('p','No matching behaviours at this cut.'));chart(run,index);}
function setSource(){const run=runs[source.value];cut.replaceChildren();run.ks.forEach((k,i)=>{const o=node('option',String(k));o.value=i;cut.append(o);});cut.value=Math.max(0,run.ks.indexOf(10));slider.max=run.ks.length-1;render();}
source.addEventListener('change',setSource);cut.addEventListener('change',render);slider.addEventListener('input',()=>{cut.value=slider.value;render();});search.addEventListener('input',render);document.getElementById('best').addEventListener('click',()=>{const run=runs[source.value];cut.value=run.metrics.reduce((best,m,i)=>m.cosine_silhouette>run.metrics[best].cosine_silhouette?i:best,0);render();});setSource();
</script></body></html>'''

if __name__ == '__main__':
    runs = load_runs()
    data = json.dumps(runs, separators=(',', ':')).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    OUT.write_text(HTML.replace('__DATA__', data))
    print(f'Wrote {OUT.relative_to(ROOT)}; PETRI {len(runs["petri"]["ks"])} cuts; WildChat {len(runs["wildchat"]["ks"])} cuts.')
