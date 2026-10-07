"""Validate saved assignments and build an unlabelled, static cluster browser."""
import csv
import html
import json
from pathlib import Path
root=Path(__file__).resolve().parent.parent/'results/final_explanations'
summary=json.loads((root/'summary.json').read_text())
parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><title>Investigator explanation clusters</title>
<style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#17242a;background:#f7f9f9}h1,h2{line-height:1.2}section{margin:40px 0}details{background:white;border:1px solid #ccd6d9;border-radius:8px;padding:14px;margin:10px 0}summary{cursor:pointer;font-weight:650}blockquote{white-space:pre-wrap;margin:12px 0;padding:16px;background:#f4f7f7}small{color:#52666e}table{border-collapse:collapse;width:100%}td,th{padding:9px;text-align:left;border-bottom:1px solid #ccd6d9}a{color:#075e73}</style>
<h1>Investigator final-explanation clusters</h1><p>Qwen3-8B investigations · Claude Opus 4.6 investigator · Numeric cluster IDs only</p>
<p>These cluster claimed explanations, not verified causes. PETRI is the alignment-relevant source. Each run is fitted independently; cluster IDs do not match across runs.</p>
<p>Method: MiniLM sentence embeddings, Ward agglomerative clustering. Candidate cuts: 5, 10, 15, 20, 30, 40, 50. Selected by cosine silhouette (sampled for the training run). Examples are the three members nearest each cluster centroid, not all members.</p>
<table><tr><th>Run</th><th>Explanations</th><th>Clusters</th><th>Silhouette</th></tr>''']
for run in summary:
    name=run['run']
    parts.append(f'<tr><td><a href="#{name}">{name}</a></td><td>{run["n_explanations"]:,}</td><td>{run["selected_k"]}</td><td>{run["selected_cosine_silhouette"]:.3f}</td></tr>')
parts.append('</table>')
checks=[]
for run in summary:
    folder=root/run['source']/run['run']
    with (folder/'assignments.csv').open() as f: assignments=list(csv.DictReader(f))
    records=[json.loads(line) for line in (folder/'explanations.jsonl').read_text().splitlines()]
    assert len(assignments)==len(records)==run['n_explanations']
    assert {r['prompt_id'] for r in assignments}=={r['prompt_id'] for r in records}
    assert all(r['source']==run['source'] and r['run']==run['run'] for r in assignments)
    ids={r['cluster_id'] for r in assignments}
    assert len(ids)==run['selected_k']
    clusters=json.loads((folder/'cluster_examples.json').read_text())
    assert sum(c['size'] for c in clusters)==len(records)
    by_id={r['prompt_id']:r for r in assignments}
    for c in clusters:
        for example in c['examples_nearest_centroid']:
            assert int(by_id[example['prompt_id']]['cluster_id'])==c['cluster_id']
    checks.append({'run':run['run'],'rows':len(records),'clusters':len(ids),'passed':True})
    rel=folder.relative_to(root).as_posix()
    parts.append(f'<section id="{run["run"]}"><h2>{run["run"]}</h2><p><a href="{rel}/assignments.csv">All assignments</a> · <a href="{rel}/diagnostics.json">Candidate diagnostics</a> · <a href="{rel}/explanations.jsonl">Full input explanations</a></p>')
    for c in sorted(clusters,key=lambda c:-c['size']):
        parts.append(f'<details><summary>Cluster {c["cluster_id"]} — {c["size"]:,} explanations</summary>')
        for example in c['examples_nearest_centroid']:
            parts.append(f'<p><b>{html.escape(example["prompt_id"])}</b><br><small>Investigation verification score: {html.escape(str(example["verification_score"]))}</small></p><blockquote>{html.escape(example["explanation"])}</blockquote>')
        parts.append('</details>')
    parts.append('</section>')
parts.append('</html>')
(root/'review.html').write_text('\n'.join(parts))
(root/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
