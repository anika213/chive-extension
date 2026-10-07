"""Within-cluster cosine rankings and exploratory hierarchies; no model calls."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import hashlib
import html
import json
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list, cut_tree, dendrogram
from scipy.spatial.distance import squareform

RUN = ROOT/'results/petri_latest_qualified'
INPUT = RUN/'petri/behaviours'
OUT = RUN/'similarity'


def esc(value):return html.escape(str(value),quote=True)
def save(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


CSS = '''
:root{--ink:#193344;--muted:#5e707d;--line:#d0dce2;--teal:#086c79}*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:30px}body{font:15px/1.6 system-ui,sans-serif;background:#f3f6f8;color:var(--ink);margin:0}
main{max-width:1400px;margin:auto;padding:28px 24px 70px}h1{font-size:clamp(27px,4vw,39px);line-height:1.2}h2{font-size:23px}h3{font-size:18px;margin:4px 0 12px}
a{color:var(--teal)}.muted,.note{color:var(--muted);font-size:13px}.eyebrow{font-size:12px;color:var(--teal);text-transform:uppercase;letter-spacing:.1em;font-weight:700}
.panel{background:white;border:1px solid var(--line);padding:22px;border-radius:12px;margin:22px 0}.stats{display:flex;flex-wrap:wrap;gap:12px}.stat{padding:12px 18px;background:white;border:1px solid var(--line);border-radius:9px}.stat strong{font-size:23px;display:block}
.controls{display:flex;gap:20px;flex-wrap:wrap;align-items:center;margin:14px 0}.controls label{display:flex;align-items:center;gap:8px}select,button{font:inherit;border:1px solid var(--line);border-radius:6px;background:white;padding:7px 10px}button{cursor:pointer;color:var(--teal)}
.visual{display:grid;grid-template-columns:minmax(300px,1fr) minmax(250px,.7fr);gap:25px}.visual svg{max-width:720px;width:100%;display:block}.pair{padding:16px;background:#f3f6f8;border-radius:8px;align-self:start}.pair p{white-space:pre-wrap;overflow-wrap:anywhere}
.pair code{display:block;margin-top:16px;overflow-wrap:anywhere}.pair a{display:block;margin-bottom:12px}
.legend{display:flex;align-items:center;gap:10px;font-size:12px}.gradient{height:12px;width:170px;background:linear-gradient(to right,hsl(205 65% 97%),hsl(205 65% 29%));border:1px solid var(--line)}
.group{border:1px solid var(--line);border-top:4px solid var(--teal);background:white;border-radius:9px;margin:20px 0;padding:20px}.group-head{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between}.group-head h3{margin:0}
.member{border-top:1px solid var(--line);padding:18px 0}.member:first-of-type{margin-top:12px}.member h4{font-size:16px;margin:0 0 6px}.member code{font-size:12px;overflow-wrap:anywhere}.text{white-space:pre-wrap;overflow-wrap:anywhere;margin:12px 0}.metrics{font-size:13px;color:var(--muted);display:flex;flex-wrap:wrap;gap:7px 20px}
.neighbours{font-size:13px}.neighbours a{display:inline-block;margin-right:14px}.badge{padding:3px 9px;background:#eaf1f4;border-radius:14px;font-size:12px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid var(--line);padding:9px;text-align:left;vertical-align:top}th{background:#eaf1f4}.scroll{overflow:auto}summary{cursor:pointer;color:var(--teal);font-weight:600;padding:9px 0}img{max-width:100%}.cardlinks{display:flex;flex-wrap:wrap;gap:10px}.cardlinks a{display:block;background:white;border:1px solid var(--line);border-radius:9px;padding:14px 20px;text-decoration:none}
@media(max-width:800px){main{padding:20px 12px}.visual{grid-template-columns:1fr}.panel,.group{padding:16px}.controls{gap:10px}.pair{max-height:400px;overflow:auto}}
@media print{.controls{display:none}.visual{display:block}.member{break-inside:avoid}body{background:white}}
'''

JS = r'''
const D=JSON.parse(document.getElementById('data').textContent);
const $=id=>document.getElementById(id);
function el(tag,text,cls){const x=document.createElement(tag);if(text!==undefined)x.textContent=text;if(cls)x.className=cls;return x}
const label=i=>`#${D.members[i].display_number}`;
const fmt=x=>x.toFixed(3);
const palette=['#086c79','#b85d24','#7253a2','#39723f','#a13e66','#536f9f','#9b7b19','#48756c','#7c6350','#7c4566','#326d92','#6b7138'];
function subgroupSets(){const k=Number($('cut').value),labels=D.cuts[String(k)];return Array.from({length:k},(_,g)=>D.leaf_order.filter(i=>labels[i]===g))}
function groupMean(indices){let sum=0,n=0;for(let a=0;a<indices.length;a++)for(let b=a+1;b<indices.length;b++){sum+=D.cosine[indices[a]][indices[b]];n++}return n?fmt(sum/n):'n/a (singleton)'}
function pair(a,b){const p=$('pair');p.replaceChildren();p.append(el('h3',a===b?`${label(a)} · same item`:`${label(a)} ↔ ${label(b)} · cosine ${fmt(D.cosine[a][b])}`));for(const i of [...new Set([a,b])]){p.append(el('code',D.members[i].prompt_id),el('p',D.members[i].behaviour_text));const link=el('a','Jump to this item');link.href=`#member-${i}`;p.append(link)}if(a===b)p.append(el('p','Self-similarity is excluded from centrality ranks.','note'))}
function svgNode(tag,attrs){const x=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs||{}))x.setAttribute(k,v);return x}
function heatmap(){const n=D.members.length,margin=30,step=12,side=n*step;const svg=svgNode('svg',{viewBox:`0 0 ${side+margin+4} ${side+margin+4}`,role:'img','aria-label':'Ordered cosine similarity matrix. Click a cell to compare two items.'});const title=svgNode('title');title.textContent='Cosine similarity, ordered to place similar members together';svg.append(title);
for(let a=0;a<n;a++){for(let b=0;b<n;b++){const i=D.leaf_order[a],j=D.leaf_order[b],v=D.cosine[i][j];const cell=svgNode('rect',{x:margin+b*step,y:margin+a*step,width:step,height:step,fill:`hsl(205 65% ${97-68*Math.max(0,Math.min(1,v))}%)`,style:'cursor:pointer'});const tip=svgNode('title');tip.textContent=`${label(i)} / ${label(j)}: ${fmt(v)}`;cell.append(tip);cell.addEventListener('click',()=>pair(i,j));svg.append(cell)}
const y=svgNode('text',{x:margin-5,y:margin+a*step+9,'text-anchor':'end','font-size':8});y.textContent=String(a+1);svg.append(y);const x=svgNode('text',{x:margin+a*step+6,y:margin-5,'text-anchor':'middle','font-size':8});x.textContent=String(a+1);svg.append(x)}
let offset=0;subgroupSets().forEach((g,i)=>{svg.append(svgNode('rect',{x:margin+offset*step,y:margin+offset*step,width:g.length*step,height:g.length*step,fill:'none',stroke:palette[i%palette.length],'stroke-width':2,'pointer-events':'none'}));offset+=g.length});$('heatmap').replaceChildren(svg)}
function render(){const groups=subgroupSets(),central=$('sort').value==='central',labels=D.cuts[$('cut').value];const container=$('members');container.replaceChildren();$('cut-info').textContent=groups.length===1?'One group: original HDBSCAN membership.':'Exploratory cut of the within-cluster hierarchy. The original HDBSCAN cluster is unchanged.';
groups.forEach((indices,g)=>{const section=el('section',undefined,'group');section.style.borderTopColor=palette[g%palette.length];const head=el('div',undefined,'group-head');head.append(el('h3',groups.length===1?'All cluster members':`Subgroup ${g+1}`),el('span',`${indices.length} items · mean pairwise cosine ${groupMean(indices)}`,'badge'));section.append(head);
const ordered=central?[...indices].sort((a,b)=>D.members[a].centrality_rank-D.members[b].centrality_rank):indices;
for(const i of ordered){const r=D.members[i],card=el('article',undefined,'member');card.id=`member-${i}`;card.append(el('h4',`${label(i)} · centrality rank ${r.centrality_rank} of ${D.members.length}`),el('code',r.prompt_id),el('p',r.behaviour_text,'text'));
const metrics=el('div',undefined,'metrics');const others=indices.filter(j=>j!==i);const within=others.length?fmt(others.reduce((s,j)=>s+D.cosine[i][j],0)/others.length):'n/a';metrics.append(el('span',`Mean to whole cluster: ${fmt(r.mean_cosine_to_others)}`),el('span',`Mean within displayed subgroup: ${within}`),el('span',`Verification: ${r.verif_score}/10`));card.append(metrics);
const nn=el('p',undefined,'neighbours');nn.append(document.createTextNode('Closest in original cluster: '));for(const q of r.neighbours){const link=el('a',`${label(q.index)} (${fmt(q.cosine)})`);link.href=`#member-${q.index}`;nn.append(link)}card.append(nn);const causes=el('a',`View this investigation alongside its ${r.cause_rows} cause rows`);causes.href=r.cause_review_url;card.append(causes);section.append(card)}container.append(section)});heatmap()}
for(let k=1;k<=D.max_cut;k++){const option=el('option',String(k));option.value=String(k);$('cut').append(option)}
$('cut').addEventListener('change',render);$('sort').addEventListener('change',render);
$('reset').addEventListener('click',()=>{$('cut').value='1';$('sort').value='neighbourhood';render()});
const rows=$('pairs');for(const p of D.top_pairs){const tr=el('tr');for(const i of [p.i,p.j]){const td=el('td');const link=el('a',`${label(i)} · ${D.members[i].prompt_id}`);link.href=`#member-${i}`;td.append(link);tr.append(td)}tr.append(el('td',fmt(p.cosine)));const td=el('td'),button=el('button','Compare');button.addEventListener('click',()=>{pair(p.i,p.j);$('pair').scrollIntoView({block:'center'})});td.append(button);tr.append(td);rows.append(tr)}
render();if(D.top_pairs.length)pair(D.top_pairs[0].i,D.top_pairs[0].j);
'''


def build_cluster(cluster_id,global_indices,records,vectors):
    folder=OUT/f'cluster_{cluster_id}';folder.mkdir(parents=True,exist_ok=True)
    members=[records[i] for i in global_indices];n=len(members)
    v=vectors[global_indices].astype(np.float64);v/=np.linalg.norm(v,axis=1,keepdims=True)
    cosine=np.clip(v@v.T,-1,1);np.fill_diagonal(cosine,1)
    distance=1-cosine;np.fill_diagonal(distance,0)
    condensed=squareform(distance,checks=True)
    tree=optimal_leaf_ordering(linkage(condensed,method='average'),condensed)
    order=leaves_list(tree).astype(int).tolist()
    assert sorted(order)==list(range(n))
    means=(cosine.sum(axis=1)-1)/(n-1)
    ranked=sorted(range(n),key=lambda i:(-means[i],members[i]['prompt_id']))
    display={index:pos+1 for pos,index in enumerate(order)};rank={index:pos+1 for pos,index in enumerate(ranked)}
    max_cut=min(12,n);cuts={}
    for k in range(1,max_cut+1):
        labels=cut_tree(tree,n_clusters=k).ravel()
        remap={int(x):j for j,x in enumerate(dict.fromkeys(labels[order].tolist()))}
        mapped=[remap[int(x)] for x in labels];assert len(set(mapped))==k
        sequence=[mapped[i] for i in order];assert sequence==sorted(sequence)
        cuts[str(k)]=mapped
    causepath=RUN/f'cluster_{cluster_id}_behaviours_hdbscan/review.html'
    if not causepath.exists():
        subprocess.run([sys.executable,str(ROOT/'examples/build_cluster5_causes_html.py'),'--cluster-id',str(cluster_id),'--method','hdbscan'],check=True)
    cause_groups=json.loads((causepath.parent/'all_items.json').read_text())
    cause_lookup={g['behaviour']['prompt_id']:(pos+1,len(g['causes'])) for pos,g in enumerate(cause_groups)}
    data_members=[]
    for i,r in enumerate(members):
        others=sorted([j for j in range(n) if j!=i],key=lambda j:(-cosine[i,j],members[j]['prompt_id']))[:3]
        assert np.isclose(means[i],np.mean([cosine[i,j] for j in range(n) if j!=i]))
        cause_pos,cause_count=cause_lookup[r['prompt_id']]
        data_members.append({**r,'local_index':i,'global_row_index':int(global_indices[i]),'display_number':display[i],
            'centrality_rank':rank[i],'mean_cosine_to_others':float(means[i]),
            'neighbours':[{'index':j,'prompt_id':members[j]['prompt_id'],'cosine':float(cosine[i,j])} for j in others],
            'cause_rows':cause_count,'cause_review_url':f'../cluster_{cluster_id}_behaviours_hdbscan/review.html#item-{cause_pos}'})
    pairs=sorted([{'i':i,'j':j,'cosine':float(cosine[i,j]),'prompt_id_i':members[i]['prompt_id'],'prompt_id_j':members[j]['prompt_id']} for i in range(n) for j in range(i+1,n)],key=lambda p:(-p['cosine'],p['i'],p['j']))
    data={'cluster_id':cluster_id,'n':n,'members':data_members,'leaf_order':order,'cuts':cuts,'max_cut':max_cut,
          'cosine':cosine.tolist(),'top_pairs':pairs[:15],
          'method':'Average linkage on cosine distance within each original HDBSCAN cluster, with optimal leaf ordering. Centrality excludes self-similarity.'}
    save(folder/'data.json',data);np.save(folder/'cosine.npy',cosine);np.save(folder/'linkage.npy',tree)
    pq.write_table(pa.Table.from_pylist([{'prompt_id':r['prompt_id'],'display_number':r['display_number'],'centrality_rank':r['centrality_rank'],
        'mean_cosine_to_others':r['mean_cosine_to_others'],'nearest_neighbour_ids':[x['prompt_id'] for x in r['neighbours']],
        'nearest_neighbour_cosines':[x['cosine'] for x in r['neighbours']],'behaviour_text':r['behaviour_text']} for r in sorted(data_members,key=lambda r:r['centrality_rank'])]),folder/'rankings.parquet')
    pq.write_table(pa.Table.from_pylist(pairs),folder/'pairs.parquet')
    fig=plt.figure(figsize=(12,13));gs=fig.add_gridspec(2,2,height_ratios=[2,9],width_ratios=[1,.025],hspace=.07,wspace=.08)
    axd=fig.add_subplot(gs[0,0]);plot=dendrogram(tree,ax=axd,no_labels=True,color_threshold=0,above_threshold_color='#086c79')
    assert plot['leaves']==order
    axd.set_ylabel('Cosine distance');axd.set_title(f'PETRI HDBSCAN behaviour cluster {cluster_id} · {n} members\nAverage-linkage hierarchy within the existing cluster',fontsize=12)
    axd.set_xlim(0,n*10);axd.spines[['top','right','bottom']].set_visible(False)
    ax=fig.add_subplot(gs[1,0]);im=ax.imshow(cosine[np.ix_(order,order)],vmin=0,vmax=1,cmap='Blues',interpolation='nearest',extent=[0,n*10,n*10,0],aspect='equal')
    pos=np.arange(n)*10+5;ax.set_xticks(pos,range(1,n+1),rotation=90,fontsize=7);ax.set_yticks(pos,range(1,n+1),fontsize=7)
    ax.set_xlabel('Member number in similarity order');ax.set_ylabel('Member number in similarity order')
    cax=fig.add_subplot(gs[1,1]);fig.colorbar(im,cax=cax,label='Cosine similarity (not silhouette)')
    fig.canvas.draw()
    heatbox=ax.get_position();treebox=axd.get_position()
    axd.set_position([heatbox.x0,treebox.y0,heatbox.width,treebox.height])
    fig.savefig(folder/'hierarchy_heatmap.png',dpi=150,bbox_inches='tight');plt.close(fig)
    all_scores=np.array([p['cosine'] for p in pairs]);summary={'cluster_id':cluster_id,'members':n,'mean_pairwise_cosine':float(all_scores.mean()),
        'min_pairwise_cosine':float(all_scores.min()),'max_pairwise_cosine':float(all_scores.max()),
        'medoid_prompt_id':members[ranked[0]]['prompt_id'],'medoid_mean_cosine':float(means[ranked[0]]),'max_exploratory_subgroups':max_cut}
    save(folder/'summary.json',summary)
    # The data script is inert JSON; escape closing-tag characters without changing data.
    safe_json=json.dumps(data,ensure_ascii=False).replace('<',r'\u003c')
    doc=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cluster {cluster_id} · similarity and subgroups</title><style>{CSS}</style></head><body><main>
    <div class="eyebrow">PETRI · latest qualifying checkpoint · HDBSCAN behaviours</div><h1>Cluster {cluster_id} · {n} members</h1>
    <p><a href="review.html">All six clusters</a> · <a href="../cluster_{cluster_id}_behaviours_hdbscan/review.html">All members alongside cause rows</a> · <a href="cluster_{cluster_id}/rankings.parquet">Rankings</a> · <a href="cluster_{cluster_id}/pairs.parquet">All pair similarities</a></p>
    <div class="stats"><div class="stat"><strong>{n}</strong>members</div><div class="stat"><strong>{summary['mean_pairwise_cosine']:.3f}</strong>mean pairwise cosine</div><div class="stat"><strong>{summary['max_pairwise_cosine']:.3f}</strong>closest pair</div><div class="stat"><strong>{summary['min_pairwise_cosine']:.3f}</strong>least similar pair</div></div>
    <p>Default order puts related items next to one another using a within-cluster hierarchy. <strong>Centrality rank</strong> instead sorts by mean cosine similarity to all other members of this cluster. Self-similarity is excluded. Member numbers stay fixed as you change the view.</p>
    <p class="note">Computed from the existing normalized MPNet behaviour embeddings, before UMAP. No model calls, reclustering of the dataset, or semantic labels. These are cosine similarities, not silhouette scores. High similarity may reflect topic or scenario overlap.</p>
    <div class="panel"><h2>Explore possible subgroups</h2><div class="controls"><label>Subgroups <select id="cut" aria-label="Number of exploratory subgroups"></select></label><label>Order inside groups <select id="sort"><option value="neighbourhood">Similar items adjacent</option><option value="central">Most central first</option></select></label><button id="reset">Reset</button></div><p id="cut-info" class="note"></p>
    <p>Try 2, 3, or 4 subgroups. Coloured boxes mark that cut of the average-linkage hierarchy; it is exploratory, not a validated choice of subgroup count.</p>
    <div class="visual"><div><h3>Pairwise similarity · click a cell</h3><div id="heatmap"></div><div class="legend"><span>0</span><div class="gradient"></div><span>1 cosine similarity</span></div></div><div id="pair" class="pair" aria-live="polite"></div></div>
    <details><summary>Dendrogram and downloadable heatmap</summary><p><a href="cluster_{cluster_id}/hierarchy_heatmap.png">Open full-size PNG</a></p><img src="cluster_{cluster_id}/hierarchy_heatmap.png" alt="Within-cluster hierarchy and ordered cosine similarity matrix"></details></div>
    <details class="panel"><summary>15 most similar pairs</summary><div class="scroll"><table><thead><tr><th>First item</th><th>Second item</th><th>Cosine</th><th>Inspect</th></tr></thead><tbody id="pairs"></tbody></table></div></details>
    <h2>Every member</h2><div id="members"></div><noscript><p>This interactive view requires JavaScript. <a href="../cluster_{cluster_id}_behaviours_hdbscan/review.html">Read all members and causes in the static view</a>.</p></noscript>
    <script type="application/json" id="data">{safe_json}</script><script>{JS}</script></main></body></html>'''
    (OUT/f'cluster_{cluster_id}.html').write_text(doc)
    assert np.allclose(cosine,cosine.T) and np.allclose(np.diag(cosine),1)
    assert len(pairs)==n*(n-1)//2
    assert json.loads(safe_json)==data
    return summary


def main():
    OUT.mkdir(exist_ok=True)
    assignments=pq.read_table(INPUT/'assignments.parquet').to_pylist()
    records=json.loads((INPUT/'records.json').read_text())
    assert [r['prompt_id'] for r in assignments]==[r['prompt_id'] for r in records]
    vectors=np.load(INPUT/'embeddings.npy');labels=np.array([r['hdbscan_cluster_id'] for r in assignments])
    assert np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-6)
    assert np.array_equal(labels,np.load(INPUT/'hdbscan_labels.npy'))
    before=hashlib.sha256((INPUT/'assignments.parquet').read_bytes()).hexdigest()
    summaries=[]
    for cluster_id in sorted(set(labels)-{-1}):
        summaries.append(build_cluster(int(cluster_id),np.flatnonzero(labels==cluster_id),assignments,vectors))
        print('Built cluster',cluster_id,flush=True)
    save(OUT/'summary.json',summaries)
    cards=''.join(f'<a href="cluster_{r["cluster_id"]}.html"><strong>Cluster {r["cluster_id"]} · {r["members"]} members</strong><br>Mean pairwise cosine {r["mean_pairwise_cosine"]:.3f}</a>' for r in summaries)
    doc=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Within-cluster similarity · PETRI</title><style>{CSS}</style></head><body><main><div class="eyebrow">PETRI · HDBSCAN behaviour clusters</div><h1>Similarity rankings and possible subgroups</h1><p>All six clusters from the latest-run dataset of 269 investigations. This covers the 172 assigned investigations; the 97 HDBSCAN noise items are not treated as a seventh cluster.</p><p>Each page provides an ordering that places similar items together, centrality ranks, three closest neighbours per item, all pairwise cosine scores, a dendrogram, and an exploratory subgroup selector. Every item links to its matching cause rows.</p><div class="cardlinks">{cards}</div><div class="panel"><h2>How to read this</h2><p><strong>Similar items adjacent:</strong> average-linkage hierarchy on cosine distance, with optimal leaf ordering within that tree. This is designed to expose blocks of related items.</p><p><strong>Most central first:</strong> descending average cosine similarity to every other item in the same original cluster, excluding itself. A central item is broadly representative; a lower-ranked item can still belong to a coherent subgroup.</p><p><strong>Subgroups:</strong> cut the within-cluster hierarchy into 1–12 groups. These are exploratory numeric partitions, not semantic labels or an automatically selected optimum. All original HDBSCAN assignments remain unchanged.</p><p>All similarities use the original normalized MPNet behaviour vectors, not the 10-dimensional UMAP coordinates. Sources, facets, and runs are not combined.</p></div><p><a href="../review.html">Back to the original clustered-samples review</a> · <a href="summary.json">Summary data</a></p></main></body></html>'''
    (OUT/'review.html').write_text(doc)
    # Add discoverable links to existing cluster pages and the main review.
    path=RUN/'review.html';page=path.read_text()
    if 'similarity/review.html' not in page:page=page.replace('</h1>','</h1><p><a href="similarity/review.html">Explore within-cluster similarity rankings and subgroups for all six HDBSCAN behaviour clusters</a></p>',1)
    path.write_text(page)
    for r in summaries:
        path=RUN/f'cluster_{r["cluster_id"]}_behaviours_hdbscan/review.html';page=path.read_text();link=f'../similarity/cluster_{r["cluster_id"]}.html'
        if link not in page:page=page.replace('</h1>',f'</h1><p><a href="{link}">Order these members by similarity and explore possible subgroups</a></p>',1)
        path.write_text(page)
    assert before==hashlib.sha256((INPUT/'assignments.parquet').read_bytes()).hexdigest()
    save(OUT/'validation.json',{'status':'passed','clusters':len(summaries),'members':sum(r['members'] for r in summaries),
        'noise_excluded':int((labels<0).sum()),'pairwise_symmetry':True,'self_excluded_from_centrality':True,'all_hierarchy_orders_are_permutations':True,
        'every_cut_has_exactly_k_contiguous_groups':True,'original_assignments_sha256':before,'original_assignments_unchanged':True,'semantic_labels_assigned':False})
    (OUT/'interaction.js').write_text(JS)
    print('COMPLETE',OUT/'review.html',flush=True)


if __name__=='__main__':main()
