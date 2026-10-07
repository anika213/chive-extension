"""Readable location cross-tabs, seed audit, and before/after text review."""
from collections import Counter, defaultdict
import html
import json
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import adjusted_rand_score

import location_seed_analysis as prep


def table(headers, rows):
    esc = lambda v: html.escape(str(v))
    return '<div class="scroll"><table><tr>' + ''.join('<th>'+esc(v)+'</th>' for v in headers) + '</tr>' + ''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows) + '</table></div>'


def main():
    out = prep.RESULT; out.mkdir(exist_ok=True)
    seed_path = out / 'seed_summary.json'
    seeds = json.loads(seed_path.read_text()) if seed_path.exists() else None
    counts = json.loads((prep.OUTPUT / 'summary.json').read_text())
    report = {'seed_proxy_analysis_completed': seeds is not None, 'sources': {}}
    body = ['<h1>Edit location and behaviour seed concentration</h1>',
            '<p>PETRI and WildChat eval only. No semantic cluster labels. Behaviour vectors and assignments are unchanged; causes were re-embedded after the deterministic location-text removal.</p>',
            '<p><a href="../checkpoint2_locations/review.html">Revised cause clustering: silhouette/null plots, ARI, sizes, noise and member samples</a> · <a href="summary.json">Full audit metrics</a></p>',
            '<h2>What the fields mean</h2>',
            '<p><code>changed_message_roles</code> records actual stored roles. <code>edit_locations</code> distinguishes system_prompt, user_message, prior_assistant_turn, and tool_output using explicit tool_response wrappers. Multiple locations are allowed. Native user-role tool responses retain user in the role field. A changed message containing both ordinary user text and a tool wrapper is flagged as mixed; its location includes both, without claiming which subregion changed.</p>',
            '<p>Exact unchanged messages anchor the sequence diff, so inserting a turn does not mark every following turn as changed. Changed baseline and experiment indices are stored. All context hashes were checked against the prior context-diff records.</p>',
            '<p>Only <code>cause_text</code> is embedded. Explicit unquoted role/turn locator phrases are deleted; <code>cause_text_with_location</code>, <code>location_text_removed</code>, and <code>cause_text_original</code> preserve an audit trail. Quoted edit payloads are retained, even if they mention a role: deleting those words would change the intervention. This controls explicit location wording, not all semantic information correlated with location.</p>',
            '<p>PETRI IDs have the form petri_&lt;scenario&gt;_e&lt;episode&gt;_t&lt;turn&gt;. Published metadata contains no explicit seed field. A scenario-prefix seed proxy requires that definition; WildChat has no corresponding scenario identifier. Conversation hashes are not substituted for seeds (agentic-source hashes are often empty). No seed is inferred from an embedding or investigator story.</p>']
    if seeds is None:
        body.append('<p><strong>Seed concentration is pending confirmation of the PETRI scenario-prefix proxy. WildChat seed concentration is unavailable without a mapping.</strong></p>')
    else:
        body.append('<p>Concentration below uses a provisional PETRI scenario-name proxy, grouping episodes and turns together. No explicit seed field is recorded; this definition is an assumption from the ID structure, not independently verified seed provenance. HDBSCAN noise is excluded and reported separately. Top-one and top-two fractions use the entire cluster as denominator. Missing seeds remain missing, never one artificial shared seed.</p>')
    if seeds:
        import os
        os.environ.setdefault('MPLCONFIGDIR', str(prep.ROOT / '.cache/matplotlib'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for i, method in enumerate(['ward', 'hdbscan']):
            stats = json.loads((out/'petri'/f'{method}_seed_concentration.json').read_text())
            axes[i,0].hist([[r['top1_fraction'] for r in stats], [r['top2_fraction'] for r in stats]],
                           bins=np.linspace(0,1,11), label=['Largest scenario', 'Top two scenarios'],
                           color=['#086c79','#c87825'])
            axes[i,0].axvline(.5, color='#555', linestyle=':', linewidth=1)
            axes[i,0].set(xlabel='Fraction of cluster', ylabel='Number of clusters', title=f'{method}: scenario concentration')
            axes[i,0].legend(fontsize=8)
            values = [r['distinct_seeds'] for r in stats]
            axes[i,1].hist(values, bins=np.arange(.5,max(values)+1.5), color='#086c79')
            axes[i,1].set(xlabel='Distinct scenario prefixes', ylabel='Number of clusters', title=f'{method}: scenario breadth')
        fig.suptitle('PETRI behaviour clusters · provisional scenario-prefix seed proxy',fontsize=12)
        fig.tight_layout(); fig.savefig(out/'petri/seed_concentration.png',dpi=160); plt.close(fig)
        body.append('<h2>PETRI scenario concentration</h2><img style="max-width:100%" src="petri/seed_concentration.png" alt="Scenario concentration and distinct scenario counts for PETRI Ward and HDBSCAN behaviour clusters">')
    for source in prep.SOURCES:
        folder = out / source; folder.mkdir(exist_ok=True)
        causes = pq.read_table(prep.OUTPUT / source / 'causes.parquet').to_pylist()
        eligible = [r for r in causes if r['clustering_eligible']]
        assignments = pq.read_table(prep.ROOT / 'results/checkpoint2_locations' / source / 'causes/assignments.parquet').to_pylist()
        behaviours = pq.read_table(prep.ROOT / 'results/checkpoint2' / source / 'behaviours/assignments.parquet').to_pylist()
        bmap = {r['prompt_id']:r for r in behaviours}
        assert all(r['prompt_id'] in bmap for r in assignments)
        cross_rows = []
        for facet in ['causes','behaviours']:
            for method in ['ward','hdbscan']:
                cross = defaultdict(list)
                for row in assignments:
                    cluster = (row if facet == 'causes' else bmap[row['prompt_id']])[method+'_cluster_id']
                    cross[cluster,row['edit_location']].append(row['prompt_id'])
                for (cluster, location), ids in sorted(cross.items()):
                    cross_rows.append({'facet':facet,'method':method,'cluster_id':cluster,
                        'edit_location':location,'cause_rows':len(ids),'distinct_investigations':len(set(ids))})
        prep.save(folder/'location_crosstabs.json',cross_rows)
        pq.write_table(pa.Table.from_pylist(cross_rows),folder/'location_crosstabs.parquet')
        old = pq.read_table(prep.ROOT/'results/checkpoint2'/source/'causes/assignments.parquet').to_pylist()
        oldmap = {(r['prompt_id'],r['experiment_number']):r for r in old}
        agreement = {}
        for method in ['ward','hdbscan']:
            pairs = [(oldmap[r['prompt_id'],r['experiment_number']][method+'_cluster_id'],r[method+'_cluster_id']) for r in assignments]
            if method == 'hdbscan':pairs=[(a,b) for a,b in pairs if a >= 0 and b >= 0]
            agreement[method]={'common_rows':len(pairs),'ari':float(adjusted_rand_score([a for a,b in pairs],[b for a,b in pairs]))}
        report['sources'][source]={'preprocessing':counts[source],'old_new_cause_cluster_agreement':agreement,
                                  'seeds':seeds[source] if seeds else {'status':'pending proxy approval' if source=='petri' else 'unavailable'}}
        body += [f'<h2>{source}</h2>',f'<p><a href="../../data/checkpoint1_locations/{source}/causes.parquet">All retained causes with metadata</a> · <a href="../../data/checkpoint1_locations/{source}/causes_for_clustering.parquet">Embedding inputs</a> · <a href="{source}/location_crosstabs.parquet">Location cross-tabs (Parquet)</a></p>',
                 table(['Eligible causes','Texts changed (eligible)','New duplicate exclusions','Location-only text flags (all retained)','Zero-cause investigations'],
                       [[counts[source][k] for k in ['eligible_causes','changed_texts_eligible','new_exact_duplicates','location_only_text']] + [counts[source]['causes_per_investigation']['zero_eligible_excluded']]]),
                 '<h3>Locations among eligible cause rows</h3>',
                 table(['Exact location combination','Cause rows'],sorted(counts[source]['edit_location_counts_eligible'].items(),key=lambda kv:-kv[1])),
                 '<p>Exact location combinations partition the rows. Cross-tab counts are experiments; distinct-investigation counts are also provided. Cluster -1 denotes HDBSCAN noise.</p>',
                 '<h3>Cause-cluster agreement before vs after location removal</h3>',
                 table(['Method','Common rows compared','ARI'],[[m,r['common_rows'],f'{r["ari"]:.4f}'] for m,r in agreement.items()])]
        body.append('<p>HDBSCAN comparison uses rows assigned by both runs. Cluster IDs are local to each run; ARI does not rely on matching their numbers.</p>')
        for facet in ['causes','behaviours']:
            for method in ['ward','hdbscan']:
                subset = [r for r in cross_rows if r['facet']==facet and r['method']==method]
                body.append(f'<details><summary>{facet} / {method}: all location cross-tab cells</summary>'+table(['Cluster','Location','Cause rows','Distinct investigations'],[[r[k] for k in ['cluster_id','edit_location','cause_rows','distinct_investigations']] for r in subset])+'</details>')
        sample = random.Random(20261004).sample([r for r in eligible if r['location_text_removed']],30)
        prep.save(folder/'text_removal_samples.json',sample)
        body.append('<details><summary>30 random changed embedding texts: before / after</summary>'+table(['Investigation / experiment','Observed location','Before removal','Embedded text'],[[r['prompt_id']+'/'+str(r['experiment_number']),r['edit_location'],r['cause_text_with_location'],r['cause_text']] for r in sample])+'</details>')
        if seeds:
            for method,summary in seeds[source]['methods'].items():
                stats=json.loads((folder/f'{method}_seed_concentration.json').read_text())
                body.append(f'<h3>Behaviour clusters / {method}: seed concentration</h3>')
                if not summary['clusters_with_known_seeds']:
                    body.append('<p>Unavailable: no scenario seed identifier in published records. Per-cluster sizes and missingness are saved; no invented seed counts or fractions.</p>')
                else:
                    fmt=lambda x:'unavailable' if x is None else f'{x:.3f}'
                    body.append(table(['Clusters','Top-one >50%','Top-two >50%','Median distinct seeds','Weighted top-one','Permutation mean','Noise rows','Singleton clusters'],
                        [[summary['clusters'],summary['clusters_top1_over_half'],summary['clusters_top2_over_half'],summary['median_distinct_seeds'],fmt(summary['row_weighted_top1_fraction']),fmt(summary['permuted_seed_weighted_top1_mean']),summary['noise_rows'],summary['singleton_clusters']]]))
                    body.append('<p>The permutation reference shuffles seed identities across assigned rows 500 times while preserving cluster sizes and seed frequencies. Singleton clusters are necessarily 100% one seed.</p>')
                body.append(f'<details><summary>Every behaviour cluster: {method}</summary>'+table(['Cluster','Size','Distinct seeds','Dominant seed','Top-one fraction','Top-two fraction','Missing seed rows'],[[r[k] for k in ['cluster_id','size','distinct_seeds','dominant_seed','top1_fraction','top2_fraction','missing_seed_rows']] for r in stats])+'</details>')
    prep.save(out/'summary.json',report)
    body.append('<p>Stopped before semantic cluster labelling. No generative LLM calls. Train and thinking deferred.</p>')
    (out/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Edit location and seed audit</title><style>body{font:16px/1.6 system-ui;color:#173044;max-width:1400px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%;margin:12px 0}td,th{border:1px solid #ccd9e0;text-align:left;vertical-align:top;padding:10px;overflow-wrap:anywhere}th{background:#eaf1f4}.scroll{overflow-x:auto}details{margin:15px 0;padding:12px;border:1px solid #ccd9e0}summary{cursor:pointer;font-weight:600}a{color:#086c79}code{font-size:.9em}</style>'+''.join(body)+'</html>')
    print('Saved',out/'review.html')


if __name__=='__main__':main()
