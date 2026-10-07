"""One Leiden/GMM comparison and matched HDBSCAN residualization; checkpoint 3."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
os.environ.setdefault('NUMBA_NUM_THREADS','4')
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'.cache/numba'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import json
import hashlib
import importlib.metadata
from collections import Counter
from html import escape
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
import igraph as ig
import leidenalg as la
import umap
import hdbscan
from compare_petri_representations import metric, seed_null, ari, distribution, cosine_distances, SEED, UMAP, HDB
from refine_behaviour_clusters import page, table, save, plt

SOURCE=ROOT/'results/petri_representation_variants/step1'
OUT=ROOT/'results/petri_representation_variants/checkpoint3'
INPUT=ROOT/'data/checkpoint1_locations/petri/behaviours.parquet'
SPECS=[('mpnet_hdbscan','mpnet_baseline','hdbscan',False),
       ('e5_hdbscan','e5_instruct','hdbscan',False),
       ('e5_leiden','e5_instruct','leiden',False),
       ('e5_gmm','e5_instruct','gmm',False),
       ('mpnet_residual_hdbscan','mpnet_baseline','hdbscan',True),
       ('e5_residual_hdbscan','e5_instruct','hdbscan',True)]
GMM={'n_components':20,'covariance_type':'full','tol':.001,'reg_covar':1e-6,'max_iter':500,'n_init':1,'init_params':'kmeans','random_state':SEED}
LEIDEN={'k':15,'resolution':1.,'partition':'RBConfigurationVertexPartition','n_iterations':-1,'seed':SEED,
        'graph':'undirected union of directed kNN, excluding self; unique edges','distance':'cosine',
        'weights':'cosine similarity; all selected edge weights must be positive','tie_break':'input row index'}


def fit(v,method,cached_umap=None):
    if method=='leiden':
        sim=np.clip(v@v.T,-1,1);np.fill_diagonal(sim,-np.inf)
        neighbors=np.argsort(-sim,axis=1,kind='stable')[:,:15]
        edges=sorted({tuple(sorted((i,int(j)))) for i in range(len(v)) for j in neighbors[i]})
        weights=np.array([sim[i,j] for i,j in edges],dtype=float)
        assert (weights>0).all(), 'Nonpositive graph weights; stop rather than alter the graph definition'
        graph=ig.Graph(n=len(v),edges=edges,directed=False)
        part=la.find_partition(graph,la.RBConfigurationVertexPartition,weights=weights.tolist(),resolution_parameter=1.,n_iterations=-1,seed=SEED)
        labels=np.array(part.membership,dtype=int)
        return labels,{'graph_edges':np.array(edges),'graph_weights':weights,'knn_indices':neighbors}, {'edges':len(edges),'connected_components':len(graph.connected_components()),'quality':float(part.quality())}
    reduced=cached_umap if cached_umap is not None else umap.UMAP(**UMAP).fit_transform(v)
    if method=='hdbscan':
        model=hdbscan.HDBSCAN(**HDB).fit(reduced)
        return model.labels_,{'umap10':reduced,'hdbscan_membership_strength':model.probabilities_},{}
    assert method=='gmm'
    model=GaussianMixture(**GMM).fit(reduced)
    assert model.converged_, 'GMM failed to converge; no retry or altered hyperparameters'
    probs=model.predict_proba(reduced);labels=probs.argmax(1)
    return labels,{'umap10':reduced,'membership_probabilities':probs,'means':model.means_,'covariances':model.covariances_,'mixture_weights':model.weights_}, {'converged':bool(model.converged_),'iterations':int(model.n_iter_),'ambiguous_max_probability_below_0_5':int((probs.max(1)<.5).sum()),'max_probability_distribution':np.quantile(probs.max(1),[0,.25,.5,.75,1]).tolist()}


def residualize(v,seeds,folder):
    residual=v.astype(np.float64).copy();counts=Counter(seeds);audit=[];centroids=[];families=sorted(counts)
    for family in families:
        ix=np.flatnonzero(seeds==family);mean=v[ix].astype(np.float64).mean(0);centroids.append(mean)
        if len(ix)>1:
            residual[ix]-=mean
            assert np.max(np.abs(residual[ix].mean(0)))<1e-12
        norms=np.linalg.norm(residual[ix],axis=1)
        assert np.all(norms>1e-10), f'Zero residual in {family}; cannot normalize'
        audit.append({'scenario_seed_proxy':family,'n':len(ix),'unchanged_singleton':len(ix)==1,'residual_norm_min':float(norms.min()),'residual_norm_median':float(np.median(norms)),'residual_norm_max':float(norms.max())})
    # Only normalize changed rows: singleton vectors remain exactly unchanged.
    changed=np.array([counts[s]>1 for s in seeds]);residual[changed]/=np.linalg.norm(residual[changed],axis=1,keepdims=True)
    result=residual.astype(np.float32)
    assert np.array_equal(result[~changed],v[~changed])
    assert np.allclose(np.linalg.norm(result,axis=1),1,atol=1e-5)
    np.save(folder/'scenario_centroids.npy',np.array(centroids));save(folder/'residualization_by_family.json',audit)
    return result,{'method':'subtract within-scenario mean from original unit embeddings, then L2 normalize','families':len(counts),'singleton_families_unchanged':sum(n==1 for n in counts.values()),'singleton_rows_unchanged':int((~changed).sum()),'two_member_families':sum(n==2 for n in counts.values()),'changed_rows':int(changed.sum()),'zero_residuals':0,'centroid_family_order':families,'full_data_transductive':True}


def expanded_metrics(labels,v,seeds):
    m,permutations=metric(labels,v,seeds)
    sizes=m['cluster_sizes'];assigned=m['assigned']
    largest=max(sizes,key=lambda r:r['size']) if sizes else None
    m['largest_cluster']={'cluster_id':largest['cluster_id'] if largest else None,'size':largest['size'] if largest else 0,'share_of_assigned':largest['size']/assigned if assigned else None}
    big=[r for r in sizes if r['size']>.15*assigned]
    trimmed=labels.copy();trimmed[np.isin(labels,[r['cluster_id'] for r in big])]=-1
    trim,trim_perms=seed_null(trimmed,seeds)
    m['seed_concentration_excluding_large_clusters']={'threshold_share_strictly_greater_than':.15,'denominator_before_exclusion':assigned,'excluded_clusters':big,'excluded_rows':sum(r['size'] for r in big),'retained_rows':int((trimmed>=0).sum()),'retained_clusters':len(set(trimmed[trimmed>=0])),'metrics':trim}
    return m,{'all_clusters':permutations,'excluding_large_clusters':trim_perms}


def sample_members(records,labels):
    rng=np.random.default_rng(SEED+30000);result=[]
    for k,size in sorted(Counter(labels[labels>=0]).items(),key=lambda x:(-x[1],x[0]))[:10]:
        ix=rng.choice(np.flatnonzero(labels==k),size=min(5,size),replace=False)
        result.append({'cluster_id':int(k),'size':size,'members':[records[int(i)] for i in ix]})
    return result


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'comparison.json').exists(), 'Existing completed run: do not overwrite'
    records=pq.read_table(INPUT).to_pylist();seeds=np.array([r['scenario_seed_proxy'] for r in records])
    assert len(records)==801 and len(set(seeds))==147 and all(r['source']=='petri' and r['verif_score']>=7 for r in records)
    source_paths=[p for p in SOURCE.rglob('*') if p.is_file()]
    original=ROOT/'results/checkpoint2_locations/petri/behaviours'
    source_paths.extend(p for p in original.iterdir() if p.is_file())
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    originals={};base_labels={};all_labels={};reports={}
    for embedding in ['mpnet_baseline','e5_instruct']:
        stored=pq.read_table(SOURCE/embedding/'assignments.parquet').to_pylist()
        assert [(r['prompt_id'],r['behaviour_text']) for r in stored]==[(r['prompt_id'],r['behaviour_text']) for r in records]
        originals[embedding]=np.load(SOURCE/embedding/'embeddings.npy');base_labels[embedding]=np.array([r['hdbscan_cluster_id'] for r in stored])
    protocol={'input':str(INPUT),'input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),'n':801,'scenario_families':147,'variants':SPECS,'umap':UMAP,'hdbscan':HDB,'gmm':GMM,'leiden':LEIDEN,'scenario_permutations':200,'embedding_null_replicates':10,'seed':SEED,'large_cluster_cutoff':'>15% of originally assigned rows, applied once','ward_dropped':True,'qwen_deferred':True,'generative_llm_calls':0,'embedding_model_calls':0,'stability_sweep':'not performed: user requested one observed Leiden and GMM fit each','null_residualization_policy':'coordinate-shuffle final normalized residual vectors and refit; matches geometry-null definition for other variants','versions':{n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','umap-learn','hdbscan','igraph','leidenalg']},'original_embedding_metadata':{n:json.loads((SOURCE/n/'embedding_metadata.json').read_text()) for n in originals},'source_hashes':hashes}
    save(OUT/'protocol.json',protocol)
    with threadpool_limits(limits=4):
        for name,embedding,method,residual in SPECS:
            print('START',name,flush=True);folder=OUT/name;folder.mkdir(exist_ok=True)
            v=originals[embedding];residual_info=None
            if residual:v,residual_info=residualize(v,seeds,folder)
            reused=method=='hdbscan' and not residual
            if reused:
                labels=base_labels[embedding].copy();artifacts={'umap10':np.load(SOURCE/embedding/'umap10.npy'),'hdbscan_membership_strength':np.load(SOURCE/embedding/'hdbscan_probabilities.npy')};extra={}
            else:
                labels,artifacts,extra=fit(v,method,np.load(SOURCE/embedding/'umap10.npy') if method=='gmm' else None)
            np.save(folder/'embeddings.npy',v)
            for filename,array in artifacts.items():np.save(folder/(filename+'.npy'),array)
            all_labels[name]=labels
            assignment=[]
            for i,r in enumerate(records):
                x={**r,'row_index':i,'cluster_id':int(labels[i])}
                if 'membership_probabilities' in artifacts:
                    prob=artifacts['membership_probabilities'][i];x.update({'membership_probabilities':prob.tolist(),'max_membership_probability':float(prob.max()),'ambiguous_membership':bool(prob.max()<.5)})
                if 'hdbscan_membership_strength' in artifacts:x['hdbscan_membership_strength']=float(artifacts['hdbscan_membership_strength'][i])
                assignment.append(x)
            pq.write_table(pa.Table.from_pylist(assignment),folder/'assignments.parquet')
            m,perms=expanded_metrics(labels,v,seeds)
            save(folder/'scenario_null_replicates.json',perms)
            m['ari_against_mpnet_hdbscan']=ari(labels,base_labels['mpnet_baseline']);m['ari_against_e5_hdbscan']=ari(labels,base_labels['e5_instruct'])
            if reused:
                old=json.loads((SOURCE/embedding/'metrics.json').read_text())['methods']['hdbscan']
                assert abs(m['cosine_silhouette']-old['cosine_silhouette'])<1e-8
                old_null=json.loads((SOURCE/embedding/'embedding_null_replicates.json').read_text())
                null=[{'replicate':r['replicate'],**r['hdbscan']} for r in old_null]
            else:
                null=[]
                for rep in range(10):
                    rng=np.random.default_rng(SEED+1000+rep)
                    shuffled=normalize(np.column_stack([rng.permutation(v[:,j]) for j in range(v.shape[1])])).astype(np.float32)
                    nl,_,ne=fit(shuffled,method);mask=nl>=0;clusters=len(set(nl[mask]))
                    score=float(silhouette_score(cosine_distances(shuffled)[np.ix_(mask,mask)],nl[mask],metric='precomputed')) if 2<=clusters<mask.sum() else None
                    null.append({'replicate':rep,'clusters':clusters,'noise_fraction':float((~mask).mean()),'silhouette':score})
                    save(folder/'embedding_null_replicates.json',null)
                    print(name,'null',rep+1,'/10',flush=True)
            save(folder/'embedding_null_replicates.json',null)
            m['embedding_shuffle_null']=distribution([r['silhouette'] for r in null]);m['embedding_shuffle_null_noise_fraction']=distribution([r['noise_fraction'] for r in null])
            # Secondary diagnostic: do residual groups remain cohesive in the original representation?
            mask=labels>=0
            m['original_unresidualized_cosine_silhouette']=float(silhouette_score(cosine_distances(originals[embedding])[np.ix_(mask,mask)],labels[mask],metric='precomputed')) if 2<=m['clusters']<mask.sum() else None
            report={'variant':name,'embedding':embedding,'method':method,'residualized':residual,'residualization':residual_info,'reused_observed_baseline_assignments':reused,'metrics':m,'method_diagnostics':extra}
            reports[name]=report;save(folder/'metrics.json',report);save(folder/'samples.json',sample_members(records,labels))
            print('DONE',name,'clusters',m['clusters'],'noise',round(m['noise_fraction'],3),'largest',m['largest_cluster'],'silhouette',m['cosine_silhouette'],flush=True)
    pairwise=[]
    for i,(name,labels) in enumerate(all_labels.items()):
        for other in list(all_labels)[i+1:]:pairwise.append({'variant_a':name,'variant_b':other,**ari(labels,all_labels[other])})
    save(OUT/'pairwise_ari.json',pairwise);save(OUT/'comparison.json',reports)
    assert hashes=={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    save(OUT/'run_validation.json',{'original_files_unchanged':True,'baseline_hash_count':len(hashes),'rows':801,'variants':len(reports),'observed_new_leiden_runs':1,'observed_new_gmm_runs':1,'observed_residual_hdbscan_runs':2,'no_sweep':True,'no_new_embeddings':True})


def render():
    reports=json.loads((OUT/'comparison.json').read_text());pairs=json.loads((OUT/'pairwise_ari.json').read_text())
    fmt=lambda x:'undefined' if x is None else f'{x:.3f}'
    pct=lambda x:'undefined' if x is None else f'{100*x:.1f}%'
    pp=lambda x:'undefined' if x is None else f'{100*x:+.1f} pp'
    headers=['Variant','Clusters','Noise','Largest: size / assigned share','Cosine silhouette / null','Seed share / null','Seed excess','After >15% exclusion: share / null','Trimmed excess','Rows retained after exclusion']
    overview=[];seedrows=[];sizes=[];arirows=[]
    for name,r in reports.items():
        m=r['metrics'];s=m['seed_concentration'];t=m['seed_concentration_excluding_large_clusters'];ts=t['metrics'];largest=m['largest_cluster'];sd=m['size_distribution']
        overview.append([name,str(m['clusters']),pct(m['noise_fraction']),f'{largest["size"]} / {pct(largest["share_of_assigned"])}',fmt(m['cosine_silhouette'])+' / '+fmt(m['embedding_shuffle_null']['mean']),pct(s['observed']['weighted_top_one_share'])+' / '+pct(s['null']['weighted_top_one_share']['mean']),pp(s['excess_over_null']['weighted_top_one_share']),pct(ts['observed']['weighted_top_one_share'])+' / '+pct(ts['null']['weighted_top_one_share']['mean']),pp(ts['excess_over_null']['weighted_top_one_share']),str(t['retained_rows'])])
        for label,sm in [('All assigned',s),('Exclude >15% clusters',ts)]:
            seedrows.append([name,label,str(sm['observed']['clusters_majority_one_scenario']),fmt(sm['null']['clusters_majority_one_scenario']['mean']),fmt(sm['observed']['median_distinct_scenarios']),fmt(sm['null']['median_distinct_scenarios']['mean']),f"[{pct(sm['null']['weighted_top_one_share']['q025'])}, {pct(sm['null']['weighted_top_one_share']['q975'])}]",str(sm['permutations'])])
        sizes.append([name,str(sd['min']) if sd else '—',fmt(sd['q25']) if sd else '—',fmt(sd['median']) if sd else '—',fmt(sd['q75']) if sd else '—',str(sd['max']) if sd else '—',str(sd['singletons']) if sd else '—'])
        for base in ['mpnet','e5']:
            a=m[f'ari_against_{base}_hdbscan'];arirows.append([name,base+' HDBSCAN',fmt(a['jointly_assigned_ari']),str(a['jointly_assigned_rows']),fmt(a['all_rows_noise_as_one_label_ari'])])
    body='<p><strong>CHECKPOINT 3 — stopped for review.</strong> Same 801 PETRI Qwen3-8B behaviour texts and 147 scenario families. Ward is dropped. Qwen embeddings remain deferred. No new embeddings or generative LLM calls.</p><p><a href="protocol.json">Complete protocol, instructions and pinned embedding revisions</a> · <a href="comparison.json">All metrics</a> · <a href="pairwise_ari.json">All method-pair ARIs</a> · <a href="report.md">Summary table</a></p>'
    body+='<h2>Comparison</h2><p>Seed share is weighted top-one scenario concentration, among assigned rows. Null values come from 200 scenario-label permutations; excess is observed minus null mean. The excess columns below are percentage-point differences. Ten coordinate-shuffled embedding refits provide each silhouette baseline.</p>'+table(headers,overview)
    body+='<img src="comparison.png" alt="Coverage, largest-cluster share, seed concentration excess, and silhouette" style="max-width:100%">'
    if (OUT/'review_notes.html').exists():
        body+='<h2>Review notes and recommendation</h2><div class="notice">'+(OUT/'review_notes.html').read_text()+'</div>'
    body+='<h2>Fixed protocol and limitations</h2><ul><li>Leiden: a single observed E5 fit, k=15 cosine-nearest neighbours; undirected union graph, cosine edge weights, RBConfiguration partition, resolution=1.0, seed=20261004, optimize to convergence. No graph sweep or stability claim.</li><li>GMM: a single observed E5 fit on the saved 10-dimensional UMAP coordinates; 20 full-covariance components, n_init=1, seed=20261004, max_iter=500, tol=0.001, reg_covar=1e-6. All 20 per-item probabilities are saved. Posterior confidence reflects this fitted mixture, not semantic certainty.</li><li>HDBSCAN settings are identical in every run: UMAP cosine, 10 dimensions, n_neighbors=30, min_dist=0, seed=20261004; HDBSCAN min_cluster_size=10, min_samples=5, Euclidean, EOM. Original assignments and null replicates are reused; residual assignments are newly fitted.</li><li>Residualization subtracts each scenario-family mean from its original unit vectors, then normalizes. Twelve singleton families/rows stay unchanged. Twenty-three two-member families become antipodal pairs by construction. Scenario residualization also removes behaviour genuinely shared within a scenario, so lower concentration is not proof of better clustering.</li><li>The >15% diagnostic excludes whole clusters using their share of the original assigned rows. It applies the threshold once, recomputes concentration on the remaining rows, and reruns the same 200 scenario permutations with this fixed mask. The saved assignments, noise fractions and silhouettes are not altered by this diagnostic. A remaining cluster may exceed 15% of the reduced denominator.</li><li>Residual seed labels participated in constructing the vectors: their permutation reference is descriptive, not an independent causal or significance test. Families and within-episode checkpoints are not independent observations.</li><li>Cosine silhouettes use each variant’s own representation, excluding noise. Residual silhouette measures relative deviations, not similarity of the original behaviour texts. An additional silhouette in the unresidualized embedding space is reported below.</li><li>Embedding nulls shuffle coordinates independently across rows, renormalize and refit the same method, ten times. For residual variants, the shuffled objects are the final residual vectors. This geometry null does not repeat scenario residualization and does not preserve its induced within-family constraints. Null fits are reference computations, not additional observed-data settings.</li><li>ARI uses jointly assigned rows; all-row ARI treating noise as one label is also saved. Random examples use the same fixed sampling seed. Undefined metrics are retained as undefined if too few clusters or rows remain.</li></ul>'
    gmm=reports['e5_gmm']['method_diagnostics']
    body+=f'<h2>GMM memberships</h2><p>{gmm["ambiguous_max_probability_below_0_5"]}/801 items have max membership probability &lt;0.5. Converged in {gmm["iterations"]} iterations. <a href="e5_gmm/assignments.parquet">Assignments with all 20 probabilities</a> · <a href="e5_gmm/membership_probabilities.npy">Probability matrix</a>.</p>'
    body+='<h2>All scenario metrics, before and after large-cluster exclusion</h2>'+table(['Variant','Scope','Clusters >50% one scenario','Null mean','Median distinct scenarios','Null mean','Seed share null 95% interval','Permutations'],seedrows)
    body+='<h2>Cluster size distributions</h2>'+table(['Variant','Minimum','Q25','Median','Q75','Maximum','Singletons'],sizes)
    body+='<h2>Silhouette in original representation and null coverage</h2>'+table(['Variant','Own-space cosine','Original-space cosine','Null valid replicates','Null mean noise','Null silhouette 95% interval'],[[name,fmt(r['metrics']['cosine_silhouette']),fmt(r['metrics']['original_unresidualized_cosine_silhouette']),str(r['metrics']['embedding_shuffle_null']['valid_replicates']),pct(r['metrics']['embedding_shuffle_null_noise_fraction']['mean']),f"[{fmt(r['metrics']['embedding_shuffle_null']['q025'])}, {fmt(r['metrics']['embedding_shuffle_null']['q975'])}]"] for name,r in reports.items()])
    body+='<h2>ARI against both original HDBSCAN baselines</h2>'+table(['Variant','Reference','ARI jointly assigned','Rows compared','All-row ARI'],arirows)
    body+='<h2>ARI between all methods and representations</h2>'+table(['Variant A','Variant B','ARI jointly assigned','Rows compared','All-row ARI'],[[r['variant_a'],r['variant_b'],fmt(r['jointly_assigned_ari']),str(r['jointly_assigned_rows']),fmt(r['all_rows_noise_as_one_label_ari'])] for r in pairs])
    body+='<h2>Five random members from the ten largest clusters</h2><p>Every variant is shown below. If fewer than ten clusters exist, show all; if a cluster has fewer than five members, show every member. No semantic labels assigned. Inspect residual groups for unrelated behaviours forced together after removing scenario means.</p>'
    for name in reports:
        body+=f'<h2>{name}</h2><p><a href="{name}/assignments.parquet">Assignments</a> · <a href="{name}/metrics.json">Metrics</a> · <a href="{name}/samples.json">Sample data</a></p>'
        for g in json.loads((OUT/name/'samples.json').read_text()):
            body+=f'<details><summary>Cluster {g["cluster_id"]} · {g["size"]} members</summary>'+table(['Investigation','Scenario','Behaviour text'],[[escape(r['prompt_id']),escape(r['scenario_seed_proxy']),escape(r['behaviour_text'])] for r in g['members']])+'</details>'
    page(OUT/'review.html','PETRI behaviour variants · checkpoint 3',body)
    lines=['# PETRI behaviour comparison — checkpoint 3','','Stopped for review. Ward excluded; Qwen deferred; no generative LLM or new embedding calls.','','|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(r)+'|' for r in overview]
    lines+=['','Largest shares use assigned rows. Excess values are percentage-point differences. See review.html for full diagnostics and random examples.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    names=list(reports);y=np.arange(len(names));fig,axes=plt.subplots(1,3,figsize=(17,6));fig.subplots_adjust(left=.20,right=.98,wspace=.4,bottom=.12,top=.9)
    vals=[(100*(1-reports[n]['metrics']['noise_fraction'])) for n in names]
    largest=[100*reports[n]['metrics']['largest_cluster']['share_of_assigned'] if reports[n]['metrics']['largest_cluster']['share_of_assigned'] is not None else 0 for n in names]
    axes[0].barh(y-.16,vals,height=.3,label='Assigned / all rows',color='#087f8c');axes[0].barh(y+.16,largest,height=.3,label='Largest / assigned',color='#dba34a');axes[0].set_title('Coverage and largest cluster');axes[0].set_xlabel('%');axes[0].legend(fontsize=8)
    for offset,key,label,color in [(-.16,'seed_concentration','All clusters','#087f8c'),(.16,'seed_concentration_excluding_large_clusters','Exclude >15%','#dba34a')]:
        val=[]
        for n in names:
            s=reports[n]['metrics'][key];s=s['metrics'] if 'metrics' in s else s;a=s['excess_over_null']['weighted_top_one_share'];val.append(100*a if a is not None else np.nan)
        axes[1].barh(y+offset,val,height=.3,label=label,color=color)
    axes[1].set_title('Seed concentration above null');axes[1].set_xlabel('Percentage points');axes[1].legend(fontsize=8)
    for offset,key,label,color in [(-.16,'cosine_silhouette','Own representation','#087f8c'),(.16,'original_unresidualized_cosine_silhouette','Original representation','#dba34a')]:
        axes[2].barh(y+offset,[reports[n]['metrics'][key] if reports[n]['metrics'][key] is not None else np.nan for n in names],height=.3,label=label,color=color)
    axes[2].set_title('Cosine silhouette');axes[2].legend(fontsize=8)
    for i,ax in enumerate(axes):
        ax.set_yticks(y,names if i==0 else ['']*len(names),fontsize=9);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.savefig(OUT/'comparison.png',dpi=160);fig.savefig(OUT/'comparison.svg');plt.close(fig)
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='report':render()
    else:run();render()
