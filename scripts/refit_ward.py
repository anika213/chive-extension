"""Keep the average-linkage pilot and refit Ward on identical saved embeddings."""
import shutil
import json
import numpy as np
from cluster_explanations import OUT, cluster, write_json
summaries=[]
for dest in sorted(OUT.glob('*/*')):
    if not (dest/'explanations.jsonl').exists():
        continue
    prior=dest/'average_linkage_pilot'
    prior.mkdir(exist_ok=True)
    for name in ['linkage.npy','assignments.csv','cluster_examples.json','diagnostics.json','summary.json']:
        if (dest/name).exists() and not (prior/name).exists():
            shutil.copy2(dest/name,prior/name)
    records=[json.loads(line) for line in (dest/'explanations.jsonl').read_text().splitlines()]
    result=cluster(records,np.load(dest/'embeddings.npy'),dest)
    result['excluded_count']=len(json.loads((dest/'excluded.json').read_text()))
    summaries.append(result)
write_json(OUT/'summary.json',summaries)
