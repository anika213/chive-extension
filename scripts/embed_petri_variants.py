"""Local embedding-only inference for the approved 801 PETRI behaviour texts."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys
import json
import hashlib
import time
import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

variant = sys.argv[1]
assert variant in ['e5_instruct', 'qwen_instruct']
device = 'cpu' if variant == 'e5_instruct' else 'mps'
if device == 'mps': assert torch.backends.mps.is_available(), 'No usable GPU'
model_path = ROOT / 'models' / ('multilingual-e5-large-instruct' if variant == 'e5_instruct' else 'Qwen3-Embedding-0.6B')
out = ROOT / 'results/petri_representation_variants/step1' / variant
out.mkdir(parents=True, exist_ok=True)
input_path = ROOT / 'data/checkpoint1_locations/petri/behaviours.parquet'
records = pq.read_table(input_path).to_pylist()
assert len(records) == 801 and all(r['source']=='petri' and r['verif_score']>=7 and r['scenario_seed_proxy'] for r in records)
instruction = 'Instruct: Represent the type of model failure described, ignoring the subject matter, people, and domain.\nQuery: '
texts = [instruction+r['behaviour_text'] for r in records]
torch.set_num_threads(4)
torch.manual_seed(20261004)
model = SentenceTransformer(str(model_path),device=device,local_files_only=True,
                            model_kwargs={'torch_dtype':torch.float32},
                            tokenizer_kwargs={'padding_side':'left'} if variant=='qwen_instruct' else {})
model.default_prompt_name = None
lengths = [len(ids) for ids in model.tokenizer(texts,add_special_tokens=True,truncation=False,padding=False)['input_ids']]
assert max(lengths) <= model.max_seq_length, 'Input exceeds model limit; no truncation allowed'
print('MODEL',variant,'device',device,'max_tokens',max(lengths),'limit',model.max_seq_length,flush=True)
vectors=[]; start=time.monotonic();batch_size=8
for offset in range(0,len(texts),batch_size):
    v=model.encode(texts[offset:offset+batch_size],batch_size=batch_size,device=device,
                   normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False,prompt='')
    assert np.isfinite(v).all()
    vectors.append(v)
    if offset%80==0 or offset+batch_size>=len(texts):
        print('EMBED',min(offset+batch_size,len(texts)), '/',len(texts),'seconds',round(time.monotonic()-start,1),flush=True)
v=np.concatenate(vectors).astype(np.float32)
assert v.shape==(801,1024) and np.allclose(np.linalg.norm(v,axis=1),1,atol=1e-5)
np.save(out/'embeddings.npy',v)
meta={'variant':variant,'model':json.loads((model_path/'download_provenance.json').read_text())['repo'],
      'revision':json.loads((model_path/'download_provenance.json').read_text())['revision'],
      'input_path':str(input_path),'input_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),
      'input_prompt_ids':[r['prompt_id'] for r in records],
      'embedding_input_sha256':hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest(),
      'instruction':instruction,'device':device,'dtype':str(next(model.parameters()).dtype),
      'pooling_config':json.loads((model_path/'1_Pooling/config.json').read_text()),
      'batch_size':batch_size,'torch_threads':4,'torch_seed':20261004,'max_sequence_length':model.max_seq_length,
      'token_lengths':{'min':min(lengths),'median':float(np.median(lengths)),'max':max(lengths)},
      'normalized':True,'truncation':False,'chunking':False,'shape':list(v.shape),
      'seconds':time.monotonic()-start,'generative_llm_calls':0}
(out/'embedding_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print('SAVED',out,flush=True)
