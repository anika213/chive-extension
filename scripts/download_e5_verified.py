"""Download pinned public embedding weights using an explicit verified CA bundle."""
import hashlib
import json
import ssl
import sys
import urllib.request
from pathlib import Path
import certifi

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'models/multilingual-e5-large-instruct'
REPO = 'intfloat/multilingual-e5-large-instruct'
REVISION = '274baa43b0e13e37fafa6428dbc7938e62e5c439'
if '--qwen' in sys.argv:
    REPO = 'Qwen/Qwen3-Embedding-0.6B'
    REVISION = '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
    OUT = ROOT / 'models/Qwen3-Embedding-0.6B'
context = ssl.create_default_context(cafile=certifi.where())
assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
OUT.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true', context=context, timeout=60) as response:
    manifest = json.load(response)
assert manifest['sha'] == REVISION
files = [r for r in manifest['siblings'] if not r['rfilename'].startswith('onnx/') and r['rfilename'] != '.gitattributes']
audit = []
for row in files:
    name = row['rfilename']; path = OUT / name
    expected = row.get('lfs', {}).get('sha256')
    if path.exists() and (not expected or hashlib.sha256(path.read_bytes()).hexdigest() == expected):
        print('CACHED',name,flush=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix+'.partial')
        print('DOWNLOAD',name,flush=True)
        with urllib.request.urlopen(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}', context=context, timeout=60) as response, temp.open('wb') as f:
            count = 0; mark = 0
            while data := response.read(4*1024*1024):
                f.write(data); count += len(data)
                if count-mark >= 128*1024*1024:
                    print(name,round(count/1024**2),'MiB',flush=True);mark=count
        actual = hashlib.sha256(temp.read_bytes()).hexdigest()
        if expected: assert actual == expected, (name,'SHA256 mismatch')
        temp.replace(path)
    audit.append({'file':name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'expected_lfs_sha256':expected})
(OUT/'download_provenance.json').write_text(json.dumps({'repo':REPO,'revision':REVISION,'https_certificate_verification':True,'ca_bundle':certifi.where(),'files':audit},indent=2)+'\n')
print('Verified download complete.',flush=True)
