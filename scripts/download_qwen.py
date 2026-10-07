"""Download only Qwen3-8B investigation artifacts at the inventoried HF revision."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / 'data'
info = json.loads((ROOT / 'hf_file_inventory.json').read_text())
revision = info['sha']
allowed = {'investigations.jsonl', 'verification.jsonl',
           'binary_counterfactual_questions.jsonl', 'eval_set_conc3_n2000.jsonl',
           'MANIFEST.txt', 'config.json', 'metadata.json', 'provenance_config.json',
           'investigations.conversion_note.txt', 'verification.conversion_note.txt',
           'binary_counterfactual_questions.build_meta.json'}
selected = []
for item in info['siblings']:
    path = item['rfilename']
    parts = path.split('/')
    if len(parts) != 3 or parts[0] != 'datasets':
        continue
    run, name = parts[1:]
    if not run.startswith(('qwen3_8b_wildchat_', 'qwen3_8b_petri_')) or name not in allowed:
        continue
    source = 'petri' if '_petri_' in run else 'wildchat'
    selected.append((path, ROOT / source / run / name))
manifest = {'repository': 'adamkarvonen/chive-data', 'revision': revision, 'files': []}
for remote, dest in sorted(selected):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        temp = dest.with_name(dest.name + '.part')
        print('Downloading', remote, flush=True)
        subprocess.run(['curl', '-fL', '--silent', '--show-error', '--retry', '5',
                        '--retry-delay', '2', '-C', '-',
                        'https://huggingface.co/datasets/adamkarvonen/chive-data/resolve/' + revision + '/' + remote,
                        '-o', str(temp)], check=True)
        temp.replace(dest)
    digest = hashlib.sha256()
    rows = 0
    with dest.open('rb') as f:
        if dest.suffix == '.jsonl':
            for line in f:
                digest.update(line)
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f'Non-object record in {dest}')
                rows += 1
        else:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
    entry = {'remote_path': remote, 'local_path': str(dest.relative_to(ROOT)),
             'bytes': dest.stat().st_size, 'sha256': digest.hexdigest(),
             'records': rows if dest.suffix == '.jsonl' else None}
    manifest['files'].append(entry)
    (ROOT / 'download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Verified', entry['local_path'], entry['bytes'], 'bytes', entry['records'], 'records', flush=True)
print('Complete:', len(manifest['files']), 'files', flush=True)
