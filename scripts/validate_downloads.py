"""Compare local download measurements with the publisher's per-run manifests."""
import json
from pathlib import Path
root = Path(__file__).resolve().parent.parent / 'data'
manifest = json.loads((root / 'download_manifest.json').read_text())
summary = {}
for entry in manifest['files']:
    path = root / entry['local_path']
    published = {}
    for line in (path.parent / 'MANIFEST.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        fields = line.split('\t')
        published[fields[0]] = fields[1:]
    if path.name != 'MANIFEST.txt':
        expected = published[path.name]
        assert path.stat().st_size == entry['bytes'] == int(expected[0]), path
        if expected[1].isdigit():
            assert entry['records'] == int(expected[1]), path
    if path.suffix == '.jsonl':
        summary.setdefault(str(path.parent.relative_to(root)), {})[path.name] = entry['records']
coverage = {}
for run in summary:
    ids = {}
    for name in ('investigations.jsonl', 'verification.jsonl'):
        with (root / run / name).open() as f:
            values = [json.loads(line)['prompt_id'] for line in f]
        ids[name] = set(values)
        assert len(values) == len(ids[name]), (run, name, 'duplicate prompt IDs')
    coverage[run] = {
        'investigations_without_verification': sorted(ids['investigations.jsonl'] - ids['verification.jsonl']),
        'verification_without_investigation': sorted(ids['verification.jsonl'] - ids['investigations.jsonl'])}
report = {'revision': manifest['revision'], 'files_checked': len(manifest['files']),
          'total_bytes': sum(e['bytes'] for e in manifest['files']),
          'publisher_sizes_and_available_counts_match': True, 'runs': summary, 'verification_coverage': coverage}
(root / 'validation_summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
