"""Add readable diagnostic tables to the completed checkpoint-2 report."""
import html
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/checkpoint2'


def main():
    summaries = json.loads((OUT / 'summary.json').read_text())
    assert len(summaries) == 4, 'Wait for both sources and both facets.'
    fmt = lambda value: 'undefined' if value is None else f'{value:.4f}'
    lines = ['## Additional diagnostics', '',
             'Null ranges below are the minimum and maximum of 10 coordinate-shuffled embedding replicates. HDBSCAN silhouette is always computed in the original embedding space on assigned rows; compare it alongside noise fraction, not alone.', '',
             '| Source | Facet | HDBSCAN cosine | Shuffled HDB cosine mean | Null range | Valid null runs | Observed noise | Null noise mean |',
             '|---|---|---:|---:|---|---:|---:|---:|']
    for r in summaries:
        h = r['hdbscan']; n = h['shuffled_null_original_cosine_silhouette']; noise = h['shuffled_null_noise_fraction']
        lines.append(f'| {r["source"]} | {r["facet"]} | {fmt(h["original_cosine_silhouette_assigned_only"])} | {fmt(n["mean"])} | {fmt(n["min"])} to {fmt(n["max"])} | {n["valid_replicates"]} | {h["noise_fraction"]:.1%} | {noise["mean"]:.1%} |')
    lines += ['', '### Cluster sizes', '', '| Source | Facet | Method | Clusters | Smallest | Median | Largest | Singletons |', '|---|---|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        for method in ['ward','hdbscan']:
            sizes = [c['size'] for c in r[method]['cluster_sizes']]
            lines.append(f'| {r["source"]} | {r["facet"]} | {method} | {len(sizes)} | {min(sizes) if sizes else "undefined"} | {statistics.median(sizes) if sizes else "undefined"} | {max(sizes) if sizes else "undefined"} | {sizes.count(1)} |')
    lines += ['', '### Near duplicates', '', 'All pairs above cosine 0.95 are saved in the per-run near_duplicates.parquet files. No near duplicates were removed.', '',
              '| Source | Facet | Pairs | Rows involved |', '|---|---|---:|---:|']
    for r in summaries:
        n = r['near_duplicates']
        lines.append(f'| {r["source"]} | {r["facet"]} | {n["pairs"]} | {n["rows_in_any_pair"]} |')
    lines += ['', '### Distinct cause clusters per investigation — positive-only denominators', '',
              'Zero-eligible-cause investigations are excluded from every diversity mean/median. For HDBSCAN, investigations whose eligible rows are all noise also have no assigned cause clusters; they are counted separately and excluded from the cluster-diversity denominator.', '',
              '| Source | Method | Positive denominator | Zero eligible causes | Eligible but only noise | Mean clusters | Median clusters | Max clusters |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        if r['facet'] != 'causes': continue
        for method, d in r['cause_diversity'].items():
            lines.append(f'| {r["source"]} | {method} | {d["investigations_in_denominator"]} | {d["investigations_with_zero_eligible_causes"]} | {d["investigations_with_eligible_causes_but_only_noise"]} | {fmt(d["mean_distinct_clusters"])} | {fmt(d["median_distinct_clusters"])} | {d["max_distinct_clusters"]} |')
    boundaries = [r['source'] + '/' + r['facet'] for r in summaries if r['ward']['best_k_at_upper_boundary']]
    lines += ['', '### Reading the results', '',
              'Ward cuts at the upper search boundary: ' + ', '.join(boundaries) + '. These runs do not establish an optimal k or a plateau beyond 200.', '',
              'Agreement between the methods reflects different cluster counts and HDBSCAN coverage as well as similarities in grouping. Higher-than-null silhouette supports structure in these embeddings; it does not validate a causal explanation or establish semantically coherent mechanisms.', '',
              'Cause rows are repeated experiments within investigations, so they are not independent investigations. Read the sampled members and near-duplicate report before interpreting fine clusters.', '',
              'The 30-row manual review samples are original investigator summaries. Only stripped edit descriptions were embedded. No semantic cluster labels have been assigned.', '']
    report_path = OUT / 'report.md'
    original = report_path.read_text().split('## Additional diagnostics')[0].rstrip()
    report_path.write_text(original + '\n\n' + '\n'.join(lines))
    esc = lambda text: html.escape(str(text))
    blocks = []; i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('|') and i+1 < len(lines) and lines[i+1].startswith('|---'):
            cells = [v.strip() for v in line.strip('|').split('|')]
            blocks.append('<div style="overflow-x:auto"><table style="min-width:1000px"><tr>' + ''.join('<th>' + esc(c) + '</th>' for c in cells) + '</tr>')
            i += 2
            while i < len(lines) and lines[i].startswith('|'):
                cells = [v.strip() for v in lines[i].strip('|').split('|')]
                blocks.append('<tr>' + ''.join('<td>' + esc(c) + '</td>' for c in cells) + '</tr>'); i += 1
            blocks.append('</table></div>'); continue
        if line.startswith('### '): blocks.append('<h3>' + esc(line[4:]) + '</h3>')
        elif line.startswith('## '): blocks.append('<h2>' + esc(line[3:]) + '</h2>')
        elif line: blocks.append('<p>' + esc(line) + '</p>')
        i += 1
    page = (OUT / 'review.html').read_text()
    page = re.sub(r'<div id="additional-diagnostics">.*?</div><!-- /additional-->', '', page, flags=re.S)
    page = page.replace('</pre>', '</pre><div id="additional-diagnostics">' + ''.join(blocks) + '</div><!-- /additional-->', 1)
    (OUT / 'review.html').write_text(page)
    print('Added full null, size, near-duplicate and positive-only diversity tables.')


if __name__ == '__main__': main()
