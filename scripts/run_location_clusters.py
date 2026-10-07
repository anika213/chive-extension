"""Re-embed revised causes using the unchanged checkpoint-2 protocol."""
import json
import shutil
from pathlib import Path
import run_checkpoint2 as run
from sentence_transformers import SentenceTransformer
from threadpoolctl import threadpool_limits
import torch


def main():
    root = Path(__file__).resolve().parent.parent
    run.INPUT = root / 'data/checkpoint1_locations'
    run.OUTPUT = root / 'results/checkpoint2_locations'
    run.OUTPUT.mkdir(parents=True, exist_ok=True)
    run.save(run.OUTPUT / 'design.json', {
        'input_directory': str(run.INPUT), 'sources': run.SOURCES, 'facets_recomputed': ['causes'],
        'behaviours': 'Existing checkpoint2 assignments and vectors reused: texts unchanged.',
        'location_fields_embedded': False, 'seed_fields_embedded': False,
        'cause_embedding_field': 'cause_text', 'semantic_labels': False,
        'protocol': 'Same MPNet CPU / Ward k10..200 / UMAP10 HDBSCAN / 10 null replicates as checkpoint2',
        'location_text_rule': 'Delete explicit unquoted role/turn locator spans; preserve quoted intervention payloads.',
    })
    torch.set_num_threads(4)
    model = SentenceTransformer(str(run.MODEL), device='cpu', local_files_only=True)
    summaries = []
    with threadpool_limits(limits=4):
        for source in run.SOURCES:
            old = root / 'results/checkpoint2' / source / 'behaviours'
            new = run.OUTPUT / source / 'behaviours'
            # Preserve the validated old behaviour run verbatim: no needless refit.
            shutil.copytree(old, new, dirs_exist_ok=True)
            summaries.append(json.loads((old / 'summary.json').read_text()))
            summaries.append(run.run_dataset(source, 'causes', model))
            run.save(run.OUTPUT / 'summary.json', summaries)
    run.build_report(summaries)
    # Correct inherited report links and identify the changed input protocol.
    for name in ['review.html', 'report.md']:
        path = run.OUTPUT / name
        content = path.read_text().replace('../../data/checkpoint1_approved/review.html', '../location_seed/review.html')
        content = content.replace('30 eligible original cause summaries per source and revised filters', 'Edit-location and seed audit')
        path.write_text(content)
    import finish_checkpoint2_report as finish
    finish.OUT = run.OUTPUT
    finish.main()
    print('Location-adjusted cause runs complete.', flush=True)


if __name__ == '__main__': main()
