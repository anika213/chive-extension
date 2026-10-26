"""Download the external datasets used by introspection/ into data/external/ (gitignored).

  sgtr-em                     atagade/sgtr-em at a pinned commit: XSum and CNN articles,
                              Claude 2.1 summaries, Qwen-32B XSum summaries, and the
                              summarization configs. Used by self_recognition/.
  introspection_self_predict  thejaminator/introspection_self_predict on Hugging Face
                              (Binder et al. 2024). Used by self_prediction/.

Already-downloaded files are skipped.

Usage:
    .venv/bin/python introspection/fetch_data.py                 # both
    .venv/bin/python introspection/fetch_data.py --only sgtr-em
"""

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chive import paths

EXTERNAL = paths.DATA_ROOT / "external"

SGTR_EM_REPO = "https://github.com/atagade/sgtr-em"
SGTR_EM_COMMIT = "5f7e789be222ce05d46c078c8bffacaa1ef60a88"
# path in the sgtr-em repo; each is saved flat under data/external/sgtr-em/
SGTR_EM_FILES = [
    "data/articles/xsum_train_articles.json",
    "data/summaries/xsum/xsum_train_claude-21_responses.json",
    "data/summaries/xsum/xsum_train_hf_qwen_32b_responses.json",
    "utils/prompts/default_xsum_summary_config.json",
    "data/articles/cnn_train_articles.json",
    "data/summaries/cnn/cnn_train_claude-21_responses.json",
    "utils/prompts/default_cnn_summary_prompts.json",
]

SELF_PREDICT_DATASET = "thejaminator/introspection_self_predict"


def fetch_sgtr_em():
    out = EXTERNAL / "sgtr-em"
    out.mkdir(parents=True, exist_ok=True)
    for src in SGTR_EM_FILES:
        dest = out / Path(src).name
        if dest.exists():
            continue
        url = f"https://raw.githubusercontent.com/atagade/sgtr-em/{SGTR_EM_COMMIT}/{src}"
        print(f"  {url}")
        urllib.request.urlretrieve(url, dest)
    (out / "SOURCE_COMMIT").write_text(f"{SGTR_EM_COMMIT}\n{SGTR_EM_REPO}\n")
    print(f"sgtr-em -> {out}")


def fetch_self_predict():
    from huggingface_hub import snapshot_download

    out = EXTERNAL / "introspection_self_predict"
    snapshot_download(SELF_PREDICT_DATASET, repo_type="dataset", local_dir=out)
    print(f"{SELF_PREDICT_DATASET} -> {out}")


SOURCES = {"sgtr-em": fetch_sgtr_em, "introspection_self_predict": fetch_self_predict}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=SOURCES, action="append", help="fetch only this source (repeatable)")
    args = parser.parse_args()
    for name in args.only or SOURCES:
        SOURCES[name]()


if __name__ == "__main__":
    main()
