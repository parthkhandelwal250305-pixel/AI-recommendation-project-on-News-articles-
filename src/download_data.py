"""
Download MIND-small: pull only MINDsmall_train.zip and MINDsmall_dev.zip
directly from the Recommenders/MIND Hugging Face repo, then extract them
ourselves. These are the original, unmodified MIND files (news.tsv,
behaviors.tsv) -- same format as the official Microsoft release, just
served from a host that's actually still up.

We deliberately do NOT use `datasets.load_dataset()` here: that repo also
contains MINDlarge_*, MINDdemo_*, and an unrelated entity-embeddings file,
and the auto-loader tries to merge all of them into one table, which
fails (different files have different columns -- that's expected, not a
bug in our code).

Usage:
    python src/download_data.py
"""
import os
import zipfile
import shutil

from huggingface_hub import hf_hub_download

REPO_ID = "Recommenders/MIND"
REPO_TYPE = "dataset"
FILES = ["MINDsmall_train.zip", "MINDsmall_dev.zip"]

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    for fname in FILES:
        print(f"Fetching {fname} from {REPO_ID} ...")
        cached_path = hf_hub_download(repo_id=REPO_ID, filename=fname, repo_type=REPO_TYPE)
        print(f"  -> cached at {cached_path}")

        local_zip = os.path.join(RAW_DIR, fname)
        shutil.copy(cached_path, local_zip)

        extract_to = os.path.join(RAW_DIR, fname.replace(".zip", ""))
        os.makedirs(extract_to, exist_ok=True)
        print(f"  Extracting to {extract_to} ...")
        with zipfile.ZipFile(local_zip, "r") as z:
            z.extractall(extract_to)

    print("\nDone. Verifying expected files exist:")
    for split in ["MINDsmall_train", "MINDsmall_dev"]:
        for needed in ["news.tsv", "behaviors.tsv"]:
            path = os.path.join(RAW_DIR, split, needed)
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"  [OK] {path} ({size / 1e6:.1f} MB)")
            else:
                print(f"  [MISSING] {path}")


if __name__ == "__main__":
    main()
