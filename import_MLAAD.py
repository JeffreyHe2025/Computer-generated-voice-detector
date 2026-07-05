"""
Download English MLAAD samples evenly across ALL engines — 200 files from each
engine, so coverage is balanced instead of biased toward whichever engines
sort first alphabetically. Existing cached files from earlier downloads are
reused (not re-downloaded).
"""

import os
import time
from huggingface_hub import HfApi, hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

REPO_ID = "mueller91/MLAAD"
LOCAL_DIR = "mlaad_clips"
SAMPLES_PER_ENGINE = 200
NUM_WORKERS = 8

api = HfApi()


def list_with_retry(path_in_repo, recursive=False):
    """list_repo_tree with exponential-backoff retry on 504s."""
    for attempt in range(5):
        try:
            return list(api.list_repo_tree(
                REPO_ID, repo_type="dataset",
                path_in_repo=path_in_repo, recursive=recursive,
            ))
        except Exception as e:
            wait = 2 ** attempt
            print(f"  retry {attempt+1}/5 for {path_in_repo} after {wait}s: {e}")
            time.sleep(wait)
    raise RuntimeError(f"Failed to list {path_in_repo} after 5 retries")


print("Listing English engine folders under fake/en/ ...")
top_level = list_with_retry("fake/en", recursive=False)
engine_folders = [e.path for e in top_level if e.path.startswith("fake/en/")]
print(f"Found {len(engine_folders)} English engine folders.\n")

print(f"Selecting up to {SAMPLES_PER_ENGINE} files from each engine...")
selected_files = []
for folder in tqdm(engine_folders, desc="Engines"):
    entries = list_with_retry(folder, recursive=False)
    audio = [e.path for e in entries if e.path.lower().endswith((".wav", ".mp3", ".flac"))]
    selected_files.extend(audio[:SAMPLES_PER_ENGINE])

print(f"\nTotal files selected: {len(selected_files)} across {len(engine_folders)} engines")
print(f"Downloading into ./{LOCAL_DIR}/  (cached files are reused, not re-downloaded)\n")
os.makedirs(LOCAL_DIR, exist_ok=True)


def download_one(path):
    return hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset",
        filename=path, local_dir=LOCAL_DIR,
    )


with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
    futures = [ex.submit(download_one, p) for p in selected_files]
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
        try:
            fut.result()
        except Exception as e:
            print(f"  Skipping due to error: {e}")

print(f"\nDone. Files are in ./{LOCAL_DIR}/")
