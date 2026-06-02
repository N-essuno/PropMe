from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

repo_id = "common-pile/comma_v0.1_training_dataset"
out_dir = Path("/work/olmotrace/common_pile_train/compressed")
out_dir.mkdir(parents=True, exist_ok=True)

all_files = list_repo_files(
    repo_id=repo_id,
    repo_type="dataset",
)

train_files = sorted(
    f for f in all_files
    if f.endswith(".jsonl.gz")
)

print(f"Found {len(train_files)} train files to download from {repo_id}")

for filename in tqdm(train_files, desc="Downloading shards"):
    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        local_dir=out_dir,
    )
    print(f"Downloaded: {local_path}")

print(f"\nDone. Files saved under: {out_dir}")

