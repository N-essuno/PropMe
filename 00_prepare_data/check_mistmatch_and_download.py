import shutil
from pathlib import Path
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile

REPO_ID = "allenai/dolma3_mix-6T"
REMOTE_PREFIX = "data"          # compare only repo/data/*
LOCAL_ROOT = Path("/work/olmotrace/dolma3/data")  # folder containing common_crawl-..., stack_edu-..., etc.

api = HfApi()

# 1. Remote files from HF under data/
remote_entries = api.list_repo_tree(
    repo_id=REPO_ID,
    repo_type="dataset",
    path_in_repo=REMOTE_PREFIX,
    recursive=True,
)

remote_files = {}
remote_sha256 = {}

for e in remote_entries:
    if isinstance(e, RepoFile):
        # Convert remote path "data/foo/bar.zst" -> local path "foo/bar.zst"
        rel = Path(e.path).relative_to(REMOTE_PREFIX).as_posix()
        remote_files[rel] = e.size

        # Available for LFS files; useful if you want hash checking too
        if e.lfs and "sha256" in e.lfs:
            remote_sha256[rel] = e.lfs["sha256"]

# 2. Local files
local_files = {}
for p in LOCAL_ROOT.rglob("*"):
    if p.is_file():
        rel = p.relative_to(LOCAL_ROOT).as_posix()

        # Ignore HF local metadata if you used snapshot_download(local_dir=...)
        if rel.startswith(".cache/huggingface/"):
            continue

        local_files[rel] = p.stat().st_size

# 3. Compare
remote_set = set(remote_files)
local_set = set(local_files)

missing = sorted(remote_set - local_set)
extra = sorted(local_set - remote_set)

size_mismatches = sorted(
    rel for rel in (remote_set & local_set)
    if remote_files[rel] != local_files[rel]
)

print(f"Remote files: {len(remote_files):,}")
print(f"Local files:  {len(local_files):,}")
print(f"Missing:      {len(missing):,}")
print(f"Extra:        {len(extra):,}")
print(f"Size mismatch:{len(size_mismatches):,}")

print("\nMissing examples:")
for x in missing[:50]:
    print("  MISSING", x, remote_files[x])

print("\nExtra examples:")
for x in extra[:50]:
    print("  EXTRA", x, local_files[x])

# 4. File-level size differences
print(f"\nFiles with size differences: {len(size_mismatches):,}")
for x in size_mismatches[:100]:  # Limit output to first 100 for brevity, adjust if needed
    local_size = local_files[x]
    remote_size = remote_files[x]
    delta = remote_size - local_size
    print(f"{x}: local={local_size:,} remote={remote_size:,} delta={delta:,}")

# 5. Folder-level size summary
remote_folder_sizes = defaultdict(int)
local_folder_sizes = defaultdict(int)

for rel, size in remote_files.items():
    folder = rel.split("/", 1)[0]
    remote_folder_sizes[folder] += size

for rel, size in local_files.items():
    folder = rel.split("/", 1)[0]
    local_folder_sizes[folder] += size

folder_problems = []
for folder in sorted(set(remote_folder_sizes) | set(local_folder_sizes)):
    r = remote_folder_sizes.get(folder, 0)
    l = local_folder_sizes.get(folder, 0)
    if r != l:
        folder_problems.append((folder, l, r, r - l))

print(f"\nFolders with size differences: {len(folder_problems):,}")
for folder, local_size, remote_size, delta in folder_problems[:100]:
    print(f"{folder}: local={local_size:,} remote={remote_size:,} delta={delta:,}")

def download_missing_and_mismatched(missing_list, mismatch_list, repo_id, remote_prefix, local_root):
    files_to_download = set(missing_list) | set(mismatch_list)
    if not files_to_download:
        print("\nNo files need downloading.")
        return

    print(f"\nStarting download for {len(files_to_download)} files...")
    for rel in files_to_download:
        remote_path = f"{remote_prefix}/{rel}"
        local_path = Path(local_root) / rel
        
        print(f"Downloading {remote_path}...")
        try:
            # Download to HF cache
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                repo_type="dataset"
            )
            
            # Ensure local directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy from cache to our actual local path
            shutil.copy2(cached_path, local_path)
            print(f"  -> Saved to {local_path}")
        except Exception as e:
            print(f"  -> Error downloading {remote_path}: {e}")

# Download the missing and mismatched files
download_missing_and_mismatched(missing, size_mismatches, REPO_ID, REMOTE_PREFIX, LOCAL_ROOT)