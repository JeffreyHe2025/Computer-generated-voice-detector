"""List the actual folder structure of the MLAAD repo so we can build correct patterns."""

from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_files("mueller91/MLAAD", repo_type="dataset")

# Collect unique top-level and second-level folders
folders = set()
for f in files:
    parts = f.split("/")
    if len(parts) >= 1:
        folders.add(parts[0])
    if len(parts) >= 2:
        folders.add("/".join(parts[:2]))
    if len(parts) >= 3:
        folders.add("/".join(parts[:3]))
    if len(parts) >= 4:
        folders.add("/".join(parts[:4]))

print(f"Total files in repo: {len(files)}\n")
print("Folder structure (up to 4 levels deep):")
for folder in sorted(folders):
    depth = folder.count("/")
    print("  " * depth + folder.split("/")[-1] + "/")

print("\nFirst 10 actual file paths:")
for f in files[:10]:
    print(" ", f)
