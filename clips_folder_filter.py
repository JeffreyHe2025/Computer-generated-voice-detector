import os
import pandas as pd
import shutil


tsv_path = "C:/Users/jeffr/OneDrive/Documents/Desktop/AI voice detector/df_high_quality_pre_placeholder.tsv"


source_folder = "C:/Users/jeffr/Downloads/human voices large collection/cv-corpus-24.0-2025-12-05/en/clips"


new_folder = "C:/Users/jeffr/OneDrive/Documents/Desktop/AI voice detector/filtered_human_clips"


os.makedirs(new_folder, exist_ok=True)


print("Reading TSV file...")
df = pd.read_csv(tsv_path, sep='\t')

# Grab all the exact filenames from the 'path' column
filenames_to_keep = df['path'].tolist()
print(f"Found {len(filenames_to_keep)} files to copy. Starting transfer...")

# 4. Copy the files over
files_copied = 0
files_missing = 0

for filename in filenames_to_keep:
    source_path = os.path.join(source_folder, filename)
    dest_path = os.path.join(new_folder, filename)
    
    # Safety check: Ensure the file actually exists in the giant folder
    if os.path.exists(source_path):
        # shutil.copy2 copies the file AND preserves its original metadata
        shutil.copy2(source_path, dest_path)
        files_copied += 1
        
        # Optional: Print progress every 1000 files so you know it hasn't frozen
        if files_copied % 1000 == 0:
            print(f"Copied {files_copied} files so far...")
    else:
        print(f"Warning: Could not find {filename} in the source folder.")
        files_missing += 1

print("-" * 40)
print(f"Success! Copied {files_copied} files into {new_folder}")
if files_missing > 0:
    print(f"Note: {files_missing} files were listed in the TSV but missing from the folder.")