import os
import shutil
from random import sample

source_dir = '/home/wangzhenyang/data/gf/datasets/mp'
target_dir = '/home/wangzhenyang/data/gf/datasets/testsmall'

folders = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

folders_to_copy = sample(folders, min(100, len(folders)))

for folder in folders_to_copy:
    src_folder_path = os.path.join(source_dir, folder)
    dest_folder_path = os.path.join(target_dir, folder)
    if not os.path.exists(dest_folder_path):
        shutil.copytree(src_folder_path, dest_folder_path)
    else:
        print(f"Folder {folder} already exists in the target directory.")

print("Folder copying complete.")