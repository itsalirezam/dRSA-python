import numpy as np
import os
import sys
from pathlib import Path

# drsa matrix shape: (n_models, subsample_tps, subsample_tps)

# ID for identifying batches of the same foler
id = sys.argv[1]

parent_dir = os.getcwd()
parent_dir = Path(parent_dir)
# make dir for saving
os.makedirs(f'batches merged-{id}', exist_ok=True)
save_path = parent_dir / f'batches merged-{id}'

# get directory paths containing the id in their name
target_subdirs = [item for item in parent_dir.iterdir() if item.is_dir() and id in item.name]

if len(target_subdirs) < 1:
    print('no directories found!')

files_by_name = {}
for folder in target_subdirs:
    for file_path in folder.glob('*.npy'):
        filename = file_path.stem
        if filename not in files_by_name:
            files_by_name[filename] = []
        files_by_name[filename].append(file_path)

# iterate over dict keys, load all the npa paths, average all items and save with the key name in a new folder 

for name, paths_list in files_by_name.items():
    single_subject = []
    for path in paths_list:
        npa = np.load(path, allow_pickle=True)
        single_subject.append(npa)
    # stack, average, save, for a single subject
    batches_stacked = np.stack(single_subject, axis=0)
    batches_averaged = np.mean(batches_stacked, axis=0)
    np.save(save_path/f'{name}-{id}-merged', batches_averaged)


# for each folder with the same id, average the files with the same name together and put them in a third foler

# for name in npy file names, find the files with the same name across the three folders in the parent folder (including the folder we are in) with an id like Q25 in their name


# to run: put in a parent directory with folders each containing a full subjects run with similar iterations - provide an argument as an id to use to identify folders as batches of the same analysis