import os, glob, random
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from easydict import EasyDict
import pickle

root = '/scratch/mkp6221/CSE586/Project-1/my-code/data'


def split_files(
    root: str,
    test_ratio: float = 0.2,
    seed: int = 42,
    pattern: str = "*_poses.npz",
) -> Tuple[List[str], List[str]]:
    files = sorted(glob.glob(os.path.join(root, 'AMASS_CMUsubset' ,pattern)))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found in {root} with pattern {pattern}")

    rng = random.Random(seed)
    rng.shuffle(files)

    n_test = int(round(len(files) * test_ratio))
    test_files = sorted(files[:n_test])
    train_files = sorted(files[n_test:])
    return train_files, test_files


train_files, test_files = split_files(root, test_ratio=0.2, seed=586)

print(len(train_files), len(test_files))
print("train example:")
for i in range(len(train_files)):
    print(train_files[i])
print("test example:")
for i in range(len(test_files)):
    print(test_files[i])


save_dir = 'data'
dataset = EasyDict()
dataset.description = 'AMASS_CMUsubset'
dataset.root = os.path.join(root, dataset.description)

dataset.partition = EasyDict()

dataset.partition.train = train_files
dataset.partition.test = test_files

with open(os.path.join(save_dir, 'dataset_all.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)