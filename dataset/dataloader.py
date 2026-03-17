import glob
import os
import pickle

import numpy as np
import torch.utils.data as data
import torch

class PoseDataset(data.Dataset):
    def __init__(self, root='data', split='train', device='cuda',
                    obs_len=30,      # 1 sec at 30 fps
                    pred_len=30,     # 1 sec at 30 fps
                    stride=1):
        data_path = os.path.join(root, 'dataset_all.pkl')
        dataset_info = pickle.load(open(data_path, 'rb+'))

        self.data_id = dataset_info.partition[split]
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.stride = stride
        self.device = device

        self.samples = []

        for seq_idx, file_path in enumerate(self.data_id):
            poses = np.load(file_path)['poses'][::4][:, 3:66]   # 120 -> 30 fps
            seq_len = poses.shape[0]
            total_len = self.obs_len + self.pred_len

            if seq_len < total_len:
                continue

            for start in range(0, seq_len - total_len + 1, self.stride):
                self.samples.append((seq_idx, start))

    
    def __getitem__(self, index):
        seq_idx, start = self.samples[index]
        file_path = self.data_id[seq_idx]

        body_pose = np.load(file_path)['poses'][::4][:, 3:66]
        body_pose = torch.from_numpy(body_pose).float()

        obs = body_pose[start : start + self.obs_len]  # (obs_len, 63)
        pred = body_pose[
            start + self.obs_len : start + self.obs_len + self.pred_len
        ]  # (pred_len, 63)

        return obs, pred

    def __len__(self):
        return len(self.samples)