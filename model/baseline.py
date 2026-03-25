import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroVelocity(nn.Module):
    def __init__(self, vp, pred_len):
        super().__init__()
        self.vp = vp
        self.pred_len = pred_len

    def forward(self, obs, targets=None):
        """
        obs:    (B, T_obs, D)
        targets: (B, T_pred, D) or None
        """

        last = obs[:, -1:, :]   # (B, 1, D)

        if targets is not None:
            T_pred = targets.shape[1]
        else:
            T_pred = self.pred_len
            
        pred = last.repeat(1, T_pred, 1)  # (B, T_pred, D)

        loss = None
        if targets is not None:
            loss = F.mse_loss(pred, targets)

        return pred
    

class ConstantVelocity(nn.Module):
    def __init__(self, vp, pred_len):
        super().__init__()
        self.vp = vp
        self.pred_len = pred_len

    def forward(self, obs, targets=None):
        """
        obs:    (B, T_obs, D)
        targets: (B, T_pred, D) or None
        """

        if obs.shape[1] < 2:
            raise ValueError("Need at least 2 frames to compute velocity")

        v = obs[:, -1, :] - obs[:, -2, :]   # (B, D)

        cur = obs[:, -1, :]   # (B, D)

        if targets is not None:
            T_pred = targets.shape[1]
        else:
            T_pred = self.pred_len

        preds = []

        for _ in range(T_pred):
            cur = cur + v
            preds.append(cur)

        pred = torch.stack(preds, dim=1)   # (B, T_pred, D)

        loss = None
        if targets is not None:
            loss = F.mse_loss(pred, targets)

        return pred