import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class PoseMLPConfig:
    obs_len: int = 30
    pred_len: int = 30
    pose_dim: int = 63
    latent_dim: int = 32
    hidden_dim: int = 512
    n_layer: int = 3
    dropout: float = 0.1


class PoseMLP(nn.Module):
    """
    Autoregressive MLP for future pose prediction.

    Input:
        obs: (B, obs_len, 63)

    Training:
        pred_pose, pred_latent = model(obs, targets=future_gt)
        -> pred_pose:   (B, pred_len, 63)
        -> pred_latent: (B, pred_len, 32)

    Inference:
        pred_pose = model(obs)
        -> autoregressively predicts pred_len frames
    """
    def __init__(self, vp, config: PoseMLPConfig):
        super().__init__()
        self.config = config

        self.obs_len = config.obs_len
        self.pred_len = config.pred_len
        self.pose_dim = config.pose_dim
        self.latent_dim = config.latent_dim
        self.hidden_dim = config.hidden_dim
        self.n_layer = config.n_layer

        in_dim = self.obs_len * self.latent_dim
        out_dim = self.latent_dim   # one next latent at a time

        layers = []
        prev_dim = in_dim
        for _ in range(self.n_layer - 1):
            layers.append(nn.Linear(prev_dim, self.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            prev_dim = self.hidden_dim

        layers.append(nn.Linear(prev_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params / 1e6:.2f}M")
        self.vp = vp


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _encode_pose_seq(self, pose_seq):
        """
        pose_seq: (B, T, 63)
        return:   (B, T, 32)
        """
        B, T, D = pose_seq.shape
        assert D == self.pose_dim, f"Expected pose dim {self.pose_dim}, got {D}"

        flat = pose_seq.reshape(B * T, D)
        latent = self.vp.encode(flat).mean   # (B*T, latent_dim)
        latent = latent.view(B, T, self.latent_dim)
        return latent

    def _decode_latent_seq(self, latent_seq):
        """
        latent_seq: (B, T, 32)
        return:     (B, T, 63)
        """
        B, T, D = latent_seq.shape
        assert D == self.latent_dim, f"Expected latent dim {self.latent_dim}, got {D}"

        flat = latent_seq.reshape(B * T, D)
        pose = self.vp.decode(flat)["pose_body"].contiguous()
        pose = pose.view(B, T, self.pose_dim)
        return pose

    def _predict_next_latent(self, context_latent):
        """
        context_latent: (B, obs_len, latent_dim)
        return:         (B, 1, latent_dim)
        """
        B, T, D = context_latent.shape
        assert T == self.obs_len, f"Expected obs_len={self.obs_len}, got {T}"
        assert D == self.latent_dim, f"Expected latent_dim={self.latent_dim}, got {D}"

        x = context_latent.reshape(B, T * D)   # (B, obs_len * latent_dim)
        next_latent = self.mlp(x)              # (B, latent_dim)
        next_latent = next_latent.unsqueeze(1) # (B, 1, latent_dim)
        return next_latent

    def forward(self, obs, targets=None):
        """
        obs:     (B, obs_len, 63)
        targets: (B, pred_len, 63) or None

        returns:
            if training:
                pred_pose, pred_latent
            if inference:
                pred_pose
        """
        B, T_obs, D = obs.shape
        assert D == self.pose_dim
        assert T_obs == self.obs_len, f"Expected obs_len={self.obs_len}, got {T_obs}"

        obs_latent = self._encode_pose_seq(obs)   # (B, obs_len, 32)

        # ----------------------------
        # Training: teacher forcing AR rollout
        # ----------------------------
        if targets is not None:
            B2, T_pred, D2 = targets.shape
            assert B2 == B and D2 == self.pose_dim
            assert T_pred == self.pred_len, f"Expected pred_len={self.pred_len}, got {T_pred}"

            tgt_latent = self._encode_pose_seq(targets)   # (B, pred_len, 32)

            pred_latents = []
            context = obs_latent

            for step in range(self.pred_len):
                next_latent = self._predict_next_latent(context)   # (B,1,32)
                pred_latents.append(next_latent)

                # teacher forcing:
                # use GT latent as the next token in the context window
                gt_next = tgt_latent[:, step:step+1, :]            # (B,1,32)
                context = torch.cat([context[:, 1:, :], gt_next], dim=1)

            pred_latent = torch.cat(pred_latents, dim=1)           # (B,pred_len,32)
            pred_pose = self._decode_latent_seq(pred_latent)       # (B,pred_len,63)

            return pred_pose, pred_latent

        # ----------------------------
        # Inference: autoregressive rollout
        # ----------------------------
        generated = []
        context = obs_latent

        for _ in range(self.pred_len):
            next_latent = self._predict_next_latent(context)       # (B,1,32)
            generated.append(next_latent)
            context = torch.cat([context[:, 1:, :], next_latent], dim=1)

        pred_latent = torch.cat(generated, dim=1)                  # (B,pred_len,32)
        pred_pose = self._decode_latent_seq(pred_latent)          # (B,pred_len,63)

        return pred_pose