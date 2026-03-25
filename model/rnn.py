import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class PoseRNNConfig:
    obs_len: int = 30
    pred_len: int = 30
    pose_dim: int = 63
    latent_dim: int = 32
    hidden_dim: int = 256
    n_layer: int = 1
    dropout: float = 0.0


class PoseRNN(nn.Module):
    """
    Autoregressive RNN for future pose prediction.

    Input:
        obs: (B, obs_len, 63)

    Training:
        pred_pose, pred_latent = model(obs, targets=future_gt)

    Inference:
        pred_pose = model(obs)
    """
    def __init__(self, vp, config: PoseRNNConfig):
        super().__init__()
        self.config = config

        self.obs_len = config.obs_len
        self.pred_len = config.pred_len
        self.pose_dim = config.pose_dim
        self.latent_dim = config.latent_dim
        self.hidden_dim = config.hidden_dim
        self.n_layer = config.n_layer

        self.input_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.rnn = nn.RNN(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layer,
            nonlinearity="tanh",
            batch_first=True,
            dropout=config.dropout if config.n_layer > 1 else 0.0,
        )
        self.output_head = nn.Linear(self.hidden_dim, self.latent_dim)

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params / 1e6:.2f}M")
        self.vp = vp


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RNN):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def _encode_pose_seq(self, pose_seq):
        """
        pose_seq: (B, T, 63)
        return:   (B, T, 32)
        """
        B, T, D = pose_seq.shape
        assert D == self.pose_dim, f"Expected pose dim {self.pose_dim}, got {D}"

        flat = pose_seq.reshape(B * T, D)
        latent = self.vp.encode(flat).mean
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

    def _run_context(self, latent_seq):
        """
        latent_seq: (B, T, 32)
        return:
            outputs: (B, T, hidden_dim)
            h_n:     (n_layer, B, hidden_dim)
        """
        x = self.input_proj(latent_seq)   # (B, T, hidden_dim)
        outputs, h_n = self.rnn(x)        # outputs: (B, T, hidden_dim)
        return outputs, h_n

    def _predict_one_step(self, latent_token, h):
        """
        latent_token: (B, 1, 32)
        h:            (n_layer, B, hidden_dim)

        return:
            next_latent: (B, 1, 32)
            next_h:      (n_layer, B, hidden_dim)
        """
        x = self.input_proj(latent_token)     # (B, 1, hidden_dim)
        out, next_h = self.rnn(x, h)          # out: (B, 1, hidden_dim)
        next_latent = self.output_head(out)   # (B, 1, 32)
        return next_latent, next_h

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

        # Encode observed sequence into hidden state
        _, h = self._run_context(obs_latent)

        # ----------------------------
        # Training: teacher forcing
        # ----------------------------
        if targets is not None:
            B2, T_pred, D2 = targets.shape
            assert B2 == B and D2 == self.pose_dim
            assert T_pred == self.pred_len, f"Expected pred_len={self.pred_len}, got {T_pred}"

            tgt_latent = self._encode_pose_seq(targets)  # (B, pred_len, 32)

            pred_latents = []
            cur_token = obs_latent[:, -1:, :]            # start from last observed latent

            for step in range(self.pred_len):
                next_latent, h = self._predict_one_step(cur_token, h)
                pred_latents.append(next_latent)

                # teacher forcing: next input is GT latent
                cur_token = tgt_latent[:, step:step+1, :]

            pred_latent = torch.cat(pred_latents, dim=1)   # (B, pred_len, 32)
            pred_pose = self._decode_latent_seq(pred_latent)

            return pred_pose, pred_latent

        # ----------------------------
        # Inference: autoregressive rollout
        # ----------------------------
        generated = []
        cur_token = obs_latent[:, -1:, :]

        for _ in range(self.pred_len):
            next_latent, h = self._predict_one_step(cur_token, h)
            generated.append(next_latent)
            cur_token = next_latent

        pred_latent = torch.cat(generated, dim=1)         # (B, pred_len, 32)
        pred_pose = self._decode_latent_seq(pred_latent)

        return pred_pose