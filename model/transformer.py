import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PoseTransformerConfig:
    obs_len: int = 30
    pred_len: int = 30
    pose_dim: int = 63          # SMPL body pose dim
    latent_dim: int = 32        # VPoser latent dim
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    @property
    def block_size(self):
        # obs + shifted future tokens
        return self.obs_len + self.pred_len


class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (
            1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
            )
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: PoseTransformerConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(self.block_size, self.block_size)).view(
                1, 1, self.block_size, self.block_size
            )
        )

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config: PoseTransformerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)

        self.mlp = nn.ModuleDict(dict(
            c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj = nn.Linear(4 * config.n_embd, config.n_embd),
            act    = NewGELU(),
            drop   = nn.Dropout(config.dropout),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.drop(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class PoseTransformer(nn.Module):
    """
    Autoregressive transformer for future pose prediction.

    Input:
        obs: (B, obs_len, 63)

    Training:
        pred = model(obs, targets=future_gt)
        -> pred shape: (B, pred_len, 63)

    Inference:
        pred = model(obs)
        -> autoregressively predicts pred_len frames
    """
    def __init__(self, vp, config: PoseTransformerConfig):
        super().__init__()
        self.vp = vp
        self.config = config

        self.obs_len = config.obs_len
        self.pred_len = config.pred_len
        self.pose_dim = config.pose_dim
        self.latent_dim = config.latent_dim
        self.block_size = config.block_size

        self.input_proj = nn.Linear(config.latent_dim, config.n_embd)
        self.pos_emb = nn.Embedding(self.block_size, config.n_embd)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.latent_dim)

        # learned start token for decoder side
        self.start_token = nn.Parameter(torch.zeros(1, 1, config.latent_dim))

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params / 1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def get_block_size(self):
        return self.block_size

    def _encode_pose_seq(self, pose_seq):
        """
        pose_seq: (B, T, 63)
        return:   (B, T, 32)
        """
        B, T, D = pose_seq.shape
        assert D == self.pose_dim, f"Expected pose dim {self.pose_dim}, got {D}"

        flat = pose_seq.reshape(B * T, D)
        latent = self.vp.encode(flat).mean   # (B*T, 32)
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
        pose = self.vp.decode(flat)['pose_body'].contiguous()   # (B*T, 63)
        pose = pose.view(B, T, self.pose_dim)
        return pose

    def _run_transformer(self, seq_latent):
        """
        seq_latent: (B, T, 32)
        return:     (B, T, 32)
        """
        B, T, D = seq_latent.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"

        device = seq_latent.device
        pos = torch.arange(0, T, device=device).unsqueeze(0)  # (1, T)

        tok_emb = self.input_proj(seq_latent)   # (B, T, n_embd)
        pos_emb = self.pos_emb(pos)             # (1, T, n_embd)

        h = tok_emb + pos_emb
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)

        pred_latent = self.head(h)              # (B, T, 32)
        return pred_latent

    def forward(self, obs, targets=None):
        """
        obs:     (B, obs_len, 63)
        targets: (B, pred_len, 63) or None

        returns:
            pred_pose: (B, pred_len, 63)
        """
        B, T_obs, D = obs.shape
        assert D == self.pose_dim
        assert T_obs == self.obs_len, f"Expected obs_len={self.obs_len}, got {T_obs}"

        obs_latent = self._encode_pose_seq(obs)   # (B, obs_len, 32)

        # ----------------------------
        # Training: teacher forcing
        # ----------------------------
        if targets is not None:
            B2, T_pred, D2 = targets.shape
            assert B2 == B and D2 == self.pose_dim
            assert T_pred == self.pred_len, f"Expected pred_len={self.pred_len}, got {T_pred}"

            tgt_latent = self._encode_pose_seq(targets)   # (B, pred_len, 32)

            # shift-right for decoder-style autoregression
            start_tok = self.start_token.expand(B, 1, self.latent_dim)   # (B,1,32)
            future_in = torch.cat([start_tok, tgt_latent[:, :-1, :]], dim=1)  # (B, pred_len, 32)

            full_in = torch.cat([obs_latent, future_in], dim=1)   # (B, obs_len + pred_len, 32)

            full_out = self._run_transformer(full_in)              # (B, obs_len + pred_len, 32)
            pred_latent = full_out[:, -self.pred_len:, :]         # only future part
            pred_pose = self._decode_latent_seq(pred_latent)      # (B, pred_len, 63)

            return pred_pose

        # ----------------------------
        # Inference: autoregressive rollout
        # ----------------------------
        generated = []
        seq = obs_latent

        for step in range(self.pred_len):
            if step == 0:
                next_in = self.start_token.expand(B, 1, self.latent_dim)
            else:
                next_in = generated[-1]

            seq_in = torch.cat([seq, next_in], dim=1)

            # keep only last block_size tokens if needed
            if seq_in.size(1) > self.block_size:
                seq_in = seq_in[:, -self.block_size:, :]

            out = self._run_transformer(seq_in)
            next_latent = out[:, -1:, :]   # predict current next token
            generated.append(next_latent)
            seq = torch.cat([seq, next_latent], dim=1)

        pred_latent = torch.cat(generated, dim=1)      # (B, pred_len, 32)
        pred_pose = self._decode_latent_seq(pred_latent)   # (B, pred_len, 63)

        return pred_pose