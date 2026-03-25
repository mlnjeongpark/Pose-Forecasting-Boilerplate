import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# -------------------------------------------------
# config
# -------------------------------------------------

@dataclass
class DiffusionTransformerConfig:
    obs_len: int = 30
    pred_len: int = 30
    pose_dim: int = 63
    latent_dim: int = 32

    n_embd: int = 256
    n_layer: int = 6
    n_head: int = 8
    dropout: float = 0.1

    diffusion_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2


# -------------------------------------------------
# timestep embedding
# -------------------------------------------------

def sinusoidal_timestep_embedding(timesteps, dim, max_period=10000):
    """
    timesteps: (B,) int64
    return:    (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) *
        torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# -------------------------------------------------
# transformer blocks
# -------------------------------------------------

class DiTBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(n_embd)

        hidden_dim = int(n_embd * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.ln_1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.ln_2(x))
        return x


class FutureDiffusionBackbone(nn.Module):
    """
    Condition on observed latent sequence and noisy future latent sequence.
    Predict epsilon for the future block.
    """
    def __init__(self, cfg: DiffusionTransformerConfig):
        super().__init__()
        self.obs_len = cfg.obs_len
        self.pred_len = cfg.pred_len
        self.latent_dim = cfg.latent_dim
        self.n_embd = cfg.n_embd
        self.total_len = cfg.obs_len + cfg.pred_len

        self.obs_proj = nn.Linear(cfg.latent_dim, cfg.n_embd)
        self.future_proj = nn.Linear(cfg.latent_dim, cfg.n_embd)

        self.pos_emb = nn.Parameter(torch.zeros(1, self.total_len, cfg.n_embd))

        # obs token / future token 구분용
        self.obs_token_type = nn.Parameter(torch.zeros(1, 1, cfg.n_embd))
        self.future_token_type = nn.Parameter(torch.zeros(1, 1, cfg.n_embd))

        self.t_embed = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.n_embd),
            nn.SiLU(),
            nn.Linear(cfg.n_embd, cfg.n_embd),
        )

        self.blocks = nn.ModuleList([
            DiTBlock(cfg.n_embd, cfg.n_head, cfg.dropout)
            for _ in range(cfg.n_layer)
        ])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.out = nn.Linear(cfg.n_embd, cfg.latent_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.obs_token_type, std=0.02)
        nn.init.normal_(self.future_token_type, std=0.02)

    def forward(self, obs_latent, noisy_future, t):
        """
        obs_latent:   (B, obs_len, latent_dim)
        noisy_future: (B, pred_len, latent_dim)
        t:            (B,)
        return:
            eps_pred:  (B, pred_len, latent_dim)
        """
        B = obs_latent.shape[0]

        obs_tok = self.obs_proj(obs_latent) + self.obs_token_type
        fut_tok = self.future_proj(noisy_future) + self.future_token_type

        x = torch.cat([obs_tok, fut_tok], dim=1)  # (B, obs+pred, n_embd)
        x = x + self.pos_emb[:, :x.shape[1], :]

        t_emb = sinusoidal_timestep_embedding(t, self.n_embd)
        t_emb = self.t_embed(t_emb).unsqueeze(1)   # (B, 1, n_embd)
        x = x + t_emb

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)

        fut_h = x[:, self.obs_len:, :]
        eps_pred = self.out(fut_h)
        return eps_pred


# -------------------------------------------------
# main model
# -------------------------------------------------

class DiffusionTransformer(nn.Module):
    """
    Conditional block diffusion transformer.

    Training:
        loss = model(obs, targets)

    Inference:
        pred_pose, pred_latent = model(obs)
    """
    def __init__(self, vp, config: DiffusionTransformerConfig):
        super().__init__()
        self.config = config

        self.obs_len = config.obs_len
        self.pred_len = config.pred_len
        self.pose_dim = config.pose_dim
        self.latent_dim = config.latent_dim
        self.diffusion_steps = config.diffusion_steps

        self.backbone = FutureDiffusionBackbone(config)

        # DDPM schedule
        betas = torch.linspace(
            config.beta_start, config.beta_end, config.diffusion_steps
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        alpha_bars_prev = torch.cat(
            [torch.ones(1, device=betas.device), alpha_bars[:-1]], dim=0
        )
        posterior_var = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        posterior_var[0] = betas[0]
        self.register_buffer("posterior_var", posterior_var)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params / 1e6:.2f}M")
        self.vp = vp

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

    # --------------------------------------------
    # diffusion helpers
    # --------------------------------------------
    def q_sample(self, x0, t, noise=None):
        """
        x0:    (B, pred_len, latent_dim)
        t:     (B,)
        noise: same shape as x0 or None
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        xt = sqrt_ab * x0 + sqrt_1mab * noise
        return xt, noise

    def p_sample(self, obs_latent, xt, t_scalar):
        """
        obs_latent: (B, obs_len, latent_dim)
        xt:         (B, pred_len, latent_dim)
        """
        B = xt.shape[0]
        t = torch.full((B,), t_scalar, device=xt.device, dtype=torch.long)

        eps_pred = self.backbone(obs_latent, xt, t)

        alpha_t = self.alphas[t].view(-1, 1, 1)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t].view(-1, 1, 1)

        mean = sqrt_recip_alpha_t * (
            xt - (beta_t / torch.sqrt(1.0 - alpha_bar_t).clamp(min=1e-8)) * eps_pred
        )

        if t_scalar > 0:
            noise = torch.randn_like(xt)
            var = self.posterior_var[t].view(-1, 1, 1)
            xt_prev = mean + torch.sqrt(var.clamp(min=1e-20)) * noise
        else:
            xt_prev = mean

        return xt_prev

    # --------------------------------------------
    # training / inference
    # --------------------------------------------
    def diffusion_loss(self, obs, targets):
        """
        obs:     (B, obs_len, 63)
        targets: (B, pred_len, 63)
        """
        obs_latent = self._encode_pose_seq(obs)
        target_latent = self._encode_pose_seq(targets)

        B = obs.shape[0]
        t = torch.randint(
            low=0,
            high=self.diffusion_steps,
            size=(B,),
            device=obs.device,
            dtype=torch.long,
        )

        noisy_future, noise = self.q_sample(target_latent, t)
        eps_pred = self.backbone(obs_latent, noisy_future, t)

        loss = F.mse_loss(eps_pred, noise)
        return loss

    @torch.no_grad()
    def sample_latent(self, obs):
        """
        obs: (B, obs_len, 63)
        return:
            pred_latent: (B, pred_len, 32)
        """
        self.eval()
        obs_latent = self._encode_pose_seq(obs)

        B = obs.shape[0]
        xt = torch.randn(
            B,
            self.pred_len,
            self.latent_dim,
            device=obs.device,
            dtype=obs.dtype,
        )

        for t in reversed(range(self.diffusion_steps)):
            xt = self.p_sample(obs_latent, xt, t)

        return xt

    @torch.no_grad()
    def sample_pose(self, obs):
        pred_latent = self.sample_latent(obs)
        pred_pose = self._decode_latent_seq(pred_latent)
        return pred_pose, pred_latent

    def forward(self, obs, targets=None):
        """
        Training:
            loss = model(obs, targets)

        Inference:
            pred_pose, pred_latent = model(obs)
        """
        B, T_obs, D = obs.shape
        assert D == self.pose_dim
        assert T_obs == self.obs_len, f"Expected obs_len={self.obs_len}, got {T_obs}"

        if targets is not None:
            B2, T_pred, D2 = targets.shape
            assert B2 == B and D2 == self.pose_dim
            assert T_pred == self.pred_len, f"Expected pred_len={self.pred_len}, got {T_pred}"
            return self.diffusion_loss(obs, targets)

        return self.sample_pose(obs)