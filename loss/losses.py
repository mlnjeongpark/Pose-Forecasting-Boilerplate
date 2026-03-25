import torch
import torch.nn.functional as F

def l1l2_loss(pred_latent, gt_latent, reg_weight=0.01):
    """
    pred_latent: (B, T, 32) 
    gt_latent: (B, T, 32) 
    reg_weight
    """
    
    # 1. Smooth L1 Loss (Huber Loss)
    recon_loss = F.smooth_l1_loss(pred_latent, gt_latent, beta=0.05)
    
    # 2. Latent Regularization (L2 Penalty)
    reg_loss = torch.mean(pred_latent ** 2)
    
    total_loss = recon_loss # + (reg_weight * reg_loss)
    
    return total_loss

def mse_loss(pred, target, reduction="mean"):
    return F.mse_loss(pred, target, reduction=reduction)


def mpjpe_loss(pred, target, reduction="mean"):
    """
    Mean Per Joint Position Error (MPJPE).

    Computes mean Euclidean distance between predicted
    and ground-truth joint positions.

    Args:
        pred:   (B, T, J, 3)
        target: (B, T, J, 3)
        reduction: "mean" | "sum" | "none"

    Returns:
        scalar MPJPE value (meters or mm depending on input units)
    """

    assert pred.shape == target.shape, "Prediction and target must have same shape"
    # assert pred.shape[-1] == 3, "Last dimension must be 3 (x,y,z)"

    error = torch.norm(pred - target, dim=-1)  # (B, T, J)

    if reduction == "mean":
        return error.mean()
    elif reduction == "sum":
        return error.sum()
    elif reduction == "none":
        return error
    else:
        raise ValueError(f"Invalid reduction: {reduction}")