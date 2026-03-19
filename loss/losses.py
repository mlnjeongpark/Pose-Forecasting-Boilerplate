import torch
import torch.nn.functional as F

def latent_loss(pred_latent, gt_latent, reg_weight=0.01):
    """
    pred_latent: (B, T, 32) - Transformer의 예측값
    gt_latent: (B, T, 32) - VPoser로 인코딩된 정답값
    reg_weight: 정규화 강도를 조절하는 하이퍼파라미터
    """
    
    # 1. Smooth L1 Loss (Huber Loss)
    # 오차가 작을 때는 세밀하게(MSE), 클 때는 덜 민감하게(MAE) 학습합니다.
    recon_loss = F.smooth_l1_loss(pred_latent, gt_latent, beta=0.0001)
    
    # 2. Latent Regularization (L2 Penalty)
    # 예측된 벡터의 크기(Norm) 자체에 페널티를 주어, 값이 불필요하게 커지는 것을 억제합니다.
    # VAE의 N(0, I) prior 특성을 유지하도록 돕습니다.
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