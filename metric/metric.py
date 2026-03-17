import numpy as np

def evaluate_metrics(pred, gt):
    """
    pred, gt: numpy arrays of shape (N, T, 63)

    returns:
        mpjpe
        ade
        fde
    """

    assert pred.shape == gt.shape

    N, T, D = pred.shape
    assert D == 63

    # reshape → joints
    pred = pred.reshape(N, T, 21, 3)
    gt = gt.reshape(N, T, 21, 3)

    # joint distance
    joint_error = np.linalg.norm(pred - gt, axis=-1)   # (N, T, 21)

    # MPJPE
    mpjpe = joint_error.mean()

    # ADE (average frame displacement)
    frame_error = joint_error.mean(axis=2)             # (N, T)
    ade = frame_error.mean()

    # FDE (final frame)
    fde = frame_error[:, -1].mean()

    return mpjpe, ade, fde


def mpjpe_at_intervals(pred, gt, fps=30):

    intervals_ms = [80, 320, 560, 720, 880, 1000]

    N, T, _ = pred.shape

    pred = pred.reshape(N, T, 21, 3)
    gt = gt.reshape(N, T, 21, 3)

    joint_error = np.linalg.norm(pred - gt, axis=-1)
    frame_error = joint_error.mean(axis=2)

    results = {}

    for ms in intervals_ms:

        frame = int((ms / 1000) * fps)

        frame = min(frame, T - 1)

        results[ms] = frame_error[:, frame].mean()

    return results