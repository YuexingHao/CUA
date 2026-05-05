"""Phase 1: Trajectory segmentation via action discontinuities."""

import numpy as np
import torch

from .data import TrajectoryData


def compute_discontinuities(actions: torch.Tensor) -> torch.Tensor:
    """Compute L2 norm of consecutive action differences.
    Args: actions (T, d_action)
    Returns: disc (T-1,)
    """
    if actions.shape[0] < 2:
        return torch.tensor([])
    return torch.norm(actions[1:] - actions[:-1], dim=1)


def detect_boundaries(disc: torch.Tensor, theta: float) -> list[int]:
    """Find boundary indices where discontinuity exceeds threshold.
    Returns action indices where new segments start (offset by 1 from disc indices).
    """
    if disc.numel() == 0:
        return []
    return (torch.where(disc > theta)[0] + 1).tolist()


def boundary_f1(pred: list[int], gt: list[int], tolerance: int = 1) -> dict:
    """Compute boundary precision, recall, F1 with tolerance window."""
    if not gt and not pred:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gt:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    if not pred:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

    gt_set = set(gt)
    pred_set = set(pred)

    # Precision: fraction of predicted boundaries near a ground-truth boundary
    tp_pred = sum(1 for p in pred_set if any(abs(p - g) <= tolerance for g in gt_set))
    precision = tp_pred / len(pred_set) if pred_set else 0.0

    # Recall: fraction of ground-truth boundaries near a predicted boundary
    tp_gt = sum(1 for g in gt_set if any(abs(g - p) <= tolerance for p in pred_set))
    recall = tp_gt / len(gt_set) if gt_set else 0.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def sweep_theta(traj_data_list: list[TrajectoryData],
                percentiles: list[float] = None) -> tuple[float, dict]:
    """Sweep theta across percentiles of discontinuity distribution.
    Returns (best_theta, best_metrics).
    """
    if percentiles is None:
        percentiles = [50, 60, 70, 75, 80, 85, 90, 95]

    # Collect all discontinuities
    all_disc = []
    for td in traj_data_list:
        disc = compute_discontinuities(td.actions)
        all_disc.append(disc)
    all_disc_cat = torch.cat(all_disc)
    all_disc_np = all_disc_cat.numpy()

    thetas = [np.percentile(all_disc_np, p) for p in percentiles]

    best_theta, best_f1, best_metrics = 0.0, -1.0, {}
    results = {}

    for theta, pct in zip(thetas, percentiles):
        precisions, recalls, f1s = [], [], []
        for td, disc in zip(traj_data_list, all_disc):
            pred_b = detect_boundaries(disc, theta)
            m = boundary_f1(pred_b, td.gt_boundaries)
            precisions.append(m["precision"])
            recalls.append(m["recall"])
            f1s.append(m["f1"])

        avg = {
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1": np.mean(f1s),
        }
        results[f"p{pct}"] = {"theta": theta, **avg}

        if avg["f1"] > best_f1:
            best_f1 = avg["f1"]
            best_theta = theta
            best_metrics = avg

    return best_theta, best_metrics, results, all_disc_cat


def segment_trajectory(actions: torch.Tensor, theta: float) -> list[torch.Tensor]:
    """Segment a trajectory into sub-sequences using detected boundaries."""
    disc = compute_discontinuities(actions)
    boundaries = detect_boundaries(disc, theta)
    boundaries = [0] + boundaries + [actions.shape[0]]
    segments = []
    for i in range(len(boundaries) - 1):
        seg = actions[boundaries[i]:boundaries[i + 1]]
        if seg.shape[0] > 0:
            segments.append(seg)
    return segments
