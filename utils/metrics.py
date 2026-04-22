import torch
import numpy as np
from sklearn.metrics import jaccard_score, f1_score


def calculate_physical_inconsistency_rate(
        preds: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        e_ire: float = 478.0,
        electrode_radius: float = 0.05
) -> float:
    """计算物理不一致率，优化：补充维度检查"""
    # 安全降维
    preds_flat = preds.flatten()
    assert len(preds_flat) == len(u) == len(d), "输入维度不匹配"

    d_cm = d / 10.0
    e_max = u / (d_cm - 2 * electrode_radius)
    s_sat = 12.5 * d - 22.5

    threshold_violations = np.sum((e_max < e_ire) & (preds_flat > 0))
    saturation_violations = np.sum(preds_flat > s_sat)
    total_violations = threshold_violations + saturation_violations

    return total_violations / len(preds_flat) * 100 if len(preds_flat) > 0 else 0.0

def calculate_segmentation_metrics(preds: np.ndarray, gts: np.ndarray) -> dict:
    """计算分割指标：IoU, DSC, Precision, Recall"""
    preds = (preds > 0.5).astype(np.int32).flatten()
    gts = gts.astype(np.int32).flatten()

    iou = jaccard_score(gts, preds)
    dsc = f1_score(gts, preds)
    precision = np.sum((preds == 1) & (gts == 1)) / np.sum(preds == 1) if np.sum(preds == 1) > 0 else 0
    recall = np.sum((preds == 1) & (gts == 1)) / np.sum(gts == 1) if np.sum(gts == 1) > 0 else 0

    return {
        "IoU": iou,
        "DSC": dsc,
        "Precision": precision,
        "Recall": recall
    }