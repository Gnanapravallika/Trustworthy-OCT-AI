"""Out-of-distribution (OOD) evaluation metrics for TrustOCT framework."""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from typing import Dict


def calculate_ood_detection_metrics(
    id_uncertainties: np.ndarray,
    ood_uncertainties: np.ndarray
) -> Dict[str, float]:
    """Evaluate OOD detection performance using uncertainty scores.

    Treats OOD detection as a binary task where OOD is the positive class.
    Higher uncertainty should correspond to OOD inputs.

    Args:
        id_uncertainties: Array of uncertainty scores for in-distribution data [N_id].
        ood_uncertainties: Array of uncertainty scores for out-of-distribution data [N_ood].

    Returns:
        Dict containing AUROC, AUPR, and False Positive Rate at 95% Recall (FPR95).
    """
    n_id = len(id_uncertainties)
    n_ood = len(ood_uncertainties)
    
    if n_id == 0 or n_ood == 0:
        return {"auroc": 0.5, "aupr_ood": 0.5, "fpr95": 1.0}

    # Binary labels: 0 for ID, 1 for OOD
    y_true = np.concatenate([np.zeros(n_id), np.ones(n_ood)])
    y_scores = np.concatenate([id_uncertainties, ood_uncertainties])

    # 1. AUROC
    auroc = roc_auc_score(y_true, y_scores)

    # 2. AUPR (OOD as positive class)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    aupr_ood = auc(recall, precision)

    # 3. FPR at 95% TPR (Recall)
    # Sort ID uncertainties in ascending order
    id_sorted = np.sort(id_uncertainties)
    # The threshold for 95% true positive rate (detecting OOD) corresponds to
    # the 5th percentile of the ID uncertainties if we want to retain 95% of ID samples.
    # Alternatively: find the uncertainty threshold where 95% of OOD is detected,
    # and compute what percentage of ID falls above that threshold (false positives).
    threshold = np.percentile(ood_uncertainties, 5)
    fpr95 = np.mean(id_uncertainties >= threshold)

    return {
        "auroc": float(auroc),
        "aupr_ood": float(aupr_ood),
        "fpr95": float(fpr95)
    }
