"""Consolidated evaluation pipeline for classification, calibration, and OOD metrics."""

from typing import Dict, List, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
    precision_recall_curve,
    auc
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# =====================================================================
# 1. Classification & Calibration Metric Functions
# =====================================================================

def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 4
) -> Dict[str, Union[float, list]]:
    """Compute standard metrics including macro accuracy, F1, MCC, and Specificity."""
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Specificity
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities = []
    for i in range(num_classes):
        temp_cm = np.delete(cm, i, axis=0)
        temp_cm = np.delete(temp_cm, i, axis=1)
        tn = temp_cm.sum()
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
    mean_specificity = float(np.mean(specificities))

    try:
        auc_score = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro", labels=list(range(num_classes))
        )
    except Exception:
        auc_score = 0.5

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": mean_specificity,
        "f1_score": float(f1),
        "mcc": float(mcc),
        "kappa": float(kappa),
        "roc_auc": float(auc_score),
        "confusion_matrix": cm.tolist()
    }


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE)."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)

    ece = 0.0
    num_samples = len(y_true)
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return float(ece)


def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int = 4) -> float:
    """Compute multi-class Brier Score."""
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0
    y_one_hot = np.zeros((n_samples, num_classes))
    y_one_hot[np.arange(n_samples), y_true] = 1.0
    brier = np.sum((y_prob - y_one_hot) ** 2) / n_samples
    return float(brier)


# =====================================================================
# 2. OOD Evaluation
# =====================================================================

def calculate_ood_detection_metrics(
    id_uncertainties: np.ndarray,
    ood_uncertainties: np.ndarray
) -> Dict[str, float]:
    """Calculate OOD detection metrics using AUROC, AUPR, and FPR95."""
    n_id = len(id_uncertainties)
    n_ood = len(ood_uncertainties)
    
    if n_id == 0 or n_ood == 0:
        return {"auroc": 0.5, "aupr_ood": 0.5, "fpr95": 1.0}

    y_true = np.concatenate([np.zeros(n_id), np.ones(n_ood)])
    y_scores = np.concatenate([id_uncertainties, ood_uncertainties])

    auroc = roc_auc_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    aupr_ood = auc(recall, precision)

    threshold = np.percentile(ood_uncertainties, 5)
    fpr95 = np.mean(id_uncertainties >= threshold)

    return {
        "auroc": float(auroc),
        "aupr_ood": float(aupr_ood),
        "fpr95": float(fpr95)
    }


# =====================================================================
# 3. Cross-Dataset Loop
# =====================================================================

def evaluate_cross_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    head_type: str = "edl",
    num_classes: int = 4
) -> Dict[str, Union[float, list]]:
    """Evaluate trained model on an external dataloader."""
    model.eval()
    
    all_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            
            if head_type.lower() == "edl":
                _, alpha = outputs
                S = torch.sum(alpha, dim=1, keepdim=True)
                probs = alpha / S
                preds = torch.argmax(alpha, dim=1)
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    
    clf_metrics = calculate_classification_metrics(y_true, y_pred, y_prob, num_classes=num_classes)
    ece = calculate_ece(y_true, y_prob)
    brier = calculate_brier_score(y_true, y_prob, num_classes=num_classes)
    
    results = {
        **clf_metrics,
        "ece": ece,
        "brier_score": brier
    }
    return results
