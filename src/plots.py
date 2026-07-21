"""Consolidated plotting script for confusion matrix and reliability diagrams."""

import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_bins: int = 10,
    save_path: str = "reliability_diagram.png"
) -> None:
    """Generate and save ECE reliability diagram."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_accs = []
    bin_sizes = []

    ece = 0.0
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        bin_sizes.append(prop_in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            bin_accs.append(accuracy_in_bin)
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
        else:
            bin_accs.append(0.0)

    plt.figure(figsize=(6, 6), dpi=300)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    plt.bar(
        bin_centers,
        bin_accs,
        width=1.0 / num_bins,
        edgecolor="black",
        color="#1f77b4",
        alpha=0.8,
        label="Accuracy"
    )

    for i in range(num_bins):
        if bin_sizes[i] > 0:
            plt.plot(
                [bin_centers[i], bin_centers[i]],
                [bin_accs[i], bin_centers[i]],
                color="red",
                linestyle="-"
            )

    plt.xlabel("Confidence", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(f"Reliability Diagram (ECE = {ece:.4f})", fontsize=14, fontweight="bold")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Reliability diagram saved to: {save_path}")


def plot_confusion_matrix(
    cm: list,
    classes: List[str],
    save_path: str = "confusion_matrix.png",
    normalize: bool = False
) -> None:
    """Generate and save a confusion matrix heatmap."""
    matrix = np.array(cm)
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(8, 8), dpi=300)
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = format(matrix[i, j], fmt)
            plt.text(
                j, i, val,
                horizontalalignment="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=12,
                fontweight="bold"
            )

    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")
