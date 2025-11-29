"""
evaluation.py

Shared evaluation helpers for all classification models.
"""

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
)

from .config import RESULTS_DIR, CONFUSION_DIR, ROC_DIR


def ensure_results_dirs():
    """Create results subdirectories if they don't exist."""
    for d in [RESULTS_DIR, CONFUSION_DIR, ROC_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)


def evaluate_classifier(
    model,
    X_test,
    y_test,
    model_name: str,
    dataset_name: str,
    save_plots: bool = True,
):
    """
    Compute and print standard classification metrics.
    Optionally save confusion matrix and ROC curve plots.
    """
    ensure_results_dirs()

    y_pred = model.predict(X_test)

    # For ROC-AUC, we need probabilities or decision function
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print(f"\n=== {model_name} on {dataset_name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    if roc is not None:
        print(f"ROC-AUC  : {roc:.4f}")
    else:
        print("ROC-AUC  : N/A (no probability output)")

    if save_plots:
        _plot_confusion_matrix(y_test, y_pred, model_name, dataset_name)
        if y_proba is not None:
            _plot_roc_curve(y_test, y_proba, model_name, dataset_name)


def _plot_confusion_matrix(y_true, y_pred, model_name: str, dataset_name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix: {model_name} ({dataset_name})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, val, ha="center", va="center")

    out_path = Path(CONFUSION_DIR) / f"{dataset_name}_{model_name}_confusion.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_roc_curve(y_true, y_proba, model_name: str, dataset_name: str):
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title(f"ROC Curve: {model_name} ({dataset_name})")

    out_path = Path(ROC_DIR) / f"{dataset_name}_{model_name}_roc.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
