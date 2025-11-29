"""
evaluation.py

Shared evaluation helpers for all classification models.
Handles:
- computing metrics (accuracy, precision, recall, F1, ROC-AUC)
- printing them nicely
- plotting and saving confusion matrices and ROC curves
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

# output paths
from .config import RESULTS_DIR, CONFUSION_DIR, ROC_DIR


def ensure_results_dirs():
    """
    Create results subdirectories if they don't exist.
    This prevents errors when saving plots.
    """
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

    Parameters
    ----------
    model : fitted sklearn Pipeline or estimator
    X_test : array-like, test features
    y_test : array-like, true labels for test set
    model_name : str, short name for the model (e.g., "rf", "logreg")
    dataset_name : str, name of dataset ("transactions" or "products")
    save_plots : bool, if True save confusion + ROC plots to disk
    """
    ensure_results_dirs() # make sure folders exist

    # Predict hard class labels
    y_pred = model.predict(X_test)


    # For ROC-AUC, we need scores/probabilities, not just class labels
    if hasattr(model, "predict_proba"):
        # Use probability of positive class (index 1)
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # Some models (like SVM) provide a decision function instead of probability
        y_proba = model.decision_function(X_test)
    else:
        # If neither is available, we can't compute ROC-AUC
        y_proba = None

    # Compute scalar metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None


    # Print metrics nicely
    print(f"\n=== {model_name} on {dataset_name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    if roc is not None:
        print(f"ROC-AUC  : {roc:.4f}")
    else:
        print("ROC-AUC  : N/A (no probability output)")

    # ------------------------------------------------------------------
    # Save metrics to CSV for easy reporting
    # ------------------------------------------------------------------
    import csv
    metrics_path = RESULTS_DIR / "metrics.csv"

    # Write header only if file does NOT exist yet
    write_header = not metrics_path.exists()

    with open(metrics_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "dataset", "model", "accuracy", "precision",
                "recall", "f1", "roc_auc"
            ])
        writer.writerow([
            dataset_name, model_name, acc, prec, rec, f1, roc
        ])
    # Optionally generate and save plots
    if save_plots:
        _plot_confusion_matrix(y_test, y_pred, model_name, dataset_name)
        if y_proba is not None:
            _plot_roc_curve(y_test, y_proba, model_name, dataset_name)


def _plot_confusion_matrix(y_true, y_pred, model_name: str, dataset_name: str):
    """
    Draw and save a confusion matrix as a PNG image.
    """
    cm = confusion_matrix(y_true, y_pred)  # 2x2 matrix for binary classification
    
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")   # show as heatmap
    ax.set_title(f"Confusion Matrix: {model_name} ({dataset_name})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # Annotate each cell with its count
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, val, ha="center", va="center")


    # Build output path like: results/confusion_matrices/transactions_rf_confusion.png
    out_path = Path(CONFUSION_DIR) / f"{dataset_name}_{model_name}_confusion.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)  # free memory


def _plot_roc_curve(y_true, y_proba, model_name: str, dataset_name: str):
    """
    Draw and save a ROC curve as a PNG image.
    """
    fig, ax = plt.subplots()
    # scikit-learn helper builds false positive rate, true positive rate, etc.
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title(f"ROC Curve: {model_name} ({dataset_name})")

    # Build output path like: results/roc_curves/products_logreg_roc.png
    out_path = Path(ROC_DIR) / f"{dataset_name}_{model_name}_roc.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)  # free memory
