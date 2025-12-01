"""
Shared evaluation helpers for all classification models.
Handles computing metrics (accuracy, precision, recall, F1, ROC-AUC)
"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)



def evaluate_classifier(
    model,
    X_test,
    y_test,
    model_name: str,
    dataset_name: str
):
    """
    Compute and print standard classification metrics.
    Optionally save confusion matrix and ROC curve plots.
    """

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