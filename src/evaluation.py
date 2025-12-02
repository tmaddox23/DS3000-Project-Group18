"""
Shared evaluation helpers for all classification models.
Handles computing metrics (accuracy, precision, recall, F1, ROC-AUC)
"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

def evaluate_and_plot_classifier(
    clf,
    X_test,
    y_test,
    model_name: str,
    dataset_name: str = "transactions",
    df=None,  # <-- NEW: pass the full dataframe if you want extra plots
):
    """
    Print metrics + plot confusion matrix, ROC and Precision–Recall curves
    for a fitted classifier. Optionally also plot a few feature distributions
    if `df` (the original DataFrame) is provided.
    """
    # ----- 1. Predictions and scores -----
    y_pred = clf.predict(X_test)

    # Continuous scores for ROC / PR
    if hasattr(clf, "predict_proba"):
        y_scores = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_scores = clf.decision_function(X_test)
    else:
        y_scores = None

    # ----- 2. Text metrics -----
    print(f"=== {model_name} ({dataset_name}) ===")
    print(classification_report(y_test, y_pred, digits=4))

    if y_scores is not None:
        roc = roc_auc_score(y_test, y_scores)
        print(f"ROC-AUC: {roc:.4f}")
    else:
        print("ROC-AUC: N/A (no probability/score output)")

    # ----- 3. Confusion matrix heatmap -----
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Not counterfeit", "Counterfeit"],
        yticklabels=["Not counterfeit", "Counterfeit"],
    )
    plt.title(f"Confusion Matrix – {model_name} ({dataset_name})")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()

    # ----- 4. ROC curve -----
    if y_scores is not None:
        plt.figure(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_test, y_scores)
        plt.title(f"ROC Curve – {model_name} ({dataset_name})")
        plt.tight_layout()
        plt.show()

        # ----- 5. Precision–Recall curve -----
        plt.figure(figsize=(5, 4))
        PrecisionRecallDisplay.from_predictions(y_test, y_scores)
        plt.title(f"Precision–Recall Curve – {model_name} ({dataset_name})")
        plt.tight_layout()
        plt.show()

    # ----- 6. Dataset-level EDA plots
    if df is not None:
        # 6a. Core numeric features in a 2x2 grid of histograms
        numeric_cols = [
            "customer_age",
            "quantity",
            "unit_price",
            "customer_history_orders",
        ]
        available_numeric = [c for c in numeric_cols if c in df.columns]

        if available_numeric:
            ax_array = df[available_numeric].hist(
                bins=15,
                figsize=(10, 7),
                layout=(2, 2),
                grid=False,
            )
            plt.suptitle("Key Numeric Feature Distributions", fontsize=14)
            plt.tight_layout()
            plt.show()

        # 6b. Total transaction amount distribution (smooth KDE)
        if "total_amount" in df.columns:
            plt.figure(figsize=(8, 4))
            sns.kdeplot(
                data=df,
                x="total_amount",
                fill=True,
                linewidth=1.2,
            )
            plt.title("Total Transaction Amount – Density")
            plt.xlabel("Total amount")
            plt.ylabel("Density")
            plt.tight_layout()
            plt.show()

        # 6c. Payment method counts (categorical)
        if "payment_method" in df.columns:
            plt.figure(figsize=(8, 4))
            order = df["payment_method"].value_counts().index
            sns.countplot(
                data=df,
                x="payment_method",
                order=order,
            )
            plt.title("Payment Method Frequency")
            plt.xlabel("Payment method")
            plt.ylabel("Number of transactions")
            plt.xticks(rotation=20)
            plt.tight_layout()
            plt.show()

        # 6d. Shipping speed counts as a horizontal bar plot
        if "shipping_speed" in df.columns:
            counts = df["shipping_speed"].value_counts()
            plt.figure(figsize=(6, 4))
            sns.barplot(
                x=counts.values,
                y=counts.index,
            )
            plt.title("Shipping Speed Breakdown")
            plt.xlabel("Number of transactions")
            plt.ylabel("Shipping speed")
            plt.tight_layout()
            plt.show()

        # 6e. Counterfeit vs non-counterfeit class balance
        if "involves_counterfeit" in df.columns:
            plt.figure(figsize=(5, 4))
            sns.countplot(
                data=df,
                x="involves_counterfeit",
            )
            plt.title("Target Class Distribution – Involves Counterfeit")
            plt.xlabel("Involves counterfeit (False / True)")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.show()

        # 6f. Geolocation mismatch vs counterfeit rate (simple 2-category bar)
        if {"geolocation_mismatch", "involves_counterfeit"}.issubset(df.columns):
            mismatch_rate = (
                df.groupby("geolocation_mismatch")["involves_counterfeit"]
                .mean()
                .rename("counterfeit_rate")
            )
            plt.figure(figsize=(6, 4))
            sns.barplot(
                x=mismatch_rate.index.astype(str),
                y=mismatch_rate.values,
            )
            plt.title("Counterfeit Rate by Geolocation Mismatch")
            plt.xlabel("Geolocation mismatch (False / True)")
            plt.ylabel("Share of transactions that are counterfeit")
            plt.tight_layout()
            plt.show()


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

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
    }