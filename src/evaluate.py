"""Evaluation metrics and reporting tools for Telco Customer Churn."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(model: Any, X_test, y_test) -> Dict[str, Any]:
    """Evaluate fitted model on test set.

    Args:
        model: Fitted estimator or pipeline.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dict[str, Any]: Summary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4) if y_prob is not None else None,
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4) if y_prob is not None else None,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def compare_models(models_dict: Dict[str, Any], X_test, y_test) -> pd.DataFrame:
    """Evaluate multiple models and compile comparison table.

    Args:
        models_dict (Dict[str, Any]): Dictionary of model_name -> fitted_model.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        pd.DataFrame: Tabular summary of model performances.
    """
    rows = []
    for name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test)
        rows.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
        })
    df_results = pd.DataFrame(rows)
    return df_results.sort_values(by="roc_auc", ascending=False).reset_index(drop=True)


def analyze_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Evaluate classification performance across multiple probability thresholds.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_prob (np.ndarray): Predicted positive class probabilities.
        thresholds (Optional[np.ndarray]): Array of cutoff thresholds (defaults to 0.10 to 0.90 in steps of 0.05).

    Returns:
        pd.DataFrame: Performance metrics across each threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.95, 0.05)

    records = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        records.append({
            "threshold": round(float(t), 2),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "customers_flagged": int(y_pred.sum()),
        })
    return pd.DataFrame(records)


def simulate_cost_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    clv_saved: float = 300.0,
    cost_intervention: float = 30.0,
    success_rate: float = 0.40,
    thresholds: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Simulate net financial profit and ROI across decision thresholds.

    Args:
        y_true (np.ndarray): True binary labels.
        y_prob (np.ndarray): Predicted probabilities.
        clv_saved (float): Financial value saved when a true churner is retained.
        cost_intervention (float): Cost incurred per targeted customer (outreach / promo).
        success_rate (float): Fraction of targeted true churners successfully retained.
        thresholds (Optional[np.ndarray]): Array of probability cutoffs.

    Returns:
        pd.DataFrame: Financial simulation metrics across thresholds.
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.95, 0.05)

    results = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        targeted = tp + fp
        churners_saved = tp * success_rate
        revenue_saved = churners_saved * clv_saved
        campaign_cost = targeted * cost_intervention
        net_profit = revenue_saved - campaign_cost
        roi = (net_profit / campaign_cost * 100) if campaign_cost > 0 else 0.0

        results.append({
            "threshold": round(float(t), 2),
            "targeted": targeted,
            "true_churners_caught": tp,
            "churners_saved": round(churners_saved, 1),
            "campaign_cost": round(campaign_cost, 2),
            "revenue_saved": round(revenue_saved, 2),
            "net_profit": round(net_profit, 2),
            "roi_pct": round(roi, 1),
        })
    return pd.DataFrame(results)


