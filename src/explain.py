"""Model explainability and SHAP visualizations for Telco Customer Churn."""

import sys
from pathlib import Path
from typing import Any, List, Optional
import matplotlib.pyplot as plt
import numpy as np

# Ensure compatibility if cv2 is absent/incompatible with numpy 2
if "cv2" not in sys.modules:
    sys.modules["cv2"] = None

import shap


def compute_shap_values(model: Any, X_sample: np.ndarray, feature_names: Optional[List[str]] = None) -> shap.Explanation:
    """Compute SHAP tree explanation values for model predictions.

    Args:
        model: Fitted tree-based model (e.g., XGBClassifier, RandomForestClassifier).
        X_sample (np.ndarray): Preprocessed feature matrix sample.
        feature_names (Optional[List[str]]): List of cleaned feature names.

    Returns:
        shap.Explanation: Explanation object containing SHAP values and base values.
    """
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_sample)
    if feature_names is not None:
        explanation.feature_names = feature_names
    return explanation


def plot_shap_summary(
    shap_values: shap.Explanation | np.ndarray,
    X_sample: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str | Path] = None,
    max_display: int = 12
) -> None:
    """Generate and save global SHAP summary plot.

    Args:
        shap_values: Calculated SHAP values or explanation object.
        X_sample (np.ndarray): Input feature matrix.
        feature_names (Optional[List[str]]): Column labels.
        save_path (Optional[str | Path]): Output image filepath.
        max_display (int): Number of top features to display.
    """
    values = shap_values.values if hasattr(shap_values, "values") else shap_values

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(values, X_sample, feature_names=feature_names, max_display=max_display, show=False)
    plt.title("Global SHAP Feature Attribution Summary", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close("all")


def plot_shap_waterfall(
    shap_explanation: shap.Explanation,
    save_path: Optional[str | Path] = None,
    max_display: int = 10
) -> None:
    """Generate and save local SHAP waterfall plot for a single customer.

    Args:
        shap_explanation (shap.Explanation): Explanation slice for a single customer instance.
        save_path (Optional[str | Path]): Output image filepath.
        max_display (int): Number of top feature contributions to display.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.waterfall(shap_explanation, max_display=max_display, show=False)
    plt.title("Individual Customer Prediction (SHAP Waterfall)", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close("all")

