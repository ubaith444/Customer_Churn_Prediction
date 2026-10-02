"""Utility functions for Streamlit application."""

import sys
from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.feature_engineering import engineer_features


def load_champion_model(model_path: str | Path | None = None) -> Any:
    """Load pre-trained champion pipeline.

    Args:
        model_path (str | Path | None): Path to model pickle file.

    Returns:
        Any: Loaded scikit-learn/xgboost pipeline.
    """
    if model_path is None:
        model_path = PROJECT_ROOT / "models" / "best_model.pkl"
    return joblib.load(model_path)


def predict_churn(
    model: Any,
    customer_dict: Dict[str, Any],
    threshold: float = 0.50
) -> Tuple[int, float, str, list]:
    """Generate churn prediction, calibrated probability, risk category, and action items.

    Args:
        model: Fitted pipeline.
        customer_dict (Dict[str, Any]): Raw customer feature inputs.
        threshold (float): Decision probability threshold.

    Returns:
        Tuple[int, float, str, list]: (prediction, probability, risk_level, recommendations)
    """
    df_raw = pd.DataFrame([customer_dict])
    df_engineered = engineer_features(df_raw, drop_customer_id=True)

    prob = float(model.predict_proba(df_engineered)[:, 1][0])
    prediction = int(prob >= threshold)

    if prob < 0.30:
        risk_level = "Low Risk"
    elif prob < 0.60:
        risk_level = "Moderate Risk"
    else:
        risk_level = "High Risk"

    # Generate tailored retention recommendations
    recommendations = []
    if customer_dict.get("Contract") == "Month-to-month":
        recommendations.append("Offer a 15% discount incentive for switching to a 1-year or 2-year contract.")
    if customer_dict.get("PaymentMethod") == "Electronic check":
        recommendations.append("Promote auto-pay enrollment with a $5/month recurring bill credit.")
    if customer_dict.get("OnlineSecurity") == "No" or customer_dict.get("TechSupport") == "No":
        recommendations.append("Provide a complimentary 6-month trial of Tech Support and Online Security.")
    if customer_dict.get("InternetService") == "Fiber optic" and prob > 0.50:
        recommendations.append("Initiate a proactive customer satisfaction check to ensure fiber speeds meet expectations.")
    if int(customer_dict.get("tenure", 0)) <= 6:
        recommendations.append("Enroll customer in the 90-Day VIP Onboarding & Loyalty check-in sequence.")

    if not recommendations:
        recommendations.append("Customer exhibits high loyalty characteristics; maintain standard service touchpoints.")

    return prediction, prob, risk_level, recommendations

