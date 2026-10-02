"""Model training script and pipeline definitions for Telco Customer Churn."""

from pathlib import Path
from typing import Dict, Any, Optional
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_baseline_models(random_state: int = 42) -> Dict[str, Any]:
    """Instantiate the 4 primary baseline classifiers.

    Args:
        random_state (int): Reproducibility seed.

    Returns:
        Dict[str, Any]: Dictionary of baseline model instances.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(random_state=random_state, eval_metric="logloss"),
    }


def get_imbalance_aware_models(
    scale_pos_weight: float = 2.7686,
    random_state: int = 42
) -> Dict[str, Any]:
    """Instantiate cost-sensitive / class-weighted classifiers to combat class imbalance.

    Args:
        scale_pos_weight (float): Ratio of majority (neg) to minority (pos) samples for XGBoost.
        random_state (int): Reproducibility seed.

    Returns:
        Dict[str, Any]: Dictionary of cost-sensitive model instances.
    """
    return {
        "Logistic Regression (Class-Weighted)": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=random_state
        ),
        "Random Forest (Class-Weighted)": RandomForestClassifier(
            class_weight="balanced", random_state=random_state
        ),
        "XGBoost (scale_pos_weight)": XGBClassifier(
            scale_pos_weight=scale_pos_weight, random_state=random_state, eval_metric="logloss"
        ),
    }


def get_tuned_models(
    scale_pos_weight: float = 2.7686,
    random_state: int = 42
) -> Dict[str, Any]:
    """Instantiate the best hyperparameter-tuned models found via 5-Fold Stratified Cross-Validation.

    Args:
        scale_pos_weight (float): Ratio of negative to positive samples.
        random_state (int): Reproducibility seed.

    Returns:
        Dict[str, Any]: Dictionary of tuned classifier instances.
    """
    return {
        "Tuned Logistic Regression (Balanced)": LogisticRegression(
            C=10.0,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "Tuned Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Tuned XGBoost (scale_pos_weight)": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="logloss",
        ),
    }


def train_model(pipeline: Any, X_train, y_train) -> Any:
    """Train pipeline on training set.

    Args:
        pipeline: Sklearn or imblearn pipeline.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Fitted pipeline.
    """
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(model_obj: Any, filepath: str | Path) -> Path:
    """Serialize and save model to disk.

    Args:
        model_obj (Any): Fitted model or pipeline.
        filepath (str | Path): Destination file path.

    Returns:
        Path: Path to saved artifact.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_obj, path)
    return path



