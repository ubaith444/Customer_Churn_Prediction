"""Feature engineering and transformation pipeline for Telco Customer Churn."""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def engineer_features(df: pd.DataFrame, drop_customer_id: bool = True) -> pd.DataFrame:
    """Engineer domain-specific features without data leakage or future information.

    Checklist implemented:
    - Churn converted to binary (0/1).
    - customerID dropped (if present and drop_customer_id=True).
    - TotalCharges numeric conversion & missing values imputed.
    - TotalServices: count of all active service subscriptions.
    - TenureGroup: binned customer lifetime cohorts.
    - MonthlyChargeGroup: binned recurring spending tiers.
    - AverageMonthlySpend: historical monthly spending run-rate.
    - ChargeDifference: current MonthlyCharges vs historical AverageMonthlySpend.
    - IsMonthToMonth: indicator for month-to-month contract.
    - SupportServices: count of protective services (OnlineSecurity + TechSupport).

    Args:
        df (pd.DataFrame): Raw or cleaned input DataFrame.
        drop_customer_id (bool): Whether to drop the unique customer identifier.

    Returns:
        pd.DataFrame: Dataframe with all engineered features.
    """
    data = df.copy()

    # 1. TotalCharges conversion and missing handling
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0.0)

    # 2. Target encoding
    if "Churn" in data.columns and data["Churn"].dtype == "object":
        data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

    # 3. TotalServices: count of active subscribed services
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    existing_services = [c for c in service_cols if c in data.columns]
    service_count = (data[existing_services] == "Yes").sum(axis=1)
    if "InternetService" in data.columns:
        service_count += (data["InternetService"] != "No").astype(int)
    data["TotalServices"] = service_count

    # 4. TenureGroup: binned customer lifetime cohorts
    if "tenure" in data.columns:
        data["TenureGroup"] = pd.cut(
            data["tenure"],
            bins=[-1, 12, 24, 36, 48, 60, 72],
            labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"],
        ).astype(str)

    # 5. MonthlyChargeGroup: binned recurring spending tiers
    if "MonthlyCharges" in data.columns:
        data["MonthlyChargeGroup"] = pd.cut(
            data["MonthlyCharges"],
            bins=[0, 35, 70, 90, 150],
            labels=["Low", "Medium", "High", "Very High"],
        ).astype(str)

    # 6. AverageMonthlySpend: TotalCharges / tenure (with MonthlyCharges for tenure == 0)
    if "TotalCharges" in data.columns and "tenure" in data.columns and "MonthlyCharges" in data.columns:
        data["AverageMonthlySpend"] = np.where(
            data["tenure"] > 0,
            data["TotalCharges"] / data["tenure"],
            data["MonthlyCharges"],
        ).round(2)

        # 7. ChargeDifference: MonthlyCharges - AverageMonthlySpend
        data["ChargeDifference"] = (data["MonthlyCharges"] - data["AverageMonthlySpend"]).round(2)

    # 8. IsMonthToMonth: binary flag for high-churn month-to-month contracts
    if "Contract" in data.columns:
        data["IsMonthToMonth"] = (data["Contract"] == "Month-to-month").astype(int)

    # 9. SupportServices: count of key protective retention services
    support_cols = [c for c in ["OnlineSecurity", "TechSupport"] if c in data.columns]
    data["SupportServices"] = (data[support_cols] == "Yes").sum(axis=1)

    # 10. Drop customerID if requested
    if drop_customer_id and "customerID" in data.columns:
        data.drop(columns=["customerID"], inplace=True)

    return data


def get_feature_types(df: pd.DataFrame, target_col: str = "Churn") -> Tuple[List[str], List[str]]:
    """Separate features into numerical and categorical column lists for preprocessors.

    Args:
        df (pd.DataFrame): Engineered dataframe.
        target_col (str): Target column to exclude.

    Returns:
        Tuple[List[str], List[str]]: (numeric_features, categorical_features)
    """
    feature_df = df.drop(columns=[target_col], errors="ignore")
    numeric_features = feature_df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_features, categorical_features


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Build scikit-learn preprocessor with scaling and one-hot encoding.

    Args:
        numeric_features (List[str]): List of numeric feature names.
        categorical_features (List[str]): List of categorical feature names.

    Returns:
        ColumnTransformer: Preprocessing transformer pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor

