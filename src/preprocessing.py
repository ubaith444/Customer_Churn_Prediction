"""Data preprocessing functions for Telco Customer Churn."""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


def clean_churn_data(
    df: pd.DataFrame,
    simplify_categories: bool = True,
    encode_target: bool = True
) -> pd.DataFrame:
    """Clean raw Telco Churn dataset with type conversion and missing value handling.

    Steps performed:
    1. Trims whitespace across string columns.
    2. Converts 'TotalCharges' from object to float64, imputing missing values (tenure == 0) with 0.0.
    3. Standardizes redundant sub-service categories ('No internet service', 'No phone service' -> 'No').
    4. Enforces proper numeric dtypes for tenure and MonthlyCharges.
    5. Optionally encodes target column 'Churn' as binary integer (0 for No, 1 for Yes).

    Args:
        df (pd.DataFrame): Raw Telco Churn dataframe.
        simplify_categories (bool): If True, replaces 'No internet/phone service' with 'No'.
        encode_target (bool): If True, maps 'Churn' ('Yes'/'No') to 1/0.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df = df.copy()

    # 1. Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # 2. Fix TotalCharges: convert to numeric and impute 0.0 for tenure == 0
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # 3. Simplify redundant category labels
    if simplify_categories:
        service_subcols = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        for col in service_subcols:
            if col in df.columns:
                df[col] = df[col].replace({"No internet service": "No"})

        if "MultipleLines" in df.columns:
            df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})

    # 4. Enforce numeric data types
    if "tenure" in df.columns:
        df["tenure"] = df["tenure"].astype(int)
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)

    # 5. Encode target variable
    if encode_target and "Churn" in df.columns:
        if df["Churn"].dtype == "object":
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def load_raw_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Load raw Telco Churn dataset.

    Args:
        file_path (Optional[str]): Path to raw CSV. Defaults to relative data/raw path.

    Returns:
        pd.DataFrame: Loaded raw dataset.
    """
    if file_path is None:
        file_path = str(Path(__file__).resolve().parent.parent / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return pd.read_csv(file_path)


def save_cleaned_data(df: pd.DataFrame, output_path: Optional[str] = None) -> Path:
    """Save cleaned DataFrame to CSV.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
        output_path (Optional[str]): Target file path.

    Returns:
        Path: Path to saved file.
    """
    if output_path is None:
        output_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "cleaned_churn_data.csv"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path

