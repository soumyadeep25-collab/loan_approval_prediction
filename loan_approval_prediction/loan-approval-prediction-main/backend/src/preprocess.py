import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from common import data_path, models_path

# Set random seed
RANDOM_SEED = 42

# Define our columns
BASE_NUMERIC_COLS = ["Age", "MonthlyIncome", "LoanAmount", "CreditScore", "ExistingLoans", "LoanTerm"]
ENGINEERED_NUMERIC_COLS = ["LoanToIncome", "DebtBurden", "CreditDebtIndex", "IncomePerTerm"]
NUMERIC_COLS = BASE_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS
CATEGORICAL_COLS = ["Gender", "EmploymentStatus", "PropertyArea"]
TARGET_COL = "LoanApproved"

def preprocess(df, is_training=True, scaler=None, medians=None, modes=None):
    """
    Preprocess the loan approval dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        is_training (bool): If True, fits the scaler. If False, uses the provided scaler/medians/modes.
        scaler (StandardScaler): Pre-fitted StandardScaler (required if is_training=False).
        medians (dict): Pre-computed medians for numeric columns (optional).
        modes (dict): Pre-computed modes for categorical columns (optional).
        
    Returns:
        X (pd.DataFrame): Preprocessed features.
        y (pd.Series or None): Target column if present, otherwise None.
        scaler (StandardScaler): Fitted or provided scaler.
        feature_cols (list): List of all feature column names in order.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Drop duplicates if training
    if is_training:
        df = df.drop_duplicates()
        
    # Separate features and target
    if TARGET_COL in df.columns:
        y = df[TARGET_COL]
        # Drop target from df for preprocessing
        df_features = df.drop(columns=[TARGET_COL])
    else:
        y = None
        df_features = df
        
    # Drop ApplicantID if present
    if "ApplicantID" in df_features.columns:
        df_features = df_features.drop(columns=["ApplicantID"])
        
    # Impute missing values (base numeric cols only; engineered cols derived after)
    if is_training:
        medians = {col: df_features[col].median() for col in BASE_NUMERIC_COLS}
        modes = {col: df_features[col].mode()[0] if not df_features[col].mode().empty else "Unknown" 
                 for col in CATEGORICAL_COLS}
    
    # Apply imputation
    for col in BASE_NUMERIC_COLS:
        fill_val = medians[col] if medians else 0
        df_features[col] = df_features[col].fillna(fill_val)
        
    for col in CATEGORICAL_COLS:
        fill_val = modes[col] if modes else "Unknown"
        df_features[col] = df_features[col].fillna(fill_val)
        
    # Ensure correct dtypes
    for col in BASE_NUMERIC_COLS:
        df_features[col] = pd.to_numeric(df_features[col])
        
    for col in CATEGORICAL_COLS:
        df_features[col] = df_features[col].astype(str)

    # ------------------------------------------------------------------ #
    # Feature engineering: derived risk signals (real ML feature work)
    # ------------------------------------------------------------------ #
    # Avoid divide-by-zero with safe denominators.
    annual_income = df_features["MonthlyIncome"] * 12
    df_features["LoanToIncome"] = df_features["LoanAmount"] / annual_income.clip(lower=1.0)
    # Debt burden: existing loans relative to income (scaled)
    df_features["DebtBurden"] = df_features["ExistingLoans"] / (df_features["MonthlyIncome"] / 1000.0).clip(lower=0.1)
    # Credit utilisation proxy: higher existing loans + lower credit = worse
    df_features["CreditDebtIndex"] = (df_features["ExistingLoans"] + 1) * (850 - df_features["CreditScore"]) / 100.0
    # Affordability: income per loan-term month
    df_features["IncomePerTerm"] = df_features["MonthlyIncome"] / df_features["LoanTerm"].clip(lower=1.0)
    # Clip engineered features to sane ranges to limit outlier influence
    for c in ["LoanToIncome", "DebtBurden", "CreditDebtIndex", "IncomePerTerm"]:
        df_features[c] = df_features[c].clip(lower=0).replace([float("inf")], 0)

    # One-hot encode categorical features
    # To be extremely deterministic, we specify categories
    categories_dict = {
        "Gender": ["Female", "Male"],
        "EmploymentStatus": ["Employed", "Self-Employed", "Unemployed"],
        "PropertyArea": ["Rural", "Semi-Urban", "Urban"]
    }
    
    # Generate dummy variables
    df_dummies_list = []
    for col in CATEGORICAL_COLS:
        # Create dummy columns for the specified categories
        for cat in categories_dict[col]:
            dummy_col_name = f"{col}_{cat}"
            df_features[dummy_col_name] = (df_features[col] == cat).astype(float)
            
    # Drop the original categorical columns
    df_features = df_features.drop(columns=CATEGORICAL_COLS)
    
    # Define exact feature order
    feature_cols = NUMERIC_COLS + [
        f"{col}_{cat}" for col in CATEGORICAL_COLS for cat in categories_dict[col]
    ]
    
    # Reindex to ensure order
    df_features = df_features.reindex(columns=feature_cols, fill_value=0.0)
    
    # Scale numeric features
    if is_training:
        scaler = StandardScaler()
        df_features[NUMERIC_COLS] = scaler.fit_transform(df_features[NUMERIC_COLS])
    else:
        if scaler is None:
            raise ValueError("A fitted scaler must be provided when is_training=False")
        df_features[NUMERIC_COLS] = scaler.transform(df_features[NUMERIC_COLS])
        
    # Validate no nulls remaining
    if df_features.isnull().sum().sum() > 0:
        raise ValueError("Data validation failed: Null values remain after preprocessing")
        
    return df_features, y, scaler, feature_cols

if __name__ == "__main__":
    print("Preprocessing raw data...")
    # Read the dataset
    dataset_path = data_path("loan_dataset.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run generate_data.py first.")
        
    df = pd.read_csv(dataset_path)
    
    # Run preprocess
    X, y, scaler, feature_cols = preprocess(df, is_training=True)
    
    # Create models directory if it doesn't exist
    os.makedirs(models_path(""), exist_ok=True)
    
    # Save scaler and feature columns
    joblib.dump(scaler, models_path("scaler.joblib"))
    with open(models_path("feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=4)
        
    print("Preprocessing completed successfully!")
    print(f"Preprocessed features shape: {X.shape}")
    print(f"Saved models/scaler.joblib and models/feature_columns.json")
