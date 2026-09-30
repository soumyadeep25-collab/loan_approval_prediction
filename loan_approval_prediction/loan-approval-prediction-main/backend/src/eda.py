import os
import pandas as pd
import numpy as np
from common import data_path

def run_eda():
    print("Running Exploratory Data Analysis...")
    dataset_path = data_path("loan_dataset.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
    df = pd.read_csv(dataset_path)
    
    # 1. Mean income by LoanApproved
    print("\n" + "="*50)
    print("1. Mean Income by Loan Approval Status")
    print("="*50)
    mean_income = df.groupby("LoanApproved")["MonthlyIncome"].mean().reset_index()
    print(mean_income.to_string(index=False))
    
    # 2. Approval rate by CreditScore bucket
    print("\n" + "="*50)
    print("2. Approval Rate by Credit Score Bucket")
    print("="*50)
    # Define buckets
    bins = [300, 500, 600, 700, 800, 900]
    labels = ["300-499", "500-599", "600-699", "700-799", "800-900"]
    df_temp = df.copy()
    df_temp["CreditScoreBucket"] = pd.cut(df_temp["CreditScore"], bins=bins, labels=labels, right=False)
    approval_by_credit = df_temp.groupby("CreditScoreBucket", observed=False)["LoanApproved"].agg(["count", "mean"]).reset_index()
    approval_by_credit.columns = ["Credit Score Bucket", "Total Applicants", "Approval Rate"]
    print(approval_by_credit.to_string(index=False))
    
    # 3. Approval rate by EmploymentStatus
    print("\n" + "="*50)
    print("3. Approval Rate by Employment Status")
    print("="*50)
    approval_by_emp = df.groupby("EmploymentStatus", dropna=False)["LoanApproved"].agg(["count", "mean"]).reset_index()
    approval_by_emp.columns = ["Employment Status", "Total Applicants", "Approval Rate"]
    print(approval_by_emp.to_string(index=False))
    
    # 4. Correlation matrix of numeric features
    print("\n" + "="*50)
    print("4. Correlation Matrix of Numeric Features")
    print("="*50)
    numeric_cols = ["Age", "MonthlyIncome", "LoanAmount", "CreditScore", "ExistingLoans", "LoanTerm", "LoanApproved"]
    correlation_matrix = df[numeric_cols].corr()
    print(correlation_matrix.round(4).to_string())
    print("="*50)

if __name__ == "__main__":
    run_eda()
