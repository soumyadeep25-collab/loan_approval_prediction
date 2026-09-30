import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from common import data_path, images_path

def run_visualizations():
    print("Generating visualizations...")
    dataset_path = data_path("loan_dataset.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
    df = pd.read_csv(dataset_path)
    
    # Create images directory
    os.makedirs(images_path(""), exist_ok=True)
    
    # Set seaborn style
    sns.set_theme(style="whitegrid")
    
    # 1. Bar chart: approval rate by EmploymentStatus
    plt.figure(figsize=(8, 5))
    # Dropna to avoid error if nan
    df_clean_emp = df.dropna(subset=["EmploymentStatus"])
    # Calculate means
    emp_approval = df_clean_emp.groupby("EmploymentStatus")["LoanApproved"].mean().reset_index()
    sns.barplot(data=emp_approval, x="EmploymentStatus", y="LoanApproved", hue="EmploymentStatus", palette="viridis", legend=False)
    plt.title("Loan Approval Rate by Employment Status", fontsize=14, pad=15)
    plt.xlabel("Employment Status", fontsize=12)
    plt.ylabel("Approval Rate", fontsize=12)
    plt.ylim(0, 1)
    for p in plt.gca().patches:
        plt.gca().annotate(f"{p.get_height()*100:.1f}%", 
                            (p.get_x() + p.get_width() / 2., p.get_height() + 0.02), 
                            ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                            textcoords='offset points')
    plt.savefig(images_path("approval_by_employment.png"), bbox_inches="tight")
    plt.close()
    print("Created images/approval_by_employment.png")
    
    # 2. Correlation heatmap of numeric features
    plt.figure(figsize=(10, 8))
    numeric_cols = ["Age", "MonthlyIncome", "LoanAmount", "CreditScore", "ExistingLoans", "LoanTerm", "LoanApproved"]
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, square=True)
    plt.title("Correlation Heatmap of Numeric Features", fontsize=14, pad=15)
    plt.savefig(images_path("correlation_heatmap.png"), bbox_inches="tight")
    plt.close()
    print("Created images/correlation_heatmap.png")
    
    # 3. Scatter plot: MonthlyIncome vs LoanAmount, color by LoanApproved
    plt.figure(figsize=(10, 6))
    # Drop rows with null in income or amount or approval
    df_scatter = df.dropna(subset=["MonthlyIncome", "LoanAmount", "LoanApproved"])
    sns.scatterplot(
        data=df_scatter, 
        x="MonthlyIncome", 
        y="LoanAmount", 
        hue="LoanApproved", 
        palette={0: "red", 1: "green"},
        alpha=0.6,
        style="LoanApproved"
    )
    plt.title("Monthly Income vs Loan Amount by Approval Status", fontsize=14, pad=15)
    plt.xlabel("Monthly Income ($)", fontsize=12)
    plt.ylabel("Loan Amount ($)", fontsize=12)
    plt.legend(title="Loan Approved", labels=["Rejected", "Approved"])
    plt.savefig(images_path("income_vs_loan.png"), bbox_inches="tight")
    plt.close()
    print("Created images/income_vs_loan.png")
    
    # 4. Pie chart: approved vs rejected
    plt.figure(figsize=(6, 6))
    approval_counts = df["LoanApproved"].value_counts()
    plt.pie(
        approval_counts, 
        labels=["Rejected", "Approved"], 
        colors=["#ff9999", "#66b3ff"], 
        autopct="%1.1f%%", 
        startangle=140,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )
    plt.title("Overall Loan Approval Distribution", fontsize=14, pad=15)
    plt.savefig(images_path("approval_pie.png"), bbox_inches="tight")
    plt.close()
    print("Created images/approval_pie.png")

if __name__ == "__main__":
    run_visualizations()
