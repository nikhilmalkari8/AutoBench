import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi


# Utility Function: Print Section Header
def print_section_header(header):
    """
    Print a clean section header in the console.
    """
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))


# 1. Distribution Plots for Numerical Features
def plot_numerical_distributions(data, save_path=None):
    """
    Plot distributions for all numerical columns in the dataset.
    """
    print_section_header("Numerical Feature Distributions")
    numeric_cols = data.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(data[col], kde=True, bins=30, color='blue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(True)
        if save_path:
            plt.savefig(f"{save_path}/distribution_{col}.png")
        plt.show()


# 2. Distribution Plots for Categorical Features
def plot_categorical_distributions(data, save_path=None):
    """
    Plot distributions for all categorical columns in the dataset.
    """
    print_section_header("Categorical Feature Distributions")
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        plt.figure(figsize=(8, 5))
        data[col].value_counts().plot(kind='bar', color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.xticks(rotation=45)
        if save_path:
            plt.savefig(f"{save_path}/distribution_{col}.png")
        plt.show()


# 3. Correlation Heatmap
def plot_correlation_heatmap(data, save_path=None):
    """
    Plot a heatmap of correlation matrix for numerical features.
    """
    print_section_header("Correlation Heatmap")
    numeric_data = data.select_dtypes(include=np.number)
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title("Correlation Heatmap")
    if save_path:
        plt.savefig(f"{save_path}/correlation_heatmap.png")
    plt.show()


# 4. Pair Plot
def plot_pairplot(data, target_column=None, save_path=None):
    """
    Plot a pairplot for numerical features. Optionally, color by target_column.
    """
    print_section_header("Pair Plot")
    pairplot = sns.pairplot(data, hue=target_column, diag_kind='kde', palette="Set2")
    if save_path:
        pairplot.savefig(f"{save_path}/pairplot.png")
    plt.show()


# 5. Boxplots for Outliers
def plot_outliers(data, save_path=None):
    """
    Plot boxplots to identify outliers in numerical features.
    """
    print_section_header("Outlier Detection")
    numeric_cols = data.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=data[col], color='orange')
        plt.title(f'Outliers in {col}')
        plt.xlabel(col)
        plt.grid(True)
        if save_path:
            plt.savefig(f"{save_path}/outliers_{col}.png")
        plt.show()


# 6. Missing Values Bar Plot
def plot_missing_values(data, save_path=None):
    """
    Plot a bar chart for missing value percentages.
    """
    print_section_header("Missing Values Analysis")
    missing = data.isnull().sum()
    missing_percentage = (missing / len(data)) * 100
    missing_percentage = missing_percentage[missing_percentage > 0]
    missing_percentage.sort_values(ascending=False, inplace=True)
    
    if not missing_percentage.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_percentage.index, y=missing_percentage.values, palette="viridis")
        plt.title("Missing Values Percentage")
        plt.xlabel("Features")
        plt.ylabel("Percentage (%)")
        plt.xticks(rotation=45)
        plt.grid(True)
        if save_path:
            plt.savefig(f"{save_path}/missing_values.png")
        plt.show()
    else:
        print("No missing values detected!")


# 7. Variance Inflation Factor (VIF) Bar Chart
def plot_vif(vif_data, save_path=None):
    """
    Plot a bar chart of Variance Inflation Factor (VIF) values.
    """
    print_section_header("Variance Inflation Factor (VIF)")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="feature", y="VIF", data=vif_data, palette="coolwarm")
    plt.title("Variance Inflation Factor (VIF)")
    plt.xlabel("Feature")
    plt.ylabel("VIF")
    plt.xticks(rotation=45)
    plt.grid(True)
    if save_path:
        plt.savefig(f"{save_path}/vif_chart.png")
    plt.show()


# 8. Class Distribution for Classification Tasks
def plot_class_distribution(target, save_path=None):
    """
    Plot class distribution for classification tasks.
    """
    print_section_header("Class Distribution")
    class_counts = target.value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="muted")
    plt.title("Class Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Counts")
    plt.grid(True)
    if save_path:
        plt.savefig(f"{save_path}/class_distribution.png")
    plt.show()


# 9. Skewness and Kurtosis
def plot_skewness_and_kurtosis(skewness, kurtosis, save_path=None):
    """
    Plot skewness and kurtosis for numerical features.
    """
    print_section_header("Skewness and Kurtosis")
    plt.figure(figsize=(12, 6))
    plt.bar(skewness.keys(), skewness.values(), alpha=0.7, label='Skewness', color='blue')
    plt.bar(kurtosis.keys(), kurtosis.values(), alpha=0.7, label='Kurtosis', color='red')
    plt.title("Skewness and Kurtosis")
    plt.xlabel("Features")
    plt.ylabel("Values")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    if save_path:
        plt.savefig(f"{save_path}/skewness_kurtosis.png")
    plt.show()


# 10. Actual vs Predicted for Regression Tasks
def plot_actual_vs_predicted(actual, predicted, save_path=None):
    """
    Scatter plot of actual vs. predicted values for regression tasks.
    """
    print_section_header("Actual vs Predicted")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=actual, y=predicted, alpha=0.6, color='blue')
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], color='red', linestyle='--', label='Perfect Fit')
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(f"{save_path}/actual_vs_predicted.png")
    plt.show()


# 11. Radar Chart for Model Comparisons
def plot_radar_chart(metrics_data, models, save_path=None):
    """
    Plot a radar chart to compare metrics across multiple models.
    """
    print_section_header("Model Comparison Radar Chart")
    metrics = list(metrics_data.keys())
    num_metrics = len(metrics)
    angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for model in models:
        values = metrics_data[model]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.25)

    ax.set_yticks([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Comparison Radar Chart")
    if save_path:
        plt.savefig(f"{save_path}/radar_chart.png")
    plt.show()
