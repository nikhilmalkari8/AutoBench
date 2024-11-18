import pandas as pd
import numpy as np
from visualizations import (
    plot_numerical_distributions,
    plot_categorical_distributions,
    plot_correlation_heatmap,
    plot_pairplot,
    plot_outliers,
    plot_missing_values,
    plot_vif,
    plot_class_distribution,
    plot_skewness_and_kurtosis,
    plot_actual_vs_predicted,
    plot_radar_chart,
)


# Load Example Dataset
def load_sample_data():
    """
    Load a sample dataset for demonstration purposes.
    """
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df


# Main Function for EDA
def main():
    # Load sample dataset
    data = load_sample_data()
    
    # Display the first few rows of the dataset
    print("\n=== Dataset Overview ===")
    print(data.head())
    print("\nDataset Info:")
    print(data.info())

    # 1. Numerical Feature Distributions
    plot_numerical_distributions(data)

    # 2. Categorical Feature Distributions (if applicable)
    if not data.select_dtypes(include=['object', 'category']).empty:
        plot_categorical_distributions(data)

    # 3. Correlation Heatmap
    plot_correlation_heatmap(data)

    # 4. Pair Plot
    plot_pairplot(data, target_column='target')

    # 5. Outlier Detection
    plot_outliers(data)

    # 6. Missing Values Analysis
    plot_missing_values(data)

    # 7. Skewness and Kurtosis
    numeric_data = data.select_dtypes(include=np.number)
    skewness = {col: numeric_data[col].skew() for col in numeric_data.columns}
    kurtosis = {col: numeric_data[col].kurt() for col in numeric_data.columns}
    print("\nSkewness:", skewness)
    print("Kurtosis:", kurtosis)
    plot_skewness_and_kurtosis(skewness, kurtosis)

    # 8. Class Distribution (if target column exists)
    if 'target' in data.columns:
        plot_class_distribution(data['target'])

    # 9. Variance Inflation Factor (VIF) Example
    vif_data = pd.DataFrame({
        "feature": numeric_data.columns,
        "VIF": np.random.uniform(1, 5, len(numeric_data.columns))  # Example VIF values
    })
    plot_vif(vif_data)

    # 10. Actual vs Predicted (Regression Example)
    actual = np.random.uniform(50, 100, 100)
    predicted = actual + np.random.normal(0, 5, 100)
    plot_actual_vs_predicted(actual, predicted)

    # 11. Radar Chart Example
    metrics_data = {
        "accuracy": {"Model_A": 0.9, "Model_B": 0.85, "Model_C": 0.87},
        "precision": {"Model_A": 0.88, "Model_B": 0.80, "Model_C": 0.86},
        "recall": {"Model_A": 0.92, "Model_B": 0.83, "Model_C": 0.89},
    }
    plot_radar_chart(metrics_data, models=["Model_A", "Model_B", "Model_C"])


# Run the main function
if __name__ == "__main__":
    main()
