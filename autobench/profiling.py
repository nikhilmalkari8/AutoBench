import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, spearmanr, chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor


def categorize_features(data):
    """
    Categorize features into numerical, categorical, ordinal, and nominal.
    """
    feature_types = {
        "categorical_nominal": [],
        "categorical_ordinal": [],
        "numerical_continuous": [],
        "numerical_discrete": [],
    }

    for col in data.columns:
        if data[col].dtype in ["object", "category"]:
            if data[col].nunique() > 5:
                feature_types["categorical_nominal"].append(col)
            else:
                feature_types["categorical_ordinal"].append(col)
        elif np.issubdtype(data[col].dtype, np.number):
            if data[col].nunique() > 10:
                feature_types["numerical_continuous"].append(col)
            else:
                feature_types["numerical_discrete"].append(col)

    return feature_types


def calculate_vif(dataframe):
    """
    Calculate Variance Inflation Factor (VIF) for each feature in the dataframe.
    """
    vif_data = []
    for i in range(dataframe.shape[1]):
        try:
            vif_value = variance_inflation_factor(dataframe.values, i)
            vif_data.append((dataframe.columns[i], vif_value))
        except Exception:
            vif_data.append((dataframe.columns[i], np.inf))
    return pd.DataFrame(vif_data, columns=["feature", "VIF"])


def detect_outliers(data, threshold=1.5):
    """
    Detect outliers using the IQR method.
    """
    outlier_stats = {}
    for col in data.columns:
        if np.issubdtype(data[col].dtype, np.number):
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            outlier_stats[col] = outliers / len(data) * 100
    return outlier_stats


def task_type_analysis(target):
    """
    Determine the task type (Regression, Binary Classification, Multi-class Classification).
    """
    unique_values = target.nunique()
    if np.issubdtype(target.dtype, np.number) and unique_values > 10:
        return "Regression"
    elif unique_values == 2:
        return "Binary Classification"
    else:
        return "Multi-class Classification"


def correlation_with_target(data, target_column):
    """
    Analyze correlation of features with the target.
    """
    correlations = {}
    target = data[target_column]
    for col in data.columns:
        if col != target_column:
            if np.issubdtype(data[col].dtype, np.number):
                corr, _ = spearmanr(data[col], target)
                correlations[col] = {"type": "Spearman", "value": corr}
            else:
                contingency_table = pd.crosstab(data[col], target)
                chi2, p, _, _ = chi2_contingency(contingency_table)
                correlations[col] = {"type": "Chi-Square", "value": chi2}
    return correlations


def quasi_constant_features(data, threshold=0.01):
    """
    Identify quasi-constant features based on variance.
    """
    low_variance_cols = [col for col in data.columns if data[col].nunique() / len(data) < threshold]
    return low_variance_cols


def distribution_analysis(data):
    """
    Analyze distribution shape (skewness and kurtosis) of numerical columns.
    """
    distribution_stats = {}
    for col in data.columns:
        if np.issubdtype(data[col].dtype, np.number):
            skewness = data[col].skew()
            kurtosis = data[col].kurt()
            distribution_stats[col] = {"skewness": skewness, "kurtosis": kurtosis}
    return distribution_stats


def dataset_overview(data):
    """
    General dataset overview including duplicates, memory usage, and more.
    """
    return {
        "shape": data.shape,
        "memory_usage": data.memory_usage(deep=True).sum(),
        "unique_rows": len(data.drop_duplicates()),
        "duplicate_rows": len(data) - len(data.drop_duplicates()),
        "empty_columns": [col for col in data.columns if data[col].isnull().all()],
        "constant_rows": len(data[data.nunique(axis=1) == 1]),
    }


def missing_value_analysis(data):
    """
    Analysis of missing values in the dataset.
    """
    missing = data.isnull().sum()
    missing_percentage = (missing / len(data)) * 100
    return {
        "count": missing.to_dict(),
        "percentage": missing_percentage.to_dict(),
    }


def advanced_statistics(numeric_data):
    """
    Compute advanced statistics for numerical columns.
    """
    return {
        "median_absolute_deviation": {
            col: np.median(np.abs(numeric_data[col] - np.median(numeric_data[col])))
            for col in numeric_data.columns
        },
        "range": {col: numeric_data[col].max() - numeric_data[col].min() for col in numeric_data.columns},
        "mode": {col: numeric_data[col].mode().iloc[0] for col in numeric_data.columns},
    }


def check_normality(numeric_data):
    """
    Perform normality tests on numerical columns.
    """
    return {
        col: {
            "statistic": shapiro(numeric_data[col].dropna())[0],
            "p_value": shapiro(numeric_data[col].dropna())[1],
            "is_normal": shapiro(numeric_data[col].dropna())[1] > 0.05,
        }
        for col in numeric_data.columns
    }


def profile_data(data, target_column=None):
    """
    Generate a comprehensive profile of the dataset.
    """
    profile = {}

    # Dataset Overview
    profile["overview"] = dataset_overview(data)

    # Feature Categorization
    profile["feature_categories"] = categorize_features(data)

    # Missing Values
    profile["missing_values"] = missing_value_analysis(data)

    # Numeric Data
    numeric_data = data.select_dtypes(include=np.number)

    # Descriptive Statistics
    profile.update(advanced_statistics(numeric_data))

    # Normality Check
    profile["normality"] = check_normality(numeric_data)

    # Multicollinearity Analysis
    filtered_numeric_data = numeric_data.loc[:, numeric_data.apply(lambda x: x.nunique() > 1, axis=0)]
    profile["multicollinearity"] = calculate_vif(filtered_numeric_data).to_dict(orient="records")

    # Outliers
    profile["outliers"] = detect_outliers(numeric_data)

    # Quasi-Constant Features
    profile["quasi_constant_features"] = quasi_constant_features(data)

    # Correlations with Target
    if target_column:
        profile["correlation_with_target"] = correlation_with_target(data, target_column)

    # Distribution Analysis
    profile["distribution_analysis"] = distribution_analysis(numeric_data)

    # Task Type Analysis
    if target_column:
        target = data[target_column]
        profile["target_analysis"] = {
            "type": task_type_analysis(target),
            "class_imbalance": target.value_counts(normalize=True).to_dict() if target.nunique() <= 10 else None,
        }

    # Recommendations
    recommendations = []
    if profile["missing_values"]["count"]:
        recommendations.append("Handle missing values in columns with significant percentages.")
    if profile["outliers"]:
        recommendations.append("Consider treating features with high outlier percentages.")
    if profile["quasi_constant_features"]:
        recommendations.append(f"Remove quasi-constant features: {profile['quasi_constant_features']}")
    profile["recommendations"] = recommendations

    return profile

def generate_profile(data, target_column=None, save_report=True, report_filename="profiling_report.json"):
    """
    High-level function to generate a dataset profile with 11 specific sections in output.
    """
    profile = profile_data(data, target_column)

    # 1. Dataset Overview
    print("\n=== Dataset Overview ===")
    for key, value in profile["overview"].items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")

    # 2. Feature Categories
    print("\n=== Feature Categories ===")
    for category, features in profile["feature_categories"].items():
        print(f"{category.replace('_', ' ').capitalize()}: {features}")

    # 3. Missing Values
    print("\n=== Missing Values ===")
    for col, count in profile["missing_values"]["count"].items():
        percentage = profile["missing_values"]["percentage"][col]
        print(f"{col}: {count} missing values ({percentage:.2f}%)")

    # 4. Target Analysis
    if target_column:
        print("\n=== Target Analysis ===")
        for key, value in profile["target_analysis"].items():
            print(f"{key.replace('_', ' ').capitalize()}: {value}")

    # 5. Outlier Detection
    print("\n=== Outlier Detection ===")
    for col, outlier_pct in profile["outliers"].items():
        print(f"{col}: {outlier_pct:.2f}%")

    # 6. Correlation with Target
    if target_column:
        print("\n=== Correlation with Target ===")
        for col, corr_info in profile["correlation_with_target"].items():
            print(f"{col} ({corr_info['type']}): {corr_info['value']:.3f}")

    # 7. Multicollinearity
    print("\n=== Multicollinearity (VIF) ===")
    for vif_info in profile["multicollinearity"]:
        print(f"{vif_info['feature']}: {vif_info['VIF']:.2f}")

    # 8. Normality Tests
    print("\n=== Normality Tests ===")
    for col, norm_test in profile["normality"].items():
        print(f"{col}: Statistic={norm_test['statistic']:.3f}, p-value={norm_test['p_value']:.3f}, Normal={norm_test['is_normal']}")

    # 9. Distribution Analysis
    print("\n=== Distribution Analysis ===")
    for col, dist_stats in profile["distribution_analysis"].items():
        print(f"{col}: Skewness={dist_stats['skewness']:.3f}, Kurtosis={dist_stats['kurtosis']:.3f}")

    # 10. Quasi-Constant Features
    print("\n=== Quasi-Constant Features ===")
    if profile["quasi_constant_features"]:
        print("Features:", profile["quasi_constant_features"])
    else:
        print("None detected.")

    # 11. Recommendations
    print("\n=== Recommendations ===")
    for rec in profile["recommendations"]:
        print(f"- {rec}")

    # Save Report
    if save_report:
        with open(report_filename, "w") as f:
            import json
            json.dump(profile, f, indent=4)
        print(f"\nDetailed report saved as '{report_filename}'")
