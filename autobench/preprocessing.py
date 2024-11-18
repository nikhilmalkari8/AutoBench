import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def auto_preprocess(
    data, 
    target_column=None, 
    encoding_method="label",  # 'label' or 'onehot'
    scaling_method="standard",  # 'standard' or 'minmax'
    outlier_method=None,  # 'iqr', 'zscore', or None
    missing_num_strategy="mean",  # 'mean', 'median', 'most_frequent', 'constant'
    missing_cat_strategy="most_frequent",  # 'most_frequent', 'constant'
    verbose=True  # Enable or disable console output
):
    """
    Automatically preprocesses the dataset for modeling.

    Parameters:
        data (pd.DataFrame): Input dataset.
        target_column (str): Target column for the task (optional).
        encoding_method (str): 'label' for Label Encoding, 'onehot' for One-Hot Encoding.
        scaling_method (str): 'standard' for StandardScaler or 'minmax' for MinMaxScaler.
        outlier_method (str): 'iqr', 'zscore', or None for no outlier treatment.
        missing_num_strategy (str): Strategy for numerical missing values ('mean', 'median', etc.).
        missing_cat_strategy (str): Strategy for categorical missing values ('most_frequent', 'constant').
        verbose (bool): Whether to print preprocessing steps.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    if verbose:
        print("\n=== Auto Preprocessing Started ===")
    preprocessed_data = data.copy()

    # 1. Handle Missing Values
    if verbose:
        print("\nHandling Missing Values...")
    num_imputer = SimpleImputer(strategy=missing_num_strategy)
    cat_imputer = SimpleImputer(strategy=missing_cat_strategy)
    for col in preprocessed_data.columns:
        if preprocessed_data[col].dtype in [np.float64, np.int64]:
            preprocessed_data[col] = num_imputer.fit_transform(preprocessed_data[[col]])
        elif preprocessed_data[col].dtype == "object":
            preprocessed_data[col] = cat_imputer.fit_transform(preprocessed_data[[col]])
    if verbose:
        print("Missing values handled.")

    # 2. Encode Categorical Features
    if encoding_method not in ["label", "onehot"]:
        raise ValueError("Invalid encoding_method. Use 'label' or 'onehot'.")
    if verbose:
        print(f"\nEncoding Categorical Features using '{encoding_method}'...")
    if encoding_method == "label":
        for col in preprocessed_data.select_dtypes(include="object").columns:
            preprocessed_data[col] = LabelEncoder().fit_transform(preprocessed_data[col])
    elif encoding_method == "onehot":
        preprocessed_data = pd.get_dummies(preprocessed_data, drop_first=True)
    if verbose:
        print("Categorical encoding complete.")

    # 3. Outlier Treatment
    if outlier_method not in [None, "iqr", "zscore"]:
        raise ValueError("Invalid outlier_method. Use 'iqr', 'zscore', or None.")
    if outlier_method:
        if verbose:
            print(f"\nApplying Outlier Treatment using '{outlier_method}'...")
        for col in preprocessed_data.select_dtypes(include=[np.float64, np.int64]).columns:
            if outlier_method == "iqr":
                q1 = preprocessed_data[col].quantile(0.25)
                q3 = preprocessed_data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                preprocessed_data[col] = np.clip(preprocessed_data[col], lower_bound, upper_bound)
            elif outlier_method == "zscore":
                mean = preprocessed_data[col].mean()
                std = preprocessed_data[col].std()
                z_scores = (preprocessed_data[col] - mean) / std
                preprocessed_data[col] = np.where(
                    z_scores > 3, mean + 3 * std,
                    np.where(z_scores < -3, mean - 3 * std, preprocessed_data[col])
                )
        if verbose:
            print("Outlier treatment complete.")

    # 4. Feature Scaling
    if scaling_method not in ["standard", "minmax"]:
        raise ValueError("Invalid scaling_method. Use 'standard' or 'minmax'.")
    if verbose:
        print(f"\nScaling Features using '{scaling_method}'...")
    scaler = StandardScaler() if scaling_method == "standard" else MinMaxScaler()
    numeric_cols = preprocessed_data.select_dtypes(include=[np.float64, np.int64]).columns
    preprocessed_data[numeric_cols] = scaler.fit_transform(preprocessed_data[numeric_cols])
    if verbose:
        print("Feature scaling complete.")

    # If target_column is specified, ensure it is added back as is
    if target_column:
        target = data[target_column]
        preprocessed_data[target_column] = target

    if verbose:
        print("\n=== Preprocessing Complete ===")
    return preprocessed_data
