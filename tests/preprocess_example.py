from autobench.preprocessing import auto_preprocess
import pandas as pd

# Load Example Dataset
def load_example_data():
    """
    Load a small example dataset for preprocessing demonstration.
    """
    data = pd.DataFrame({
        "Age": [25, 30, 35, 40, None],
        "Income": [50000, 60000, 75000, None, 100000],
        "Gender": ["Male", "Female", "Female", "Male", None],
        "Purchased": [0, 1, 0, 1, 1]
    })
    print("\n=== Example Dataset ===")
    print(data)
    return data

# Preprocessing Demonstration
def demonstrate_preprocessing():
    """
    Demonstrate how to use the auto_preprocess function.
    """
    # Step 1: Load Data
    data = load_example_data()

    # Step 2: Preprocess Data
    preprocessed_data = auto_preprocess(
        data=data,
        target_column="Purchased",
        encoding_method="onehot",  # Specify One-hot encoding
        scaling_method="minmax",  # Specify Min-max scaling
        outlier_method="iqr",  # Specify IQR-based outlier treatment
        missing_num_strategy="median",  # Use median for missing numerical values
        missing_cat_strategy="constant",  # Fill missing categorical values with a placeholder
        verbose=True  # Enable detailed console output
    )

    # Step 3: Display Preprocessed Data
    print("\n=== Preprocessed Data ===")
    print(preprocessed_data)

# Run the demonstration
if __name__ == "__main__":
    demonstrate_preprocessing()
