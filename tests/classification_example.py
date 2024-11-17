import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

# Import the Benchmark class
from autobench.trainer import Benchmark

# Load the Iris dataset
data = load_iris()
X = data.data
Y = data.target

# Feature scaling for certain models
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the Benchmark class with cross-validation
benchmark = Benchmark(X_train, X_test, Y_train, Y_test, cv=5)

# Train models and evaluate their performance
results = benchmark.train_and_evaluate()

# Save results to a JSON file
benchmark.save_results_to_file("classification_results.json")

# Display results in a tabular format
print("\nModel Performance:")
benchmark.display_results()

# Visualization of results
def visualize_results(results):
    df = pd.DataFrame(results).T
    # Filter numeric columns for visualization
    numeric_df = df[["cv_mean_accuracy", "accuracy", "training_time"]].astype(float)

    # Accuracy and Cross-validation mean accuracy
    numeric_df[["cv_mean_accuracy", "accuracy"]].plot(kind="bar", figsize=(10, 6))
    plt.title("Model Accuracy and Cross-Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Training time visualization
    numeric_df["training_time"].plot(kind="bar", figsize=(10, 6), color="orange")
    plt.title("Model Training Time")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Visualize the results
visualize_results(results)
