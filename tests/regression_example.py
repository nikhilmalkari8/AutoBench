import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from autobench.trainer import Benchmark

# Generate synthetic regression dataset
print("=== Regression Task ===")
X, Y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize Benchmark for regression
benchmark = Benchmark(X_train, X_test, Y_train, Y_test, task="regression", cv=5)

# Train and evaluate models
results = benchmark.train_and_evaluate()

# Save results to a file
benchmark.save_results_to_file("regression_results.json")

# Display results
benchmark.display_results()

# Visualize feature importance for RandomForestRegressor
benchmark.visualize_feature_importance("RandomForestRegressor")

# Generate a PDF report
benchmark.generate_pdf_report("regression_report.pdf")
