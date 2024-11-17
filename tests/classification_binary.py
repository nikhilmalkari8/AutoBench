from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from autobench.trainer import Benchmark

# Load the Iris dataset for binary classification
print("=== Binary Classification ===")
iris = load_iris()
X_binary = iris.data[iris.target != 2]  # Filter to only two classes (0 and 1)
Y_binary = iris.target[iris.target != 2]

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_binary, Y_binary, test_size=0.2, random_state=42)

# Initialize Benchmark for binary classification
benchmark = Benchmark(X_train, X_test, Y_train, Y_test, task="binary", cv=5)

# Train and evaluate models
results = benchmark.train_and_evaluate()

# Save results to a file
benchmark.save_results_to_file("binary_results.json")

# Display results
benchmark.display_results()

# Visualize results
benchmark.visualize_results()

# Visualize confusion matrices
benchmark.visualize_confusion_matrix()

# Perform hyperparameter tuning for RandomForest
param_grid = {"n_estimators": [10, 50, 100], "max_depth": [3, 5, 10]}
best_model = benchmark.hyperparameter_optimization("RandomForest", param_grid)
