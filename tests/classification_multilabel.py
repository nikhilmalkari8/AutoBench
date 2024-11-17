from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from autobench.trainer import Benchmark

# Generate a synthetic multi-label classification dataset
print("=== Multi-Label Classification ===")
X, Y_raw = make_multilabel_classification(
    n_samples=500, n_features=20, n_classes=5, n_labels=3, random_state=42
)

# Convert labels to multi-label indicator format
mlb = MultiLabelBinarizer()
Y_multilabel = mlb.fit_transform(Y_raw)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_multilabel, test_size=0.2, random_state=42)

# Initialize Benchmark for multi-label classification
benchmark = Benchmark(X_train, X_test, Y_train, Y_test, task="multilabel", cv=5)

# Train and evaluate models
results = benchmark.train_and_evaluate()

# Save results to a file
benchmark.save_results_to_file("multilabel_results.json")

# Display results
benchmark.display_results()

# Visualize results (Accuracy and Training Time)
benchmark.visualize_results()

# Perform hyperparameter tuning for GradientBoosting
param_grid = {"n_estimators": [50, 100], "max_depth": [3, 5], "learning_rate": [0.01, 0.1]}
best_model = benchmark.hyperparameter_optimization("MultiOutputGradientBoosting", param_grid)
