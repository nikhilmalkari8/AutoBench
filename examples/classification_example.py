import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from autobench.trainer import Benchmark

# Load sample dataset
data = load_iris()
X = data.data
Y = data.target

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Run the benchmark
benchmark = Benchmark(X_train, X_test, Y_train, Y_test)
results = benchmark.train_and_evaluate()

# Display results
print("Model Performance:")
for model, metrics in results.items():
    print(f"{model}: {metrics}")
