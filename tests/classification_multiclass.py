from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from autobench.trainer import Benchmark

# Multi-Class Classification Example
print("=== Multi-Class Classification ===")
iris = load_iris()
X_multiclass = iris.data
Y_multiclass = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X_multiclass, Y_multiclass, test_size=0.2, random_state=42)

benchmark_multiclass = Benchmark(X_train, X_test, Y_train, Y_test, task="multiclass", cv=5)
results_multiclass = benchmark_multiclass.train_and_evaluate()
benchmark_multiclass.display_results()
