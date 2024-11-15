from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class Benchmark:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        """
        Initialize the benchmark class.
        :param X_train: Training features.
        :param X_test: Testing features.
        :param Y_train: Training labels.
        :param Y_test: Testing labels.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.models = {
            "LogisticRegression": LogisticRegression(),
            "DecisionTree": DecisionTreeClassifier(),
            "RandomForest": RandomForestClassifier(),
        }
        self.results = {}

    def train_and_evaluate(self):
        """
        Train models and evaluate performance using accuracy.
        :return: Dictionary of results.
        """
        for name, model in self.models.items():
            model.fit(self.X_train, self.Y_train)
            predictions = model.predict(self.X_test)
            accuracy = accuracy_score(self.Y_test, predictions)
            self.results[name] = {"accuracy": accuracy}
        return self.results
