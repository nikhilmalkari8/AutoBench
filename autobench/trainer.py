import warnings
import time
import logging
import joblib
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import shap
from fpdf import FPDF  # For PDF reporting


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Benchmark:
    def __init__(self, X_train, X_test, Y_train, Y_test, task="binary", cv=5):
        """
        Initialize the benchmark class.
        :param X_train: Training features.
        :param X_test: Testing features.
        :param Y_train: Training labels.
        :param Y_test: Testing labels.
        :param task: Task type ('binary', 'multiclass', 'multilabel', 'regression').
        :param cv: Number of cross-validation folds.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.task = task
        self.cv = cv
        self.results = {}
        self.models = self.initialize_models()

    def initialize_models(self):
        """
        Initialize models based on the task type.
        """
        if self.task in ["binary", "multiclass"]:
            return {
                "LogisticRegression": LogisticRegression(),
                "DecisionTree": DecisionTreeClassifier(),
                "RandomForest": RandomForestClassifier(),
                "SupportVectorMachine": SVC(probability=True),
                "GradientBoosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "KNearestNeighbors": KNeighborsClassifier(),
                "NeuralNetwork": MLPClassifier(max_iter=500),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
                "LightGBM": LGBMClassifier(verbose=-1),
            }
        elif self.task == "multilabel":
            return {
                "MultiOutputLogisticRegression": MultiOutputClassifier(LogisticRegression()),
                "MultiOutputDecisionTree": MultiOutputClassifier(DecisionTreeClassifier()),
                "MultiOutputRandomForest": MultiOutputClassifier(RandomForestClassifier()),
                "MultiOutputGradientBoosting": MultiOutputClassifier(GradientBoostingClassifier()),
                "MultiOutputSVC": MultiOutputClassifier(SVC()),
            }
        elif self.task == "regression":
            return {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "LightGBMRegressor": LGBMRegressor(verbose=-1),
            }
        else:
            raise ValueError(f"Unsupported task type: {self.task}")

    def train_and_evaluate(self):
        """
        Train models and evaluate performance.
        :return: Dictionary of results.
        """
        warnings.filterwarnings("ignore")  # Ignore warnings globally

        if not self.models:
            raise ValueError("Models are not initialized. Ensure the task type is correct.")

        for name, model in self.models.items():
            try:
                # Start timing
                start_time = time.time()

                # Cross-validation
                cv_scores = cross_val_score(model, self.X_train, self.Y_train, cv=self.cv)

                # Train and predict
                model.fit(self.X_train, self.Y_train)
                predictions = model.predict(self.X_test)

                # End timing
                end_time = time.time()
                training_time = end_time - start_time

                # Classification metrics
                if self.task in ["binary", "multiclass"]:
                    self.results[name] = {
                        "cv_mean_accuracy": cv_scores.mean(),
                        "cv_std_accuracy": cv_scores.std(),
                        "accuracy": accuracy_score(self.Y_test, predictions),
                        "precision": precision_score(self.Y_test, predictions, average="weighted"),
                        "recall": recall_score(self.Y_test, predictions, average="weighted"),
                        "f1_score": f1_score(self.Y_test, predictions, average="weighted"),
                        "roc_auc": roc_auc_score(self.Y_test, model.predict_proba(self.X_test), multi_class="ovo")
                        if hasattr(model, "predict_proba")
                        else None,
                        "training_time": training_time,
                        "confusion_matrix": confusion_matrix(self.Y_test, predictions).tolist(),
                    }

                # Regression metrics
                elif self.task == "regression":
                    self.results[name] = {
                        "cv_mean_r2": cv_scores.mean(),
                        "cv_std_r2": cv_scores.std(),
                        "mse": mean_squared_error(self.Y_test, predictions),
                        "mae": mean_absolute_error(self.Y_test, predictions),
                        "r2": r2_score(self.Y_test, predictions),
                        "training_time": training_time,
                    }

                logging.info(f"{name}: Training completed successfully.")

            except Exception as e:
                self.results[name] = {"error": str(e)}
                logging.error(f"{name}: Training failed with error: {str(e)}")

        return self.results

    def visualize_feature_importance(self, model_name):
        """
        Visualize feature importance using SHAP for tree-based models.
        """
        if model_name not in self.models:
            print(f"Model '{model_name}' not found.")
            return

        model = self.models[model_name]
        if not hasattr(model, "feature_importances_"):
            print(f"Model '{model_name}' does not support feature importance.")
            return

        # SHAP Explanation
        explainer = shap.Explainer(model, self.X_train)
        shap_values = explainer(self.X_test)
        shap.summary_plot(shap_values, self.X_test, show=False)
        plt.savefig(f"{model_name}_feature_importance.png")
        plt.close()

    def save_results_to_file(self, filename="results.json"):
        """
        Save the results to a JSON file.
        :param filename: Name of the file to save results.
        """
        with open(filename, "w") as file:
            json.dump(self.results, file, indent=4)

    def display_results(self):
        """
        Display the results in a tabular format.
        """
        df = pd.DataFrame(self.results).T
        print(df)

    def generate_pdf_report(self, filename="report.pdf"):
        """
        Generate a PDF report summarizing results and metrics.
        """
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Add summary
        pdf.cell(200, 10, txt="Benchmark Report", ln=True, align="C")
        for model, metrics in self.results.items():
            pdf.multi_cell(0, 10, txt=f"{model}: {metrics}", align="L")

        pdf.output(filename)
        logging.info(f"PDF report saved as {filename}.")
