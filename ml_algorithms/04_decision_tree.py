"""Decision Tree classifier for the Alzheimer dataset-manifest ML benchmark.

Trained on the shared, pre-split features from data_preprocessing.py so every
algorithm in this benchmark sees an identical train/test split.
"""
import json
import time

from data_preprocessing import load_and_preprocess_data, RESULTS_DIR
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

ALGORITHM_NAME = "Decision Tree"
RESULT_FILE = RESULTS_DIR / "decision_tree.json"

PARAM_GRID = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"],
}


def main():
    X_train, X_test, y_train, y_test, label_names = load_and_preprocess_data()

    base_model = DecisionTreeClassifier(random_state=42)
    search = GridSearchCV(base_model, PARAM_GRID, cv=5, scoring="accuracy", n_jobs=-1)

    start = time.perf_counter()
    search.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = {
        "algorithm": ALGORITHM_NAME,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "train_time_sec": train_time,
        "best_params": {k: v for k, v in search.best_params_.items()},
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "label_names": label_names,
    }
    metrics["best_params"] = {k: (v if not isinstance(v, tuple) else list(v)) for k, v in metrics["best_params"].items()}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"=== {ALGORITHM_NAME} ===")
    print(f"Best Params: {search.best_params_}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))
    print(f"Training Time: {train_time:.3f}s")


if __name__ == "__main__":
    main()
