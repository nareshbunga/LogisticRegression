"""
Social Network Advertisement Purchase Prediction
Logistic Regression classification project.

Run from the project root:
    python src/train_model.py

The script trains and evaluates a Logistic Regression model, predicts the
future_prediction.csv records, and saves metrics, predictions, and plots.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
IMAGE_DIR = PROJECT_ROOT / "images"

TRAIN_FILE = DATA_DIR / "logit_classification.csv"
FUTURE_FILE = DATA_DIR / "future_prediction.csv"

FEATURES = ["Age", "EstimatedSalary"]
TARGET = "Purchased"
RANDOM_STATE = 0
TEST_SIZE = 0.25


def load_training_data(path: Path) -> pd.DataFrame:
    """Load and validate the labelled training dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")

    data = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Training dataset is missing columns: {missing}")

    data = data.dropna(subset=required).copy()
    data[TARGET] = data[TARGET].astype(int)

    labels = set(data[TARGET].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"{TARGET} must contain only 0 and 1. Found: {labels}")

    return data


def build_model() -> Pipeline:
    """Create a pipeline so training and future data use identical scaling."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=RANDOM_STATE)),
        ]
    )


def save_decision_boundary(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    title: str,
    output_path: Path,
) -> None:
    """Save a two-feature decision-boundary plot in original feature units."""
    age_min, age_max = X["Age"].min() - 5, X["Age"].max() + 5
    salary_min, salary_max = (
        X["EstimatedSalary"].min() - 5000,
        X["EstimatedSalary"].max() + 5000,
    )

    age_grid, salary_grid = np.meshgrid(
        np.linspace(age_min, age_max, 250),
        np.linspace(salary_min, salary_max, 250),
    )
    grid = pd.DataFrame(
        {
            "Age": age_grid.ravel(),
            "EstimatedSalary": salary_grid.ravel(),
        }
    )
    predictions = model.predict(grid).reshape(age_grid.shape)

    cmap = ListedColormap(["#f5aaaa", "#a8ddb5"])
    plt.figure(figsize=(9, 6))
    plt.contourf(age_grid, salary_grid, predictions, alpha=0.45, cmap=cmap)

    point_colors = {0: "#c62828", 1: "#168447"}
    for label in sorted(y.unique()):
        mask = y == label
        plt.scatter(
            X.loc[mask, "Age"],
            X.loc[mask, "EstimatedSalary"],
            color=point_colors[label],
            label=f"Purchased = {label}",
            edgecolor="white",
            linewidth=0.4,
            s=28,
        )

    plt.title(title)
    plt.xlabel("Age")
    plt.ylabel("Estimated Salary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    data = load_training_data(TRAIN_FILE)
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_probability = model.predict_proba(X_test)[:, 1]

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    matrix = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    report = classification_report(
        y_test,
        y_test_pred,
        labels=[0, 1],
        target_names=["Not Purchased (0)", "Purchased (1)"],
        output_dict=True,
        zero_division=0,
    )
    auc = roc_auc_score(y_test, y_test_probability)

    print("\n=== Model Evaluation ===")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy:     {test_accuracy:.4f}")
    print(f"ROC-AUC:           {auc:.4f}")
    print("\nConfusion matrix:")
    print(matrix)
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_test_pred,
            labels=[0, 1],
            target_names=["Not Purchased (0)", "Purchased (1)"],
            zero_division=0,
        )
    )

    metrics = {
        "training_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "roc_auc": float(auc),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "features": FEATURES,
        "target": TARGET,
    }
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Confusion matrix plot
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Not Purchased", "Purchased"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Logistic Regression - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    # ROC curve
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test, y_test_probability
    )
    plt.figure(figsize=(7, 5))
    plt.plot(false_positive_rate, true_positive_rate, label=f"ROC-AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "roc_curve.png", dpi=160)
    plt.close()

    # Decision boundaries use the original Age and EstimatedSalary units.
    save_decision_boundary(
        model,
        X_train,
        y_train,
        "Logistic Regression - Training Set",
        IMAGE_DIR / "training_decision_boundary.png",
    )
    save_decision_boundary(
        model,
        X_test,
        y_test,
        "Logistic Regression - Test Set",
        IMAGE_DIR / "test_decision_boundary.png",
    )

    # Predict future records with the SAME fitted pipeline/scaler.
    if FUTURE_FILE.exists():
        future_data = pd.read_csv(FUTURE_FILE)
        missing = [column for column in FEATURES if column not in future_data.columns]
        if missing:
            raise ValueError(f"Future dataset is missing columns: {missing}")

        future_predictions = model.predict(future_data[FEATURES])
        future_probabilities = model.predict_proba(future_data[FEATURES])[:, 1]

        results = future_data.copy()
        results["Predicted_Purchased"] = future_predictions
        results["Purchase_Probability"] = future_probabilities.round(4)
        results["Prediction_Label"] = results["Predicted_Purchased"].map(
            {0: "Not Purchased", 1: "Purchased"}
        )
        results.to_csv(
            OUTPUT_DIR / "future_predictions.csv",
            index=False,
        )
        print("\n=== Future Predictions ===")
        print(results.to_string(index=False))

    print(f"\nFiles saved in: {PROJECT_ROOT}")


if __name__ == "__main__":
    main()
