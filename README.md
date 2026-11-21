# Social Network Advertisement Purchase Prediction

A beginner-friendly **binary classification** project using **Logistic Regression** and Scikit-learn.

## 1. Project overview

A social network advertising dataset contains user information and whether each user purchased a product after seeing an advertisement. The model uses two features:

- `Age`
- `EstimatedSalary`

It predicts the target:

- `Purchased = 0`: the user did not purchase.
- `Purchased = 1`: the user purchased.

> This is a learning exercise using the supplied dataset. Predictions indicate patterns learned from the data; they do not establish why an individual made a purchase.

## 2. Repository structure

```text
LogisticRegression-Project/
├── data/
│   ├── logit_classification.csv
│   └── future_prediction.csv
├── images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── training_decision_boundary.png
│   └── test_decision_boundary.png
├── outputs/
│   ├── metrics.json
│   └── future_predictions.csv
├── src/
│   └── train_model.py
├── .gitignore
├── README.md
└── requirements.txt
```

## 3. Machine learning workflow

1. Load and validate the training data.
2. Select `Age` and `EstimatedSalary` as input features.
3. Select `Purchased` as the target.
4. Split the data into training and test sets (75% / 25%).
5. Apply `StandardScaler` to the input features.
6. Train a Logistic Regression classifier.
7. Predict labels for the test set.
8. Evaluate using accuracy, confusion matrix, precision, recall, F1-score and ROC-AUC.
9. Visualize the decision boundaries, confusion matrix and ROC curve.
10. Predict the supplied future records and export the results.

## 4. Why use a Pipeline?

The project uses a Scikit-learn `Pipeline` containing `StandardScaler` followed by `LogisticRegression`.

The scaler is fitted only on the training data. When test or future records are passed to the fitted pipeline, the same training-set mean and standard deviation are applied. This prevents inconsistent scaling and data leakage.

## 5. Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## 6. Run the project

Run this command from the repository root:

```bash
python src/train_model.py
```

The script prints the model evaluation and future predictions. It creates or updates the files in `outputs/` and `images/`.

## 7. Evaluation metrics

- **Accuracy:** proportion of all predictions that are correct.
- **Confusion matrix:** counts true negatives, false positives, false negatives and true positives.
- **Precision:** among records predicted as purchased, the proportion actually purchased.
- **Recall:** among actual purchasers, the proportion correctly identified.
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC:** measures how well the model ranks positive examples above negative examples across classification thresholds.

The generated values are in `outputs/metrics.json`. Results can vary if the split strategy, random seed, features or model settings change.

## 8. Training accuracy vs. test accuracy

The script reports training accuracy and test accuracy separately. These are performance measurements, **not bias and variance themselves**. A large gap between training and test performance can be a warning sign of overfitting, but a proper bias-variance assessment requires more analysis, such as cross-validation and learning curves.

## 9. Important implementation notes

- `User ID` and `Gender` are not used as model features in this exercise.
- The scaler is not refitted on the future dataset.
- Future predictions are generated with the same fitted preprocessing and model pipeline.
- The CSV export uses `index=False` to avoid an unnecessary index column.
- The model is trained and evaluated on a held-out test set.

## 10. Dataset

The two CSV files in `data/` are the files supplied for this exercise. If you redistribute this repository publicly, verify that you have permission to publish the dataset and that it contains no personal or confidential information.

## 11. Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Logistic Regression
- Standardization
- Binary Classification

## 12. Learning outcomes

After completing this project, you should be able to explain:

- The difference between linear regression and logistic regression.
- Why logistic regression is used for classification.
- How a train/test split works.
- Why feature scaling is useful.
- Why the scaler must be fitted only on training data.
- How to interpret a confusion matrix and classification report.
- How to use a fitted model to predict unseen records.
