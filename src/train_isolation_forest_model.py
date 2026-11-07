
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "processed" / "training_features.csv"
VALID_PATH = ROOT / "data" / "processed" / "validation_features.csv"

MODEL_DIR = ROOT / "models" / "shared"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
valid_df = pd.read_csv(VALID_PATH)

# Separate features from identifiers and labels
X_train = train_df.drop(columns=["seq_id"])
X_valid = valid_df.drop(columns=["seq_id", "anomaly"])

# Ground truth (only used for evaluation)
y_valid = valid_df["anomaly"].astype(int)


# --------------------------------------------------
# 3. DATA VALIDATION
# --------------------------------------------------

# Ensure training and validation have identical features
assert list(X_train.columns) == list(X_valid.columns), \
    "Training and validation features do not match"

# Convert all features to numeric
X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_valid = X_valid.apply(pd.to_numeric, errors="coerce")

# Check for infinite values
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_valid = X_valid.replace([np.inf, -np.inf], np.nan)

print("Training shape:", X_train.shape)
print("Validation shape:", X_valid.shape)

print("\nMissing values before imputation:")
print("Training:", X_train.isna().sum().sum())
print("Validation:", X_valid.isna().sum().sum())


# --------------------------------------------------
# 4. HANDLE MISSING VALUES
# --------------------------------------------------

# Learn feature medians from training data only
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_valid = imputer.transform(X_valid)

print("\nAfter imputation:")
print("Training NaNs:", np.isnan(X_train).sum())
print("Validation NaNs:", np.isnan(X_valid).sum())


# --------------------------------------------------
# 5. INITIALIZE ISOLATION FOREST
# --------------------------------------------------

model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42
)


# --------------------------------------------------
# 6. TRAIN MODEL
# --------------------------------------------------

print("\nTraining Isolation Forest...")

# Only healthy training recordings are used
model.fit(X_train)

print("Training completed!")


# --------------------------------------------------
# 7. PREDICT ON VALIDATION DATA
# --------------------------------------------------

# Isolation Forest:
#  1 = Normal
# -1 = Anomaly

raw_predictions = model.predict(X_valid)

# Convert to project convention:
# 0 = Normal
# 1 = Anomaly

predictions = (raw_predictions == -1).astype(int)


# --------------------------------------------------
# 8. EVALUATE MODEL
# --------------------------------------------------

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_valid,
        predictions,
        target_names=["Normal", "Anomaly"],
        zero_division=0
    )
)

cm = confusion_matrix(y_valid, predictions)

print("\nCONFUSION MATRIX")
print(cm)

print("\nINDIVIDUAL METRICS")
print("Precision:", precision_score(y_valid, predictions, zero_division=0))
print("Recall:", recall_score(y_valid, predictions, zero_division=0))
print("F1-score:", f1_score(y_valid, predictions, zero_division=0))


# --------------------------------------------------
# 9. ANOMALY SCORES
# --------------------------------------------------

# Lower decision_function scores indicate more anomalous samples.
# We negate the scores so higher values indicate more anomalous.

anomaly_scores = -model.decision_function(X_valid)

pr_auc = average_precision_score(y_valid, anomaly_scores)

print("\nPR-AUC:", pr_auc)


# --------------------------------------------------
# 10. CONFUSION MATRIX VISUALIZATION
# --------------------------------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Anomaly"]
)

display.plot(cmap="Blues", values_format="d")

plt.title("Isolation Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    MODEL_DIR / "isolation_forest_confusion_matrix.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 11. SAVE MODEL AND PREPROCESSING
# --------------------------------------------------

joblib.dump(
    {
        "model": model,
        "imputer": imputer,
        "features": list(train_df.drop(columns=["seq_id"]).columns)
    },
    MODEL_DIR / "isolation_forest.joblib"
)

print("\nModel and imputer saved successfully!")
