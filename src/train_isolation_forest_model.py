
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
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

y_valid = valid_df["anomaly"].astype(int).to_numpy()

print("Training shape:", X_train.shape)
print("Validation shape:", X_valid.shape)


# --------------------------------------------------
# 3. DATA VALIDATION
# --------------------------------------------------

assert list(X_train.columns) == list(X_valid.columns), \
    "Training and validation features do not match"

X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_valid = X_valid.apply(pd.to_numeric, errors="coerce")

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_valid = X_valid.replace([np.inf, -np.inf], np.nan)


# Get positions of the two rule-based flags
constant_column = list(X_train.columns).index("is_constant")
stuck_column = list(X_train.columns).index("is_stuck")


# --------------------------------------------------
# 4. HANDLE MISSING VALUES
# --------------------------------------------------

# Learn medians from training data only

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_valid = imputer.transform(X_valid)


# --------------------------------------------------
# 5. SPLIT VALIDATION DATA
# --------------------------------------------------

# Same split as the One-Class SVM implementation:
# 297 tuning recordings + 297 testing recordings.

X_tune, X_test, y_tune, y_test = train_test_split(
    X_valid,
    y_valid,
    test_size=0.5,
    stratify=y_valid,
    random_state=42
)

print("\nData split:")
print("Training recordings:", len(X_train))
print("Tuning recordings:", len(X_tune))
print("Testing recordings:", len(X_test))


# --------------------------------------------------
# 6. TRAINING FUNCTION
# --------------------------------------------------

def training(X_train, n_estimators, max_samples, contamination):

    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=42
    )

    model.fit(X_train)

    return model


# --------------------------------------------------
# 7. PREDICTION FUNCTION
# --------------------------------------------------

def predict_labels(model, X):

    # Isolation Forest:
    #  1 = Normal
    # -1 = Anomaly

    raw_predictions = model.predict(X)

    # Convert to project convention:
    # 0 = Normal
    # 1 = Anomaly

    predictions = (raw_predictions == -1).astype(int)

    # Apply team rule:
    # Constant or stuck signals are always anomalies.

    flagged = (
        (X[:, constant_column] == 1) |
        (X[:, stuck_column] == 1)
    )

    predictions[flagged] = 1

    return predictions


# --------------------------------------------------
# 8. HYPERPARAMETER TUNING
# --------------------------------------------------

# Try 12 combinations:
# 2 n_estimators x 3 max_samples x 2 contamination

records = []

for n_estimators in [100, 200]:

    for max_samples in [256, 512, 1024]:

        for contamination in ["auto", 0.05]:

            model = training(
                X_train,
                n_estimators,
                max_samples,
                contamination
            )

            tune_pred = predict_labels(model, X_tune)

            records.append({
                "n_estimators": n_estimators,
                "max_samples": max_samples,
                "contamination": contamination,
                "tune_f1": f1_score(
                    y_tune,
                    tune_pred,
                    zero_division=0
                )
            })


tuning_results = pd.DataFrame(records)

tuning_results = tuning_results.sort_values(
    "tune_f1",
    ascending=False
)

print("\nHYPERPARAMETER TUNING RESULTS")
print(tuning_results.to_string(index=False))


# --------------------------------------------------
# 9. TRAIN FINAL MODEL
# --------------------------------------------------

best = tuning_results.iloc[0]

best_params = {
    "n_estimators": int(best["n_estimators"]),
    "max_samples": int(best["max_samples"]),
    "contamination": best["contamination"]
}

print("\nBest settings:", best_params)

final_model = training(
    X_train,
    best_params["n_estimators"],
    best_params["max_samples"],
    best_params["contamination"]
)


# --------------------------------------------------
# 10. FINAL EVALUATION ON TEST DATA
# --------------------------------------------------

test_pred = predict_labels(final_model, X_test)

# Baseline: always predict normal

baseline_pred = np.full_like(y_test, 0)

print("\nFINAL RESULTS ON TEST DATA")
print("(Anomaly = positive class)")

for name, pred in [
    ("Baseline (always predicts normal)", baseline_pred),
    ("Isolation Forest", test_pred)
]:

    cm = confusion_matrix(y_test, pred, labels=[0, 1])

    print("\n", name)

    print("Accuracy :", round(np.mean(pred == y_test), 3))

    print("Precision:", round(
        precision_score(y_test, pred, zero_division=0), 3
    ))

    print("Recall   :", round(
        recall_score(y_test, pred, zero_division=0), 3
    ))

    print("F1       :", round(
        f1_score(y_test, pred, zero_division=0), 3
    ))

    print("Normal recordings kept normal:", cm[0, 0])
    print("Normal recordings wrongly flagged:", cm[0, 1])
    print("Anomalies missed:", cm[1, 0])
    print("Anomalies found:", cm[1, 1])


# Detailed classification report

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        test_pred,
        target_names=["Normal", "Anomaly"],
        zero_division=0
    )
)


# --------------------------------------------------
# 11. ANOMALY SCORES AND PR-AUC
# --------------------------------------------------

# Lower decision_function values indicate more anomalous samples.
# Negate them so higher scores indicate more anomalous samples.

anomaly_scores = -final_model.decision_function(X_test)

# Account for the hard rule-based flags in the scores too.

flagged = (
    (X_test[:, constant_column] == 1) |
    (X_test[:, stuck_column] == 1)
)

if flagged.any():
    anomaly_scores[flagged] = anomaly_scores.max() + 1

pr_auc = average_precision_score(y_test, anomaly_scores)

print("\nPR-AUC:", round(pr_auc, 3))


# --------------------------------------------------
# 12. CONFUSION MATRIX VISUALIZATION
# --------------------------------------------------

cm = confusion_matrix(y_test, test_pred, labels=[0, 1])

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
# 13. SAVE MODEL AND PREPROCESSING
# --------------------------------------------------

joblib.dump(
    {
        "model": final_model,
        "imputer": imputer,
        "features": list(train_df.drop(columns=["seq_id"]).columns),
        "best_params": best_params
    },
    MODEL_DIR / "isolation_forest.joblib"
)

print("\nModel and imputer saved successfully!")


# --------------------------------------------------
# 14. INVESTIGATE MISCLASSIFICATIONS
# --------------------------------------------------

results = valid_df.iloc[
    np.flatnonzero(np.isin(
        valid_df.index,
        valid_df.index
    ))
].copy()

# Use the original validation indices corresponding to the test split.
# train_test_split preserves the original array indices through its outputs.

_, test_indices = train_test_split(
    np.arange(len(valid_df)),
    test_size=0.5,
    stratify=y_valid,
    random_state=42
)

results = valid_df.iloc[test_indices][
    ["seq_id", "anomaly", "is_constant", "is_stuck"]
].copy()

results["prediction"] = test_pred
results["anomaly_score"] = anomaly_scores

# False negatives: actual anomalies predicted as normal

false_negatives = results[
    (results["anomaly"] == 1) &
    (results["prediction"] == 0)
]

# False positives: actual normal predicted as anomaly

false_positives = results[
    (results["anomaly"] == 0) &
    (results["prediction"] == 1)
]

print("\nFALSE NEGATIVES:")
print(false_negatives)

print("\nFALSE POSITIVES:")
print(false_positives)


# Constant signal analysis

constant_signals = results[
    results["is_constant"] == 1
]

print("\nCONSTANT SIGNAL ANALYSIS")
print("Total:", len(constant_signals))
print(
    "Correctly detected:",
    (constant_signals["prediction"] == 1).sum()
)
print(
    "Missed:",
    (constant_signals["prediction"] == 0).sum()
)


# Stuck signal analysis

stuck_signals = results[
    results["is_stuck"] == 1
]

print("\nSTUCK SIGNAL ANALYSIS")
print("Total:", len(stuck_signals))
print(
    "Correctly detected:",
    (stuck_signals["prediction"] == 1).sum()
)
print(
    "Missed:",
    (stuck_signals["prediction"] == 0).sum()
)

print("\nInvestigation completed!")
