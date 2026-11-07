import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score
)

# Load feature datasets
train_df = pd.read_csv("data/processed/training_features.csv")
valid_df = pd.read_csv("data/processed/validation_features.csv")

# Separate features
X_train = train_df.drop(columns=["seq_id"])
X_valid = valid_df.drop(columns=["seq_id", "anomaly"])

# Ground truth
y_valid = valid_df["anomaly"].astype(int)

# Handle missing values using training medians
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_valid = imputer.transform(X_valid)

print("Training shape:", X_train.shape)
print("Validation shape:", X_valid.shape)
print("Remaining NaNs:", np.isnan(X_valid).sum())
