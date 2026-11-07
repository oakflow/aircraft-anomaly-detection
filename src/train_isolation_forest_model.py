
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

# Load datasets
train_df = pd.read_csv("data/processed/training_features.csv")
valid_df = pd.read_csv("data/processed/validation_features.csv")

# Separate features
X_train = train_df.drop(columns=["seq_id"])
X_valid = valid_df.drop(columns=["seq_id", "anomaly"])

y_valid = valid_df["anomaly"].astype(int)

print("TRAINING DTYPES:")
print(X_train.dtypes)

print("\nVALIDATION DTYPES:")
print(X_valid.dtypes)

# Check exact problematic values
print("\nTRAINING DATA TYPES:")
print(X_train.map(type).stack().value_counts())

print("\nVALIDATION DATA TYPES:")
print(X_valid.map(type).stack().value_counts())






print("Invalid training values:")

for col in X_train.columns:
    invalid = pd.to_numeric(X_train[col], errors="coerce").isna()

    if invalid.any():
        print(f"\nColumn: {col}")
        print(X_train.loc[invalid, col])





# Convert to numeric
X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_valid = X_valid.apply(pd.to_numeric, errors="coerce")

print("\nAfter conversion:")
print("Training NaNs:", X_train.isna().sum().sum())
print("Validation NaNs:", X_valid.isna().sum().sum())

# Imputation
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_valid = imputer.transform(X_valid)

print("\nFINAL SHAPES:")
print("Training:", X_train.shape)
print("Validation:", X_valid.shape)
print("Remaining NaNs:", np.isnan(X_valid).sum())
