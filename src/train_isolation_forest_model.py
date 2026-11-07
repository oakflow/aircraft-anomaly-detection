
import pandas as pd

train_df = pd.read_csv("data/processed/training_features.csv")
valid_df = pd.read_csv("data/processed/validation_features.csv")
labels_df = pd.read_csv("data/raw/dfvalid_groundtruth.csv")

# 1. Separate features from identifiers and labels
X_train = train_df.drop(columns=["seq_id"])

X_valid = valid_df.drop(columns=["seq_id", "anomaly"])

y_valid = labels_df.set_index("seqID")["anomaly"]

# 2. Check feature consistency
print("Same features:", list(X_train.columns) == list(X_valid.columns))

# 3. Check validation labels match by sequence ID
valid_labels = valid_df.set_index("seq_id")["anomaly"]
print("Labels match:",
      valid_labels.sort_index().equals(y_valid.sort_index()))

# 4. Check missing values
print("\nMissing values:")
print("Training:", X_train.isna().sum().sum())
print("Validation:", X_valid.isna().sum().sum())

# 5. Check class distribution
print("\nValidation class distribution:")
print(y_valid.value_counts())

# 6. Check training feature dimensions
print("\nTraining shape:", X_train.shape)
print("Validation shape:", X_valid.shape)








# A. Investigate label mismatch

valid_labels = (
    valid_df.set_index("seq_id")["anomaly"]
    .sort_index()
    .astype(float)
)

ground_labels = (
    labels_df.set_index("seqID")["anomaly"]
    .sort_index()
    .astype(float)
)

print("\n--- LABEL DIAGNOSTICS ---")
print("Same sequence IDs:",
      valid_labels.index.equals(ground_labels.index))

print("Validation label dtype:", valid_labels.dtype)
print("Ground truth dtype:", ground_labels.dtype)

# Compare actual values, ignoring dtype differences
mismatches = valid_labels[valid_labels != ground_labels]

print("Actual label mismatches:", len(mismatches))
print(mismatches.head(10))

# B. Investigate missing feature values

print("\n--- MISSING VALUE DIAGNOSTICS ---")

missing_by_column = X_valid.isna().sum().sort_values(ascending=False)

print("\nMissing values per feature:")
print(missing_by_column[missing_by_column > 0])

# Number of missing values in each recording
missing_by_row = X_valid.isna().sum(axis=1)

print("\nMissing values per recording:")
print(missing_by_row.describe())

# Show recordings with the most missing values
problem_rows = missing_by_row[missing_by_row > 0].sort_values(
    ascending=False
)

print("\nRecordings with missing values:")
print(problem_rows.head(15))

# C. Check whether missing values are associated with anomalies

print("\nMissing values by actual class:")

for label in [0, 1]:
    rows = y_valid[y_valid == label].index
    print(
        f"Class {label}:",
        X_valid.loc[rows].isna().sum().sum(),
        "missing values"
    )









# Identify all affected recordings
affected = X_valid[X_valid.isna().any(axis=1)].copy()

print("Affected recordings:", len(affected))

# Check the features that are NOT missing
print("\nOther available features:")
print(
    affected[
        ["mean", "std", "min", "max", "rms",
         "is_constant", "is_stuck"]
    ].head(30)
)

# Check the missingness pattern
print("\nMissingness pattern:")
print(affected.isna().sum(axis=1).value_counts())

# Check whether the affected recordings have constant signals
print("\nConstant signal flags:")
print(valid_df.loc[affected.index, "is_constant"].value_counts())

# Show the sequence IDs
print("\nAffected sequence IDs:")
print(valid_df.loc[affected.index, "seq_id"].tolist())
