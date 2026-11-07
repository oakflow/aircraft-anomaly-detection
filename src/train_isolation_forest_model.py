
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
