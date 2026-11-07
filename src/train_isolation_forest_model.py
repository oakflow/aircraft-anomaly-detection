
import pandas as pd

# Load processed datasets
train_df = pd.read_csv("data/processed/training_features.csv")
valid_df = pd.read_csv("data/processed/validation_features.csv")

# Load actual validation labels
labels_df = pd.read_csv("data/raw/dfvalid_groundtruth.csv")

# Inspect training data
print("TRAINING DATA")
print(train_df.head())
print(train_df.shape)
print(train_df.columns.tolist())

# Inspect validation data
print("\nVALIDATION DATA")
print(valid_df.head())
print(valid_df.shape)

# Inspect labels
print("\nGROUND TRUTH LABELS")
print(labels_df.head())
print(labels_df.shape)
print(labels_df.columns.tolist())
