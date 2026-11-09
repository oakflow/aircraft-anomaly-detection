import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Load the feature files
# training_features.csv: 1,677 recordings, all normal.
# validation_features.csv: 594 recordings, normal and anomalous,
# with an "anomaly" column (0 = normal, 1 = anomaly).
train_df = pd.read_csv("data/processed/training_features.csv")
valid_df = pd.read_csv("data/processed/validation_features.csv")

# Separate the features from the id and the label
# What: X keeps the 30 features, y_valid keeps the labels.
# Why: seq_id is only an id, and the model must never see the label (the answer).
X_train = train_df.drop(columns = ["seq_id"])
X_valid = valid_df.drop(columns = ["seq_id", "anomaly"])
y_valid = valid_df["anomaly"].to_numpy()

# Column positions of the two flags, used by the rule in predict_labels
constant_column = list(X_valid.columns).index("is_constant")
stuck_column = list(X_valid.columns).index("is_stuck") 

# Fill missing values with the training medians
# What: replaces each empty value with that feature's training median.
# Why: 28 flat validation signals have empty features, and the model cannot use empty values.
# Medians come from training data only, so validation data is not used to prepare the model.
imputer = SimpleImputer(strategy = "median")
X_train = imputer.fit_transform(X_train)
X_valid = imputer.transform(X_valid)

# Split the labelled data in half: tune half and test half
# What: 594 labelled recordings -> tune half (297) and test half (297).
# Why: settings are chosen on the tune half, and the test half gives a fair final score
# (Tutorial 3, Hyperparameter Search). stratify keeps the class ratio, random_state fixes the split.
X_tune, X_test, y_tune, y_test = train_test_split(
    X_valid, y_valid, test_size = 0.5, stratify = y_valid, random_state = 42)


# Build and train the model: scaling + One Class SVM
# What: scales the features, then trains a One-Class SVM on normal recordings only.
# Why: the rbf kernel uses distances, so features need the same scale (tutorial 1).
# The training data has only normal recordings, so the model learns a boundary around them.
# nu: max share of training recordings outside the boundary. gamma: how tight the boundary is.
def training(X_train, nu, gamma):
    model = Pipeline([
        ("scale", StandardScaler()),
        ("svm", OneClassSVM(kernel = "rbf", nu = nu, gamma = gamma))])
    model.fit(X_train)
    return model

# Convert the predictions to the project labels, then apply the team rule
# What: One Class SVM gives 1 = normal, -1 = anomaly; this changes them to 0 = normal, 1 = anomaly.
# Then any recording with is_constant == 1 or is_stuck == 1 is set to anomaly.
# Why: a healthy sensor is never constant or stuck in flight (rule agreed by the team).
def predict_labels(model, X):
    raw_predictions = model.predict(X)
    predictions = (raw_predictions == -1).astype(int)
    flagged = (X[:, constant_column] == 1) | (X[:, stuck_column] == 1)
    predictions[flagged] = 1
    return predictions 

# Try every combination of settings on the tune half
# What: trains 12 models (3 nu x 4 gamma) and ranks them by F1 on the tune half.
# Why: the best settings are unknown in advance. F1 is high only if precision and recall are both high.
records = []
for nu in [0.01, 0.05, 0.1]:
    for gamma in ["scale", 0.001, 0.01, 0.1]:
        model = training(X_train, nu, gamma)
        tune_pred = predict_labels(model, X_tune)
        records.append({
            "nu": nu,
            "gamma": gamma,
            "tune_f1": f1_score(y_tune, tune_pred, zero_division = 0)})

results = pd.DataFrame(records)
results = results.sort_values("tune_f1", ascending = False)
print(results.to_string(index = False))

# Train the best model and test it once on the test half
# What: trains the final model with the best settings and predicts the test half.
# Why: trained on normal data only (the tune half has anomalies). The test half is used only here.
best = results.iloc[0]
print("\nBest settings: nu =", best["nu"], ", gamma =", best["gamma"])
final_model = training(X_train, best["nu"], best["gamma"])
test_pred = predict_labels(final_model, X_test)

# Baseline: always predict "normal"
# What: predicts 0 (normal) for every test recording.
# Why: the report requires a majority-vote baseline; our model must beat it (tutorial 1).
baseline_pred = np.full_like(y_test, 0)

# Print the results of the baseline and the one class SVM on the test half
# Anomaly (1) is the positive class. 
print("\nFinal results on the test half (anomaly = positive class)")
for name, pred in [("Baseline (always predicts normal)", baseline_pred), ("One-Class SVM", test_pred)]:
    cm = confusion_matrix(y_test, pred, labels = [0, 1])
    print("\n" + name)
    print("  Accuracy  (correct predictions overall)   :", round(np.mean(pred == y_test), 3))
    print("  Precision (flagged that were anomalies)   :", round(precision_score(y_test, pred, zero_division = 0), 3))
    print("  Recall    (anomalies that were found)     :", round(recall_score(y_test, pred, zero_division = 0), 3))
    print("  F1        (balance of precision, recall)  :", round(f1_score(y_test, pred, zero_division = 0), 3))
    print("  Normal recordings kept normal             :", cm[0, 0])
    print("  Normal recordings wrongly flagged         :", cm[0, 1])
    print("  Anomalies missed                          :", cm[1, 0])
    print("  Anomalies found                           :", cm[1, 1])