import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parent.parent
train = root / 'data' / 'raw' / 'dftrain.h5'
valid = root / 'data' / 'raw' / 'dfvalid.h5'
ground_truth = root / 'data' / 'raw' / 'dfvalid_groundtruth.csv'

def load_hdf5(path):
    with h5py.File(path, 'r') as file:
        dataset = file[list(file.keys())[0]]['block0_values']
        data = dataset[:]

        return data

def load_labels(path):
    labels = pd.read_csv(path)
    return labels

# load_labels(ground_truth)
# load_hdf5(train)
X_train = load_hdf5(train)
X_valid = load_hdf5(valid)

y_valid = load_labels(ground_truth)

assert X_valid.shape[0] == y_valid.shape[0], "validation data and labels don't match"




def plot_signal(signal, title):
    time = np.arange(len(signal)) / 1024 # 1024 hz

    plt.figure(figsize=(10,4))
    plt.plot(time, signal, linewidth=0.5)
    plt.xlabel("time (seconds)")
    plt.ylabel("acceleration")
    plt.title(title)
    plt.show()


# first training seq
plot_signal(X_train[0], "training sequence (dftrain.h5, sequence no. 0)") # healthy, all training is healthy

# first healthy validation seq
normal_idx = y_valid[y_valid['anomaly'] == 0].index[0]
plot_signal(X_valid[normal_idx], f"healthy validation sequence (dfvalid.h5, sequence no. {normal_idx})")

# first anomalous validation seq
anomaly_idx = y_valid[y_valid['anomaly'] == 1].index[0]
plot_signal(X_valid[anomaly_idx], f"anomalous validation sequence (dfvalid.h5, sequence no. {anomaly_idx})")
