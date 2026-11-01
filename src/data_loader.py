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



def plot_signal(signal, title):
    time = np.arange(len(signal)) / 1024 # 1024 hz

    plt.figure(figsize=(10,4))
    plt.plot(time, signal, linewidth=0.5)
    plt.xlabel("time (seconds)")
    plt.ylabel("acceleration")
    plt.title(title)
    plt.show()

def preview(X_train, X_valid, y_valid):
    assert X_valid.shape[0] == y_valid.shape[0],\
    "validation data and labels don't match"

    train_idx = np.random.randint(len(X_train))

    normal_idx = np.random.choice(
        y_valid[y_valid['anomaly'] == 0].index
        )
    
    anomaly_idx = np.random.choice( 
        y_valid[y_valid['anomaly'] == 1].index
        )

    # random training seq, healthy, all training is healthy
    plot_signal(X_train[train_idx], \
                f"training sequence (dftrain.h5, sequence no. {train_idx})") 

    # random healthy validation seq
    plot_signal(X_valid[normal_idx], \
                f"healthy validation sequence (dfvalid.h5, sequence no. {normal_idx})")

    # random anomalous validation seq
    plot_signal(X_valid[anomaly_idx],\
                f"anomalous validation sequence (dfvalid.h5, sequence no. {anomaly_idx})")


if __name__ == "__main__":
    X_train = load_hdf5(train)
    X_valid = load_hdf5(valid)
    
    y_valid = load_labels(ground_truth)

    preview(X_train, X_valid, y_valid)