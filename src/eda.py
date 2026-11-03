from data_loader import (load_hdf5, load_labels, train, valid, ground_truth)
from scipy.stats import skew, kurtosis
from scipy.signal import welch, correlate
import scipy.integrate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X_train = load_hdf5(train)
X_valid = load_hdf5(valid)
y_valid = load_labels(ground_truth)

def extract_features(signal):

    std = np.std(signal)
    is_constant = std < 1e-10
    centered_signal = signal - np.mean(signal)

    time_features = {
        'mean': np.mean(signal),
        'std': std,
        'min': np.min(signal),
        'max': np.max(signal),
        'rms': np.sqrt(np.mean(signal**2)),
        'skewness': np.nan if is_constant else skew(signal),
        'kurtosis': np.nan if is_constant else kurtosis(signal),
        'is_constant': int(is_constant),
        'peak_to_peak': np.max(signal) - np.min(signal),
        'mean_absolute_difference': np.mean(np.abs(np.diff(signal))),
        'zero_crossing_rate': np.mean(centered_signal[:-1] * centered_signal[1:] < 0),
    }

    frequency_features = extract_frequency_features(signal)

    return {
        **time_features,
        **frequency_features
    }

def extract_frequency_features(signal):

    hz = 1024

    if np.std(signal) < 1e-10:
        return {
            'dominant_frequency': 0.0,
            'spectral_centroid': np.nan,
            'spectral_entropy': np.nan,
            'spectral_bandwidth': np.nan,
            'spectral_rolloff': np.nan,
            'spectral_flatness': np.nan,
            'low_band_power': np.nan,
            'mid_band_power': np.nan,
            'high_band_power': np.nan,
            'autocorrelation_lag1': np.nan,
            'autocorrelation_lag10': np.nan
        }

    centered_signal = signal - np.mean(signal)

    # FFT
    fft = np.fft.rfft(centered_signal)
    magnitude = np.abs(fft)
    frequencies = np.fft.rfftfreq(len(centered_signal), d=1/hz)

    # existing frequency features
    dominant_frequency = frequencies[np.argmax(magnitude)]
    
    spectral_centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)

    power = magnitude**2
    prob = power / np.sum(power)

    spectral_entropy = -np.sum(prob * np.log2(prob + 1e-12))

    # welch, power spectral density
    frequencies_psd, psd = welch(
        centered_signal,
        fs=hz,
        nperseg=4096,
        noverlap=2048
    )

    # bandwidth
    psd_sum = np.sum(psd)
    psd_prob = psd / psd_sum
    psd_centroid = np.sum(frequencies_psd * psd_prob)

    spectral_bandwidth = np.sqrt(
        np.sum( (frequencies_psd - psd_centroid)**2 * psd_prob )
    )

    # rolloff (til 85%)
    cumulative_pwr = np.cumsum(psd)
    rolloff_threshold = .85 * cumulative_pwr[-1]

    spectral_rolloff = frequencies_psd[
        np.searchsorted(cumulative_pwr, rolloff_threshold)
        ]

    # flatness
    spectral_flatness = (
        np.exp(np.mean(np.log(psd+1e-12))) / np.mean(psd)
    )

    # welch band powers
    low_band_power = scipy.integrate.trapezoid(
        psd[(frequencies_psd >= 0) & (frequencies_psd < 128)],
        frequencies_psd[(frequencies_psd >= 0) & (frequencies_psd < 128)]
    )

    mid_band_power = scipy.integrate.trapezoid(
        psd[(frequencies_psd >= 128) & (frequencies_psd < 256)],
        frequencies_psd[(frequencies_psd >= 128) & (frequencies_psd < 256)]
    )

    high_band_power = scipy.integrate.trapezoid(
        psd[(frequencies_psd >= 256) & (frequencies_psd <= 512)],
        frequencies_psd[(frequencies_psd >= 256) & (frequencies_psd <= 512)]
    )

    # autocorrelation
    autocorrelation = correlate(
        centered_signal,
        centered_signal,
        mode='full',
        method='fft'
    )

    autocorrelation = autocorrelation[ len(centered_signal) - 1 :]
    
    # normalize against lag0
    autocorrelation = autocorrelation / autocorrelation[0]
    
    autocorrelation_lag1 = autocorrelation[1]
    autocorrelation_lag10 = autocorrelation[10]

    return {
        'dominant_frequency' : dominant_frequency,
        'spectral_centroid' : spectral_centroid,
        'spectral_entropy' : spectral_entropy,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_rolloff': spectral_rolloff,
        'spectral_flatness': spectral_flatness,
        'low_band_power': low_band_power,
        'mid_band_power': mid_band_power,
        'high_band_power': high_band_power,
        'autocorrelation_lag1': autocorrelation_lag1,
        'autocorrelation_lag10': autocorrelation_lag10
    }


def post_feature_extraction(t_feature_set: pd.DataFrame, v_feature_set: pd.DataFrame):

    std_threshold = t_feature_set["std"].quantile(0.01)
    mean_threshold = t_feature_set["mean"].abs().quantile(0.999)

    for fs in [t_feature_set, v_feature_set]:
        low_std = fs["std"] < std_threshold
        large_offset = fs["mean"].abs() > mean_threshold
        is_constant = fs["is_constant"] == 1

        fs["is_stuck"] = (is_constant | (low_std & large_offset)).astype(int)


training_features = []
for signal in X_train:
        training_features.append(extract_features(signal))
training_features = pd.DataFrame(training_features)

validation_features = []
for signal in X_valid:
        validation_features.append(extract_features(signal))
validation_features = pd.DataFrame(validation_features)
validation_features['anomaly'] = y_valid['anomaly'].astype(int)

post_feature_extraction(training_features, validation_features)

# print(training_features.head())
# print(validation_features.head())

# print("val:", validation_features.shape)
# print("train:", training_features.shape)