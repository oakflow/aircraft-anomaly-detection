from data_loader import (load_hdf5, load_labels, train, valid, ground_truth)
from scipy.stats import skew, kurtosis
from scipy.signal import welch, correlate, spectrogram
import scipy.integrate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

root = Path(__file__).resolve().parent.parent

X_train = load_hdf5(train)
X_valid = load_hdf5(valid)
y_valid = load_labels(ground_truth)

def extract_features(signal):
    """
    Extract time domain features,
    then frequency domain features through extract_frequency_features(),
    and return a dict of both feature sets for the given signal
    """

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
    """
    Extract frequency domain features from a signal
    """

    hz = 1024

    if np.std(signal) < 1e-10:
        return {
            'dominant_frequency': 0.0,
            'spectral_centroid': np.nan,
            'spectral_entropy': np.nan,
            'spectral_bandwidth': np.nan,
            'spectral_rolloff': np.nan,
            'spectral_flatness': np.nan,
            'below_1hz_power': np.nan,
            '1_2hz_power': np.nan,
            '2_4hz_power': np.nan,
            '4_8hz_power': np.nan,
            '8_16hz_power': np.nan,
            '16_32hz_power': np.nan,
            '32_64hz_power': np.nan,
            '64_128hz_power': np.nan,
            '128_256hz_power': np.nan,
            '256_512hz_power': np.nan,
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
    bands = [
        (0, 1, 'below_1hz'),
        (1, 2, '1_2hz'),
        (2, 4, '2_4hz'),
        (4, 8, '4_8hz'),
        (8, 16, '8_16hz'),
        (16, 32, '16_32hz'),
        (32, 64, '32_64hz'),
        (64, 128, '64_128hz'),
        (128, 256, '128_256hz'),
        (256, 512, '256_512hz')
    ]

    band_features = {}

    for low, high, name in bands:

        mask = (frequencies_psd >= low) & (frequencies_psd < high)

        band_power = scipy.integrate.trapezoid(
            psd[mask], frequencies_psd[mask]
        )

        band_features[f'{name}_power'] = band_power

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
        **band_features,
        'autocorrelation_lag1': autocorrelation_lag1,
        'autocorrelation_lag10': autocorrelation_lag10
    }


def post_feature_extraction(t_feature_set: pd.DataFrame, v_feature_set: pd.DataFrame):
    """
    Post-processing on the feature sets, 
    needed for any features that rely on feature set stats,
    only stuck sensor detection is implemented here, but more could be added
    """

    std_threshold = t_feature_set["std"].quantile(0.01)
    mean_threshold = t_feature_set["mean"].abs().quantile(0.999)

    for fs in [t_feature_set, v_feature_set]:
        low_std = fs["std"] < std_threshold
        large_offset = fs["mean"].abs() > mean_threshold
        is_constant = fs["is_constant"] == 1 

        fs["is_stuck"] = (low_std & large_offset & ~is_constant).astype(int)



# build the feature sets
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

# preview
# print(training_features.head())
# print(validation_features.head())

# quick sanity check
# print("val:", validation_features.shape)
# print("train:", training_features.shape)




# save to csv
# training_features.to_csv(
#      root/'data'/'processed'/'training_features.csv',
#      index=True,
#      index_label='seq_id'
# )

# validation_features.to_csv(
#      root/'data'/'processed'/'validation_features.csv',
#      index=True,
#      index_label='seq_id'
# )








# ------------------ boxplots ------------------

# features = [
#     'mean', 'std', 'min', 'max', 'rms',
#     'skewness', 'kurtosis', 'peak_to_peak',
#     'dominant_frequency', 'spectral_centroid',
#     'spectral_entropy',
#     'spectral_bandwidth', 'spectral_rolloff',
#     'spectral_flatness', 'below_1hz_power',
#     '1_2hz_power', '2_4hz_power', '4_8hz_power',
#     '8_16hz_power', '16_32hz_power', '32_64hz_power',
#     '64_128hz_power', '128_256hz_power', '256_512hz_power',
#     'autocorrelation_lag1', 'autocorrelation_lag10',
#     'mean_absolute_difference', 'zero_crossing_rate'
# ]

# for feature in features:
#     validation_features.boxplot(column=feature, by='anomaly')

#     if feature.endswith('_power'): plt.yscale('log')

#     plt.title(f'{feature} by anomaly class')
#     plt.suptitle('')
#     plt.xlabel('Anomaly (0 = Normal, 1 = Anomalous)')
#     plt.ylabel(feature)

#     plt.tight_layout()
#     plt.savefig(f'{root}/findings/comparisons/{feature}_boxplot.png', dpi=300, bbox_inches='tight')
#     plt.close()


# flag_counts = validation_features.groupby('anomaly')[
#     ['is_constant', 'is_stuck']
# ].sum()

# flag_counts.plot(kind='bar', color=["#747474", "#000000"])

# plt.title('Stuck Sensor Flags by Anomaly Class')
# plt.xlabel('Anomaly (0 = Normal, 1 = Anomalous)')
# plt.ylabel('Number of Signals Flagged')
# plt.xticks(rotation=0)

# plt.tight_layout()
# plt.savefig(f'{root}/findings/comparisons/stuck_sensor_counts.png', dpi=300)
# plt.close()






# ------------------ correlation matrix ------------------

# correlation_matrix = training_features.drop(columns=['is_constant', 'is_stuck']).corr()

# plt.figure(figsize=(20, 16))

# sns.heatmap(
#     correlation_matrix,
#     cmap='Greys',
#     vmin=-1,
#     vmax=1,
#     center=0,
#     annot=True,
#     fmt='.2f',
#     square=True,
#     linewidths=0.3
# )

# plt.title('Feature Correlation Matrix (Training Dataset)')
# plt.tight_layout()

# plt.savefig(
#     f'{root}/findings/correlations/correlation_matrix.png',
#     dpi=300,
#     bbox_inches='tight'
# )

# plt.close()






# ------------------ spectrogram ------------------
# (1 random healthy, 1 random anomalous)

# def plot_spectrogram(signal, title, ax, vmin, vmax):
#     frequencies, times, Sxx = spectrogram(
#         signal,
#         fs=1024,
#         nperseg=2048,
#         noverlap=1024
#     )

#     # convert to dB
#     Sxx_db = 10 * np.log10(Sxx + 1e-12)    
#     vmin = 10 * np.log10(vmin + 1e-12)
#     vmax = 10 * np.log10(vmax + 1e-12)

#     mesh = ax.pcolormesh(
#         times,
#         frequencies,
#         Sxx_db,
#         shading="auto",
#         cmap="viridis",
#         vmin=vmin,
#         vmax=vmax
#     )

#     ax.set_title(title)
#     ax.set_xlabel("time (s)")
#     ax.set_ylabel("frequency (Hz)")
#     ax.set_ylim(0, 512)

#     return mesh

# normal_idx = np.random.choice(
#     y_valid[y_valid['anomaly'] == 0].index
#     )

# anomaly_idx = np.random.choice( 
#     y_valid[y_valid['anomaly'] == 1].index
#     )

# normal_signal = X_valid[normal_idx]
# anomaly_signal = X_valid[anomaly_idx]

# signals = [
#     (normal_signal, "Normal Validation Signal", "normal_spectrogram.png"),
#     (anomaly_signal, "Anomalous Validation Signal", "anomalous_spectrogram.png")
# ]

# # calculate both spectrograms to establish a shared colour range
# _, _, normal_power = spectrogram(
#     normal_signal, fs=1024, nperseg=2048, noverlap=1024
# )

# _, _, anomaly_power = spectrogram(
#     anomaly_signal, fs=1024, nperseg=2048, noverlap=1024
# )

# vmin = min(normal_power.min(), anomaly_power.min())
# vmax = max(normal_power.max(), anomaly_power.max())

# for s in signals:
#     signal, title, filename = s

#     fig, ax = plt.subplots(figsize=(10, 4))
#     mesh = plot_spectrogram(signal, title, ax, vmin, vmax)
#     fig.colorbar(mesh, ax=ax, label="power (dB)")

#     plt.tight_layout()
#     plt.savefig(f'{root}/findings/comparisons/{filename}', dpi=300)
#     plt.close()