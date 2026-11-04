# EDA and Feature Engineering Findings

## 1. Quick review

The dataset consists of:

* **Training:** 1,677 normal signals.
* **Validation:** 594 signals containing both normal and anomalous examples.
* **Signal length:** 61,440 measurements per signal (60 seconds at 1,024 Hz).

I developed a feature extraction pipeline to capture different characteristics of each signal in both the time and frequency domains.

## 2. Feature Extraction

The following feature groups were extracted:

| Category               | Features                                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------------------------- |
| Time-domain            | Mean, standard deviation, minimum, maximum, RMS, skewness, kurtosis, peak-to-peak                       |
| Signal variation       | Mean absolute difference, zero-crossing rate                                                            |
| Frequency-domain       | Dominant frequency, spectral centroid, spectral entropy, bandwidth, spectral rolloff, spectral flatness |
| Frequency bands        | Power across ten frequency bands from 0–512 Hz                                                          |
| Autocorrelation        | Lag 1 and lag 10                                                                                        |
| Signal condition flags | `is_constant`, `is_stuck`                                                                               |

**Total of 30 features for the training set.** 

The signal condition flags are heuristic indicators:

* `is_constant` identifies signals with negligible variation (std < 1e-10).
* `is_stuck` identifies signals with extremely low variation (std below 1st percentile) and an extremely high mean offset relative to the training distribution (mean above 99.9th percentile).

## 3. Exploratory Findings (Boxplots)

### Time-domain features

Lots of time domain features showed noticeable differences between normal and anomalous validation signals.

These features showed interesting distributions: rms, skewness, min, max, mean, and peak-to-peak.

### Frequency-domain features

Frequency domain features and frequency band powers were explored to capture differences in vibration patterns.

Some frequency features showed noticeable differences between the validation classes, such as the frequency bands and spectral flatness, while others showed lots of overlap. How useful they are should be determined with model evaluation.

**You can view all boxplots under findings/comparisons/**

## 4. Feature correlation

The correlation matrix was calculated only using training features. This helps us understand the relationships between the features under completely normal conditions, it gives us a baseline to identify redundancies.

Very correlated features are likely redundant, you could look into whether removing some of them affects model performance or not.

Some relationships to look out for:

* Standard deviation and RMS: approximately 1.00.
* Minimum and maximum: approximately -0.97.
* Peak-to-peak and maximum: approximately 0.99.
* Zero-crossing rate and spectral rolloff: approximately 0.87.

The `is_constant` and `is_stuck` flags were excluded because they have no variance in the training set, so their correlations were undefined.

**You can view the matrix under findings/correlations/**

## 5. Next Steps

1. Establish a baseline anomaly detection model using the extracted features, I think isolation forest is a good starting point.
2. (Optional, but really nice to have) Compare with a different approach (one class SVM, we mentioned this in project proposal).
3. Investigate feature selection and the impact of removing highly correlated features.
4. Experiment with more features or modify existing ones then evaluate.
5. Split the validation data into two halves: use one half to tune the model and the other half to test the performance. Ensure each half has similar proportions of normal and anomalous signals. The model should be trained using only the normal training signals.
6. Evaluate models using appropriate metrics (precision, recall, F1-score, PR-AUC, accuracy, and false-positive rate).

The idea is to use the current features as a starting point and let the model results guide any further improvements.