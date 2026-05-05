# Session 3

## Goal

Session 3 focuses on peak detection and heart-rate comparison for ECG and PPG
signals. The implemented code now supports:

- ECG R-peak detection with a Pan-Tompkins-inspired method.
- PPG systolic peak detection with an ensemble-inspired method.
- False positive reduction with minimum distance rules and adaptive thresholds.
- Heart-rate calculation from ECG and PPG peaks.
- Difference between ECG-based heart rate and PPG-based heart rate.

## ECG R-Peak Detection

The ECG peak detection is implemented in `src/peaks.py` with
`detect_ecg_r_peaks`.

The method follows the main idea of Pan-Tompkins:

1. Filter the ECG in a QRS-focused band from `5-18 Hz`.
2. Take the derivative to highlight fast changes.
3. Square the derivative to make all values positive and emphasize large peaks.
4. Smooth the result with a moving integration window of about `150 ms`.
5. Detect candidate QRS regions with an adaptive threshold.
6. Move each candidate back to the local maximum in the ECG signal.
7. Remove peaks that are too close together.

The minimum distance between ECG R-peaks is `250 ms`. This helps reject false
positives because two real R-peaks cannot normally occur extremely close
together.

## PPG Peak Detection

The PPG peak detection is implemented in `src/peaks.py` with
`detect_ppg_peaks`.

The method is inspired by papers that combine multiple PPG peak-detection
strategies. It uses three simple detectors:

- Local maxima with adaptive prominence.
- First derivative with adaptive thresholding.
- Slope-sum function with adaptive thresholding.

The final PPG peak is accepted only when at least two methods detect a peak in
approximately the same time region. This acts like a small ensemble and reduces
false positives compared with using only one detector.

After voting, the peak location is refined to the local maximum in the original
PPG signal. Peaks that are too close together are removed with a `350 ms`
minimum distance rule.

## Heart-Rate Calculation

Heart rate is calculated with `calculate_heart_rate_from_peaks`.

The function uses the intervals between consecutive peaks:

```text
interval_seconds = difference_between_peaks / sampling_rate
heart_rate_bpm = 60 / mean_interval_seconds
```

Only physiological intervals between `0.3 s` and `1.5 s` are used. This
corresponds roughly to `40-200 bpm` and helps reduce the effect of false
positive or missed peaks.

## ECG Versus PPG Heart Rate

The function `compare_heart_rates` returns the absolute difference between ECG
heart rate and PPG heart rate:

```text
difference = abs(ECG_HR - PPG_HR)
```

For patient `p000188`, segment `23`, the current implementation finds:

```text
ECG R-peaks: 42
PPG systolic peaks: 42
ECG heart rate: about 82.97 bpm
PPG heart rate: about 82.91 bpm
Difference: about 0.07 bpm
```

## How To Run

From the project root:

```powershell
python -B src\main.py
```

The script prints the detected peak counts, ECG heart rate, PPG heart rate and
the difference between them. It also opens plots with the detected ECG and PPG
peaks marked.
