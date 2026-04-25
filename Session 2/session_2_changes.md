# Session 2

## Goal

Session 2 focuses on preprocessing and noise reduction for the ECG and PPG
signals. The implemented code now supports:

- Butterworth bandpass filtering in literature-based bandwidths.
- Baseline wander removal through the high-pass part of the bandpass filter.
- Frequency-domain inspection with a Fourier transform.
- Visual comparison of raw and filtered signals.
- Motion artifact detection with a MAD-based robust z-score.

## Current Signal Shapes

For patient `p000188`, the loaded signal arrays have shape `(30, 3750)`.

This means:

- 30 signal segments.
- 3750 samples per segment.
- At `125 Hz`, each segment is `3750 / 125 = 30` seconds long.

## Butterworth Bandpass Filtering

The filter code is implemented in `src/filters.py`.

The project uses a fourth-order Butterworth bandpass filter:

- ECG: `0.5-20 Hz`
- PPG: `0.5-5 Hz`
- Sampling rate: `125 Hz`
- Filter order: `4`

Butterworth was chosen because it has a flat passband, meaning it does not add
ripple inside the frequency range we want to keep. This is a conservative
choice for biomedical signals where waveform shape matters.

The implementation uses SciPy second-order sections:

```python
signal.butter(..., output="sos")
signal.sosfiltfilt(...)
```

`sosfiltfilt` applies the filter forward and backward. This gives zero-phase
filtering, so the ECG and PPG peaks are not shifted in time by the filter.

## Baseline Wander Removal

Baseline wander is slow movement of the signal around the vertical axis. It is
mainly low-frequency noise.

The lower cutoff of the bandpass filter removes this:

- ECG frequencies below `0.5 Hz` are attenuated.
- PPG frequencies below `0.5 Hz` are attenuated.

This means a separate baseline correction step is not needed for the current
pipeline.

## Powerline Interference Check

Powerline interference appears as a narrow noise peak around `50 Hz` in Europe
or `60 Hz` in some other regions.

The code adds frequency-domain plots with `FrequencyDomainPlotter` in
`src/plotter.py`. It computes a one-sided Fourier spectrum with:

```python
np.fft.rfft(...)
np.fft.rfftfreq(...)
```

The plots show frequencies from `0 Hz` up to `62.5 Hz`, because the sampling
rate is `125 Hz` and the Nyquist frequency is `125 / 2 = 62.5 Hz`.

The filtered signal should strongly reduce 50 Hz or 60 Hz noise because the ECG
filter keeps only up to `20 Hz` and the PPG filter keeps only up to `5 Hz`.

## Time-Domain Plots

`src/main.py` currently creates:

- Raw synchronized ECG/PPG time-domain plot.
- Filtered synchronized ECG/PPG time-domain plot.

The filtered time-domain plot uses the same plotting function as the raw plot,
but passes `filtered_ecg` and `filtered_ppg` instead of the raw arrays.

## Frequency-Domain Plots

`src/main.py` also creates:

- Raw ECG frequency spectrum.
- Raw PPG frequency spectrum.
- Filtered ECG frequency spectrum.
- Filtered PPG frequency spectrum.

These plots are useful for checking which frequencies are present before and
after filtering. They also support the Session 2 requirement to inspect whether
there is a powerline noise peak around `50 Hz` or `60 Hz`.

## Motion Artifact Detection

Motion artifacts are handled with detection, not correction. The implementation
is in `src/artifacts.py`.

The reason for this choice is that motion artifacts can distort the waveform so
strongly that trying to "repair" the signal may create fake physiological
features. For this project, it is safer to mark suspicious segments.

The algorithm uses a MAD-based robust z-score:

```python
centered = segment - np.median(segment)
mad = np.median(np.abs(centered))
robust_z = centered / (1.4826 * mad)
```

MAD means median absolute deviation. It is more robust than standard deviation
because a few large motion spikes do not dominate the estimate.

The current detection settings are:

- `z_threshold = 20.0`
- `max_spike_ratio = 0.01`

For a segment of 3750 samples, `0.01` means that more than about 37 extreme
samples will mark the segment as suspicious. A threshold of `20.0` is used
because normal ECG R-peaks are sharp and can look like outliers if the threshold
is too low. The value can still be changed when calling the function.

The main functions are:

- `robust_z_score(segment)`
- `has_motion_artifact(segment, z_threshold=20.0, max_spike_ratio=0.01)`
- `find_artifact_segments(signal_data, z_threshold=20.0, max_spike_ratio=0.01)`

`src/main.py` prints the ECG and PPG segment indexes that are flagged as likely
motion artifacts.

## How To Run

From the project root:

```powershell
python -B src\main.py
```

The script prints the signal shapes, label for the selected segment, and the
artifact-heavy segment indexes. It then opens the time-domain and
frequency-domain plots.
