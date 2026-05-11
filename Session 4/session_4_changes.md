# Session 4

## Goal

Session 4 focuses on Signal Quality Index (SQI) calculation and ECG/PPG heart
rate fusion. The implemented code now supports:

- Quality score per segment for ECG and PPG.
- ECG SQI based on R-peak interval regularity, R-peak amplitude stability and
  estimated noise level.
- PPG SQI based on pulse interval regularity, systolic peak amplitude stability
  and estimated noise level.
- Weight ratio calculation from ECG and PPG SQI scores.
- Fused heart-rate calculation.
- CSV and plot outputs for all segments.

## Signal Quality Metrics

The SQI code is implemented in `src/quality.py`.

Each metric is normalized between `0` and `1`, where `1` means better signal
quality:

- `interval_regularity`: checks how stable the intervals between detected peaks
  are. For ECG this uses R-R intervals. For PPG this uses pulse-to-pulse
  intervals.
- `amplitude_stability`: checks how stable the detected peak amplitudes are.
  ECG uses R-peak amplitudes. PPG uses systolic peak amplitudes.
- `noise_level`: estimates high-frequency residual noise after subtracting a
  moving average from the signal.
- `overall_score`: weighted average of the individual metrics.

The ECG overall score uses these weights:

```text
45% interval regularity
35% amplitude stability
20% noise level
```

The PPG overall score uses these weights:

```text
40% interval regularity
35% amplitude stability
25% noise level
```

PPG gets slightly more noise weight because PPG waveform quality is often more
sensitive to movement and sensor contact.

## Fusion Logic

The fusion code is implemented in `src/fusion.py`.

For every segment, the pipeline:

1. Detects ECG R-peaks and PPG systolic peaks.
2. Calculates ECG HR and PPG HR.
3. Calculates ECG SQI and PPG SQI.
4. Converts SQI values into weights:

```text
w_ECG = SQI_ECG / (SQI_ECG + SQI_PPG)
w_PPG = SQI_PPG / (SQI_ECG + SQI_PPG)
```

5. Calculates fused heart rate:

```text
HR_fused = w_ECG * HR_ECG + w_PPG * HR_PPG
```

If one heart-rate value is invalid, the valid signal receives all the weight.
If both heart-rate values are invalid, the fused HR is reported as invalid.

## Outputs

Running the main script creates these Session 4 files:

- `outputs/session4/p000188_quality_fusion.csv`
- `outputs/session4/p000188_sqi_per_segment.png`
- `outputs/session4/p000188_heart_rate_fusion.png`

The CSV contains one row per segment with:

- ECG peak count
- PPG peak count
- ECG HR
- PPG HR
- ECG SQI
- PPG SQI
- ECG weight
- PPG weight
- fused HR

## Example Result

For patient `p000188`, segment `23`, the current output is:

```text
ECG SQI: 0.838
PPG SQI: 0.878
ECG weight: 0.488
PPG weight: 0.512
Fused heart rate: 82.94 bpm
```

## How To Run

From the project root:

```powershell
python -B src\main.py
```

The script prints the selected segment's SQI and fused HR, saves the full
per-segment CSV table, saves the Session 4 plots and opens the figures.

## Limitations

The SQI is heuristic and meant for project-level signal comparison, not clinical
validation. The selected metrics are interpretable and match the Session 4
requirements, but they should be tuned further if the project is evaluated on
many patients or noisy recordings.
