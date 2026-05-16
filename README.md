# ECG-PPG Fusion Backend

This backend contains the signal-processing code for the 6-DSP Roadmap v2
project. It loads synchronized ECG and PPG data, preprocesses the signals,
detects peaks, calculates HR and SQI, fuses ECG/PPG heart rate, calculates HRV,
and exports measurements for the GUI.

## Folder Setup

Expected project layout:

```text
C:\DSP_prac
|-- dsp_backend
|   |-- data
|   |-- src
|   `-- requirements.txt
`-- dsp_frontend
```

The frontend connects to this backend through
`dsp_frontend/backend_client.py`, which imports backend modules from
`dsp_backend/src`.

## Data Layout

The backend expects `.npy` files grouped by signal type:

```text
dsp_backend\data
|-- ecg
|   `-- p000188_ecg.npy
|-- ppg
|   `-- p000188_ppg.npy
|-- abp
|   `-- p000188_abp.npy
`-- labels
    `-- p000188_labels.npy
```

Patient IDs must match across folders. For example, patient `p000188` should
have ECG, PPG, ABP, and label files with the same prefix.

## Install Dependencies

From the backend folder:

```powershell
cd C:\DSP_prac\dsp_backend
python -m pip install -r requirements.txt
```

Main dependencies:

- NumPy
- SciPy
- Matplotlib

## Main Modules

- `src/loader.py`: loads ECG, PPG, ABP, and label arrays.
- `src/filters.py`: ECG/PPG Butterworth bandpass preprocessing.
- `src/artifacts.py`: MAD-based artifact segment detection.
- `src/peaks.py`: ECG R-peak and PPG systolic peak detection.
- `src/quality.py`: ECG/PPG signal quality index metrics.
- `src/fusion.py`: SQI weights and fused heart-rate calculation.
- `src/hrv.py`: ECG, PPG, and fused HRV statistics.
- `src/plotter.py`: signal, peak, SQI, HR, and frequency plots.
- `src/app_pipeline.py`: app-facing backend API used by the GUI.
- `src/main.py`: demo script for running the pipeline on `p000188`.

## Run The Backend Demo

```powershell
cd C:\DSP_prac\dsp_backend
python -B src\main.py
```

The demo loads patient `p000188`, runs preprocessing, peak detection, fusion,
and HRV analysis, then saves output tables/plots under `outputs`.

## Use From Python

The GUI uses `src/app_pipeline.py`. A minimal example:

```python
from app_pipeline import run_app_analysis, export_app_results

result = run_app_analysis("p000188", data_path="C:/DSP_prac/dsp_backend/data")
export_app_results(result, "p000188_measurements.csv")
```

`run_app_analysis(...)` returns:

- raw ECG and PPG arrays
- filtered ECG and PPG arrays
- artifact segment lists
- ECG HR, PPG HR, and fused HR
- ECG SQI and PPG SQI
- ECG/PPG fusion weights
- ECG HRV, PPG HRV, and fused HRV metrics

## Exported Measurements

The app-facing exporter writes one row per segment with:

- patient ID
- segment index
- ECG HR
- PPG HR
- fused HR
- ECG SQI
- PPG SQI
- ECG/PPG weights
- ECG HRV metrics
- PPG HRV metrics
- fused HRV metrics

These exports are intended for the final GUI and portfolio evidence.

## Notes And Limitations

- The current demo script uses `p000188`; the GUI can select from all available
  backend patient files.
- The frontend and backend must remain in the expected sibling folder layout,
  unless `backend_client.py` is updated.
