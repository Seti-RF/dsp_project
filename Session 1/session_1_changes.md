# Session 1

## What Was Not Done Yet

The project could already load ECG, PPG, ABP and label arrays for one patient.
The missing deliverable from Session 1 was the synchronized ECG and PPG plot.

The slide also mentions manual identification of R-peaks in ECG and systolic
peaks in PPG. This implementation supports that by plotting both signals on the
same time axis. It does not add an automatic peak detector yet.

## Changes Made

- Added `src/plotter.py` with `plot_synchronized_ecg_ppg`.
- Updated `src/main.py` so it loads one patient, prints the data shapes and
  shows the synchronized ECG/PPG plot in a popup window.
- Improved `src/loader.py` documentation with JavaDoc-style Python docstrings.
- Added `matplotlib==3.9.2` to `requirements.txt` because plotting depends on it.
- Plot saving is currently paused. Running the script opens the plot window with
  `plt.show()` instead of writing a PNG to `outputs/`.

## How To Run

From the project root:

```powershell
python -B src\main.py
```

Expected console output:

```text
ECG vorm: (30, 3750)
PPG vorm: (30, 3750)
ABP vorm: (30, 3750)
Labels vorm: (30, 2)
Eerste label [SBP, DBP]: [...]
```

## Code Explanation

### `src/loader.py`

`PROJECT_ROOT = Path(__file__).resolve().parents[1]` finds the root of the
project from the location of `loader.py`. This makes loading data independent
from the folder where the script is started.

`load_patient(patient_id, data_path=None)` loads four NumPy arrays:

- ECG from `data/ecg/{patient_id}_ecg.npy`
- PPG from `data/ppg/{patient_id}_ppg.npy`
- ABP from `data/abp/{patient_id}_abp.npy`
- Labels from `data/labels/{patient_id}_labels.npy`

For patient `p000188`, each signal has shape `(30, 3750)`. This means there are
30 signal segments, and every segment contains 3750 samples. The labels have
shape `(30, 2)`, one `[SBP, DBP]` pair per segment.

### `src/plotter.py`

`plot_synchronized_ecg_ppg(...)` receives ECG and PPG arrays and chooses one
segment with `segment_index`.

The time axis is created with:

```python
time_seconds = np.arange(ecg_segment.shape[0]) / sampling_rate_hz
```

With the default sampling rate of 125 Hz, 3750 samples become 30 seconds. ECG
and PPG are plotted in two stacked subplots with `sharex=True`, so both signals
use exactly the same time axis.

The first subplot shows ECG. This is where R-peaks can be manually inspected.
The second subplot shows PPG. This is where systolic peaks can be manually
inspected.

If `output_path` is provided, the function can still save the plot as a PNG
image. `main.py` does not pass `output_path`, so the Session 1 workflow shows
the plot in a popup window instead.

### `src/main.py`

`main.py` defines the Session 1 example settings:

- `PATIENT_ID = "p000188"`
- `SEGMENT_INDEX = 0`
- `SAMPLING_RATE_HZ = 125`
When the script runs, it loads the patient, prints the array shapes, prints the
first label pair, calls `plot_synchronized_ecg_ppg` to create the Session 1
plot, and then calls `plt.show()` to display it in a popup window.
