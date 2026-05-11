# Session 5

## Goal

Session 5 focuses on heart-rate variability (HRV) analysis on top of the
existing ECG/PPG pipeline.

The main objective is to move from average heart-rate estimation to beat-to-beat
variability analysis using the peaks detected in Session 3 and the quality/fusion
logic added in Session 4.

## Current Project State

The project currently contains this pipeline:

1. Load ECG, PPG, ABP and label arrays for one patient.
2. Inspect synchronized ECG/PPG segments in the time domain.
3. Filter ECG and PPG with Butterworth bandpass preprocessing.
4. Flag possible motion-artifact segments with a MAD-based detector.
5. Detect ECG R-peaks and PPG systolic peaks.
6. Calculate ECG and PPG heart rate per segment.
7. Calculate ECG and PPG signal quality scores.
8. Compute SQI-based ECG/PPG fusion weights.
9. Calculate fused heart rate and save Session 4 plots/CSV output.

## Session 5 Expected Work

Session 5 will likely add:

- Conversion from detected peaks to beat-to-beat intervals.
- HRV metrics for ECG peak intervals.
- HRV metrics for PPG peak intervals.
- A clear strategy for fused HRV, if Session 4 provides enough information.
- A reusable module/function that returns HR and HRV statistics for a selected
  patient or segment.
- Documentation of the implementation and outputs.

## Important Note About Fused HRV

Fused heart rate from Session 4 is a weighted average of two bpm values:

```text
HR_fused = w_ECG * HR_ECG + w_PPG * HR_PPG
```

That is enough for fused HR, but not automatically enough for fused HRV.

HRV needs a beat-to-beat interval series, not only one averaged bpm value.
Because of that, Session 5 should first implement ECG and PPG HRV cleanly, then
decide how fused HRV should be defined in this project.

## Work Log

### 2026-05-07

- Confirmed that the working branch is `session-5`, created from
  `origin/session-4`.
- Reviewed the full session progression from Session 1 through Session 4.
- Reviewed the current source modules:
  - `src/loader.py`
  - `src/filters.py`
  - `src/artifacts.py`
  - `src/peaks.py`
  - `src/quality.py`
  - `src/fusion.py`
  - `src/plotter.py`
  - `src/main.py`
- Confirmed the current end-to-end project flow is:
  `load -> inspect -> preprocess -> artifact check -> peak detect -> HR -> SQI -> fusion -> plots/CSV`
- Started this Session 5 tracking file so all further work can be documented in
  the same style as previous session reports.
- Added `src/hrv.py` with Session 5 HRV logic.
- Implemented peak-to-interval conversion with physiological interval filtering
  (`0.3-1.5 s`).
- Implemented time-domain HRV metrics:
  - `mean NN`
  - `SDNN`
  - `RMSSD`
  - `pNN50`
- Added `SegmentHrvResult` and `HrvMetrics` dataclasses for structured output.
- Added `analyze_hrv(...)` for all segments and `analyze_segment_hrv(...)` for
  one segment.
- Added `load_patient_hrv_statistics(...)` as a reusable Session 5 helper that
  loads patient data, preprocesses the signals and returns HR/HRV statistics.
- Added `save_hrv_results_csv(...)` to export Session 5 results.
- Integrated Session 5 into `src/main.py`:
  - calculate HRV for all segments
  - print a short HRV summary for the selected segment
  - save `outputs/session5/{patient_id}_hrv_statistics.csv`
- Implemented a practical fused-HRV strategy:
  - if ECG and PPG interval counts are nearly aligned, build a weighted fused
    interval series from the Session 4 SQI weights
  - otherwise, fall back to the higher-weight valid modality
  - store the used source in `fused_hrv_source`
- Verified Python syntax with:
  `python3 -m py_compile src/hrv.py src/main.py src/fusion.py src/quality.py src/peaks.py`
- Full runtime validation was partially blocked by the local environment:
  - system Python has a NumPy architecture mismatch
  - bundled workspace Python does not currently include `scipy`

## Next Planned Step

Run the Session 5 pipeline in a Python environment where `numpy` and `scipy`
are both available, inspect the generated CSV output, and tune the fused-HRV
strategy further if the project later requires a stricter beat-alignment method.
