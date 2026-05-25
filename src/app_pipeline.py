from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np

from artifacts import find_artifact_segments
from filters import preprocess_ecg, preprocess_ppg
from fusion import analyze_quality_and_fusion
from hrv import analyze_hrv
from loader import load_patient


@dataclass(frozen=True)
class AppSegmentResult:
    """
    One segment's final Roadmap v2 measurements for the GUI.
    """

    segment_index: int
    fusion: object
    hrv: object


@dataclass(frozen=True)
class AppPatientResult:
    """
    Full ECG/PPG fusion result for one patient.
    """

    patient_id: str
    sampling_rate_hz: float
    ecg_shape: tuple
    ppg_shape: tuple
    bad_ecg_segments: list
    bad_ppg_segments: list
    raw_ecg: np.ndarray
    raw_ppg: np.ndarray
    filtered_ecg: np.ndarray
    filtered_ppg: np.ndarray
    segments: list


def list_patient_ids(data_path=None):
    """
    List patient ids that have both ECG and PPG files.
    """
    data_dir = Path(data_path) if data_path is not None else _project_root() / "data"
    ecg_ids = _ids_from_folder(data_dir / "ecg", "_ecg.npy")
    ppg_ids = _ids_from_folder(data_dir / "ppg", "_ppg.npy")
    return sorted(ecg_ids & ppg_ids)


def run_app_analysis(patient_id, data_path=None, sampling_rate_hz=125):
    """
    Run the final Roadmap v2 backend pipeline for one patient.

    The app needs one stable function that performs all analysis steps:
    load data, preprocess, calculate HR/SQI/fusion and calculate HRV.
    """
    ecg, ppg, _, _ = load_patient(patient_id, data_path=data_path)
    return run_app_analysis_from_arrays(
        patient_id=patient_id,
        ecg=ecg,
        ppg=ppg,
        sampling_rate_hz=sampling_rate_hz,
    )


def run_app_analysis_from_files(ecg_path, ppg_path, sampling_rate_hz=125):
    """
    Run the app pipeline from two user-selected ECG and PPG .npy files.
    """
    ecg_file = Path(ecg_path)
    ppg_file = Path(ppg_path)
    ecg = np.load(ecg_file)
    ppg = np.load(ppg_file)
    patient_id = f"{ecg_file.stem} / {ppg_file.stem}"
    return run_app_analysis_from_arrays(
        patient_id=patient_id,
        ecg=ecg,
        ppg=ppg,
        sampling_rate_hz=sampling_rate_hz,
    )


def run_app_analysis_from_arrays(patient_id, ecg, ppg, sampling_rate_hz=125):
    """
    Run the final Roadmap v2 backend pipeline for loaded ECG and PPG arrays.
    """
    ecg = _normalize_signal_array(ecg, "ECG")
    ppg = _normalize_signal_array(ppg, "PPG")

    if ecg.shape != ppg.shape:
        raise ValueError(
            f"ECG and PPG arrays must have the same shape. "
            f"Got ECG {ecg.shape} and PPG {ppg.shape}."
        )

    filtered_ecg = preprocess_ecg(ecg, sampling_rate_hz=sampling_rate_hz)
    filtered_ppg = preprocess_ppg(ppg, sampling_rate_hz=sampling_rate_hz)

    fusion_results = analyze_quality_and_fusion(
        filtered_ecg=filtered_ecg,
        filtered_ppg=filtered_ppg,
        sampling_rate_hz=sampling_rate_hz,
    )
    hrv_results = analyze_hrv(
        filtered_ecg=filtered_ecg,
        filtered_ppg=filtered_ppg,
        sampling_rate_hz=sampling_rate_hz,
    )

    segments = [
        AppSegmentResult(
            segment_index=index,
            fusion=fusion_results[index],
            hrv=hrv_results[index],
        )
        for index in range(len(fusion_results))
    ]

    return AppPatientResult(
        patient_id=patient_id,
        sampling_rate_hz=float(sampling_rate_hz),
        ecg_shape=ecg.shape,
        ppg_shape=ppg.shape,
        bad_ecg_segments=find_artifact_segments(ecg),
        bad_ppg_segments=find_artifact_segments(ppg),
        raw_ecg=ecg,
        raw_ppg=ppg,
        filtered_ecg=filtered_ecg,
        filtered_ppg=filtered_ppg,
        segments=segments,
    )


def export_app_results(result, output_path):
    """
    Export all final app measurements to one CSV file.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "patient_id",
                "segment_index",
                "ecg_hr_bpm",
                "ppg_hr_bpm",
                "fused_hr_bpm",
                "ecg_sqi",
                "ppg_sqi",
                "ecg_weight",
                "ppg_weight",
                "ecg_mean_nn_seconds",
                "ecg_sdnn_seconds",
                "ecg_rmssd_seconds",
                "ecg_pnn50_percent",
                "ppg_mean_nn_seconds",
                "ppg_sdnn_seconds",
                "ppg_rmssd_seconds",
                "ppg_pnn50_percent",
                "fused_hrv_source",
                "fused_mean_nn_seconds",
                "fused_sdnn_seconds",
                "fused_rmssd_seconds",
                "fused_pnn50_percent",
            ]
        )

        for segment in result.segments:
            writer.writerow(
                [
                    result.patient_id,
                    segment.segment_index,
                    _format_float(segment.fusion.ecg_heart_rate_bpm),
                    _format_float(segment.fusion.ppg_heart_rate_bpm),
                    _format_float(segment.fusion.fused_heart_rate_bpm),
                    _format_float(segment.fusion.ecg_sqi),
                    _format_float(segment.fusion.ppg_sqi),
                    _format_float(segment.fusion.ecg_weight),
                    _format_float(segment.fusion.ppg_weight),
                    _format_float(segment.hrv.ecg_hrv.mean_nn_seconds),
                    _format_float(segment.hrv.ecg_hrv.sdnn_seconds),
                    _format_float(segment.hrv.ecg_hrv.rmssd_seconds),
                    _format_float(segment.hrv.ecg_hrv.pnn50_percent),
                    _format_float(segment.hrv.ppg_hrv.mean_nn_seconds),
                    _format_float(segment.hrv.ppg_hrv.sdnn_seconds),
                    _format_float(segment.hrv.ppg_hrv.rmssd_seconds),
                    _format_float(segment.hrv.ppg_hrv.pnn50_percent),
                    segment.hrv.fused_hrv_source,
                    _format_float(segment.hrv.fused_hrv.mean_nn_seconds),
                    _format_float(segment.hrv.fused_hrv.sdnn_seconds),
                    _format_float(segment.hrv.fused_hrv.rmssd_seconds),
                    _format_float(segment.hrv.fused_hrv.pnn50_percent),
                ]
            )


def _ids_from_folder(folder, suffix):
    if not folder.exists():
        return set()
    return {path.name.removesuffix(suffix) for path in folder.glob(f"*{suffix}")}


def _project_root():
    return Path(__file__).resolve().parents[1]


def _normalize_signal_array(signal, signal_name):
    signal = np.asarray(signal)
    if signal.ndim == 1:
        return signal.reshape(1, -1)
    if signal.ndim != 2:
        raise ValueError(
            f"{signal_name} array must be 1D or 2D. Got shape {signal.shape}."
        )
    return signal


def _format_float(value):
    if not np.isfinite(value):
        return ""
    return f"{value:.4f}"
