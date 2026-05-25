from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np

from filters import preprocess_ecg, preprocess_ppg
from fusion import calculate_fused_heart_rate, calculate_sqi_weights
from loader import load_patient
from peaks import (
    calculate_heart_rate_from_peaks,
    detect_ecg_r_peaks,
    detect_ppg_peaks,
)
from quality import calculate_ecg_sqi, calculate_ppg_sqi


@dataclass(frozen=True)
class HrvMetrics:
    """
    Time-domain HRV metrics for one interval series.
    """

    interval_count: int
    mean_nn_seconds: float
    sdnn_seconds: float
    rmssd_seconds: float
    pnn50_percent: float


@dataclass(frozen=True)
class SegmentHrvResult:


    segment_index: int
    ecg_peak_count: int
    ppg_peak_count: int
    ecg_heart_rate_bpm: float
    ppg_heart_rate_bpm: float
    fused_heart_rate_bpm: float
    ecg_sqi: float
    ppg_sqi: float
    fused_hrv_source: str
    ecg_hrv: HrvMetrics
    ppg_hrv: HrvMetrics
    fused_hrv: HrvMetrics


def peaks_to_intervals_seconds(
    peaks,
    sampling_rate_hz=125,
    min_interval_seconds=0.3,
    max_interval_seconds=1.5,
):
    """
    Convert peak indexes into physiological beat-to-beat intervals in seconds.
    """
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size < 2:
        return np.array([], dtype=float)

    # Consecutive peak distances in samples become seconds by dividing by the
    # sampling frequency.
    intervals_seconds = np.diff(peaks) / float(sampling_rate_hz)

    # Remove intervals outside a broad physiological range. This reduces the
    # effect of missed peaks or false double detections.
    valid_mask = (
        np.isfinite(intervals_seconds)
        & (intervals_seconds >= min_interval_seconds)
        & (intervals_seconds <= max_interval_seconds)
    )
    return intervals_seconds[valid_mask]


def calculate_hrv_metrics(intervals_seconds):
    """
    Calculate basic time-domain HRV metrics from one interval series.
    """
    intervals_seconds = np.asarray(intervals_seconds, dtype=float)

    # HRV formulas only make sense for positive finite beat intervals.
    intervals_seconds = intervals_seconds[
        np.isfinite(intervals_seconds) & (intervals_seconds > 0.0)
    ]

    if intervals_seconds.size == 0:
        return HrvMetrics(
            interval_count=0,
            mean_nn_seconds=float("nan"),
            sdnn_seconds=float("nan"),
            rmssd_seconds=float("nan"),
            pnn50_percent=float("nan"),
        )

    successive_differences = np.diff(intervals_seconds)

    sdnn_seconds = float("nan")
    if intervals_seconds.size >= 2:
        # SDNN is the standard deviation of normal-to-normal intervals.
        sdnn_seconds = float(np.std(intervals_seconds, ddof=1))

    rmssd_seconds = float("nan")
    pnn50_percent = float("nan")
    if successive_differences.size >= 1:
        # RMSSD emphasizes short-term beat-to-beat variability.
        squared_differences = successive_differences**2
        rmssd_seconds = float(np.sqrt(np.mean(squared_differences)))

        # pNN50 counts how often adjacent intervals differ by more than 50 ms.
        pnn50_percent = float(
            100.0 * np.mean(np.abs(successive_differences) > 0.05)
        )

    return HrvMetrics(
        interval_count=int(intervals_seconds.size),
        mean_nn_seconds=float(np.mean(intervals_seconds)),
        sdnn_seconds=sdnn_seconds,
        rmssd_seconds=rmssd_seconds,
        pnn50_percent=pnn50_percent,
    )


def analyze_hrv(filtered_ecg, filtered_ppg, sampling_rate_hz=125):
    """
    Calculate HR and HRV results for all segments.
    """
    if filtered_ecg.shape != filtered_ppg.shape:
        raise ValueError("filtered_ecg and filtered_ppg must have the same shape")

    results = []
    for segment_index in range(filtered_ecg.shape[0]):
        results.append(
            analyze_segment_hrv(
                ecg_segment=filtered_ecg[segment_index],
                ppg_segment=filtered_ppg[segment_index],
                segment_index=segment_index,
                sampling_rate_hz=sampling_rate_hz,
            )
        )

    return results


def analyze_segment_hrv(
    ecg_segment,
    ppg_segment,
    segment_index=0,
    sampling_rate_hz=125,
):
    """
    Calculate HR and HRV results for one segment pair.
    """
    ecg_peaks = detect_ecg_r_peaks(ecg_segment, sampling_rate_hz=sampling_rate_hz)
    ppg_peaks = detect_ppg_peaks(ppg_segment, sampling_rate_hz=sampling_rate_hz)

    ecg_heart_rate = calculate_heart_rate_from_peaks(
        ecg_peaks,
        sampling_rate_hz=sampling_rate_hz,
    )
    ppg_heart_rate = calculate_heart_rate_from_peaks(
        ppg_peaks,
        sampling_rate_hz=sampling_rate_hz,
    )

    ecg_sqi = calculate_ecg_sqi(
        ecg_segment,
        ecg_peaks,
        sampling_rate_hz=sampling_rate_hz,
    )
    ppg_sqi = calculate_ppg_sqi(
        ppg_segment,
        ppg_peaks,
        sampling_rate_hz=sampling_rate_hz,
    )

    ecg_weight, ppg_weight = calculate_sqi_weights(
        ecg_sqi=ecg_sqi.overall_score,
        ppg_sqi=ppg_sqi.overall_score,
        ecg_heart_rate_bpm=ecg_heart_rate.beats_per_minute,
        ppg_heart_rate_bpm=ppg_heart_rate.beats_per_minute,
    )
    fused_heart_rate_bpm = calculate_fused_heart_rate(
        ecg_heart_rate_bpm=ecg_heart_rate.beats_per_minute,
        ppg_heart_rate_bpm=ppg_heart_rate.beats_per_minute,
        ecg_weight=ecg_weight,
        ppg_weight=ppg_weight,
    )

    ecg_intervals_seconds = peaks_to_intervals_seconds(
        ecg_peaks,
        sampling_rate_hz=sampling_rate_hz,
    )
    ppg_intervals_seconds = peaks_to_intervals_seconds(
        ppg_peaks,
        sampling_rate_hz=sampling_rate_hz,
    )

    ecg_hrv = calculate_hrv_metrics(ecg_intervals_seconds)
    ppg_hrv = calculate_hrv_metrics(ppg_intervals_seconds)
    fused_intervals_seconds, fused_source = calculate_fused_intervals(
        ecg_intervals_seconds=ecg_intervals_seconds,
        ppg_intervals_seconds=ppg_intervals_seconds,
        ecg_weight=ecg_weight,
        ppg_weight=ppg_weight,
    )
    fused_hrv = calculate_hrv_metrics(fused_intervals_seconds)

    return SegmentHrvResult(
        segment_index=segment_index,
        ecg_peak_count=ecg_heart_rate.peak_count,
        ppg_peak_count=ppg_heart_rate.peak_count,
        ecg_heart_rate_bpm=ecg_heart_rate.beats_per_minute,
        ppg_heart_rate_bpm=ppg_heart_rate.beats_per_minute,
        fused_heart_rate_bpm=fused_heart_rate_bpm,
        ecg_sqi=ecg_sqi.overall_score,
        ppg_sqi=ppg_sqi.overall_score,
        fused_hrv_source=fused_source,
        ecg_hrv=ecg_hrv,
        ppg_hrv=ppg_hrv,
        fused_hrv=fused_hrv,
    )


def calculate_fused_intervals(
    ecg_intervals_seconds,
    ppg_intervals_seconds,
    ecg_weight,
    ppg_weight,
):
    """
    Build a fused interval series for HRV.

    If ECG and PPG interval counts are nearly aligned, a weighted interval series
    is created. Otherwise the higher-weight valid modality is used as fallback.
    """
    ecg_intervals_seconds = np.asarray(ecg_intervals_seconds, dtype=float)
    ppg_intervals_seconds = np.asarray(ppg_intervals_seconds, dtype=float)

    ecg_valid = ecg_intervals_seconds.size > 0
    ppg_valid = ppg_intervals_seconds.size > 0

    # Weighted fusion only makes sense when both series have nearly the same
    # number of beat intervals. Otherwise the beats are probably not aligned.
    if ecg_valid and ppg_valid and abs(ecg_intervals_seconds.size - ppg_intervals_seconds.size) <= 1:
        aligned_count = min(ecg_intervals_seconds.size, ppg_intervals_seconds.size)
        fused_intervals_seconds = (
            ecg_weight * ecg_intervals_seconds[:aligned_count]
            + ppg_weight * ppg_intervals_seconds[:aligned_count]
        )
        return fused_intervals_seconds, "weighted_ecg_ppg"

    if ecg_valid and (not ppg_valid or ecg_weight >= ppg_weight):
        return ecg_intervals_seconds, "ecg_only"

    if ppg_valid:
        return ppg_intervals_seconds, "ppg_only"

    return np.array([], dtype=float), "none"


def load_patient_hrv_statistics(
    patient_id,
    data_path=None,
    sampling_rate_hz=125,
):
    """
    Load one patient, preprocess the signals and return Session 5 statistics.
    """
    ecg, ppg, _, _ = load_patient(patient_id, data_path=data_path)
    filtered_ecg = preprocess_ecg(ecg, sampling_rate_hz=sampling_rate_hz)
    filtered_ppg = preprocess_ppg(ppg, sampling_rate_hz=sampling_rate_hz)
    return analyze_hrv(
        filtered_ecg=filtered_ecg,
        filtered_ppg=filtered_ppg,
        sampling_rate_hz=sampling_rate_hz,
    )


def save_hrv_results_csv(results, output_path):
    """
    Save Session 5 HR and HRV results per segment to CSV.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "segment_index",
                "ecg_peak_count",
                "ppg_peak_count",
                "ecg_heart_rate_bpm",
                "ppg_heart_rate_bpm",
                "fused_heart_rate_bpm",
                "ecg_sqi",
                "ppg_sqi",
                "fused_hrv_source",
                "ecg_interval_count",
                "ecg_mean_nn_seconds",
                "ecg_sdnn_seconds",
                "ecg_rmssd_seconds",
                "ecg_pnn50_percent",
                "ppg_interval_count",
                "ppg_mean_nn_seconds",
                "ppg_sdnn_seconds",
                "ppg_rmssd_seconds",
                "ppg_pnn50_percent",
                "fused_interval_count",
                "fused_mean_nn_seconds",
                "fused_sdnn_seconds",
                "fused_rmssd_seconds",
                "fused_pnn50_percent",
            ]
        )

        for result in results:
            writer.writerow(
                [
                    result.segment_index,
                    result.ecg_peak_count,
                    result.ppg_peak_count,
                    _format_float(result.ecg_heart_rate_bpm),
                    _format_float(result.ppg_heart_rate_bpm),
                    _format_float(result.fused_heart_rate_bpm),
                    _format_float(result.ecg_sqi),
                    _format_float(result.ppg_sqi),
                    result.fused_hrv_source,
                    result.ecg_hrv.interval_count,
                    _format_float(result.ecg_hrv.mean_nn_seconds),
                    _format_float(result.ecg_hrv.sdnn_seconds),
                    _format_float(result.ecg_hrv.rmssd_seconds),
                    _format_float(result.ecg_hrv.pnn50_percent),
                    result.ppg_hrv.interval_count,
                    _format_float(result.ppg_hrv.mean_nn_seconds),
                    _format_float(result.ppg_hrv.sdnn_seconds),
                    _format_float(result.ppg_hrv.rmssd_seconds),
                    _format_float(result.ppg_hrv.pnn50_percent),
                    result.fused_hrv.interval_count,
                    _format_float(result.fused_hrv.mean_nn_seconds),
                    _format_float(result.fused_hrv.sdnn_seconds),
                    _format_float(result.fused_hrv.rmssd_seconds),
                    _format_float(result.fused_hrv.pnn50_percent),
                ]
            )


def _format_float(value):
    """
    Keep CSV output readable by writing invalid numeric values as empty cells.
    """
    if not np.isfinite(value):
        return ""

    return f"{value:.4f}"
