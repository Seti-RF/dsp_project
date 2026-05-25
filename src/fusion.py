from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np

from peaks import (
    calculate_heart_rate_from_peaks,
    compare_heart_rates,
    detect_ecg_r_peaks,
    detect_ppg_peaks,
)
from quality import calculate_ecg_sqi, calculate_ppg_sqi


@dataclass(frozen=True)
class SegmentFusionResult:
    """
    Session 4 quality and fused heart-rate result for one segment.
    """

    segment_index: int
    ecg_peak_count: int
    ppg_peak_count: int
    ecg_heart_rate_bpm: float
    ppg_heart_rate_bpm: float
    heart_rate_difference_bpm: float
    ecg_sqi: float
    ppg_sqi: float
    ecg_weight: float
    ppg_weight: float
    fused_heart_rate_bpm: float


def analyze_quality_and_fusion(filtered_ecg, filtered_ppg, sampling_rate_hz=125):
    """
    Calculate ECG/PPG SQI, quality weights and fused HR for every segment.
    """
    if filtered_ecg.shape != filtered_ppg.shape:
        raise ValueError("filtered_ecg and filtered_ppg must have the same shape")

    results = []
    for segment_index in range(filtered_ecg.shape[0]):
        results.append(
            analyze_segment_quality_and_fusion(
                ecg_segment=filtered_ecg[segment_index],
                ppg_segment=filtered_ppg[segment_index],
                segment_index=segment_index,
                sampling_rate_hz=sampling_rate_hz,
            )
        )

    return results


def analyze_segment_quality_and_fusion(
    ecg_segment,
    ppg_segment,
    segment_index=0,
    sampling_rate_hz=125,
):
    """
    Calculate Session 4 metrics for one ECG/PPG segment pair.
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
    fused_heart_rate = calculate_fused_heart_rate(
        ecg_heart_rate_bpm=ecg_heart_rate.beats_per_minute,
        ppg_heart_rate_bpm=ppg_heart_rate.beats_per_minute,
        ecg_weight=ecg_weight,
        ppg_weight=ppg_weight,
    )

    return SegmentFusionResult(
        segment_index=segment_index,
        ecg_peak_count=ecg_heart_rate.peak_count,
        ppg_peak_count=ppg_heart_rate.peak_count,
        ecg_heart_rate_bpm=ecg_heart_rate.beats_per_minute,
        ppg_heart_rate_bpm=ppg_heart_rate.beats_per_minute,
        heart_rate_difference_bpm=compare_heart_rates(ecg_heart_rate, ppg_heart_rate),
        ecg_sqi=ecg_sqi.overall_score,
        ppg_sqi=ppg_sqi.overall_score,
        ecg_weight=ecg_weight,
        ppg_weight=ppg_weight,
        fused_heart_rate_bpm=fused_heart_rate,
    )


def calculate_sqi_weights(
    ecg_sqi,
    ppg_sqi,
    ecg_heart_rate_bpm,
    ppg_heart_rate_bpm,
):
    """
    Calculate ECG and PPG fusion weights from SQI scores.

    If only one HR is valid, that signal receives all weight. If both HR values
    are valid but both SQI scores are zero, equal weights are used.
    """
    ecg_valid = np.isfinite(ecg_heart_rate_bpm)
    ppg_valid = np.isfinite(ppg_heart_rate_bpm)

    if ecg_valid and not ppg_valid:
        return 1.0, 0.0
    if ppg_valid and not ecg_valid:
        return 0.0, 1.0
    if not ecg_valid and not ppg_valid:
        return 0.0, 0.0

    ecg_score = max(0.0, float(ecg_sqi))
    ppg_score = max(0.0, float(ppg_sqi))
    score_sum = ecg_score + ppg_score

    if score_sum == 0:
        return 0.5, 0.5

    return ecg_score / score_sum, ppg_score / score_sum


def calculate_fused_heart_rate(
    ecg_heart_rate_bpm,
    ppg_heart_rate_bpm,
    ecg_weight,
    ppg_weight,
):
    """
    Calculate HR_fused = w_ECG * HR_ECG + w_PPG * HR_PPG.
    """
    if ecg_weight == 0 and ppg_weight == 0:
        return float("nan")

    ecg_value = 0.0 if not np.isfinite(ecg_heart_rate_bpm) else ecg_heart_rate_bpm
    ppg_value = 0.0 if not np.isfinite(ppg_heart_rate_bpm) else ppg_heart_rate_bpm

    return float(ecg_weight * ecg_value + ppg_weight * ppg_value)


def save_fusion_results_csv(results, output_path):
    """
    Save Session 4 per-segment quality and fused HR values to CSV.
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
                "heart_rate_difference_bpm",
                "ecg_sqi",
                "ppg_sqi",
                "ecg_weight",
                "ppg_weight",
                "fused_heart_rate_bpm",
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
                    _format_float(result.heart_rate_difference_bpm),
                    _format_float(result.ecg_sqi),
                    _format_float(result.ppg_sqi),
                    _format_float(result.ecg_weight),
                    _format_float(result.ppg_weight),
                    _format_float(result.fused_heart_rate_bpm),
                ]
            )


def _format_float(value):
    if not np.isfinite(value):
        return ""

    return f"{value:.4f}"
