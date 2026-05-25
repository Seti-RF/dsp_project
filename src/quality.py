from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignalQualityMetrics:
    """
    SQI metrics for one signal segment.

    Scores are normalized between 0 and 1, where 1 means better quality.
    """

    interval_regularity: float
    amplitude_stability: float
    noise_level: float
    overall_score: float


def calculate_ecg_sqi(ecg_segment, r_peaks, sampling_rate_hz=125):
    """
    Calculate ECG signal quality for one segment.

    ECG quality is based on:
    - R-R interval regularity.
    - R-peak amplitude stability.
    - Noise level compared with the signal amplitude.

    NOTE: THESE MEASURES WERE FOUND ONLINE BUT WORK PRETTY WELL
    """
    return _calculate_peak_based_sqi(
        signal_segment=ecg_segment,
        peaks=r_peaks,
        sampling_rate_hz=sampling_rate_hz,
        metric_weights=(0.45, 0.35, 0.20),
    )


def calculate_ppg_sqi(ppg_segment, systolic_peaks, sampling_rate_hz=125):
    """
    Calculate PPG signal quality for one segment.

    PPG quality is based on:
    - Pulse interval regularity.
    - Systolic peak amplitude stability.
    - Noise level compared with the pulse waveform amplitude.

    NOTE: THESE MEASURES WERE FOUND ONLINE BUT WORK PRETTY WELL
    """
    return _calculate_peak_based_sqi(
        signal_segment=ppg_segment,
        peaks=systolic_peaks,
        sampling_rate_hz=sampling_rate_hz,
        metric_weights=(0.40, 0.35, 0.25),
    )


def interval_regularity_score(peaks, sampling_rate_hz=125):
    """
    Score how regular the beat-to-beat intervals are.

    The score uses the coefficient of variation: std(intervals) / mean(intervals).
    A stable rhythm gives a score near 1. Strongly irregular intervals reduce it.
    """
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size < 3:
        return 0.0

    intervals_seconds = np.diff(peaks) / sampling_rate_hz
    intervals_seconds = intervals_seconds[
        (intervals_seconds >= 0.3) & (intervals_seconds <= 1.5)
    ]
    if intervals_seconds.size < 2:
        return 0.0

    coefficient_of_variation = _safe_ratio(
        np.std(intervals_seconds),
        np.mean(intervals_seconds),
    )
    return _score_from_coefficient_of_variation(coefficient_of_variation)


def amplitude_stability_score(signal_segment, peaks):
    """
    Score how stable the detected peak amplitudes are.

    A clean segment should have peak amplitudes that are reasonably consistent.
    Large amplitude variation lowers the score.
    """
    signal_segment = np.asarray(signal_segment, dtype=float)
    peaks = _valid_peak_indexes(peaks, signal_segment.shape[0])
    if peaks.size < 3:
        return 0.0

    peak_amplitudes = signal_segment[peaks]
    coefficient_of_variation = _safe_ratio(
        np.std(peak_amplitudes),
        np.abs(np.mean(peak_amplitudes)),
    )
    return _score_from_coefficient_of_variation(coefficient_of_variation)


def noise_level_score(signal_segment):
    """
    Estimate noise using the residual after subtracting a moving average.

    This is a simple heuristic: if the high-frequency residual is small compared
    with the full signal range, quality is higher.
    """
    signal_segment = np.asarray(signal_segment, dtype=float)
    if signal_segment.size < 3:
        return 0.0

    window_size = max(3, int(0.20 * signal_segment.size / 30))
    if window_size % 2 == 0:
        window_size += 1

    smoothed = _moving_average(signal_segment, window_size)
    residual = signal_segment - smoothed
    residual_ratio = _safe_ratio(
        np.std(residual),
        np.ptp(signal_segment),
    )

    return float(np.clip(1.0 - residual_ratio * 5.0, 0.0, 1.0))


def _calculate_peak_based_sqi(
    signal_segment,
    peaks,
    sampling_rate_hz,
    metric_weights,
):
    """
    Combine interval, amplitude and noise scores into one SQI value.

    ECG and PPG use the same three ingredients but different weights. This keeps
    the code consistent while allowing PPG to penalize noise slightly more.
    """
    interval_score = interval_regularity_score(peaks, sampling_rate_hz)
    amplitude_score = amplitude_stability_score(signal_segment, peaks)
    noise_score = noise_level_score(signal_segment)

    overall_score = float(
        np.average(
            [interval_score, amplitude_score, noise_score],
            weights=metric_weights,
        )
    )

    return SignalQualityMetrics(
        interval_regularity=interval_score,
        amplitude_stability=amplitude_score,
        noise_level=noise_score,
        overall_score=overall_score,
    )


def _score_from_coefficient_of_variation(coefficient_of_variation):
    """
    Convert relative variation into a bounded 0-1 quality score.

    A coefficient of variation near zero means the values are stable, so the
    score is near 1. Larger variation moves the score toward 0.
    """
    if not np.isfinite(coefficient_of_variation):
        return 0.0

    return float(np.clip(1.0 - coefficient_of_variation, 0.0, 1.0))


def _moving_average(signal_segment, window_size):
    padding = window_size // 2
    padded = np.pad(signal_segment, padding, mode="edge")
    window = np.ones(window_size) / window_size
    return np.convolve(padded, window, mode="valid")


def _safe_ratio(numerator, denominator):
    if denominator == 0 or not np.isfinite(denominator):
        return float("inf")

    return float(numerator / denominator)


def _valid_peak_indexes(peaks, signal_length):
    peaks = np.asarray(peaks, dtype=int)
    return peaks[(peaks >= 0) & (peaks < signal_length)]
