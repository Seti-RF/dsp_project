from dataclasses import dataclass

import numpy as np
from scipy import signal

from filters import ButterworthBandpassSettings, butterworth_bandpass


@dataclass(frozen=True)
class HeartRateResult:
    """
    Heart-rate calculation result for one segment.
    """

    beats_per_minute: float
    peak_count: int
    mean_interval_seconds: float


def detect_ecg_r_peaks(ecg_segment, sampling_rate_hz=125):
    """
    Detect ECG R-peaks with a Pan-Tompkins-inspired pipeline.

    The signal is filtered, differentiated, squared and smoothed with a moving
    integration window. Candidate QRS regions are detected on the integrated
    signal, then the final R-peak is placed on the local maximum in the ECG.

    Args:
        ecg_segment: One ECG segment with shape (samples,).
        sampling_rate_hz: Sampling rate in Hz.

    Returns:
        NumPy array with R-peak sample indexes.
    """
    ecg_segment = np.asarray(ecg_segment, dtype=float) # # input to numpy array
    qrs_signal = _prepare_ecg_qrs_signal(ecg_segment, sampling_rate_hz) # Pan-Tompkins

    threshold = np.mean(qrs_signal) + 0.5 * np.std(qrs_signal) # Dynamic tresh
    minimum_distance = int(0.25 * sampling_rate_hz) # Dist between peaks

    # 2 outputs
    # 1) candidate_peaks = x indexes of the peaks
    # 2) properties of the peak but we ignore them
    candidate_peaks, _ = signal.find_peaks(
        qrs_signal,
        height=threshold,
        distance=minimum_distance, # amount of peaks between amount of samples
    )

    # It shifts(shiftdelay) the orig signal and makes it wider, this compensates
    search_radius = int(0.08 * sampling_rate_hz)
    r_peaks = _refine_to_local_maxima(ecg_segment, candidate_peaks, search_radius)

    # If there are 2 peaks, take the one with highest amplitude
    return _remove_close_peaks(
        ecg_segment,
        r_peaks,
        minimum_distance=minimum_distance,
    )


def detect_ppg_peaks(ppg_segment, sampling_rate_hz=125):
    """
    Detect systolic PPG peaks with a small ensemble of simple detectors.

    The ensemble combines three strategies from the PPG literature:
    local maxima with adaptive prominence, derivative-based detection and a
    slope-sum detector. Peaks that are supported by at least two strategies are
    accepted, then physiological distance rules reduce false positives.

    Args:
        ppg_segment: One PPG segment with shape (samples,).
        sampling_rate_hz: Sampling rate in Hz.

    Returns:
        NumPy array with systolic PPG peak sample indexes.
    """
    ppg_segment = np.asarray(ppg_segment, dtype=float) # input to numpy array
    # PPG signal differs for everyone(position,segment), so normalise
    normalized_segment = _normalize(ppg_segment)

    local_maxima = _detect_ppg_local_maxima(normalized_segment, sampling_rate_hz)
    derivative_peaks = _detect_ppg_derivative_peaks(normalized_segment, sampling_rate_hz)
    slope_sum_peaks = _detect_ppg_slope_sum_peaks(normalized_segment, sampling_rate_hz)

    merged_peaks = _majority_vote_peaks(
        [local_maxima, derivative_peaks, slope_sum_peaks],
        tolerance_samples=int(0.12 * sampling_rate_hz), #(15 samples) 3 diffr methodes,3 diffr indexes,so approximately same signal
        required_votes=2, # 2 methodes have to detect the peak
    )

    # So two final PPG peaks must be at least about 43 samples apart.
    # To not detect the dicrotic notch / diastolic wave!!!!
    minimum_distance = int(0.35 * sampling_rate_hz)
    refined_peaks = _refine_to_local_maxima(ppg_segment, merged_peaks, int(0.12 * sampling_rate_hz))

    # Delete close peaks
    return _remove_close_peaks(
        ppg_segment,
        refined_peaks,
        minimum_distance=minimum_distance,
    )


def calculate_heart_rate_from_peaks(peaks, sampling_rate_hz=125):
    """
    The function converts peak values into time intervals, removes
    unrealistic values and uses these to calculate the average heart
    rate in bpm.

    Args:
        peaks: Sample indexes of detected beats.
        sampling_rate_hz: Sampling rate in Hz.

    Returns:
        HeartRateResult with bpm, number of peaks and mean interval.
    """
    peaks = np.asarray(peaks, dtype=int)

    if peaks.size < 2:
        return HeartRateResult(
            beats_per_minute=float("nan"),
            peak_count=int(peaks.size),
            mean_interval_seconds=float("nan"),
        )

    # np.diff => difference between sequential peak indices
    intervals_seconds = np.diff(peaks) / sampling_rate_hz
    # Time intervals between the heart beats, filters between 0.3s <=interval<= 1.5s
    valid_intervals = intervals_seconds[
        (intervals_seconds >= 0.3) & (intervals_seconds <= 1.5)
    ]

    if valid_intervals.size == 0:
        return HeartRateResult(
            beats_per_minute=float("nan"),
            peak_count=int(peaks.size),
            mean_interval_seconds=float("nan"),
        )

    mean_interval_seconds = float(np.mean(valid_intervals))
    beats_per_minute = 60.0 / mean_interval_seconds

    return HeartRateResult(
        beats_per_minute=beats_per_minute,
        peak_count=int(peaks.size),
        mean_interval_seconds=mean_interval_seconds,
    )


def compare_heart_rates(ecg_heart_rate, ppg_heart_rate):
    """
    Return the absolute difference between ECG and PPG heart rate.
    """
    if np.isnan(ecg_heart_rate.beats_per_minute) or np.isnan(ppg_heart_rate.beats_per_minute):
        return float("nan")

    # To see how far those two are from each other
    return abs(ecg_heart_rate.beats_per_minute - ppg_heart_rate.beats_per_minute)


def _prepare_ecg_qrs_signal(ecg_segment, sampling_rate_hz):
    settings = ButterworthBandpassSettings(
        low_cut_hz=5.0,
        high_cut_hz=18.0,
        sampling_rate_hz=sampling_rate_hz,
        order=2,
    )
    filtered_ecg = butterworth_bandpass(ecg_segment, settings)
    derivative = np.gradient(filtered_ecg)
    squared = derivative**2

    window_size = max(1, int(0.15 * sampling_rate_hz)) # window of 19 sampl,0.15 time R peak
    integration_window = np.ones(window_size) / window_size # mean filter
    # moving window integration (moving avrage see theory)
    return np.convolve(squared, integration_window, mode="same") # same = in=out


def _detect_ppg_local_maxima(ppg_segment, sampling_rate_hz):
    minimum_distance = int(0.35 * sampling_rate_hz) # avoid dicrotic notch
    # how much does my peak sticking out of the environment = adaptive and minimum edge
    prominence = max(0.15, 0.35 * np.std(ppg_segment))

    # filter only the peaks, ignore the prop
    peaks, _ = signal.find_peaks(
        ppg_segment,
        distance=minimum_distance,
        prominence=prominence,
        height=np.mean(ppg_segment),
    )
    return peaks


def _detect_ppg_derivative_peaks(ppg_segment, sampling_rate_hz):
    derivative = np.gradient(ppg_segment)
    zero_crossing_candidates = np.where(
        (derivative[:-1] > 0) & (derivative[1:] <= 0)
    )[0] + 1 # derv(i) and derv(i+1)
    threshold = np.mean(ppg_segment) + 0.25 * np.std(ppg_segment)
    candidates = zero_crossing_candidates[ppg_segment[zero_crossing_candidates] > threshold]

    return _remove_close_peaks(
        ppg_segment,
        candidates,
        minimum_distance=int(0.35 * sampling_rate_hz),
    )


def _detect_ppg_slope_sum_peaks(ppg_segment, sampling_rate_hz):
    derivative = np.gradient(ppg_segment)
    positive_derivative = np.maximum(derivative, 0.0)# only +, neg to zero

    window_size = max(1, int(0.12 * sampling_rate_hz)) # 0.12 for increase time PPG puls
    # moving avrage(slope-sum function (SSF))
    window = np.ones(window_size) / window_size
    slope_sum = np.convolve(positive_derivative, window, mode="same")

    # median bc its robuster against outliers
    threshold = np.median(slope_sum) + 0.7 * np.std(slope_sum)

    # areas where the increase is significant
    candidate_regions, _ = signal.find_peaks(
        slope_sum,
        height=threshold,
        distance=int(0.35 * sampling_rate_hz),
    )

    # now we look for the real PPG peak
    return _refine_to_local_maxima(
        ppg_segment,
        candidate_regions,
        search_radius=int(0.18 * sampling_rate_hz),
    )


def _majority_vote_peaks(peak_sets, tolerance_samples, required_votes):
    # all peaks in one list, if no peaks empty array
    all_peaks = np.sort(np.concatenate([np.asarray(peaks, dtype=int) for peaks in peak_sets]))
    if all_peaks.size == 0:
        return np.array([], dtype=int)

    # final out
    accepted_peaks = []
    used_indexes = np.zeros(all_peaks.shape, dtype=bool)

    # peak_index=list, peak = sample index
    for peak_index, peak in enumerate(all_peaks):
        if used_indexes[peak_index]: # skip if it is already in a group
            continue

        # look for peaks that are close enough
        close_mask = np.abs(all_peaks - peak) <= tolerance_samples
        close_peaks = all_peaks[close_mask]

        # how many methodes detect these group
        vote_count = _count_peak_set_votes(close_peaks, peak_sets, tolerance_samples)

        # if enought methodes detect, we calc peak and add it to the list
        if vote_count >= required_votes:
            accepted_peaks.append(int(np.round(np.median(close_peaks))))

        # mark the used peak indexes
        used_indexes = used_indexes | close_mask

    return np.asarray(accepted_peaks, dtype=int)


def _count_peak_set_votes(close_peaks, peak_sets, tolerance_samples):
    """
        Count how many detection methods support a group of nearby peaks.

        Each method gets at most one vote if it has at least one peak within
        tolerance_samples of any peak in close_peaks. This is used for ensemble
        voting, where a peak is accepted only if enough methods agree.

        """
    vote_count = 0

    for peak_set in peak_sets:
        peak_set = np.asarray(peak_set, dtype=int)
        if peak_set.size == 0:
            continue
        distances = np.abs(peak_set[:, None] - close_peaks[None, :])
        if np.any(distances <= tolerance_samples):
            vote_count += 1

    return vote_count


def _refine_to_local_maxima(signal_segment, candidate_peaks, search_radius):
    """
    Refine candidate peaks by aligning them to the nearest local maxima.

    For each candidate peak, a local window is defined around its position.
    The function then searches within this window for the maximum value in the
    original signal and updates the peak location accordingly.

    """
    signal_segment = np.asarray(signal_segment, dtype=float)
    refined_peaks = []

    for peak in np.asarray(candidate_peaks, dtype=int):
        # calc window
        start_index = max(0, peak - search_radius)
        end_index = min(signal_segment.shape[0], peak + search_radius + 1)

        # check window
        if start_index >= end_index:
            continue

        # absolut index = startindex + index of the biggest value in the window
        local_peak = start_index + int(np.argmax(signal_segment[start_index:end_index]))
        refined_peaks.append(local_peak)

    return np.asarray(sorted(set(refined_peaks)), dtype=int)


def _remove_close_peaks(signal_segment, peaks, minimum_distance):
    """
    Remove peaks that are too close to each other and keep only the most prominent ones.

    Peaks are first sorted and deduplicated. The function then iterates through
    the peaks and enforces a minimum distance between consecutive peaks. If two
    peaks are closer than the specified minimum distance, only the peak with the
    higher signal amplitude is retained.
    """

    peaks = np.asarray(sorted(set(np.asarray(peaks, dtype=int))), dtype=int)
    if peaks.size <= 1:
        return peaks

    kept_peaks = []
    for peak in peaks:
        if not kept_peaks:
            kept_peaks.append(peak)
            continue

        previous_peak = kept_peaks[-1]
        if peak - previous_peak >= minimum_distance:
            kept_peaks.append(peak)
        elif signal_segment[peak] > signal_segment[previous_peak]:
            kept_peaks[-1] = peak

    return np.asarray(kept_peaks, dtype=int)


def _normalize(signal_segment):
    """
        Normalize a signal segment using z-score normalization.

        The signal is centered by subtracting its mean and scaled by its
        standard deviation, resulting in a signal with approximately zero mean
        and unit variance. If the standard deviation is zero, only mean-centering
        is applied to avoid division by zero.
        """
    signal_segment = np.asarray(signal_segment, dtype=float)
    standard_deviation = np.std(signal_segment)

    if standard_deviation == 0:
        return signal_segment - np.mean(signal_segment)

    return (signal_segment - np.mean(signal_segment)) / standard_deviation
