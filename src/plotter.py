from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class FrequencyDomainPlotter:
    """
    Plot signal content in the frequency domain using a Fourier transform.
    """

    def __init__(self, sampling_rate_hz=125):
        self.sampling_rate_hz = sampling_rate_hz

    def plot_signal_spectrum(
        self,
        signal_data,
        signal_name,
        patient_id,
        segment_index=0,
        max_frequency_hz=None,
        output_path=None,
    ):
        """
        Show which frequency components are present in a raw segment.

        This is mainly used during preprocessing checks: a strong low-frequency
        component points to baseline drift, while a narrow peak around 50/60 Hz
        would suggest powerline noise.
        """
        if segment_index < 0 or segment_index >= signal_data.shape[0]:
            raise IndexError(
                f"segment_index {segment_index} is outside available range 0-{signal_data.shape[0] - 1}"
            )

        signal_segment = signal_data[segment_index]
        frequencies, magnitude = self._fft_magnitude(signal_segment)

        figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
        axis.plot(frequencies, magnitude, color="#1f77b4", linewidth=0.9)
        axis.set_title(f"{patient_id} - {signal_name} frequency spectrum segment {segment_index}")
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Magnitude")
        axis.grid(True, alpha=0.3)

        if max_frequency_hz is not None:
            axis.set_xlim(0, max_frequency_hz)

        if output_path is not None:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_file, dpi=150)

        return figure

    def plot_pre_post_filter_spectrum(
        self,
        raw_signal,
        filtered_signal,
        signal_name,
        patient_id,
        segment_index=0,
        max_frequency_hz=None,
        output_path=None,
    ):
        """
        Compare the spectrum before and after filtering.

        If the preprocessing is working, unwanted frequency content should be
        reduced in the filtered curve while the physiological band remains.
        """
        if raw_signal.shape != filtered_signal.shape:
            raise ValueError("raw_signal and filtered_signal must have the same shape")
        if segment_index < 0 or segment_index >= raw_signal.shape[0]:
            raise IndexError(
                f"segment_index {segment_index} is outside available range 0-{raw_signal.shape[0] - 1}"
            )

        raw_segment = raw_signal[segment_index]
        filtered_segment = filtered_signal[segment_index]
        raw_frequencies, raw_magnitude = self._fft_magnitude(raw_segment)
        filtered_frequencies, filtered_magnitude = self._fft_magnitude(filtered_segment)

        figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
        axis.plot(raw_frequencies, raw_magnitude, color="#606060", linewidth=0.9, label="Raw")
        axis.plot(
            filtered_frequencies,
            filtered_magnitude,
            color="#1f77b4",
            linewidth=0.9,
            label="Filtered",
        )
        axis.set_title(f"{patient_id} - {signal_name} frequency spectrum segment {segment_index}")
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Magnitude")
        axis.grid(True, alpha=0.3)
        axis.legend()

        if max_frequency_hz is not None:
            axis.set_xlim(0, max_frequency_hz)

        if output_path is not None:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_file, dpi=150)

        return figure

    def _fft_magnitude(self, signal_segment):
        """
        Return positive FFT frequencies and normalized magnitudes.

        rfft is used because ECG/PPG signals are real-valued. Subtracting the
        mean removes the DC offset so the zero-frequency bin does not dominate.
        """
        centered_segment = np.asarray(signal_segment, dtype=float)
        centered_segment = centered_segment - np.mean(centered_segment)
        frequencies = np.fft.rfftfreq(
            centered_segment.shape[0],
            d=1.0 / self.sampling_rate_hz,
        )
        magnitude = np.abs(np.fft.rfft(centered_segment)) / centered_segment.shape[0]
        return frequencies, magnitude


def plot_synchronized_ecg_ppg(
    ecg,
    ppg,
    patient_id,
    segment_index=0,
    sampling_rate_hz=125,
    output_path=None,
):
    """
    Plot ECG and PPG on the same time axis for visual inspection.

    The two signals are measured over the same segment, so plotting them with a
    shared x-axis makes it easier to compare beat timing and signal quality.
    """
    if segment_index < 0 or segment_index >= ecg.shape[0]:
        raise IndexError(
            f"segment_index {segment_index} is outside available range 0-{ecg.shape[0] - 1}"
        )

    ecg_segment = ecg[segment_index]
    ppg_segment = ppg[segment_index]
    time_seconds = np.arange(ecg_segment.shape[0]) / sampling_rate_hz

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(12, 6),
        constrained_layout=True,
    )

    axes[0].plot(time_seconds, ecg_segment, color="#1f77b4", linewidth=0.9)
    axes[0].set_title(f"{patient_id} - ECG segment {segment_index}")
    axes[0].set_ylabel("ECG amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_seconds, ppg_segment, color="#d62728", linewidth=0.9)
    axes[1].set_title(f"{patient_id} - PPG segment {segment_index}")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("PPG amplitude")
    axes[1].grid(True, alpha=0.3)

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_file, dpi=150)

    return figure


def plot_detected_peaks(
    signal_data,
    peaks,
    signal_name,
    patient_id,
    segment_index=0,
    sampling_rate_hz=125,
    output_path=None,
):
    """
    Plot a segment and mark the detected beat locations.

    This is the main visual check for peak detection. Correct markers should sit
    on ECG R-peaks or PPG systolic peaks, not on noise or secondary waves.
    """
    if segment_index < 0 or segment_index >= signal_data.shape[0]:
        raise IndexError(
            f"segment_index {segment_index} is outside available range 0-{signal_data.shape[0] - 1}"
        )

    signal_segment = signal_data[segment_index]
    peaks = np.asarray(peaks, dtype=int)
    time_seconds = np.arange(signal_segment.shape[0]) / sampling_rate_hz

    figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    axis.plot(time_seconds, signal_segment, color="#1f77b4", linewidth=0.9)
    axis.scatter(
        time_seconds[peaks],
        signal_segment[peaks],
        color="#d62728",
        s=28,
        label="Detected peaks",
        zorder=3,
    )
    axis.set_title(f"{patient_id} - detected {signal_name} peaks segment {segment_index}")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel(f"{signal_name} amplitude")
    axis.grid(True, alpha=0.3)
    axis.legend()

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_file, dpi=150)

    return figure


def plot_pre_post_filter(
    raw_signal,
    filtered_signal,
    signal_name,
    patient_id,
    segment_index=0,
    sampling_rate_hz=125,
    output_path=None,
):
    """
    Show the time-domain effect of preprocessing on one segment.

    The raw trace keeps the original waveform, while the filtered trace should
    have less drift and noise without destroying the peaks used later.
    """
    if raw_signal.shape != filtered_signal.shape:
        raise ValueError("raw_signal and filtered_signal must have the same shape")
    if segment_index < 0 or segment_index >= raw_signal.shape[0]:
        raise IndexError(
            f"segment_index {segment_index} is outside available range 0-{raw_signal.shape[0] - 1}"
        )

    raw_segment = raw_signal[segment_index]
    filtered_segment = filtered_signal[segment_index]
    time_seconds = np.arange(raw_segment.shape[0]) / sampling_rate_hz

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(12, 6),
        constrained_layout=True,
    )

    axes[0].plot(time_seconds, raw_segment, color="#606060", linewidth=0.9)
    axes[0].set_title(f"{patient_id} - raw {signal_name} segment {segment_index}")
    axes[0].set_ylabel("Raw amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_seconds, filtered_segment, color="#1f77b4", linewidth=0.9)
    axes[1].set_title(f"{patient_id} - filtered {signal_name} segment {segment_index}")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Filtered amplitude")
    axes[1].grid(True, alpha=0.3)

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_file, dpi=150)

    return figure


def plot_sqi_per_segment(fusion_results, patient_id, output_path=None):
    """
    Plot ECG and PPG SQI scores for all segments.
    """
    segment_indexes = np.asarray([result.segment_index for result in fusion_results])
    ecg_sqi = np.asarray([result.ecg_sqi for result in fusion_results])
    ppg_sqi = np.asarray([result.ppg_sqi for result in fusion_results])

    figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    axis.plot(segment_indexes, ecg_sqi, marker="o", color="#1f77b4", label="ECG SQI")
    axis.plot(segment_indexes, ppg_sqi, marker="o", color="#d62728", label="PPG SQI")
    axis.set_title(f"{patient_id} - signal quality per segment")
    axis.set_xlabel("Segment index")
    axis.set_ylabel("SQI score")
    axis.set_ylim(0, 1.05)
    axis.grid(True, alpha=0.3)
    axis.legend()

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_file, dpi=150)

    return figure


def plot_heart_rates_per_segment(fusion_results, patient_id, output_path=None):
    """
    Plot ECG HR, PPG HR and fused HR for all segments.
    """
    segment_indexes = np.asarray([result.segment_index for result in fusion_results])
    ecg_hr = np.asarray([result.ecg_heart_rate_bpm for result in fusion_results])
    ppg_hr = np.asarray([result.ppg_heart_rate_bpm for result in fusion_results])
    fused_hr = np.asarray([result.fused_heart_rate_bpm for result in fusion_results])

    figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    axis.plot(segment_indexes, ecg_hr, marker="o", color="#1f77b4", label="ECG HR")
    axis.plot(segment_indexes, ppg_hr, marker="o", color="#d62728", label="PPG HR")
    axis.plot(segment_indexes, fused_hr, marker="o", color="#2ca02c", label="Fused HR")
    axis.set_title(f"{patient_id} - heart rate fusion per segment")
    axis.set_xlabel("Segment index")
    axis.set_ylabel("Heart rate (bpm)")
    axis.grid(True, alpha=0.3)
    axis.legend()

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_file, dpi=150)

    return figure
