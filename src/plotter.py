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
        Plot the frequency spectrum for one raw signal segment.

        Args:
            signal_data: Signal array with shape (segments, samples).
            signal_name: Signal label, for example "ECG" or "PPG".
            patient_id: Patient identifier, for example "p000188".
            segment_index: Segment number to plot.
            max_frequency_hz: Optional upper x-axis limit in Hz.
            output_path: Optional file path where the plot is saved.

        Returns:
            The matplotlib Figure object containing the frequency-domain plot.
        """
        if segment_index < 0 or segment_index >= signal_data.shape[0]:
            raise IndexError(
                f"segment_index {segment_index} is outside available range 0-{signal_data.shape[0] - 1}"
            )


        # Hetzelfde al de tijdsdomein maar dan met de fft toegepast
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
        Plot raw and filtered signal spectra for one segment.

        Args:
            raw_signal: Raw signal array with shape (segments, samples).
            filtered_signal: Filtered signal array with the same shape.
            signal_name: Signal label, for example "ECG" or "PPG".
            patient_id: Patient identifier, for example "p000188".
            segment_index: Segment number to plot.
            max_frequency_hz: Optional upper x-axis limit in Hz.
            output_path: Optional file path where the plot is saved.

        Returns:
            The matplotlib Figure object containing the frequency-domain plot.
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
    Plot synchronized ECG and PPG signals for one patient segment.

    Args:
        ecg: ECG signal array with shape (segments, samples).
        ppg: PPG signal array with shape (segments, samples).
        patient_id: Patient identifier, for example "p000188".
        segment_index: Segment number to plot.
        sampling_rate_hz: Sampling rate used to convert samples to seconds.
        output_path: Optional file path where the plot is saved.

    Returns:
        The matplotlib Figure object containing the synchronized plot.

    Raises:
        IndexError: If segment_index is outside the available segment range.
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
    Plot one raw segment above its filtered version.

    Args:
        raw_signal: Raw signal array with shape (segments, samples).
        filtered_signal: Filtered signal array with the same shape.
        signal_name: Signal label, for example "ECG" or "PPG".
        patient_id: Patient identifier, for example "p000188".
        segment_index: Segment number to plot.
        sampling_rate_hz: Sampling rate used to convert samples to seconds.
        output_path: Optional file path where the plot is saved.

    Returns:
        The matplotlib Figure object containing the comparison plot.
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
