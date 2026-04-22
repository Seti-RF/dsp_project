from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
