from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class ButterworthBandpassSettings:
    """
    Settings for one Butterworth bandpass filter.

    The lower cutoff removes baseline wander. The upper cutoff removes
    high-frequency noise, including powerline interference when it is above the
    selected physiological bandwidth.
    """

    low_cut_hz: float
    high_cut_hz: float
    sampling_rate_hz: float = 125.0
    order: int = 4


ECG_BUTTERWORTH_SETTINGS = ButterworthBandpassSettings(
    low_cut_hz=0.5,
    high_cut_hz=20.0,
)

PPG_BUTTERWORTH_SETTINGS = ButterworthBandpassSettings(
    low_cut_hz=0.5,
    high_cut_hz=5.0,
)


def butterworth_bandpass(data, settings):
    """
    Apply a zero-phase Butterworth bandpass filter.

    Args:
        data: Signal array with shape (samples,) or (segments, samples).
        settings: Butterworth filter settings.

    Returns:
        Filtered signal with the same shape as the input.
    """
    _validate_settings(settings)
    sos = signal.butter(
        settings.order,
        [settings.low_cut_hz, settings.high_cut_hz],
        btype="bandpass",
        fs=settings.sampling_rate_hz,
        output="sos",
    )
    return signal.sosfiltfilt(sos, np.asarray(data, dtype=float), axis=-1)


def preprocess_ecg(ecg, sampling_rate_hz=125.0):
    """
    Filter ECG with a 0.5-20 Hz Butterworth bandpass.

    This removes baseline wander below 0.5 Hz and attenuates high-frequency
    noise above the ECG analysis bandwidth.
    """
    settings = ButterworthBandpassSettings(
        low_cut_hz=ECG_BUTTERWORTH_SETTINGS.low_cut_hz,
        high_cut_hz=ECG_BUTTERWORTH_SETTINGS.high_cut_hz,
        sampling_rate_hz=sampling_rate_hz,
        order=ECG_BUTTERWORTH_SETTINGS.order,
    )
    return butterworth_bandpass(ecg, settings)


def preprocess_ppg(ppg, sampling_rate_hz=125.0):
    """
    Filter PPG with a 0.5-5 Hz Butterworth bandpass.

    This removes slow baseline drift and attenuates high-frequency/motion noise
    outside the usual pulse waveform bandwidth.
    """
    settings = ButterworthBandpassSettings(
        low_cut_hz=PPG_BUTTERWORTH_SETTINGS.low_cut_hz,
        high_cut_hz=PPG_BUTTERWORTH_SETTINGS.high_cut_hz,
        sampling_rate_hz=sampling_rate_hz,
        order=PPG_BUTTERWORTH_SETTINGS.order,
    )
    return butterworth_bandpass(ppg, settings)


def _validate_settings(settings):
    if settings.order < 1:
        raise ValueError("Butterworth filter order must be at least 1")
    if settings.low_cut_hz <= 0:
        raise ValueError("low_cut_hz must be positive")
    if settings.high_cut_hz <= settings.low_cut_hz:
        raise ValueError("high_cut_hz must be greater than low_cut_hz")

    nyquist_hz = settings.sampling_rate_hz / 2.0
    if settings.high_cut_hz >= nyquist_hz:
        raise ValueError(
            f"high_cut_hz must be below Nyquist ({nyquist_hz:g} Hz); "
            f"got {settings.high_cut_hz:g} Hz"
        )
