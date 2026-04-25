import numpy as np


def robust_z_score(segment):
    """
    Compute MAD-based robust z-scores for one signal segment.

    MAD is the median absolute deviation. It is less sensitive to large motion
    spikes than a normal standard deviation.
    """
    segment = np.asarray(segment, dtype=float)
    centered = segment - np.median(segment)
    mad = np.median(np.abs(centered))

    if mad == 0:
        return np.full(segment.shape, np.inf)

    return centered / (1.4826 * mad)


def has_motion_artifact(segment, z_threshold=20.0, max_spike_ratio=0.01):
    """
    Detect likely motion artifacts in one segment.

    Args:
        segment: One ECG or PPG segment with shape (samples,).
        z_threshold: Absolute robust z-score above which a sample is a spike.
        max_spike_ratio: Maximum allowed fraction of spike samples.

    Returns:
        True if the segment likely contains too many motion spikes.
    """
    z_scores = np.abs(robust_z_score(segment))
    spike_ratio = np.mean(z_scores > z_threshold)

    return spike_ratio > max_spike_ratio


def find_artifact_segments(signal_data, z_threshold=20.0, max_spike_ratio=0.01):
    """
    Return indexes of segments that likely contain motion artifacts.

    Args:
        signal_data: Signal array with shape (segments, samples).
        z_threshold: Absolute robust z-score above which a sample is a spike.
        max_spike_ratio: Maximum allowed fraction of spike samples.

    Returns:
        List of segment indexes flagged as artifact-heavy.
    """
    bad_segments = []

    for segment_index, segment in enumerate(signal_data):
        if has_motion_artifact(
            segment,
            z_threshold=z_threshold,
            max_spike_ratio=max_spike_ratio,
        ):
            bad_segments.append(segment_index)

    return bad_segments
