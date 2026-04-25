import matplotlib.pyplot as plt

from artifacts import find_artifact_segments
from filters import preprocess_ecg, preprocess_ppg
from loader import load_patient
from plotter import FrequencyDomainPlotter, plot_synchronized_ecg_ppg

PATIENT_ID = "p000188"
SEGMENT_INDEX = 23
SAMPLING_RATE_HZ = 125

if __name__ == "__main__":
    ecg, ppg, abp, labels = load_patient(PATIENT_ID)
    print(f"ECG vorm: {ecg.shape}")
    print(f"PPG vorm: {ppg.shape}")
    print(f"ABP vorm: {abp.shape}")
    print(f"Labels vorm: {labels.shape}")
    print(f"Eerste label [SBP, DBP]: {labels[SEGMENT_INDEX]}")

    bad_ecg_segments = find_artifact_segments(ecg)
    bad_ppg_segments = find_artifact_segments(ppg)
    print(f"ECG segmenten met mogelijke bewegingsartefacten: {bad_ecg_segments}")
    print(f"PPG segmenten met mogelijke bewegingsartefacten: {bad_ppg_segments}")

    figure = plot_synchronized_ecg_ppg(
        ecg=ecg,
        ppg=ppg,
        patient_id=PATIENT_ID,
        segment_index=SEGMENT_INDEX,
        sampling_rate_hz=SAMPLING_RATE_HZ,
    )
    figure.canvas.manager.set_window_title(f"{PATIENT_ID} ECG/PPG segment {SEGMENT_INDEX}")

    frequency_plotter = FrequencyDomainPlotter(sampling_rate_hz=SAMPLING_RATE_HZ)

    ecg_frequency_figure = frequency_plotter.plot_signal_spectrum(
        signal_data=ecg,
        signal_name="ECG",
        patient_id=PATIENT_ID,
        segment_index=SEGMENT_INDEX,
        max_frequency_hz=62.5,
    )
    ecg_frequency_figure.canvas.manager.set_window_title(
        f"{PATIENT_ID} ECG frequency spectrum segment {SEGMENT_INDEX}"
    )

    ppg_frequency_figure = frequency_plotter.plot_signal_spectrum(
        signal_data=ppg,
        signal_name="PPG",
        patient_id=PATIENT_ID,
        segment_index=SEGMENT_INDEX,
        max_frequency_hz=62.5,
    )
    ppg_frequency_figure.canvas.manager.set_window_title(
        f"{PATIENT_ID} PPG frequency spectrum segment {SEGMENT_INDEX}"
    )

    filtered_ecg = preprocess_ecg(ecg, sampling_rate_hz=SAMPLING_RATE_HZ)
    filtered_ppg = preprocess_ppg(ppg, sampling_rate_hz=SAMPLING_RATE_HZ)

    filtered_figure = plot_synchronized_ecg_ppg(
        ecg=filtered_ecg,
        ppg=filtered_ppg,
        patient_id=PATIENT_ID,
        segment_index=SEGMENT_INDEX,
        sampling_rate_hz=SAMPLING_RATE_HZ,
    )
    filtered_figure.canvas.manager.set_window_title(
        f"{PATIENT_ID} filtered ECG/PPG segment {SEGMENT_INDEX}"
    )

    filtered_ecg_frequency_figure = frequency_plotter.plot_signal_spectrum(
        signal_data=filtered_ecg,
        signal_name="Filtered ECG",
        patient_id=PATIENT_ID,
        segment_index=SEGMENT_INDEX,
        max_frequency_hz=62.5,
    )
    filtered_ecg_frequency_figure.canvas.manager.set_window_title(
        f"{PATIENT_ID} filtered ECG frequency spectrum segment {SEGMENT_INDEX}"
    )

    filtered_ppg_frequency_figure = frequency_plotter.plot_signal_spectrum(
        signal_data=filtered_ppg,
        signal_name="Filtered PPG",
        patient_id=PATIENT_ID,
        segment_index=SEGMENT_INDEX,
        max_frequency_hz=62.5,
    )
    filtered_ppg_frequency_figure.canvas.manager.set_window_title(
        f"{PATIENT_ID} filtered PPG frequency spectrum segment {SEGMENT_INDEX}"
    )

    plt.show()
