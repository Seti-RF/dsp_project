import matplotlib.pyplot as plt

from loader import load_patient
from plotter import plot_synchronized_ecg_ppg


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

    figure = plot_synchronized_ecg_ppg(
        ecg=ecg,
        ppg=ppg,
        patient_id=PATIENT_ID,
        segment_index=SEGMENT_INDEX,
        sampling_rate_hz=SAMPLING_RATE_HZ,
    )
    figure.canvas.manager.set_window_title(f"{PATIENT_ID} ECG/PPG segment {SEGMENT_INDEX}")

    plt.show()
