import numpy as np


def load_patient(patient_id, data_path="data"):
    """
    Laad alle signalen voor één patiënt.

    patient_id: bv. "p000188"
    data_path: pad naar de data map
    """
    ecg = np.load(f"{data_path}/ecg/{patient_id}_ecg.npy")
    ppg = np.load(f"{data_path}/ppg/{patient_id}_ppg.npy")
    abp = np.load(f"{data_path}/abp/{patient_id}_abp.npy")
    labels = np.load(f"{data_path}/labels/{patient_id}_labels.npy")

    return ecg, ppg, abp, labels


# Test
if __name__ == "__main__":
    ecg, ppg, abp, labels = load_patient("p000188")
    print(f"ECG vorm: {ecg.shape}")
    print(f"PPG vorm: {ppg.shape}")
    print(f"Labels: {labels}")