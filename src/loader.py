from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_patient(patient_id, data_path=None):
    """
    Load all available signals and labels for one patient.

    The dataset stores each modality in a separate folder but uses the same
    patient id in every filename. This function gathers the four arrays so older
    backend scripts can work from a single call.

    NOTE: WE LOAD THE ECG AND THE PPG FILE SEPARATELY
    """
    data_dir = Path(data_path) if data_path is not None else PROJECT_ROOT / "data"

    ecg = np.load(data_dir / "ecg" / f"{patient_id}_ecg.npy")
    ppg = np.load(data_dir / "ppg" / f"{patient_id}_ppg.npy")
    abp = np.load(data_dir / "abp" / f"{patient_id}_abp.npy")
    labels = np.load(data_dir / "labels" / f"{patient_id}_labels.npy")

    return ecg, ppg, abp, labels


if __name__ == "__main__":
    ecg, ppg, abp, labels = load_patient("p000188")
    print(f"ECG vorm: {ecg.shape}")
    print(f"PPG vorm: {ppg.shape}")
    print(f"ABP vorm: {abp.shape}")
    print(f"Labels vorm: {labels.shape}")
