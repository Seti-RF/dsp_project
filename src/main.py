from loader import load_patient

if __name__ == "__main__":
    ecg, ppg, abp, labels = load_patient("p000188")
    print(f"ECG vorm: {ecg.shape}")
    print(f"PPG vorm: {ppg.shape}")
    print(f"ABP vorm: {abp.shape}")
    print(f"Labels vorm: {labels.shape}")