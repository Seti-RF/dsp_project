import matplotlib.pyplot as plt

from artifacts import find_artifact_segments
from filters import preprocess_ecg, preprocess_ppg
from fusion import analyze_quality_and_fusion, save_fusion_results_csv
from hrv import analyze_hrv, save_hrv_results_csv
from loader import load_patient
from peaks import (
    calculate_heart_rate_from_peaks,
    compare_heart_rates,
    detect_ecg_r_peaks,
    detect_ppg_peaks,
)
from plotter import (
    FrequencyDomainPlotter,
    plot_detected_peaks,
    plot_heart_rates_per_segment,
    plot_sqi_per_segment,
    plot_synchronized_ecg_ppg,
)

"""
This script was to test our implementations
"""
PATIENT_ID = "p000188"
SEGMENT_INDEX = 23
SAMPLING_RATE_HZ = 125
SESSION4_OUTPUT_DIR = "outputs/session4"
SESSION5_OUTPUT_DIR = "outputs/session5"

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

    ecg_r_peaks = detect_ecg_r_peaks(
        filtered_ecg[SEGMENT_INDEX],
        sampling_rate_hz=SAMPLING_RATE_HZ,
    )
    ppg_peaks = detect_ppg_peaks(
        filtered_ppg[SEGMENT_INDEX],
        sampling_rate_hz=SAMPLING_RATE_HZ,
    )
    ecg_heart_rate = calculate_heart_rate_from_peaks(
        ecg_r_peaks,
        sampling_rate_hz=SAMPLING_RATE_HZ,
    )
    ppg_heart_rate = calculate_heart_rate_from_peaks(
        ppg_peaks,
        sampling_rate_hz=SAMPLING_RATE_HZ,
    )
    heart_rate_difference = compare_heart_rates(ecg_heart_rate, ppg_heart_rate)

    print(f"ECG R-peaks gevonden: {ecg_heart_rate.peak_count}")
    print(f"PPG systolische pieken gevonden: {ppg_heart_rate.peak_count}")
    print(f"ECG hartslag: {ecg_heart_rate.beats_per_minute:.2f} bpm")
    print(f"PPG hartslag: {ppg_heart_rate.beats_per_minute:.2f} bpm")
    print(f"Verschil ECG/PPG hartslag: {heart_rate_difference:.2f} bpm")

    fusion_results = analyze_quality_and_fusion(
        filtered_ecg=filtered_ecg,
        filtered_ppg=filtered_ppg,
        sampling_rate_hz=SAMPLING_RATE_HZ,
    )
    save_fusion_results_csv(
        fusion_results,
        output_path=f"{SESSION4_OUTPUT_DIR}/{PATIENT_ID}_quality_fusion.csv",
    )
    selected_fusion_result = fusion_results[SEGMENT_INDEX]

    print("Session 4 kwaliteit en fusie:")
    print(f"ECG SQI: {selected_fusion_result.ecg_sqi:.3f}")
    print(f"PPG SQI: {selected_fusion_result.ppg_sqi:.3f}")
    print(f"ECG gewicht: {selected_fusion_result.ecg_weight:.3f}")
    print(f"PPG gewicht: {selected_fusion_result.ppg_weight:.3f}")
    print(f"Gefuseerde hartslag: {selected_fusion_result.fused_heart_rate_bpm:.2f} bpm")
    print(
        "Session 4 tabel opgeslagen als "
        f"{SESSION4_OUTPUT_DIR}/{PATIENT_ID}_quality_fusion.csv"
    )

    hrv_results = analyze_hrv(
        filtered_ecg=filtered_ecg,
        filtered_ppg=filtered_ppg,
        sampling_rate_hz=SAMPLING_RATE_HZ,
    )
    save_hrv_results_csv(
        hrv_results,
        output_path=f"{SESSION5_OUTPUT_DIR}/{PATIENT_ID}_hrv_statistics.csv",
    )
    selected_hrv_result = hrv_results[SEGMENT_INDEX]

    print("Session 5 HRV:")
    print(f"ECG mean NN: {selected_hrv_result.ecg_hrv.mean_nn_seconds:.3f} s")
    print(f"ECG SDNN: {selected_hrv_result.ecg_hrv.sdnn_seconds:.3f} s")
    print(f"ECG RMSSD: {selected_hrv_result.ecg_hrv.rmssd_seconds:.3f} s")
    print(f"ECG pNN50: {selected_hrv_result.ecg_hrv.pnn50_percent:.1f} %")
    print(f"PPG mean NN: {selected_hrv_result.ppg_hrv.mean_nn_seconds:.3f} s")
    print(f"PPG SDNN: {selected_hrv_result.ppg_hrv.sdnn_seconds:.3f} s")
    print(f"PPG RMSSD: {selected_hrv_result.ppg_hrv.rmssd_seconds:.3f} s")
    print(f"PPG pNN50: {selected_hrv_result.ppg_hrv.pnn50_percent:.1f} %")
    print(f"Fused HRV source: {selected_hrv_result.fused_hrv_source}")
    print(f"Fused mean NN: {selected_hrv_result.fused_hrv.mean_nn_seconds:.3f} s")
    print(f"Fused SDNN: {selected_hrv_result.fused_hrv.sdnn_seconds:.3f} s")
    print(f"Fused RMSSD: {selected_hrv_result.fused_hrv.rmssd_seconds:.3f} s")
    print(f"Fused pNN50: {selected_hrv_result.fused_hrv.pnn50_percent:.1f} %")
    print(
        "Session 5 tabel opgeslagen als "
        f"{SESSION5_OUTPUT_DIR}/{PATIENT_ID}_hrv_statistics.csv"
    )

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

    ecg_peak_figure = plot_detected_peaks(
        signal_data=filtered_ecg,
        peaks=ecg_r_peaks,
        signal_name="ECG R",
        patient_id=PATIENT_ID,
        segment_index=SEGMENT_INDEX,
        sampling_rate_hz=SAMPLING_RATE_HZ,
    )
    ecg_peak_figure.canvas.manager.set_window_title(
        f"{PATIENT_ID} ECG R-peaks segment {SEGMENT_INDEX}"
    )

    ppg_peak_figure = plot_detected_peaks(
        signal_data=filtered_ppg,
        peaks=ppg_peaks,
        signal_name="PPG systolic",
        patient_id=PATIENT_ID,
        segment_index=SEGMENT_INDEX,
        sampling_rate_hz=SAMPLING_RATE_HZ,
    )
    ppg_peak_figure.canvas.manager.set_window_title(
        f"{PATIENT_ID} PPG peaks segment {SEGMENT_INDEX}"
    )

    sqi_figure = plot_sqi_per_segment(
        fusion_results,
        patient_id=PATIENT_ID,
        output_path=f"{SESSION4_OUTPUT_DIR}/{PATIENT_ID}_sqi_per_segment.png",
    )
    sqi_figure.canvas.manager.set_window_title(f"{PATIENT_ID} SQI per segment")

    heart_rate_fusion_figure = plot_heart_rates_per_segment(
        fusion_results,
        patient_id=PATIENT_ID,
        output_path=f"{SESSION4_OUTPUT_DIR}/{PATIENT_ID}_heart_rate_fusion.png",
    )
    heart_rate_fusion_figure.canvas.manager.set_window_title(
        f"{PATIENT_ID} heart rate fusion"
    )

    plt.show()
