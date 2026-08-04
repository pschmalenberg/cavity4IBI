from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = 2000
PEAK_THRESHOLD = 0.4
MIN_PEAK_DISTANCE = 616


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute metrics for retained continuous traces.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "reference_results" / "P12",
    )
    parser.add_argument(
        "--skip-dtw",
        action="store_true",
        help="Skip the quadratic-time DTW calculation for a quick check.",
    )
    return parser.parse_args()


def get_peaks(signal: np.ndarray) -> np.ndarray:
    peaks, _ = find_peaks(signal, height=PEAK_THRESHOLD, distance=MIN_PEAK_DISTANCE)
    return peaks


def get_heartbeat_matches(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    matches = np.zeros((2, reference.shape[0]))
    for index in range(1, reference.shape[0] - 1):
        left_gap = (reference[index] - reference[index - 1]) / 2
        right_gap = (reference[index + 1] - reference[index]) / 2
        candidates = prediction[
            (prediction > reference[index] - left_gap)
            & (prediction < reference[index] + right_gap)
        ]
        if len(candidates) == 1:
            matches[:, index] = reference[index], candidates[0]
    return matches[:, np.any(matches, axis=0)]


def get_timing_metrics(matches: np.ndarray, total_peaks: int) -> tuple[float, float]:
    match_ratio = matches.shape[1] / total_peaks
    selected_count = math.floor(total_peaks * (match_ratio - 0.0001))
    differences = np.abs(matches[0] - matches[1])
    selected = np.sort(np.argpartition(differences, selected_count)[:selected_count])
    selected_matches = matches[:, selected]
    reference_intervals = np.diff(selected_matches[0]) / SAMPLE_RATE
    prediction_intervals = np.diff(selected_matches[1]) / SAMPLE_RATE
    return match_ratio, float(np.mean(np.abs(reference_intervals - prediction_intervals)))


def main() -> None:
    args = parse_args()
    prediction = np.loadtxt(args.results_dir / "continuous_pred_2831.csv", delimiter=",")
    reference = np.loadtxt(args.results_dir / "continuous_real_2831.csv", delimiter=",")
    if prediction.shape != reference.shape:
        raise ValueError(f"Trace shapes differ: {prediction.shape} and {reference.shape}")

    reference_peaks = get_peaks(reference)
    prediction_peaks = get_peaks(prediction)
    matches = get_heartbeat_matches(reference_peaks, prediction_peaks)
    match_ratio, timing_mae = get_timing_metrics(matches, len(reference_peaks) - 2)

    duration = (prediction_peaks[-2] - prediction_peaks[1]) / SAMPLE_RATE
    reference_hr = (len(reference_peaks) - 2) * 60 / duration
    prediction_hr = (len(prediction_peaks) - 2) * 60 / duration
    pearson_correlation, pearson_p_value = pearsonr(reference, prediction)

    print(f"R-peak timing MAE: {timing_mae:.17f} s")
    print(f"Heartbeat match: {match_ratio * 100:.2f}%")
    print(f"Ground-truth heart rate: {reference_hr:.1f} BPM")
    print(f"Predicted heart rate: {prediction_hr:.1f} BPM")
    print(f"Pearson correlation: {pearson_correlation:.4f} (p={pearson_p_value:.2e})")

    if not args.skip_dtw:
        from dtw import dtw

        downsample_factor = 10
        result = dtw(reference[::downsample_factor], prediction[::downsample_factor])
        normalized_distance = result.distance / len(reference[::downsample_factor])
        print(f"DTW distance: {result.distance:.4f}")
        print(f"Normalized DTW distance: {normalized_distance:.6f}")


if __name__ == "__main__":
    main()