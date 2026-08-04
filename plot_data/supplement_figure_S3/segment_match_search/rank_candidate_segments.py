from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate


SAMPLE_RATE = 2000
SEGMENT_SAMPLES = 4 * SAMPLE_RATE
TOP_COUNT = 12


@dataclass
class Match:
    source: str
    start_sample: int
    correlation: float
    affine_nrmse: float
    scale: float
    offset: float
    prediction: np.ndarray
    raw_input: np.ndarray | None
    name: str = ""


def load_rendered_target(root: Path) -> np.ndarray:
    path = root.parent / "retained_source_export" / "rendered_upper_samples.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        target = np.asarray(
            [float(row["predicted_ecg_rendered"]) for row in csv.DictReader(handle)],
            dtype=np.float64,
        )
    if target.size != SEGMENT_SAMPLES:
        raise ValueError(f"Expected {SEGMENT_SAMPLES} target samples, found {target.size}")
    return target


def score_segment(target: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float, float]:
    target_centered = target - target.mean()
    candidate_centered = candidate - candidate.mean()
    denominator = np.linalg.norm(target_centered) * np.linalg.norm(candidate_centered)
    correlation = float(np.dot(target_centered, candidate_centered) / denominator) if denominator else 0.0

    design = np.column_stack((candidate, np.ones(candidate.size)))
    scale, offset = np.linalg.lstsq(design, target, rcond=None)[0]
    fitted = scale * candidate + offset
    target_range = np.ptp(target)
    affine_nrmse = float(np.sqrt(np.mean((target - fitted) ** 2)) / target_range)
    return correlation, affine_nrmse, float(scale), float(offset)


def rank_known_windows(
    target: np.ndarray,
    predictions: np.ndarray,
    raw_inputs: np.ndarray | None,
    source: str,
    names: list[str] | None = None,
) -> list[Match]:
    matches: list[Match] = []
    for index, prediction in enumerate(predictions):
        prediction = np.asarray(prediction, dtype=np.float64).squeeze()
        if prediction.size != SEGMENT_SAMPLES:
            continue
        raw_input = None if raw_inputs is None else np.asarray(raw_inputs[index], dtype=np.float64).squeeze()
        correlation, nrmse, scale, offset = score_segment(target, prediction)
        matches.append(
            Match(
                source=source,
                start_sample=index * SEGMENT_SAMPLES,
                correlation=correlation,
                affine_nrmse=nrmse,
                scale=scale,
                offset=offset,
                prediction=prediction,
                raw_input=raw_input,
                name="" if names is None else str(names[index]),
            )
        )
    return matches


def normalized_correlation_scan(target: np.ndarray, signal: np.ndarray) -> np.ndarray:
    target_centered = target - target.mean()
    numerator = correlate(signal, target_centered, mode="valid", method="fft")
    kernel = np.ones(target.size, dtype=np.float64)
    window_sum = np.convolve(signal, kernel, mode="valid")
    window_square_sum = np.convolve(signal * signal, kernel, mode="valid")
    window_energy = np.maximum(window_square_sum - window_sum * window_sum / target.size, 0.0)
    denominator = np.sqrt(window_energy) * np.linalg.norm(target_centered)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def rank_continuous_trace(
    target: np.ndarray,
    prediction: np.ndarray,
    raw_input: np.ndarray | None,
    source: str,
    count: int,
) -> list[Match]:
    correlations = normalized_correlation_scan(target, prediction)
    candidate_order = np.argsort(correlations)[::-1]
    selected: list[int] = []
    minimum_separation = SEGMENT_SAMPLES // 2
    for start in candidate_order:
        start = int(start)
        if all(abs(start - existing) >= minimum_separation for existing in selected):
            selected.append(start)
        if len(selected) == count:
            break

    matches: list[Match] = []
    for start in selected:
        segment = prediction[start : start + SEGMENT_SAMPLES]
        raw_segment = None if raw_input is None else raw_input[start : start + SEGMENT_SAMPLES]
        correlation, nrmse, scale, offset = score_segment(target, segment)
        matches.append(
            Match(
                source=source,
                start_sample=start,
                correlation=correlation,
                affine_nrmse=nrmse,
                scale=scale,
                offset=offset,
                prediction=segment,
                raw_input=raw_segment,
            )
        )
    return matches


def save_results(output_dir: Path, target: np.ndarray, matches: list[Match]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    matches.sort(key=lambda item: (-item.correlation, item.affine_nrmse))
    matches = matches[:TOP_COUNT]

    with (output_dir / "ranked_matches.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "source",
                "name",
                "start_sample",
                "start_time_sec",
                "correlation",
                "affine_nrmse",
                "affine_scale",
                "affine_offset",
            ]
        )
        for rank, match in enumerate(matches, start=1):
            writer.writerow(
                [
                    rank,
                    match.source,
                    match.name,
                    match.start_sample,
                    match.start_sample / SAMPLE_RATE,
                    match.correlation,
                    match.affine_nrmse,
                    match.scale,
                    match.offset,
                ]
            )
            np.save(output_dir / f"rank_{rank:02d}_prediction.npy", match.prediction)
            if match.raw_input is not None and match.raw_input.size == SEGMENT_SAMPLES:
                np.save(output_dir / f"rank_{rank:02d}_raw_input.npy", match.raw_input)

    time = np.arange(target.size) / SAMPLE_RATE
    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex=True)
    for rank, (axis, match) in enumerate(zip(axes.flat, matches), start=1):
        fitted = match.scale * match.prediction + match.offset
        axis.plot(time, target, color="black", linewidth=1.0, label="Paper SVG")
        axis.plot(time, fitted, color="#d62728", linewidth=0.8, alpha=0.85, label="Candidate, affine fit")
        axis.set_title(
            f"#{rank}: r={match.correlation:.4f}, NRMSE={match.affine_nrmse:.4f}\n"
            f"{match.source}, t={match.start_sample / SAMPLE_RATE:.3f} s",
            fontsize=9,
        )
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (s)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Amplitude")
    fig.suptitle("Figure S3 forensic segment search: strongest surviving waveform matches")
    fig.tight_layout()
    fig.savefig(output_dir / "top_matches_overlay.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "top_matches_overlay.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    workspace = root.parents[3]
    target = load_rendered_target(root)
    results: list[Match] = []

    reference = workspace / "2026_nature" / "reference_results" / "P12"
    continuous_prediction = np.loadtxt(reference / "continuous_pred_2831.csv", delimiter=",")
    continuous_real = np.loadtxt(reference / "continuous_real_2831.csv", delimiter=",")
    results.extend(
        rank_continuous_trace(
            target,
            continuous_prediction,
            continuous_real,
            "P12 continuous trace",
            TOP_COUNT,
        )
    )

    prediction_windows = np.load(reference / "[PRED_ECG]2025-11-22_23h02min.npy")
    real_windows = np.load(reference / "[REAL_ECG]2025-11-22_23h02min.npy")
    with (reference / "[NAMES_INF]2025-11-22_23h02min.pkl").open("rb") as handle:
        names = pickle.load(handle)
    results.extend(rank_known_windows(target, prediction_windows, real_windows, "P12 inference windows", names))

    save_results(root, target, results)
    print(f"Ranked {len(results)} candidates; results saved to {root}")


if __name__ == "__main__":
    main()