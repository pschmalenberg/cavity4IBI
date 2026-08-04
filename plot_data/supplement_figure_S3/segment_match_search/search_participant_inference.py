from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from scipy.signal import correlate


SAMPLE_RATE = 2000
SEGMENT_SAMPLES = 4 * SAMPLE_RATE
HOP_SAMPLES = SEGMENT_SAMPLES // 2
TOP_COUNT = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search participant inference for the Figure S3 waveform.")
    parser.add_argument("participant", help="Participant folder name, for example P11")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def load_target(script_dir: Path) -> np.ndarray:
    path = script_dir.parent / "retained_source_export" / "rendered_upper_samples.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        return np.asarray(
            [float(row["predicted_ecg_rendered"]) for row in csv.DictReader(handle)],
            dtype=np.float64,
        )


def load_inputs(input_dir: Path) -> tuple[list[Path], torch.Tensor]:
    paths = sorted(input_dir.glob("*.wav"))
    signals: list[torch.Tensor] = []
    for path in paths:
        signal, sample_rate = torchaudio.load(path)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Unexpected sample rate {sample_rate} in {path}")
        signal = signal.squeeze()[:SEGMENT_SAMPLES]
        signal = signal / signal.max()
        if signal.numel() != SEGMENT_SAMPLES:
            raise ValueError(f"Unexpected sample count {signal.numel()} in {path}")
        signals.append(signal)
    return paths, torch.stack(signals)


def run_inference(workspace: Path, inputs: torch.Tensor, batch_size: int) -> np.ndarray:
    sys.path.insert(0, str(workspace))
    from models import ConvTasNet

    checkpoint = workspace / "ckpt" / "2025-11-22_23h02min" / "ckpt" / "2025-11-22_23h02min.pth"
    model = ConvTasNet()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            prediction = model(inputs[start : start + batch_size])[0]
            prediction = torchaudio.functional.lowpass_biquad(prediction, SAMPLE_RATE, 20)
            predictions.extend(prediction.cpu().numpy())
            print(f"Processed {min(start + batch_size, len(inputs))}/{len(inputs)} windows")
    return np.asarray(predictions)


def reconstruct_continuous(windows: np.ndarray) -> np.ndarray:
    continuous = np.zeros(len(windows) * HOP_SAMPLES + SEGMENT_SAMPLES, dtype=np.float64)
    for index, window in enumerate(windows):
        start = index * HOP_SAMPLES
        continuous[start : start + SEGMENT_SAMPLES] += window
    continuous[:HOP_SAMPLES] *= 2
    continuous = continuous[: len(windows) * HOP_SAMPLES]
    return continuous / np.max(np.abs(continuous))


def normalized_correlation_scan(target: np.ndarray, signal: np.ndarray) -> np.ndarray:
    target_centered = target - target.mean()
    numerator = correlate(signal, target_centered, mode="valid", method="fft")
    cumulative = np.concatenate(([0.0], np.cumsum(signal, dtype=np.float64)))
    cumulative_squares = np.concatenate(([0.0], np.cumsum(signal * signal, dtype=np.float64)))
    window_sum = cumulative[target.size :] - cumulative[: -target.size]
    window_square_sum = cumulative_squares[target.size :] - cumulative_squares[: -target.size]
    window_energy = np.maximum(window_square_sum - window_sum * window_sum / target.size, 0.0)
    denominator = np.sqrt(window_energy) * np.linalg.norm(target_centered)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def select_matches(target: np.ndarray, continuous: np.ndarray) -> list[tuple[int, float, float, float, float]]:
    correlations = normalized_correlation_scan(target, continuous)
    order = np.argsort(correlations)[::-1]
    starts: list[int] = []
    for start in order:
        start = int(start)
        if all(abs(start - existing) >= HOP_SAMPLES for existing in starts):
            starts.append(start)
        if len(starts) == TOP_COUNT:
            break

    results: list[tuple[int, float, float, float, float]] = []
    for start in starts:
        candidate = continuous[start : start + SEGMENT_SAMPLES]
        design = np.column_stack((candidate, np.ones(candidate.size)))
        scale, offset = np.linalg.lstsq(design, target, rcond=None)[0]
        fitted = scale * candidate + offset
        nrmse = np.sqrt(np.mean((target - fitted) ** 2)) / np.ptp(target)
        results.append((start, float(correlations[start]), float(nrmse), float(scale), float(offset)))
    return results


def save_results(
    output_dir: Path,
    participant: str,
    paths: list[Path],
    windows: np.ndarray,
    continuous: np.ndarray,
    target: np.ndarray,
    matches: list[tuple[int, float, float, float, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "prediction_windows.npy", windows)
    np.save(output_dir / "continuous_prediction.npy", continuous)
    (output_dir / "window_names.txt").write_text("\n".join(path.name for path in paths), encoding="utf-8")

    with (output_dir / "ranked_matches.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["rank", "participant", "start_sample", "start_time_sec", "correlation", "affine_nrmse", "scale", "offset"]
        )
        for rank, (start, correlation, nrmse, scale, offset) in enumerate(matches, start=1):
            writer.writerow([rank, participant, start, start / SAMPLE_RATE, correlation, nrmse, scale, offset])
            np.save(output_dir / f"rank_{rank:02d}_prediction.npy", continuous[start : start + SEGMENT_SAMPLES])

    time = np.arange(SEGMENT_SAMPLES) / SAMPLE_RATE
    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex=True)
    for rank, (axis, match) in enumerate(zip(axes.flat, matches), start=1):
        start, correlation, nrmse, scale, offset = match
        candidate = continuous[start : start + SEGMENT_SAMPLES]
        axis.plot(time, target, color="black", linewidth=1.0, label="Paper SVG")
        axis.plot(time, scale * candidate + offset, color="#d62728", linewidth=0.8, label="Candidate, affine fit")
        axis.set_title(f"#{rank}: r={correlation:.4f}, NRMSE={nrmse:.4f}\nt={start / SAMPLE_RATE:.3f} s", fontsize=9)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (s)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Amplitude")
    fig.suptitle(f"Figure S3 candidate search: {participant} model predictions")
    fig.tight_layout()
    fig.savefig(output_dir / "top_matches_overlay.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "top_matches_overlay.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    workspace = script_dir.parents[3]
    data_root = Path(r"C:\Users\Admin\Desktop\2025 Cavity\2025 Cavity Data Collection")
    input_dir = data_root / "raw_ECG_inference_dataset" / "Inference_by_p" / args.participant / "[input]"
    output_dir = script_dir / args.participant
    output_dir.mkdir(parents=True, exist_ok=True)

    target = load_target(script_dir)
    paths, inputs = load_inputs(input_dir)
    prediction_cache = output_dir / "prediction_windows.npy"
    if prediction_cache.exists():
        predictions = np.load(prediction_cache)
        print(f"Loaded cached predictions from {prediction_cache}")
    else:
        predictions = run_inference(workspace, inputs, args.batch_size)
        np.save(prediction_cache, predictions)
        print(f"Cached predictions at {prediction_cache}")
    del inputs
    gc.collect()
    continuous = reconstruct_continuous(predictions)
    matches = select_matches(target, continuous)
    save_results(output_dir, args.participant, paths, predictions, continuous, target, matches)
    print(f"Best {args.participant} correlation: {matches[0][1]:.6f}")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()