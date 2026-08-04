from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SAMPLE_RATE = 2000
PUBLISHED_DURATION_SECONDS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the retained continuous ECG excerpt used for Figure S1."
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "figure_S1.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real = np.loadtxt(args.data_dir / "continuous_real_2831.csv", delimiter=",")
    prediction = np.loadtxt(
        args.data_dir / "continuous_pred_2831.csv", delimiter=","
    )
    if real.shape != prediction.shape:
        raise ValueError(f"Trace shapes differ: {real.shape} and {prediction.shape}")

    sample_count = SAMPLE_RATE * PUBLISHED_DURATION_SECONDS
    if real.size < sample_count:
        raise ValueError(f"Expected at least {sample_count} samples, found {real.size}")

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(real[:sample_count], label="real ecg")
    ax.plot(prediction[:sample_count], label="prediction")
    ax.set_xlabel("Time[sec]")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()