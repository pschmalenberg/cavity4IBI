from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROW_OFFSETS = {
    "recording_01": (0.0, 0.5),
    "recording_02": (-1.0, -0.5),
    "recording_03": (-2.0, -1.5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the structured Main Figure 5 dataset.")
    parser.add_argument(
        "--time-axis",
        choices=("published", "native"),
        default="published",
        help="Use the retained 0-36 s paper axis or the 0-12 s axis implied by 2 kHz sampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "figure_05_python.png",
    )
    return parser.parse_args()


def load_plot_data(path: Path) -> dict[str, dict[str, np.ndarray]]:
    columns = (
        "native_time_seconds",
        "published_time_seconds",
        "measured_ecg_normalized",
        "predicted_ecg_normalized",
        "difference_scaled",
    )
    grouped: dict[str, dict[str, list[float]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            values = grouped.setdefault(row["recording_id"], {column: [] for column in columns})
            for column in columns:
                values[column].append(float(row[column]))
    return {
        recording_id: {column: np.asarray(values) for column, values in columns_by_name.items()}
        for recording_id, columns_by_name in grouped.items()
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    recordings = load_plot_data(root / "data" / "figure5_plot_data.csv")
    time_column = f"{args.time_axis}_time_seconds"

    fig, ax = plt.subplots(figsize=(12, 7))
    for recording_id, values in recordings.items():
        signal_offset, difference_offset = ROW_OFFSETS[recording_id]
        ax.plot(
            values[time_column],
            values["measured_ecg_normalized"] + signal_offset,
            color="#77AC30",
            alpha=0.75,
            linewidth=1.5,
            label="ECG Measurement" if recording_id == "recording_01" else None,
        )
        ax.plot(
            values[time_column],
            values["predicted_ecg_normalized"] + signal_offset,
            color="#D95319",
            alpha=0.75,
            linestyle=":",
            linewidth=1.5,
            label="ECG Prediction" if recording_id == "recording_01" else None,
        )
        ax.plot(
            values[time_column],
            values["difference_scaled"] + difference_offset,
            color="black",
            linewidth=1.5,
            label="Difference" if recording_id == "recording_01" else None,
        )

    duration = 36 if args.time_axis == "published" else 12
    ax.set_xlim(0, duration)
    ax.set_yticks([])
    ax.set_xlabel("Seconds [s]")
    ax.set_ylabel("Normalized Value [arb. unit]")
    ax.grid(axis="y", color="black", linestyle="--", alpha=0.75)
    ax.legend(loc="upper right")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"Saved {args.output} using the {args.time_axis} time axis")


if __name__ == "__main__":
    main()