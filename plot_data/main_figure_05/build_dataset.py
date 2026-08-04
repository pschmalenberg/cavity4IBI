from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


SAMPLE_RATE_HZ = 2000
EPOCH_DURATION_SECONDS = 4
PUBLISHED_DURATION_SECONDS = 36
DIFFERENCE_SCALE = 0.15

SOURCE_MAP = {
    "recording_01": (("a.csv", "d.csv"), ("b.csv", "e.csv"), ("c.csv", "f.csv")),
    "recording_02": (("aa.csv", "dd.csv"), ("bb.csv", "ee.csv"), ("cc.csv", "ff.csv")),
    "recording_03": (("aaa.csv", "ddd.csv"), ("bbb.csv", "eee.csv"), ("ccc.csv", "fff.csv")),
}


def normalize_zero_to_one(values: np.ndarray) -> np.ndarray:
    value_range = values.max() - values.min()
    if value_range == 0:
        raise ValueError("Cannot normalize a constant signal")
    return (values - values.min()) / value_range


def load_epoch(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",").reshape(-1)
    expected_samples = SAMPLE_RATE_HZ * EPOCH_DURATION_SECONDS
    if values.size != expected_samples:
        raise ValueError(
            f"{path.name}: expected {expected_samples} samples, found {values.size}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name}: signal contains non-finite values")
    return values


def write_epoch_file(
    output_path: Path,
    measured: np.ndarray,
    predicted: np.ndarray,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample_index", "time_seconds", "measured_ecg", "predicted_ecg"))
        for sample_index, (measured_value, predicted_value) in enumerate(
            zip(measured, predicted, strict=True)
        ):
            writer.writerow(
                (
                    sample_index,
                    f"{sample_index / SAMPLE_RATE_HZ:.7f}",
                    f"{measured_value:.17g}",
                    f"{predicted_value:.17g}",
                )
            )


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    epoch_dir = data_dir / "epochs"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    plot_data_path = data_dir / "figure5_plot_data.csv"
    source_manifest_rows = []
    recording_rows = []

    for recording_number, (recording_id, epoch_pairs) in enumerate(SOURCE_MAP.items(), start=1):
        measured_epochs = []
        predicted_epochs = []

        for epoch_number, (measured_name, predicted_name) in enumerate(epoch_pairs, start=1):
            measured = load_epoch(root / measured_name)
            predicted = load_epoch(root / predicted_name)
            measured_epochs.append(measured)
            predicted_epochs.append(predicted)

            professional_name = f"{recording_id}_epoch_{epoch_number:02d}.csv"
            write_epoch_file(epoch_dir / professional_name, measured, predicted)
            source_manifest_rows.append(
                (
                    recording_id,
                    epoch_number,
                    professional_name,
                    measured_name,
                    predicted_name,
                    measured.size,
                    SAMPLE_RATE_HZ,
                    EPOCH_DURATION_SECONDS,
                )
            )

        measured_raw = np.concatenate(measured_epochs)
        predicted_raw = np.concatenate(predicted_epochs)
        measured_normalized = normalize_zero_to_one(measured_raw)
        predicted_normalized = normalize_zero_to_one(predicted_raw)
        difference_scaled = DIFFERENCE_SCALE * (predicted_normalized - measured_normalized)
        native_time = np.arange(measured_raw.size) / SAMPLE_RATE_HZ
        published_time = np.linspace(0, PUBLISHED_DURATION_SECONDS, measured_raw.size)

        for sample_index in range(measured_raw.size):
            recording_rows.append(
                (
                    recording_id,
                    recording_number,
                    sample_index // (SAMPLE_RATE_HZ * EPOCH_DURATION_SECONDS) + 1,
                    sample_index,
                    f"{native_time[sample_index]:.7f}",
                    f"{published_time[sample_index]:.7f}",
                    f"{measured_raw[sample_index]:.17g}",
                    f"{predicted_raw[sample_index]:.17g}",
                    f"{measured_normalized[sample_index]:.17g}",
                    f"{predicted_normalized[sample_index]:.17g}",
                    f"{difference_scaled[sample_index]:.17g}",
                )
            )

    with plot_data_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "recording_id",
                "recording_number",
                "epoch_number",
                "sample_index",
                "native_time_seconds",
                "published_time_seconds",
                "measured_ecg_raw",
                "predicted_ecg_raw",
                "measured_ecg_normalized",
                "predicted_ecg_normalized",
                "difference_scaled",
            )
        )
        writer.writerows(recording_rows)

    with (data_dir / "source_file_map.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "recording_id",
                "epoch_number",
                "professional_file",
                "legacy_measured_file",
                "legacy_predicted_file",
                "samples",
                "sample_rate_hz",
                "duration_seconds",
            )
        )
        writer.writerows(source_manifest_rows)

    metadata = {
        "title": "Main Figure 5 ECG reconstruction data",
        "description": "Measured and neural-network reconstructed ECG epochs used in Main Figure 5.",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "epoch_duration_seconds": EPOCH_DURATION_SECONDS,
        "epochs_per_recording": 3,
        "native_recording_duration_seconds": 12,
        "published_axis_duration_seconds": PUBLISHED_DURATION_SECONDS,
        "published_axis_note": (
            "The retained MATLAB producer uses linspace(0, 36, 24000). At 2 kHz, "
            "24000 samples contain 12 seconds; both axes are retained explicitly."
        ),
        "normalization": "Independent min-max normalization over each 24000-sample measured or predicted recording.",
        "difference": "0.15 * (predicted_ecg_normalized - measured_ecg_normalized)",
        "recording_labels": "Neutral identifiers; participant or source recording names were not retained with the legacy files.",
    }
    (data_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(source_manifest_rows)} paired epochs and {len(recording_rows)} plot rows")


if __name__ == "__main__":
    main()