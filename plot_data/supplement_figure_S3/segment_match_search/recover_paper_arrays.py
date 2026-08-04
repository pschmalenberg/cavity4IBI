from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom


SAMPLE_RATE = 2000
SEGMENT_SAMPLES = 4 * SAMPLE_RATE
HOP_SAMPLES = SEGMENT_SAMPLES // 2
PARTICIPANT = "P13"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rendered_data(script_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = script_dir.parent / "retained_source_export" / "rendered_upper_samples.csv"
    predicted: list[float] = []
    importance: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            predicted.append(float(row["predicted_ecg_rendered"]))
            importance.append(float(row["jet_colormap_position_approx"]))
    return np.asarray(predicted), np.asarray(importance)


def reconstruct_continuous(windows: np.ndarray) -> np.ndarray:
    continuous = np.zeros(len(windows) * HOP_SAMPLES + SEGMENT_SAMPLES, dtype=np.float64)
    for index, window in enumerate(windows):
        start = index * HOP_SAMPLES
        continuous[start : start + SEGMENT_SAMPLES] += np.asarray(window).squeeze()
    continuous[:HOP_SAMPLES] *= 2
    continuous = continuous[: len(windows) * HOP_SAMPLES]
    return continuous / np.max(np.abs(continuous))


def load_reference_windows(label_dir: Path) -> tuple[list[Path], np.ndarray]:
    paths = sorted(label_dir.glob("*.csv"))
    windows = []
    for path in paths:
        frame = pd.read_csv(path)
        window = frame.iloc[:SEGMENT_SAMPLES, 1].to_numpy(dtype=np.float64)
        if window.size != SEGMENT_SAMPLES:
            raise ValueError(f"Unexpected sample count {window.size} in {path}")
        windows.append(window)
    return paths, np.asarray(windows)


def recompute_grad_cam(workspace: Path, input_segment: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    sys.path.insert(0, str(workspace))
    from models import ConvTasNet

    checkpoint = workspace / "ckpt" / "2025-11-22_23h02min" / "ckpt" / "2025-11-22_23h02min.pth"
    model = ConvTasNet()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        captured["activations"] = output

    def backward_hook(_module, _grad_inputs, grad_outputs):
        captured["gradients"] = grad_outputs[0]

    target_layer = model.gen_masks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    input_tensor = torch.tensor(input_segment, dtype=torch.float32).unsqueeze(0)
    input_tensor.requires_grad_(True)
    output = model(input_tensor)[0]
    model.zero_grad()
    output.sum().backward()
    forward_handle.remove()
    backward_handle.remove()

    activations = captured["activations"]
    gradients = captured["gradients"]
    weights = gradients.mean(dim=-1, keepdim=True)
    raw = torch.relu((weights * activations).sum(dim=1)).squeeze().detach().numpy()
    upsampled = zoom(raw, SEGMENT_SAMPLES / len(raw))
    upsampled /= np.max(upsampled) + 1e-8
    return raw, upsampled, "gen_masks"


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    workspace = script_dir.parents[3]
    participant_dir = script_dir / PARTICIPANT
    output_dir = script_dir / "recovered_paper_arrays"
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_windows = np.load(participant_dir / "prediction_windows.npy")
    continuous_prediction = reconstruct_continuous(prediction_windows)
    label_dir = Path(
        r"C:\Users\Admin\Desktop\2025 Cavity\2025 Cavity Data Collection"
    ) / "raw_ECG_inference_dataset" / "Inference_by_p" / PARTICIPANT / "[wide_gaussian]"
    label_paths, reference_windows = load_reference_windows(label_dir)
    continuous_reference = reconstruct_continuous(reference_windows)

    segment_start = len(continuous_prediction) // 2
    segment_end = segment_start + SEGMENT_SAMPLES
    prediction_segment = continuous_prediction[segment_start:segment_end]
    input_segment = continuous_reference[segment_start:segment_end]
    rendered_prediction, rendered_importance = load_rendered_data(script_dir)

    prediction_error = prediction_segment - rendered_prediction
    prediction_correlation = float(np.corrcoef(prediction_segment, rendered_prediction)[0, 1])
    prediction_max_error = float(np.max(np.abs(prediction_error)))
    prediction_mae = float(np.mean(np.abs(prediction_error)))

    grad_cam_raw, grad_cam_upsampled, target_layer = recompute_grad_cam(workspace, input_segment)
    importance_correlation = float(np.corrcoef(grad_cam_upsampled, rendered_importance)[0, 1])
    threshold_agreement = float(
        np.mean((grad_cam_upsampled > 0.3) == (rendered_importance > 0.3))
    )

    output_paths = {
        "input_segment": output_dir / "paper_input_segment.npy",
        "prediction_segment": output_dir / "paper_prediction_segment.npy",
        "grad_cam_raw": output_dir / "paper_grad_cam_raw.npy",
        "grad_cam_upsampled": output_dir / "paper_grad_cam_upsampled.npy",
        "reference_windows": output_dir / "P13_reference_windows.npy",
        "prediction_windows": output_dir / "P13_prediction_windows.npy",
    }
    np.save(output_paths["input_segment"], input_segment)
    np.save(output_paths["prediction_segment"], prediction_segment)
    np.save(output_paths["grad_cam_raw"], grad_cam_raw)
    np.save(output_paths["grad_cam_upsampled"], grad_cam_upsampled)
    np.save(output_paths["reference_windows"], reference_windows)
    np.save(output_paths["prediction_windows"], prediction_windows)

    metadata = {
        "status": "recovered by exact waveform match to retained paper SVG",
        "participant": PARTICIPANT,
        "recording": label_paths[0].name.split("[")[1:4],
        "sample_rate_hz": SAMPLE_RATE,
        "segment_start_sample": segment_start,
        "segment_start_time_sec_within_reconstructed_trace": segment_start / SAMPLE_RATE,
        "segment_samples": SEGMENT_SAMPLES,
        "target_layer": target_layer,
        "backward_target": "sum of ConvTasNet output waveform",
        "prediction_svg_correlation": prediction_correlation,
        "prediction_svg_mae": prediction_mae,
        "prediction_svg_max_abs_error": prediction_max_error,
        "grad_cam_vs_svg_color_approx_correlation": importance_correlation,
        "grad_cam_threshold_0_3_agreement": threshold_agreement,
        "label_first_file": label_paths[0].name,
        "label_last_file": label_paths[-1].name,
        "checkpoint": "ckpt/2025-11-22_23h02min/ckpt/2025-11-22_23h02min.pth",
        "files": {},
    }
    for name, path in output_paths.items():
        metadata["files"][name] = {
            "path": path.name,
            "sha256": sha256(path),
            "shape": list(np.load(path, mmap_mode="r").shape),
        }
    (output_dir / "recovery_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Prediction correlation with paper SVG: {prediction_correlation:.15f}")
    print(f"Prediction maximum absolute error: {prediction_max_error:.3e}")
    print(f"Grad-CAM correlation with SVG color approximation: {importance_correlation:.6f}")
    print(f"Grad-CAM threshold agreement: {threshold_agreement:.6f}")
    print(f"Recovered arrays saved to {output_dir}")


if __name__ == "__main__":
    main()