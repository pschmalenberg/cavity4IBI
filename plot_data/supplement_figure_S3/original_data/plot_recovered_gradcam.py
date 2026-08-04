from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageDraw


SAMPLE_RATE = 2000
HIGH_ACTIVATION_THRESHOLD = 0.3


def draw_overlay(axis, time, prediction, grad_cam, add_colorbar=True):
    axis.plot(time, prediction, color="blue", linewidth=0.8, label="Predicted ECG")
    axis.fill_between(
        time,
        prediction.min(),
        prediction.max(),
        where=grad_cam > HIGH_ACTIVATION_THRESHOLD,
        alpha=0.3,
        color="red",
        label="High Grad-CAM activation",
    )
    scatter = axis.scatter(time, prediction, c=grad_cam, cmap="jet", s=1, zorder=5)
    axis.set_ylabel("Amplitude")
    axis.set_title("Grad-CAM on Predicted ECG (layer: gen_masks)")
    axis.legend(loc="upper right")
    axis.grid(True)
    if add_colorbar:
        plt.colorbar(scatter, ax=axis, label="Grad-CAM Importance")
    return scatter


def draw_activation_map(axis, time, reference, prediction, grad_cam):
    scale = np.max(np.abs(prediction))
    axis.plot(time, reference, color="green", linewidth=0.8, alpha=0.7, label="Real ECG")
    axis.plot(time, prediction, color="blue", linewidth=0.8, alpha=0.7, label="Predicted ECG")
    axis.fill_between(
        time,
        0,
        grad_cam * scale,
        step="mid",
        alpha=0.3,
        color="red",
        label="Grad-CAM weight",
    )
    axis.set_xlabel("Time [sec]")
    axis.set_ylabel("Amplitude / Importance")
    axis.set_title("Grad-CAM Activation Map vs Real & Predicted ECG")
    axis.legend(loc="upper right")
    axis.grid(True)


def make_confirmation_sheet(original_path: Path, regenerated_path: Path, output_path: Path):
    original = Image.open(original_path).convert("RGB")
    regenerated = Image.open(regenerated_path).convert("RGB")
    target_height = max(original.height, regenerated.height)

    def fit_height(image):
        width = round(image.width * target_height / image.height)
        return image.resize((width, target_height), Image.Resampling.LANCZOS)

    original = fit_height(original)
    regenerated = fit_height(regenerated)
    header = 80
    gap = 20
    sheet = Image.new(
        "RGB",
        (original.width + regenerated.width + gap, target_height + header),
        "white",
    )
    sheet.paste(original, (0, header))
    sheet.paste(regenerated, (original.width + gap, header))
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 25), "Retained paper source", fill="black")
    draw.text((original.width + gap + 20, 25), "Regenerated from recovered P13 arrays", fill="black")
    sheet = ImageOps.contain(sheet, (6000, 2400))
    sheet.save(output_path, quality=95)


def main():
    root = Path(__file__).resolve().parent
    prediction = np.load(root / "paper_prediction_segment.npy")
    reference = np.load(root / "paper_input_segment.npy")
    grad_cam = np.load(root / "paper_grad_cam_upsampled.npy")
    time = np.arange(prediction.size) / SAMPLE_RATE

    fig, axes = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
    draw_overlay(axes[0], time, prediction, grad_cam)
    draw_activation_map(axes[1], time, reference, prediction, grad_cam)
    fig.tight_layout()
    combined_png = root / "recovered_grad_cam_predicted_ecg.png"
    fig.savefig(combined_png, dpi=150, bbox_inches="tight")
    fig.savefig(root / "recovered_grad_cam_predicted_ecg.svg", bbox_inches="tight")
    plt.close(fig)

    original = root.parent / "retained_source_export" / "grad_cam_predicted_ecg.png"
    make_confirmation_sheet(
        original,
        combined_png,
        root / "retained_vs_recovered_comparison.png",
    )

    fig, axis = plt.subplots(figsize=(20, 5))
    draw_overlay(axis, time, prediction, grad_cam)
    axis.set_xlabel("Time [sec]")
    fig.tight_layout()
    fig.savefig(root / "recovered_grad_cam_overlay.png", dpi=300, bbox_inches="tight")
    fig.savefig(root / "recovered_grad_cam_overlay.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(20, 5))
    draw_activation_map(axis, time, reference, prediction, grad_cam)
    fig.tight_layout()
    fig.savefig(root / "recovered_grad_cam_activation_map.png", dpi=300, bbox_inches="tight")
    fig.savefig(root / "recovered_grad_cam_activation_map.svg", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved regenerated figures to {root}")


if __name__ == "__main__":
    main()