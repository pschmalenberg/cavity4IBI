from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from loaders import CavityDataset
from models import ConvTasNet


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the released Conv-TasNet checkpoint.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Processed directory containing [input] and [wide_gaussian].",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "ckpt" / "2025-11-22_23h02min.pth",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dataset = CavityDataset(str(args.dataset_root), args.split, sr=2000)
    if len(dataset) == 0:
        raise RuntimeError(f"No {args.split} windows found under {args.dataset_root}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = ConvTasNet(N=512, L=16, B=128, H=512, P=3, X=8, R=3, num_spks=1)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    predictions: list[np.ndarray] = []
    references: list[np.ndarray] = []
    with torch.inference_mode():
        for pcg, ecg, _ in tqdm(loader, desc="Inference"):
            prediction = model(pcg.to(device))[0]
            predictions.append(prediction.cpu().numpy())
            references.append(ecg.numpy())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = args.checkpoint.stem
    prediction_path = args.output_dir / f"[PRED_ECG]{checkpoint_name}.npy"
    reference_path = args.output_dir / f"[REAL_ECG]{checkpoint_name}.npy"
    names_path = args.output_dir / f"[NAMES_INF]{checkpoint_name}.pkl"

    np.save(prediction_path, np.concatenate(predictions, axis=0))
    np.save(reference_path, np.concatenate(references, axis=0))
    names = [Path(path).stem for path in dataset.pcg_paths]
    with names_path.open("wb") as file:
        pickle.dump(names, file)

    print(f"Saved {len(dataset)} predictions to {args.output_dir}")


if __name__ == "__main__":
    main()