from __future__ import annotations

import csv
import hashlib
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "ckpt" / "2025-11-22_23h02min.pth"
EXPECTED_CHECKPOINT_SIZE = 45_260_109
EXPECTED_CHECKPOINT_SHA256 = (
    "14732b97b405d781406278bc5c4be0f5e91e1acc4888f0c209bf12d8c23ef5c7"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_similarity_metrics(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = csv.DictReader(file)
        return {row["Metric"]: float(row["Value"]) for row in rows}


def extract_number(text: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}:\s*([0-9.]+)", text)
    if match is None:
        raise ValueError(f"Missing metric: {label}")
    return float(match.group(1))


def require_close(actual: float, expected: float, label: str, tolerance: float = 1e-9) -> None:
    if abs(actual - expected) > tolerance:
        raise ValueError(f"{label}: expected {expected}, found {actual}")


def verify_source_manifest(manifest_path: Path, source_dir: Path) -> None:
    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    listed_names = {row["filename"] for row in rows}
    actual_names = {path.name for path in source_dir.glob("*.m")}
    if len(rows) != 14 or listed_names != actual_names:
        raise ValueError("MATLAB source manifest does not match the 14 included .m files")

    for row in rows:
        path = source_dir / row["filename"]
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"MATLAB source size does not match: {path.name}")
        if sha256(path) != row["sha256"]:
            raise ValueError(f"MATLAB source SHA-256 does not match: {path.name}")


def main() -> int:
    required = [
        ROOT / "models" / "convtasnet.py",
        ROOT / "data" / "figshare_manifest.csv",
        ROOT / "preprocessing_matlab" / "source_manifest.csv",
        ROOT / "reference_results" / "P12" / "continuous_pred_2831.csv",
        ROOT / "reference_results" / "P12" / "continuous_real_2831.csv",
        ROOT / "reference_results" / "P12" / "[NAMES_INF]2025-11-22_23h02min.pkl",
        ROOT / "reference_results" / "P11" / "global_HRI_and_HR.txt",
        ROOT / "reference_results" / "P11" / "similarity_metrics.csv",
        CHECKPOINT,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))

    if CHECKPOINT.stat().st_size != EXPECTED_CHECKPOINT_SIZE:
        raise ValueError("Checkpoint size does not match the manifest")
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Checkpoint SHA-256 does not match the manifest")

    manifest_path = ROOT / "data" / "figshare_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as file:
        recordings = list(csv.DictReader(file))
    if len(recordings) != 13:
        raise ValueError(f"Expected 13 Figshare recordings, found {len(recordings)}")
    expected_participants = {f"P{index}" for index in range(1, 14)}
    if {row["participant"] for row in recordings} != expected_participants:
        raise ValueError("Figshare manifest does not contain exactly P1-P13")

    verify_source_manifest(
        ROOT / "preprocessing_matlab" / "source_manifest.csv",
        ROOT / "preprocessing_matlab",
    )

    summary_dir = ROOT / "reference_results" / "P11"
    global_text = (summary_dir / "global_HRI_and_HR.txt").read_text(encoding="utf-8")
    require_close(extract_number(global_text, "Global MAE"), 0.022837349397590363, "MAE")
    require_close(extract_number(global_text, "Ground truth HR"), 74.2, "Ground-truth HR")
    require_close(extract_number(global_text, "PREDICTION HR"), 66.8, "Predicted HR")

    similarity = read_similarity_metrics(summary_dir / "similarity_metrics.csv")
    require_close(similarity["Pearson Correlation"], 0.7408, "Pearson correlation")
    require_close(similarity["DTW Normalized Distance"], 0.045654, "DTW")
    require_close(similarity["Downsample Factor (DTW)"], 10.0, "DTW downsample factor")

    print(
        "PASS: release files, checkpoint, data and MATLAB manifests, "
        "and retained summaries verified"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, OSError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)