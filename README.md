<div align="center">

<img src="assets/banner.png" alt="Cavity4IBI Banner" width="200"/>

# Cavity4IBI: Non-Invasive Cardiac Sensing via Acoustic Helmholtz Resonator Cavity

[![Python](https://img.shields.io/badge/Python-3.11.4-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the source code, data processing pipelines, trained model checkpoints, evaluation scripts, numerical models, circuit model, and figure processing for the accompanying the paper: (LINK to paper TBD)

> *Toyota Research Institute of North America (TRINA)*

</div>

---
## Overview

**Cavity4IBI** is the software component of an end-to-end non-invasive cardiac monitoring system that combines a custom-designed **acoustic Helmholtz resonator cavity** with deep neural networks to reconstruct electrocardiogram (ECG) waveforms directly from acoustic heart sound signals. The Helmholtz resonator cavity is an acoustic sensor tuned to the [50–120] Hz frequency range where cardiac-generated pressure waves reside. A pressure-acoustic sensor inside the cavity captures heart sounds through layers of clothing and seat foam without any direct body contact. The captured acoustic signal is then processed by a **Conv-TasNet** (Convolutional Time-domain Audio Separation Network, originally developed for speech separation) which is repurposed here to reconstruct the full ECG waveform including identifiable PQRST morphology.

---

## Table of Contents

- [What Can Be Reproduced](#what-can-be-reproduced)
- [Submission Documentation](#submission-documentation)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Datasets](#datasets)
  - [Supported Data Sources](#supported-data-sources)
  - [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [Models](#models)
- [Training](#training)
- [Inference](#inference)
  - [Single-File Prediction](#single-file-prediction)
  - [Analysis Notebooks](#analysis-notebooks)
- [Citation](#citation)
- [Related Work](#related-work)
- [Reproducibility Boundaries](#reproducibility-boundaries)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## What Can Be Reproduced

| Level | Inputs | Command or tool |
|-------|--------|-----------------|
| Inference | Checkpoint + processed Figshare data | `python run_inference.py --dataset-root PATH` |
| Training | Processed train/validation data | `python wav2ecg.py` |
| Raw preprocessing | Figshare recordings | `preprocessing_matlab/` (MATLAB required) |
| Circuit model | Included MATLAB source | `circuit_model/circuit_hemholtz.m` (MATLAB required) |
| Numerical model | Included `.mph` files | COMSOL Multiphysics 6.3 required |

---

## Submission Documentation

- [`NATURE_ML_REPORTING_SUMMARY.md`](NATURE_ML_REPORTING_SUMMARY.md) — evidence-linked answers for every Nature Machine Learning Checklist V1.1 item
- [`ML-reporting-summary-completed.pdf`](ML-reporting-summary-completed.pdf) — completed checklist
- [`DATASET_CARD.md`](DATASET_CARD.md) — dataset identity, collection scope, exclusions, splits, intended use
- [`MODEL_CARD.md`](MODEL_CARD.md) — model scope, architecture, checkpoint, evaluation, limitations
- [`COMPUTE_RESOURCES.md`](COMPUTE_RESOURCES.md) — reported hardware and compute costs

---

## Installation

### Prerequisites

- **Python:** 3.11.4 (verified baseline)
- **CUDA:** Optional (CPU inference supported; GPU recommended for training). The pinned requirements select CUDA 11.7 wheels; use the corresponding CPU wheels on machines without CUDA.
- **MATLAB:** Required only for raw data preprocessing and figure generation
- **COMSOL Multiphysics 6.3:** Required only to open the numerical models
- **Conda:** [Anaconda](https://www.anaconda.com/)

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/pschmalenberg/cavity4IBI.git
   cd cavity4IBI
   ```

2. **Create and activate a conda environment:**

   ```bash
   conda create -n cavity4ibi python=3.11.4
   conda activate cavity4ibi
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   Or install the package in editable mode with `pip install -e .`. The verified stack is:

   | Package | Version |
   |---------|---------|
   | `torch` | 2.0.1 |
   | `torchaudio` | 2.0.2 |
   | `auraloss` | 0.4.0 |
   | `neurokit2` | 0.2.5 |
   | `dtw-python` | 1.5.3 |
   | `scipy` | 1.10.0 |
   | `numpy` | 1.25.2 |
   | `pandas` | 2.0.3 |
   | `matplotlib` | 3.7.2 |
   | `tqdm` | 4.66.0 |

---

## Project Structure

```
cavity4IBI/
├── wav2ecg.py                          # Main training script
├── run_inference.py                    # Portable inference CLI (recommended)
├── predict.py                          # Single-file ECG prediction from audio
├── inference.py                        # Legacy batch inference (analysis provenance)
├── inference_mc_mae.py                 # Legacy multi-channel MAE analysis
├── print_global_mae.py                 # Legacy global HRI MAE analysis
├── methods.py                          # Auxiliary helper methods
├── utils.py                            # Global configuration and parameters
├── setup.py                            # Package installation script
├── requirements.txt                    # Pinned dependencies
│
├── loaders/                            # Dataset loading classes
│   ├── __init__.py                     # get_dataset()
│   └── cavity.py                       # CavityDataset (4 kHz pressure-acoustic signal)
│
├── models/                             # Neural network architectures
│   ├── __init__.py                     # Model exports
│   └── convtasnet.py                   # Conv-TasNet
│
├── metrics/                            # Evaluation metrics
│   ├── heart_rate.py                   # RR intervals and heart rate (BPM)
│   ├── mse_rec.py                      # Mean Squared Error (MSE)
│   └── r_peaks.py                      # R-peak localization accuracy
│
├── preprocessing_matlab/               # Raw → processed dataset generation (MATLAB)
├── ckpt/                               # Trained checkpoint (.pth)
├── circuit_model/                      # MATLAB Helmholtz circuit model
├── numerical_model/                    # COMSOL 6.3 model files
└── notebooks/                          # Exploratory / evaluation notebooks
```

---

## Configuration

All global parameters are configured in **`utils.py`**. Modify these before training or inference:

```python
# Data parameters
data_name      = "cavity_data"    # Only "cavity_data" is available in this release
test_name      = "cavity_data"    # Test dataset
sample_rate    = 2000             # Target sample rate (Hz)
batch_size     = 8                # Training batch size
peak_threshold = 0.4              # Threshold for R-peak detection

# Model parameters
model_name    = "conv-tasnet"     # Only "conv-tasnet" is available in this release
learning_rate = 1e-5              # Initial learning rate
num_epochs    = 50                # Number of training epochs
```

Paths are **not** hard-coded. Set them via environment variables (they default to
`./data/processed` and `./outputs`):

```bash
export CAVITY_DATASET_DIR=/path/to/processed   # dataset root
export CAVITY_OUTPUT_DIR=/path/to/outputs      # checkpoint save directory
```

```powershell
$env:CAVITY_DATASET_DIR = "C:\path\to\processed"
$env:CAVITY_OUTPUT_DIR  = "C:\path\to\outputs"
```

---

## Datasets

### Supported Data Sources

Raw synchronized recordings are hosted on Figshare rather than duplicated in Git:
<https://doi.org/10.6084/m9.figshare.31855450>

The record contains 13 recordings (P1–P13), approximately 2.96 GB total. File sizes, MD5 hashes, direct URLs, and participant mappings are published with the Figshare record.

| Dataset | Sensor Type | Native Sample Rate | Format | Description |
|---------|------------|-------------------|--------|-------------|
| **Cavity** (`cavity_data`) | Acoustic Helmholtz resonator cavity | 4,000 Hz | `.wav` (PCG) + `.csv` (ECG) | In-vehicle cardiac monitoring via cavity-mounted shear transducer. 13 participants, ~7 hours of data. |

### Data Preprocessing Pipeline

The raw → processed conversion is implemented in MATLAB under
[`preprocessing_matlab/`](preprocessing_matlab/README.md). Run
`generate_dataset.m` (training) or `generate_inference_dataset.m` (inference) to
produce the directory structure the loaders expect:

```text
processed/
|-- [input]/          # 4-second, 2 kHz acoustic WAV windows
`-- [wide_gaussian]/  # paired 8,000-sample ECG CSV files
```

The pipeline applies:

1. **Resample** to a unified 2 kHz sample rate
2. **Segment** recordings into 4-second windows (8,000 samples)
   - Training / Validation: 50% overlap between consecutive segments
   - Testing: No overlap
3. **Discard** incomplete trailing segments
4. **Filter** signals to the frequency range of interest:
   - **ECG:** 0.5 Hz high-pass Butterworth filter (5th order) + powerline filtering via `neurokit2`
   - **PCG:** 25–50 Hz band-pass Butterworth filter (5th order)
5. **Save** pre-processed segments to `_processed/` folders:
   - ECG → `.csv` format
   - PCG → `.wav` format

**During training**, additional augmentation is applied:

6. **Normalize** each segment by dividing by its `max()` value
7. **Add white Gaussian noise** at a randomly selected SNR level from {0, 30, 60, 90} dB — applied to PCG only

---

## Models

| Model | Key | Architecture | Parameters | Description |
|-------|-----|-------------|------------|-------------|
| **Conv-TasNet** | `conv-tasnet` | 1D Encoder → TCN Separator → Decoder | N=512, L=16, B=128, H=512, P=3, X=8, R=3 | Convolutional Time-domain Audio Separation Network using dilated depthwise separable convolutions. Default and recommended model. |

---

## Training

1. **Configure** your training parameters in `utils.py` (dataset, model, learning rate, epochs, paths).

2. **Run the training script:**

   ```bash
   python wav2ecg.py
   ```
---

## Inference

### Batch Inference on Test Set (recommended)

```bash
python run_inference.py \
  --dataset-root data/processed \
  --checkpoint ckpt/2025-11-22_23h02min.pth \
  --output-dir outputs/P12 \
  --device auto
```

This writes prediction, reference, and source-name arrays using the canonical
checkpoint stem. It uses `model.eval()` and `torch.inference_mode()`.

The legacy `inference.py`, `inference_mc_mae.py`, `predict.py`, and
`print_global_mae.py` scripts are retained as analysis provenance and still
contain experiment-specific control flow. Prefer `run_inference.py` for
portable runs.

### Single-File Prediction

Reconstruct ECG from a single audio file (`.wav` or `.mat`):

```bash
python predict.py path/to/your/recording.wav
```

**Supported input formats:**
- `.wav` — Single-channel PCG audio
- `.mat` — MATLAB file with `data` variable (channel 0 = ECG, channel 1 = PCG)

The script will:
1. Load and resample the input to 2 kHz
2. Segment into overlapping 4-second windows
3. Run inference on each segment
4. Merge overlapping predictions via averaging
5. Save the reconstructed ECG to `results/` in numpy (`.npy`) format

---

### Analysis Notebooks

The project includes several Jupyter notebooks for detailed result analysis:

- **`check_nn_performance.ipynb`** — Evaluate model predictions, compute metrics, and visualize reconstructed waveforms
- **`inspect_cavity_data.ipynb`** — Inspect the cavity dataset, reconstruct continuous signals, and perform heartbeat matching analysis
- **`dynamic_time_warping.ipynb`** — DTW-based temporal alignment analysis between predicted and ground truth ECG

---

## Citation

If you use this code, framework, or data pipeline in your research, please cite the following paper:

```
TBD

```
---

## Related Work

This project builds on and references the following prior work:

- **Conv-TasNet:** Luo, Y., & Mesgarani, N. (2019). *Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation.* IEEE/ACM Transactions on Audio, Speech, and Language Processing, 27(8), 1256–1266. [DOI: 10.1109/TASLP.2019.2915167](https://doi.org/10.1109/TASLP.2019.2915167)

- **FastNVG R-Peak Detection:** Emrich, J., Koka, T., Wirth, S., & Muma, M. (2023). *Accelerated sample-accurate R-peak detectors based on visibility graphs.* Proceedings of EUSIPCO 2023. [DOI: 10.23919/EUSIPCO58844.2023.10290007](https://doi.org/10.23919/EUSIPCO58844.2023.10290007)

- **NeuroKit2:** Makowski, D. et al. (2021). *NeuroKit2: A Python toolbox for neurophysiological signal processing.* Behavior Research Methods, 53(4), 1689–1696. [DOI: 10.3758/s13428-020-01516-y](https://doi.org/10.3758/s13428-020-01516-y)

- **AdamW:** Loshchilov, I., & Hutter, F. (2017). *Decoupled weight decay regularization.* arXiv:1711.05101.

---

## Reproducibility Boundaries

The Python implementation in this repository is the ground truth for training and
evaluation; the MATLAB sources under `preprocessing_matlab/` are the ground truth
for raw preprocessing.

**No random seed is set**, so exact bitwise retraining is not guaranteed.
Reference outputs, per-figure source data, and full per-window output arrays are
archived externally rather than in Git.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **Toyota Research Institute of North America (TRINA)** — Electronics Research Department
- **MIRISE Technologies** — Research support

---

<div align="center">

</div>
