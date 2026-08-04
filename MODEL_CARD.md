# Conv-TasNet model card

## Model

- Task: reconstruct an ECG waveform from a synchronous acoustic heart-sound
  waveform measured with the Helmholtz cavity.
- Framework: PyTorch 2.0.1 / Torchaudio 2.0.2.
- Input: one 4-second acoustic window at 2 kHz (8,000 samples).
- Output: one reconstructed ECG waveform of the same duration.
- Architecture: `ConvTasNet(N=512, L=16, B=128, H=512, P=3, X=8, R=3,
  num_spks=1)`.
- Parameters and buffers in the state dictionary: 11,280,569 float32 values.

## Checkpoint

- File: `ckpt/2025-11-22_23h02min.pth`
- Size: 45,260,109 bytes.
- SHA-256:
  `14732b97b405d781406278bc5c4be0f5e91e1acc4888f0c209bf12d8c23ef5c7`
- Format: plain 370-entry PyTorch `OrderedDict` state dictionary.
- Compatibility: exact load into the included default `ConvTasNet` was verified.

The training entry point constructs Conv-TasNet directly. No external
pretrained state is loaded by that code path, although a stale upstream
checkpoint identifier remains in `utils.py` for provenance.

## Evaluation

Reported evaluation includes R-peak timing MAE, heart rate, Pearson correlation,
and normalized DTW after decimation from 2 kHz to 200 Hz. The final local
inference path applies a 20 Hz Torchaudio low-pass biquad to predictions, and
SciPy peak detection is used for the reported continuous analysis.

The retained result directory was reused across participants. P11 summary values
and P12 signal arrays are therefore stored separately in this release. See
`reference_results/README.md`.

## Intended use and limitations

This model is a research proof of concept for the studied cavity, acquisition
chain, participants, and recording conditions. It is not a medical device and
must not be used for diagnosis, treatment, or safety-critical monitoring.

Generalization has not been established for different cavity geometries,
transducers, body positions, clothing, vehicles, motion conditions, clinical
populations, or acquisition sample rates. Reconstructed morphology and detected
R-peaks can be wrong even when aggregate metrics appear acceptable.