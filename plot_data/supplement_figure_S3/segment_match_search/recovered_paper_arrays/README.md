# Recovered Figure S3 paper arrays

These arrays reproduce the retained March 9, 2026 Figure S3 source export.
They were recovered by searching participant-specific model predictions against
the 8,000-sample waveform calibrated from the exact retained SVG.

## Identified source

- Participant: P13
- Recording: `[B2][2025-10-07][13h48min]`
- Reconstructed continuous-trace segment: samples 230,000--237,999
- Segment time within reconstructed trace: 115.0--119.0 s
- Sampling rate: 2,000 Hz
- Model checkpoint: `ckpt/2025-11-22_23h02min/ckpt/2025-11-22_23h02min.pth`
- Grad-CAM target layer: `model.gen_masks`
- Backward target: sum of the ConvTasNet output waveform

The reconstructed prediction matches the paper SVG waveform with correlation
`0.9999999999999998`, mean absolute error `2.50e-9`, and maximum absolute error
`5.87e-9`. The recomputed Grad-CAM agrees with the SVG color-derived importance
at correlation `0.999955`, with `99.7875%` agreement for the original `> 0.3`
high-activation threshold.

## Files

- `paper_input_segment.npy`: exact 8,000-sample reference input passed to the model.
- `paper_prediction_segment.npy`: exact 8,000-sample plotted prediction.
- `paper_grad_cam_raw.npy`: 999-sample Grad-CAM before interpolation.
- `paper_grad_cam_upsampled.npy`: normalized 8,000-sample Grad-CAM used for plotting.
- `P13_reference_windows.npy`: 115 paired P13 reference windows used for reconstruction.
- `P13_prediction_windows.npy`: 115 P13 model predictions used for reconstruction.
- `recovery_metadata.json`: validation metrics, shapes, and SHA-256 hashes.

Regenerate the arrays from the retained P13 dataset and checkpoint with:

```powershell
C:\Users\Admin\anaconda3\envs\wav2ecg\python.exe ..\recover_paper_arrays.py
```
