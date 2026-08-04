# Supplement Figure S1

## Status

Retained plot-ready continuous P12 traces, the exact legacy source PNG, the
image embedded in the supplementary PDF, and a standalone plot script. No
inference or retraining is required. The paper uses the first 60,000 samples,
corresponding to 30 seconds at 2 kHz.

## Files

- `continuous_real_2831.csv`: retained continuous measured ECG.
- `continuous_pred_2831.csv`: retained continuous reconstructed ECG.
- `figure_S1.png`: exact retained 2000 by 400 pixel source PNG used in the paper.
- `figure_S1_paper.png`: lossless extraction of the 1636 by 359 pixel image embedded on supplementary PDF page 8.
- `plot_s1.py`: standalone recreation of the legacy first-30-second plotting path.
- `print_global_mae.py`: preserved legacy producer containing the original continuous-trace plotting path.

Run `python plot_s1.py` from this directory. The script uses the original 20 by
4 inch, 100 dpi canvas and intentionally retains the legacy sample-index x-axis,
including its `Time[sec]` label, because that is how the paper image was made.

The retained arrays belong to P12 recording
`[2025-10-07][10h56min][ECG+HS].txt`. Original release paths are
`reference_results/P12/`, `print_global_mae.py`, and
`reproduce_reference_metrics.py`. The audited supplementary PDF has SHA-256
`2906bd2d218fc94c26241050f83eb27551d7eea8f865cd2975f86feb1c7608c7`.