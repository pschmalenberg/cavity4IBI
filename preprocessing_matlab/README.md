# MATLAB preprocessing

These are the local dataset-generation sources used with the 4 kHz Biopac text
recordings. MATLAB Signal Processing Toolbox functions are required.

For raw ECG targets:

- `generate_dataset_raw_ECG.m`: create train, validation, and test windows.
- `generate_inference_dataset_raw_ECG.m`: create final `[B2]` inference windows.

Before running either script, set its `dir_root` to the directory containing the
13 Figshare `.txt` files and `dir_save` to the desired processed output. Keep all
helper `.m` files in this directory on the MATLAB path.

The scripts read 15 header lines, normalize ECG and acoustic channels, resample
4 kHz to 2 kHz, apply an 8-128 Hz order-8 acoustic bandpass and wavelet
denoising, and apply a fourth-order 25 Hz ECG low-pass with `filtfilt`.

`generate_dataset_raw_ECG.m` assigns approximately the first 80% of each
recording to training (`[A]`), the next 10% to validation (`[B1]`), and the
final 10% to test (`[B2]`). Files marked `[P01EXTRA]` are training-only.
Four-second windows use 0.3 s hops for training and validation and a 2 s hop
for test.

