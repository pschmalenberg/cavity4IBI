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

MATLAB inclusive indexing writes 8,001 samples for each nominal four-second
pair. The Python loader truncates both channels to the first 8,000 samples
before model use. This is the behavior used by the local generated dataset.

The directory name `[wide_gaussian]` is historical: the `raw_ECG` scripts save
the processed raw ECG target there. The 14 included `.m` files are byte-identical
to the top-level sources in the local `dataset_generation_code` directory.
Their sizes and SHA-256 hashes are recorded in
[source_manifest.csv](source_manifest.csv). See
`REPRODUCIBILITY_NOTES.md` for the paper text that must be corrected before
submission.