# Implementation-grounded reproducibility notes

This file records the project decision that the local Python implementation is
the ground truth for model training and evaluation. The MATLAB sources in
`preprocessing_matlab/` are the ground truth for converting the raw 4 kHz
BIOPAC recordings into the paired files consumed by Python. Manuscript or
Supplementary Information text that disagrees with those implementations
should be corrected as described below.

The final checkpoint confirms the included architecture and parameters, but a
plain PyTorch state dictionary does not store the optimizer, learning rate,
augmentation, split, or preprocessing configuration. Those settings are
therefore grounded in the author-designated local source, not inferred from the
checkpoint.

## Resolved settings

| Topic | Implementation ground truth | Required paper correction | Resolution |
| --- | --- | --- | --- |
| Learning rate | `utils.py` sets `learning_rate = 1e-5`; `wav2ecg.py` passes it to AdamW | Main Methods and Supplement Section 1.2: replace `0.001` / `1 x 10^-3` with `1 x 10^-5` | Resolved |
| Noise SNR | `loaders/cavity.py` randomly selects from 100 evenly spaced values from 0 through 30 dB for each training sample load | Main Methods and Supplement Section 1.1: replace discrete `{0, 30, 60, 90}` dB with a random choice from 100 linearly spaced values over `[0, 30]` dB | Resolved |
| Split labels | MATLAB writes `[A]`, `[B1]`, and `[B2]`; Python maps them to training, validation, and test, respectively | Use the same label mapping wherever split names are described | Resolved |
| Temporal split | `generate_dataset_raw_ECG.m` assigns approximately the first 80% of each recording to `[A]`, the next 10% to `[B1]`, and the final 10% to `[B2]`; `[P01EXTRA]` files are forced into training | Main Methods and Supplement Algorithm 1: replace the recording-date split with this within-recording temporal split and state the P01-extra exception | Resolved |
| Window step | MATLAB uses 4 s windows with 0.3 s hops for `[A]` and `[B1]`, and 2 s hops for `[B2]` | Main Methods and Supplement Sections 1.1/Algorithm 1: replace “50% overlap for train/validation; no overlap for test” with 0.3 s train/validation hops and a 2 s test hop | Resolved |
| Acoustic preprocessing | MATLAB resamples 4 kHz to 2 kHz, applies an order-8 IIR bandpass with 8 and 128 Hz half-power frequencies using `filter`, then applies `wt_noise_reduction` | Main Methods: replace the 5 Hz high-pass plus order-2 25-50 Hz bandpass. Supplement Section 1.1: replace the order-5 25-50 Hz bandpass | Resolved |
| ECG target preprocessing | MATLAB resamples 4 kHz to 2 kHz and applies a fourth-order 25 Hz Butterworth low-pass with `filtfilt` | Main Methods: replace the 30 Hz low-pass. Supplement Section 1.1: replace NeuroKit2 `ecg_clean` / 0.5-40 Hz | Resolved |
| Prediction post-processing | `inference_mc_mae.py` and `methods.lowpass_filter` apply `torchaudio.functional.lowpass_biquad` at 20 Hz before saving predictions and detecting peaks | Main Methods and Supplement Section 1.3: replace NeuroKit2 `ecg_clean` / 0.5-40 Hz with the 20 Hz Torchaudio low-pass biquad | Resolved |
| Normalization | MATLAB divides full raw ECG and PA channels by their maximum absolute amplitude, renormalizes filtered PA, and normalizes each generated ECG/PA window by maximum absolute amplitude. The Python loader additionally divides each loaded PA window by its positive maximum before augmentation | Expand the preprocessing description to report these stages; do not state only a single generic amplitude-normalization step | Resolved |
| Model input length | Inclusive MATLAB indexing writes 8,001 samples for each nominal 4 s window. `CavityDataset.__getitem__` truncates both PA and ECG to the first 8,000 samples before model use | State that the model receives 8,000 samples at 2 kHz; optionally note that generated files contain an inclusive endpoint that is discarded on load | Resolved |
| DTW | Evaluation decimates by taking every tenth sample, from 2 kHz to 200 Hz, before DTW | No correction required | Consistent |
| Random seeds | No Python, NumPy, or PyTorch random seed is set in the local training path | State that training and augmentation were not run with a fixed seed; exact bitwise retraining is not guaranteed | Resolved limitation |

## Exact text to fix in the main paper

The current main-paper Methods text should be corrected in these locations:

1. **Data collection and pre-processing (PDF pages 11-12):** replace the
  reported 5 Hz high-pass and order-2 25-50 Hz PA bandpass with the MATLAB
  order-8 8-128 Hz bandpass followed by wavelet denoising.
2. **Data collection and pre-processing (PDF page 11):** replace the reported
  30 Hz ECG low-pass with a fourth-order 25 Hz Butterworth low-pass applied
  using zero-phase `filtfilt`.
3. **Training the neural network (PDF page 12):** replace 50% overlap for
  train/validation and no overlap for test with 4 s windows using 0.3 s hops
  for train/validation and a 2 s hop for test.
4. **Training the neural network (PDF page 12):** add that each recording is
  split temporally into approximately 80% training, 10% validation, and 10%
  test data; `[P01EXTRA]` recordings are assigned only to training.
5. **Training the neural network (PDF page 12):** replace SNR values
  `{0, 30, 60, 90}` dB with random selection from 100 linearly spaced values
  between 0 and 30 dB, inclusive.
6. **Training the neural network (PDF page 12):** replace learning rate
  `0.001` with `0.00001`.
7. **Training the neural network (PDF page 12):** replace NeuroKit2 ECG
  cleaning at inference with a Torchaudio low-pass biquad at 20 Hz.

Suggested consolidated replacement:

> Synchronous PA and ECG recordings were normalized by maximum absolute
> amplitude and resampled from 4 kHz to 2 kHz in MATLAB. PA was processed with
> an order-8 IIR bandpass filter with half-power frequencies of 8 and 128 Hz,
> followed by wavelet denoising. ECG targets were processed with a fourth-order
> 25 Hz Butterworth low-pass filter using zero-phase filtering. Each recording
> was split temporally into approximately the first 80% for training, the next
> 10% for validation, and the final 10% for testing; recordings marked
> `[P01EXTRA]` were assigned only to training. Four-second windows used 0.3 s
> hops for training and validation and a 2 s hop for testing. MATLAB generated
> 8,001-point inclusive windows, and the Python loader retained the first 8,000
> samples. During training, each PA window was augmented with white Gaussian
> noise at an SNR randomly selected from 100 linearly spaced values between 0
> and 30 dB, inclusive. Conv-TasNet was trained for 50 epochs using AdamW with
> an initial learning rate of 1 x 10^-5, default betas and epsilon, weight decay
> 0.01, batch size 8, and log-cosh loss. The learning rate was reduced by a
> factor of 10 after five validation epochs without improvement. For reported
> peak analysis, reconstructed ECG predictions were low-pass filtered at 20 Hz
> using `torchaudio.functional.lowpass_biquad`.

## Exact text to fix in the Supplementary Information

1. **Section 1.1, Data Pipeline (PDF page 3):** replace PA 25-50 Hz order-5
  filtering with order-8 8-128 Hz filtering plus wavelet denoising.
2. **Section 1.1 (PDF page 3):** replace NeuroKit2 ECG cleaning at 0.5-40 Hz
  with the MATLAB fourth-order 25 Hz low-pass using `filtfilt`.
3. **Section 1.1 (PDF page 3):** replace 50% train/validation overlap and zero
  test overlap with 0.3 s train/validation hops and a 2 s test hop.
4. **Section 1.1 (PDF page 3):** replace `{0, 30, 60, 90}` dB augmentation
  with random selection from 100 linearly spaced values over 0-30 dB.
5. **Section 1.2 (PDF page 3):** replace `eta_0 = 1 x 10^-3` with
  `eta_0 = 1 x 10^-5`.
6. **Section 1.3 (PDF page 3):** replace NeuroKit2 post-processing with the
  20 Hz Torchaudio low-pass biquad used by the final inference path.
7. **Algorithm 1, lines 2-5 (PDF page 4):** update preprocessing, hops, and
  split logic to match the implementation above. Delete “06/07/08
  (train/val), 09 (test).”
8. **Algorithm 1, line 19 (PDF page 4):** replace NeuroKit2 `ecg_clean` with
  the 20 Hz Torchaudio low-pass biquad.

## Split and window details

`generate_dataset_raw_ECG.m` creates the split files and
`loaders/cavity.py` consumes them:

| Prefix | Python split | Recording region | MATLAB hop | Overlap of a 4 s window |
| --- | --- | --- | --- | --- |
| `[A]` | Training | Approximately first 80% | 0.3 s | 92.5% |
| `[B1]` | Validation | Approximately next 10% | 0.3 s | 92.5% |
| `[B2]` | Test | Approximately final 10% | 2 s | 50% |

The generator starts `[B1]` four seconds after the nominal 80% boundary and
`[B2]` four seconds after the nominal 90% boundary. This prevents direct
window overlap across adjacent split boundaries. The same participant
recordings still contribute to multiple splits, so this is not a
participant-held-out evaluation.

## Source provenance

The 14 top-level `.m` files in `preprocessing_matlab/` were compared by SHA-256
with:

`C:\Users\Admin\Desktop\2025 Cavity\2025 Cavity Data Collection\dataset_generation_code`

All 14 release copies are byte-identical. The nested `New Folder` contains
figure-development files and CSVs, not dependencies of the raw-ECG dataset
generator, and is intentionally excluded.

## Packaging corrections

The following runtime corrections were applied only in `2026_nature`:

- Added model and loader package initializers containing only included classes.
- Replaced workstation paths in `utils.py` with environment variables and
  release-relative defaults.
- Corrected `torch.cuda.is_available()` invocation.
- Limited training construction to the included Conv-TasNet.
- Skipped non-finite batches and excluded them from epoch averages.
- Disabled gradient tracking during validation.
- Added portable inference, metric recomputation, and integrity-check commands.

These packaging corrections do not change the author-designated learning rate,
epoch count, model dimensions, split mapping, preprocessing filters,
normalization, or augmentation values documented above.

## Retained output provenance

The sole compact result directory is not one complete participant run. P11
summary metrics were written at 11:32 on 2026-03-10. P12 names, arrays, and MAE
files were written at 11:34, and P12 continuous traces at 11:35. Grad-CAM files
span still earlier timestamps. The release separates these artifacts instead
of claiming a false end-to-end P11 reproduction.

Complete retained P1-P13 inference outputs were not found. The paper tables
remain the authoritative aggregate report until those outputs are recovered or
inference is rerun from the public recordings.