# Nature Machine Learning Checklist V1.1 response

Paper: **Non-invasive cardiac sensing via an acoustic Helmholtz resonator
cavity with electrocardiogram waveform reconstruction**

- Corresponding author: Paul D. Schmalenberg
- Last updated: 2026-08-03
- Checklist source: `ML-reporting-summary.pdf`, three pages, version 1.1

This document is a completion guide for the Nature form. A checked item means
that the answer is supported by the manuscript, Supplementary Information, or
this release. Conditional answers assume that this `2026_nature` directory is
included with the submission or published at a stable public URL. Scientific
facts that are absent or contradictory are not inferred.

Status terms:

- **Supported**: enough evidence exists to select the recommended box.
- **Partial**: select the stated box and include the qualification.
- **No / gap**: leave the positive claim unchecked or select No.
- **Paper correction pending**: the implementation is established, but the
  submitted text must be corrected before selecting Yes.

## 1. Availability and reproducibility of code and data

### CodeOcean capsule

- [ ] Code will be included in a CodeOcean capsule.
- **Answer:** No CodeOcean capsule is provided.
- **Status:** No / gap.

### Source code

- [x] The source code is included in the submission or available in a public
  repository.
- **Form text:** `https://github.com/pschmalenberg/cavity4IBI`; include or
  publish this expanded release with the submission.
- **Paper evidence:** main manuscript, PDF page 12, *Code availability*.
- **Release evidence:** [README.md](README.md), [models](models),
  [loaders](loaders), and [metrics](metrics).
- **Status:** Supported, conditional on sharing this expanded release.

### Compiled standalone software

- [ ] A compiled standalone version of the software is included.
- **Form text:** Not applicable. The release contains installable Python and
  MATLAB source, not a compiled executable, container, or standalone binary.
- **Release evidence:** [requirements.txt](requirements.txt) and
  [setup.py](setup.py).
- **Status:** No / gap. An installable source package is not a compiled
  standalone version.

### Test dataset and reproduction scripts

- [x] A test dataset and instructions/scripts for replicating the results are
  included or publicly available.
- **Form text:** Raw synchronized recordings are available at
  `https://doi.org/10.6084/m9.figshare.31855450`. Instructions and scripts are
  provided in this release. Retained P12 traces support a fast metric check;
  complete P1-P13 outputs must be regenerated.
- **Paper evidence:** main manuscript, PDF page 12, *Data availability*.
- **Release evidence:** [data/README.md](data/README.md),
  [data/figshare_manifest.csv](data/figshare_manifest.csv),
  [preprocessing_matlab/README.md](preprocessing_matlab/README.md),
  [run_inference.py](run_inference.py), and
  [reproduce_reference_metrics.py](reproduce_reference_metrics.py).
- **Status:** Partial. Raw data and scripts are available, but processed split
  datasets and complete retained P1-P13 inference outputs are not separately
  hosted.

### Installation and execution README

- [x] A README with instructions for installing and running the code is
  included or publicly available.
- **Release evidence:** [README.md](README.md), including quick start, data
  layout, inference, training entry point, and reproducibility boundaries.
- **Status:** Supported.

### Reviewer access

- [x] The code is made available to reviewers during review.
- **Form text:** Public source and data are available at the URLs above. Attach
  or provide reviewer access to this expanded release so reviewers also receive
  the portable entry points, exact checkpoint, manifests, and model card.
- **Status:** Supported only after this expanded release is shared.

### Pretrained models

- [ ] Pretrained models are used and accessible.
- [ ] Pretrained models are used and inaccessible.
- **Form text:** No external pretrained model was used. Training starts from a
  newly initialized Conv-TasNet. The final model trained in this study is
  included as `ckpt/2025-11-22_23h02min.pth`.
- **Release evidence:** [MODEL_CARD.md](MODEL_CARD.md) and
  [ckpt/README.md](ckpt/README.md).
- **Status:** Not applicable. Do not classify the study's final checkpoint as
  an external pretrained model.

### Post-publication access

- [x] The paper contains information on obtaining code and data after
  publication.
- **Paper evidence:** main manuscript, PDF page 12, *Data availability* and
  *Code availability*.
- **Required correction:** replace the account-only Figshare URL in the
  manuscript with `https://doi.org/10.6084/m9.figshare.31855450`.
- **Status:** Supported after the data URL is corrected and this expanded
  release is made public or attached to the submission.

## 2. Datasets

### A. All data sources are listed

- [x] Yes.
- **Form text:** The study uses synchronized PA and ECG recordings collected
  from 13 consenting participants with the described cavity, transducer, and
  BIOPAC acquisition chain. No external training dataset was used.
- **Paper evidence:** main manuscript, PDF pages 11-12, *Data collection and
  pre-processing*; Supplementary Information, PDF page 3, *Data Pipeline*.
- **Release evidence:** [data/figshare_manifest.csv](data/figshare_manifest.csv).
- **Status:** Supported.

### B. Train, validation, and test data are public

- [x] Yes, with qualification.
- **Form text:** All 13 source recordings are public at
  `https://doi.org/10.6084/m9.figshare.31855450`; the released MATLAB scripts
  generate the processed train, validation, and test windows. The processed
  windows are not hosted separately.
- **Paper evidence:** main manuscript, PDF page 12, *Data availability*.
- **Release evidence:** [data/README.md](data/README.md),
  [data/figshare_manifest.csv](data/figshare_manifest.csv), and
  [preprocessing_matlab/generate_dataset_raw_ECG.m](preprocessing_matlab/generate_dataset_raw_ECG.m).
- **Status:** Partial because processed split files are not hosted separately.
  The confirmed split implementation is documented in Section 3D below.

### C. Potential dataset biases and mitigation

- [ ] Yes.
- [x] No.
- **Form text:** The paper identifies a quiet-room laboratory setting, one
  acquisition chain, movement-related exclusions, and expected degradation in
  a dynamic vehicle. It does not report participant demographics or analyze
  subgroup performance, selection bias, or the effect of excluding noisy
  movement segments. Noise augmentation is described, but this is not a full
  dataset-bias analysis.
- **Paper evidence:** main manuscript, PDF page 7, *Discussion*, and PDF page
  12, *Data collection and pre-processing*.
- **Release evidence:** [MODEL_CARD.md](MODEL_CARD.md), *Intended use and
  limitations*.
- **Status:** No / gap. Add an explicit dataset limitations paragraph to the
  manuscript if the journal expects Yes.

### D. Cleaning and preprocessing are described

- [x] Yes.
- [ ] No.
- **Form text:** MATLAB: PA 8-128 Hz plus wavelet denoising; ECG 25 Hz
  low-pass. Python: SNR sampled over 0-30 dB.
- **Paper evidence:** main manuscript, PDF pages 11-12; Supplementary
  Information, PDF pages 3-4, Sections 1.1-1.3.
- **Release evidence:** [preprocessing_matlab/README.md](preprocessing_matlab/README.md)
  and [loaders/cavity.py](loaders/cavity.py).
- **Status:** Supported by the included code pipeline. The authoritative values
  and exact paper edits are in
  [REPRODUCIBILITY_NOTES.md](REPRODUCIBILITY_NOTES.md).

### E. Combining data from multiple sources

- [x] Yes: not applicable.
- [ ] No.
- **Form text:** No independent datasets were combined. PA and ECG are
  synchronized channels from the same BIOPAC recording sessions; ECG supplies
  the supervised reference target.
- **Paper evidence:** main manuscript, PDF page 11, *Data collection and
  pre-processing*.
- **Status:** Not applicable.

## 3. Model and training

### A. Model architecture basis

- **Answer:** Conv-TasNet, adapted from a time-domain audio source-separation
  architecture to supervised PA-to-ECG waveform regression.
- **Paper evidence:** main manuscript, PDF pages 6 and 12; Supplementary
  Information, PDF pages 2-3, Sections 1 and 1.2.
- **Release evidence:** [models/convtasnet.py](models/convtasnet.py) and
  [MODEL_CARD.md](MODEL_CARD.md).
- **Status:** Supported.

### B. Model Card

- [x] Yes.
- [ ] No.
- **Release evidence:** [MODEL_CARD.md](MODEL_CARD.md).
- **Status:** Supported when this expanded release is shared.

### C. Separate training, validation, and test sets

- [x] Yes.
- [ ] No.
- **Form text:** The implementation uses `[A]` for training, `[B1]` for
  validation, and `[B2]` for testing. Aggregate durations are 215 min 32 s,
  45 min 18 s, and 168 min 06 s, respectively.
- **Paper evidence:** main manuscript, PDF page 6 and Table 2 on PDF page 19;
  Supplementary Information, PDF pages 3-4.
- **Release evidence:** [loaders/cavity.py](loaders/cavity.py).
- **Status:** Supported as three distinct file sets.

### D. Data splitting method is clearly stated

- [x] Yes.
- [ ] No.
- **Form text:** Within each recording: approximately 80% training, 10%
  validation, and 10% testing, with four-second boundary gaps. `[P01EXTRA]`
  files are training-only.
- **Paper evidence:** main manuscript, Table 2 on PDF page 19;
  Supplementary Information, PDF page 4, Algorithm 1.
- **Release evidence:**
  [preprocessing_matlab/generate_dataset_raw_ECG.m](preprocessing_matlab/generate_dataset_raw_ECG.m)
  and [REPRODUCIBILITY_NOTES.md](REPRODUCIBILITY_NOTES.md).
- **Status:** Supported by the included MATLAB and Python pipeline. The current
  supplement's recording-date statement must still be corrected.

### E. Split mimics anticipated real-world applications

- [ ] Yes.
- [x] No.
- **Form text:** The intended application is monitoring in a dynamic vehicle,
  but all recordings were acquired in a quiet room. A temporal split from the
  same recording sessions does not test the expected vehicle-domain shift.
- **Paper evidence:** main manuscript, PDF page 7, *Discussion*, and PDF page
  12, *Data collection and pre-processing*.
- **Status:** No / gap.

### F. Split chosen to avoid data leakage

- [ ] Yes.
- [x] No.
- **Form text:** The released MATLAB pipeline leaves four-second gaps between
  sequential split regions, preventing direct overlap at split boundaries.
  However, the same participant recordings contribute to multiple splits and
  no participant-level leakage analysis is reported. This does not establish
  performance on unseen participants.
- **Release evidence:**
  [preprocessing_matlab/generate_dataset_raw_ECG.m](preprocessing_matlab/generate_dataset_raw_ECG.m)
  and [loaders/cavity.py](loaders/cavity.py).
- **Status:** Partial evidence; recommend No for participant-independent
  generalization.

### G. Interpretability studied and validated

- [x] Yes, with qualification.
- [ ] No.
- **Form text:** One-dimensional Grad-CAM attributions were projected to the
  time axis and compared qualitatively with beat-related ECG transients and the
  expected electromechanical delay. This is post-hoc, qualitative validation;
  no quantitative attribution benchmark was reported.
- **Paper evidence:** main manuscript, PDF pages 6-7; Supplementary
  Information, PDF pages 5-6, Section 3.1 and Figures S3-S4.
- **Status:** Partial but sufficient to report that interpretability was
  studied, provided the qualitative limitation is retained.

## 4. Evaluation

### A. Metrics are described and justified

- [x] Yes.
- [ ] No.
- **Form text:** R-peak timing MAE measures beat-level timing; heart-rate error
  gives a clinically interpretable aggregate; Pearson correlation and
  normalized DTW assess waveform morphology.
- **Paper evidence:** main manuscript, PDF pages 6-7; Supplementary
  Information, PDF pages 4-7, Sections 2 and 4.
- **Release evidence:** [metrics](metrics) and
  [reproduce_reference_metrics.py](reproduce_reference_metrics.py).
- **Status:** Supported.

### B. Cross-validation

- [ ] Yes.
- [x] No.
- **Form text:** No k-fold, grouped, or repeated cross-validation is reported;
  the study uses one train/validation/test split.
- **Status:** No / gap.

### C. Community-accepted benchmark dataset or task

- [ ] Yes.
- [x] No.
- **Form text:** No community benchmark exists or was used for this custom
  cavity PA-to-ECG reconstruction task.
- **Status:** No; effectively not applicable to the novel acquisition task.

### D. Simple or trivial baseline comparisons

- [ ] Yes.
- [x] No.
- **Form text:** The paper explains that simple acoustic thresholding is not
  adequate but provides no quantified threshold, classical signal-processing,
  or simpler-model baseline.
- **Paper evidence:** main manuscript, PDF page 6, immediately before
  *Data-driven ECG waveform reconstruction*.
- **Status:** No / gap.

### E. Current state-of-the-art benchmarks

- [ ] Yes.
- [x] No.
- **Form text:** Related non-invasive sensing methods are discussed, but no
  side-by-side benchmark on the same data and metrics is reported.
- **Paper evidence:** main manuscript, PDF pages 2-4, *Introduction*.
- **Status:** No / gap.

### F. Ablation experiments

- [ ] Yes.
- [x] No.
- **Form text:** No architecture, preprocessing, augmentation, or loss ablation
  results are reported.
- **Status:** No / gap.

### G. Fully independent dataset

- [ ] Yes.
- [x] No.
- **Form text:** Evaluation uses held-out portions of the same 13-participant,
  quiet-room acquisition collection. No independent site, vehicle-condition,
  acquisition-chain, or clinical dataset was tested.
- **Paper evidence:** main manuscript, PDF pages 7 and 11-12.
- **Release evidence:** [MODEL_CARD.md](MODEL_CARD.md), *Intended use and
  limitations*.
- **Status:** No / gap.

## 5. Computational resources

### A. Hardware and computing resources

- [x] Yes.
- [ ] No.
- **Form text:** Python 3.11.4, PyTorch 2.0.1, Torchaudio 2.0.2, CUDA 11.7,
  and cuDNN 8.5.0 on Windows 10 Pro; workstation with three NVIDIA GeForce RTX
  3090 TURBO GPUs (24 GB VRAM each), an AMD EPYC 7282 16-core processor, and
  255 GB system memory.
- **Paper evidence:** Supplementary Information, PDF page 4, Section 1.4.
- **Release evidence:** [requirements.txt](requirements.txt) and
  [README.md](README.md).
- **Status:** Supported.

### B. Computational cost, time, parallelization, or carbon

- [ ] Yes.
- [x] No.
- **Form text:** The supplement reports up to 50 epochs, batch size 8, and
  checkpointing every two hours, but it does not report total wall-clock time,
  GPU utilization/parallelization, inference latency, financial cost, energy,
  or carbon emissions. These values cannot be reconstructed from checkpoint
  timestamps.
- **Paper evidence:** Supplementary Information, PDF pages 3-4.
- **Status:** No / gap; author measurement or confirmation is required.

## Required actions before submission

1. Publish or attach this expanded release and use its stable URL in the form.
2. Replace the Figshare account URL in the manuscript with the public DOI.
3. Apply the implementation-grounded main-paper and supplement corrections in
  [REPRODUCIBILITY_NOTES.md](REPRODUCIBILITY_NOTES.md).
4. Add an explicit dataset-bias/limitations paragraph to the paper if Section
   2C is to be marked Yes.
5. Report measured training time and parallelization if Section 5B is to be
   marked Yes.

Baseline comparisons, ablation experiments, cross-validation, and a fully
independent dataset require new experiments. They must remain marked No unless
those experiments are performed and reported.