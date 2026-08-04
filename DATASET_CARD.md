# Cavity PA-ECG dataset card

## Dataset identity

- Public record: <https://doi.org/10.6084/m9.figshare.31855450>
- License reported by the paper and Figshare record: MIT
- Contents: 13 synchronized pressure-acoustic (PA) and ECG recordings, mapped
  to P1-P13 in [data/figshare_manifest.csv](data/figshare_manifest.csv)
- Total public download size: approximately 2.96 GB
- Acquisition rate: 4 kHz; the model pipeline operates at 2 kHz

Each source file contains time, reference ECG, and acoustic cavity channels.
The ECG and PA channels were acquired synchronously through a BIOPAC MP160
system. The PA transducer was mounted in the medium-sized cavity integrated
into a vehicle seat, and reference ECG used a three-electrode Lead II
configuration.

## Participants and collection conditions

The manuscript reports 13 consenting participants. Collection occurred in a
quiet room while participants wore one or two layers of typical fall clothing.
The paper reports approval under Toyota Legal One guidelines and the Toyota
Motor North America Office of Privacy.

The public paper and release do not report participant age, sex, gender, race,
ethnicity, body composition, health status, medication, or recruitment
criteria. These values must not be inferred from participant identifiers.

## Cleaning and exclusions

The paper states that segments with excessive noise caused by body movement or
sensor repositioning were excluded. The resulting aggregate duration was
7 hours 8 minutes. Counts, thresholds, participant distribution, and exact
timestamps for excluded segments are not reported.

The main paper and supplement contain values that disagree with the local
implementation. The project designates Python as authoritative for training
and evaluation and MATLAB as authoritative for raw-data generation. Exact
paper corrections are listed in
[REPRODUCIBILITY_NOTES.md](REPRODUCIBILITY_NOTES.md). No random seeds are set,
so exact bitwise retraining is not guaranteed.

## Splits

The paper reports aggregate durations of 215 minutes 32 seconds for training,
45 minutes 18 seconds for validation, and 168 minutes 06 seconds for testing.
It does not publish a participant-level split table.

The confirmed MATLAB implementation assigns approximately the first 80% of
each recording to `[A]`, the next 10% to `[B1]`, and the final 10% to `[B2]`.
The Python loader maps these to training, validation, and test. Files marked
`[P01EXTRA]` are assigned only to training. Four-second windows use 0.3 s hops
for training and validation and a 2 s hop for testing.

The inspected local generated dataset contains 116,328 paired files: 107,064
training, 8,083 validation, and 1,181 test windows. MATLAB inclusive indexing
writes 8,001 samples per file at 2 kHz; the Python loader retains the first
8,000 samples for model input.

## Intended use

The dataset supports research on reconstructing ECG waveforms and deriving
cardiac timing measures from the studied passive acoustic cavity system. It may
also support reproducibility checks of the accompanying Conv-TasNet model.

This is a proof-of-concept research dataset. It is not a clinical dataset and
must not be used to diagnose, treat, or make safety-critical decisions.

## Known limitations and potential biases

- The sample contains 13 participants and has no reported demographic or
  clinical characterization.
- Data were collected in one quiet-room laboratory setting with one reported
  acquisition chain and the studied medium-sized cavity configuration.
- The target application includes dynamic vehicle operation, but no moving-
  vehicle data are reported.
- Removing movement-corrupted segments may make the retained data less
  representative of real vehicle use.
- Subgroup performance and sensitivity to clothing, body position, cavity
  geometry, transducer, vehicle, motion, and clinical condition were not
  quantified.
- All participants appear in the per-participant evaluation table, and no
  fully independent external dataset was evaluated.

These limitations are documented rather than mitigated by the release. White-
noise augmentation may improve noise robustness, but it does not establish
fairness, subgroup performance, or generalization to vehicle noise and motion.

## Reproduction files

- [data/README.md](data/README.md): download and local layout
- [data/figshare_manifest.csv](data/figshare_manifest.csv): file identities,
  sizes, checksums, and stable direct URLs
- [preprocessing_matlab](preprocessing_matlab): raw-data preprocessing
- [run_inference.py](run_inference.py): checkpoint inference
- [reference_results/README.md](reference_results/README.md): retained-output
  provenance and limitations