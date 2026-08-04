# Figure data package

This directory packages the retained data and code associated with every
published main and supplementary figure. It was assembled from existing files;
no model retraining or new inference was performed. The package distinguishes
reproducible numerical plots, partial scientific sources, rendered-only legacy
figures, and unavailable illustration or photograph sources.

## Published figure inventory

| Figure | Content | Package status |
| --- | --- | --- |
| Main Figure 1 | Cardiac sensing framework | Source artwork not located |
| Main Figure 2 | Resonator geometry, photograph, and pressure fields | Partial: COMSOL models located; artwork and photograph missing |
| Main Figure 3 | Measured, simulated, and circuit transfer functions | Complete MATLAB code and data |
| Main Figure 4 | Raw and processed ECG/audio waveforms | Complete MATLAB code and data; final annotations are manual |
| Main Figure 5 | Measured and reconstructed ECG traces | Complete MATLAB code and 18 CSV inputs |
| Main Figure 6 | Mechanical and equivalent-circuit schematic | Source artwork not located |
| Main Figure 7 | Anechoic-chamber validation schematic | Source artwork not located |
| Main Figure 8 | Conv-TasNet architecture | Partial: implementation located; drawing source missing |
| Main Figure 9 | FFT of the Figure 4 processed-audio trace | Complete MATLAB code and data |
| Figure S1 | Continuous-ECG boundary evaluation | Complete retained traces and standalone plot script |
| Figure S2 | ECG and heart-sound timing relationship | Source artwork not located |
| Figure S3 | Grad-CAM activation overlay | Complete recovered P13 input, prediction, and Grad-CAM arrays; exact PDF figure and retained SVG |
| Figure S4 | Time-resolved Grad-CAM attribution | Complete recovered P13 input, prediction, and Grad-CAM arrays; exact PDF figure and retained SVG |
| Figures S5-S7 | Experimental photographs | Original photographs not located |

See each packaged figure's `README.md` for source paths, reproduction steps,
and limitations. Every figure folder contains `DATA_FILES.csv`, which classifies
each local input, producer, output, and unavailable original. Reproduction does
not depend on data paths elsewhere in the repository. `MISSING_SOURCES.md`
records the unavailable assets.
`manifest.csv` and `checksums.sha256` inventory and hash every packaged file;
regenerate them with `python generate_manifest.py`.

## Git package exclusions

Three classes of file are listed in the per-figure `DATA_FILES.csv` tables but
are deliberately not committed to Git, because they are exact duplicates of
assets already present elsewhere in the repository or are method-development
evidence that is not the published figure:

- `main_figure_02/*.mph` — identical to `numerical_model/inWall_resonator.mph`
  and `numerical_model/largeResonator_quarter.mph`.
- `supplement_figure_S3/candidate_recomputation_not_paper/` and
  `supplement_figure_S4/candidate_recomputation_not_paper/` — a later,
  non-paper run. Their checkpoint is identical to
  `ckpt/2025-11-22_23h02min.pth`.

`manifest.csv` and `checksums.sha256` reflect the committed file set only.

The S3/S4 `original_data/` folders contain the recovered P13 paper arrays and
regeneration code. Their prediction matches the retained SVG at correlation
1.0, and the recovered Grad-CAM matches its color-derived importance at
correlation 0.999955. The separate render-derived tables can still be regenerated
with `python extract_retained_svg_data.py`; they preserve displayed SVG geometry
and are not substitutes for the recovered raw arrays.
