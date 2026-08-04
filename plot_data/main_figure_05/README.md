# Main Figure 5

This folder contains a self-contained Python reproduction package and the
retained MATLAB source. The 18 anonymous, headerless legacy CSV files have been
converted into nine professionally named paired epochs and one tidy plot table.

## Python workflow

```bash
python build_dataset.py
python plot_figure_05.py
```

The first command validates the legacy files and rebuilds `data/`. The second
reads only `data/figure5_plot_data.csv` and creates `figure_05_python.png` using
the published 0-36 second axis. To use sample timing implied by the 2 kHz sample
rate instead:

```bash
python plot_figure_05.py --time-axis native --output figure_05_native_time.png
```

Install the pinned Python dependencies with `pip install -r requirements.txt`.

## Structured data

- `data/epochs/recording_01_epoch_01.csv` through
	`recording_03_epoch_03.csv`: paired measured and predicted four-second epochs.
- `data/figure5_plot_data.csv`: tidy long-form table containing all raw,
	normalized, difference, epoch, sample, and time values used by the plot.
- `data/source_file_map.csv`: mapping from legacy names to professional names.
- `data/metadata.json`: machine-readable sampling, normalization, and timing
	metadata.
- `DATA_DICTIONARY.md`: column definitions and transformation equations.

Neutral recording identifiers are used because participant and source-recording
names were not retained with `a.csv` through `fff.csv`.

## Timing note

Each epoch contains 8,000 samples. At the repository's 2 kHz sample rate, each
epoch is four seconds and each three-epoch recording is 12 seconds. The retained
MATLAB producer instead uses `linspace(0, 36, 24000)`, and the paper displays
0-36 seconds. The structured table preserves both `native_time_seconds` and
`published_time_seconds`; the Python plot defaults to the published axis for
figure reproduction without presenting it as native sample timing.

## Retained source

`fig5_gen.m` and the 18 legacy CSV files are preserved unchanged for provenance.
Run `fig5_gen.m` in MATLAB to reproduce the original plotting path. Original
release path: `figures/fig5/`.