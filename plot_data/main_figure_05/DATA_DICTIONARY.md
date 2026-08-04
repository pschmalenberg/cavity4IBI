# Figure 5 data dictionary

## Epoch tables

Each `data/epochs/recording_NN_epoch_NN.csv` file contains one paired epoch.

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `sample_index` | integer | sample | Zero-based index within the epoch |
| `time_seconds` | float | s | `sample_index / 2000` |
| `measured_ecg` | float | normalized arbitrary unit | Retained processed experimental ECG |
| `predicted_ecg` | float | normalized arbitrary unit | Retained neural-network ECG reconstruction |

## Plot-ready table

`data/figure5_plot_data.csv` contains one row per sample and recording.

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `recording_id` | string | none | Neutral identifier `recording_01` through `recording_03` |
| `recording_number` | integer | none | One-based recording order in the paper |
| `epoch_number` | integer | none | One-based epoch number within the recording |
| `sample_index` | integer | sample | Zero-based index within the concatenated recording |
| `native_time_seconds` | float | s | Sample time at 2 kHz; spans 0 to just under 12 s |
| `published_time_seconds` | float | s | MATLAB `linspace(0, 36, 24000)` axis used in the paper |
| `measured_ecg_raw` | float | normalized arbitrary unit | Concatenated retained measured samples |
| `predicted_ecg_raw` | float | normalized arbitrary unit | Concatenated retained prediction samples |
| `measured_ecg_normalized` | float | 0-1 | Recording-level min-max normalization |
| `predicted_ecg_normalized` | float | 0-1 | Recording-level min-max normalization |
| `difference_scaled` | float | arbitrary unit | Scaled normalized prediction error |

For signal $x$, min-max normalization is

$$
x_{01} = \frac{x - \min(x)}{\max(x) - \min(x)}.
$$

The plotted difference is

$$
d = 0.15\left(\hat{x}_{01} - x_{01}\right).
$$

Normalization is performed independently over each 24,000-sample measured or
predicted recording, matching `fig5_gen.m`.