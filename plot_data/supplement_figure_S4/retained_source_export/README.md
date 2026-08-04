# Retained pre-PDF export

These PNG/SVG files are the retained source export before placement in the
supplementary PDF. Its canvas differs from the final embedded image.

Use `../figure_S4.png` for the exact figure as published.

## Machine-readable rendered data

The following tables were deterministically recovered from the Matplotlib SVG:

| File | Rows | Content |
| --- | ---: | --- |
| `rendered_signal_paths.csv` | 1,512 | Simplified real- and predicted-ECG SVG path vertices |
| `rendered_activation_bars.csv` | 8,000 | Grad-CAM bar geometry, plotted height, and approximate normalized height |
| `rendered_data_metadata.json` | - | Source hash/date, SVG selectors, axis calibrations, row counts, and caveats |

The `svg_*` columns preserve the original page coordinates. Columns ending in
`_rendered` are calibrated plot coordinates. `gradcam_scaled_amplitude_rendered`
is the directly plotted bar height in amplitude-axis units.
`gradcam_normalized_approx` divides that height by the peak absolute amplitude
retained in the simplified predicted path. Small negative values are retained:
they are visible cubic-interpolation undershoot from the plotted array.

These files are **render-derived data, not raw ECG or Grad-CAM arrays**. The
recovered raw paper arrays are preserved separately in `../original_data/`.
The signal paths were simplified during SVG export and therefore contain fewer
vertices than the original 8,000-sample arrays. Regenerate these tables together
with the S3 tables from `plot_data/` using:

```bash
python extract_retained_svg_data.py
```