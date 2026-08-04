# Retained pre-PDF export

These March 9 PNG/SVG files are the retained two-panel working export. The upper
panel supplied Figure S3, but PDF placement resampled and cropped it.

Use `../figure_S3.png` for the exact figure as published.

## Machine-readable rendered data

The following tables were deterministically recovered from the Matplotlib SVG:

| File | Rows | Content |
| --- | ---: | --- |
| `rendered_upper_samples.csv` | 8,000 | Upper-panel predicted ECG coordinates, marker RGB, and nearest standard `jet` colormap position |
| `rendered_upper_high_activation_regions.csv` | 13 | Rendered extents of regions selected by the source expression `grad_cam_upsampled > 0.3` |
| `rendered_lower_signal_paths.csv` | 1,485 | Simplified real- and predicted-ECG SVG path vertices |
| `rendered_lower_activation_bars.csv` | 8,000 | Lower-panel Grad-CAM bar geometry, plotted height, and approximate normalized height |
| `rendered_data_metadata.json` | - | Source hash/date, SVG selectors, axis calibrations, row counts, and caveats |

The `svg_*` columns preserve the original page coordinates. Columns ending in
`_rendered` are calibrated plot coordinates. Columns ending in `_approx` are
derived estimates and are not original model values. In particular,
`jet_colormap_position_approx` is obtained by matching the exported marker RGB
to the nearest entry in the standard 256-level Matplotlib `jet` lookup table.
The lower ECG paths were simplified during SVG export and therefore contain
fewer vertices than the original 8,000-sample arrays.

These files are **render-derived data, not raw ECG or Grad-CAM arrays**. The
recovered raw paper arrays are preserved separately in `../original_data/`.
Regenerate these tables together with the S4 tables from `plot_data/` using:

```bash
python extract_retained_svg_data.py
```