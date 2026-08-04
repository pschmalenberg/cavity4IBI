# Reference result provenance

The source output directory was reused for more than one participant. The files
in this release are separated by their content and write time so that unrelated
runs are not presented as one reproducible bundle.

## P11

`P11/` contains only the summaries written at 11:32 on 2026-03-10. Their values
match the P11 rows reported in Tables 3, S1, and S2. The P11 window arrays that
produced these summaries are no longer present in the retained run directory,
so these values are reference evidence rather than independently recomputable
outputs.

## P12

`P12/` contains the arrays and names written at 11:34 and the continuous traces
written at 11:35. The names identify the recording
`[2025-10-07][10h56min][ECG+HS].txt`, mapped to P12 in the data manifest. Run:

```bash
python reproduce_reference_metrics.py --skip-dtw
python reproduce_reference_metrics.py
```

The first command recomputes peak, heart-rate, and correlation metrics. The
second also computes normalized DTW and is slower.

## Unassigned visualizations

`legacy_unassigned/` preserves Grad-CAM files from several earlier write times.
They cannot be assigned to a participant from their content and are excluded
from metric verification.