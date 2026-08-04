# Missing and partial figure sources

The repository and the adjacent retained data collection were searched by
filename, content, and visual comparison with manuscript page renders. The
following original assets were not located and therefore cannot be packaged or
regenerated exactly from the retained files.

| Figure | Missing source | What remains available |
| --- | --- | --- |
| Main Figure 1 | Editable cardiac-sensing framework artwork | Manuscript rendering only |
| Main Figure 2 | CAD-style geometry renders, original resonator photograph, and final panel layout | Two COMSOL model files in `main_figure_02` |
| Main Figure 4 | Editable final callouts and annotations | Plot-ready MAT files and MATLAB plotting code in `main_figure_04` |
| Main Figure 6 | Editable mechanical and equivalent-circuit schematic | Circuit calculation code elsewhere in the release, but it does not draw the figure |
| Main Figure 7 | Editable anechoic-chamber validation schematic | Manuscript rendering only |
| Main Figure 8 | Editable architecture drawing | Conv-TasNet implementation in `main_figure_08` |
| Figure S2 | Editable ECG and heart-sound timing illustration | Manuscript rendering only |
| Figures S5-S7 | Original JPG/TIFF photographs and editable annotations | Manuscript renderings only |

The exploratory `get_time_domain.m` and `get_power_density_spectrum.m` scripts
in the adjacent data collection use an August 2025 recording and do not produce
Figure S1 or Main Figure 9. They are intentionally excluded from this package.

Rendered manuscript page screenshots are audit evidence and are excluded. The
losslessly extracted image objects actually embedded in the supplementary PDF
are included for Figures S1, S3, and S4 and are labeled as published renderings,
not raw numerical data.

Figures S3 and S4 were removed from this missing-source list after recovering
their exact P13 input, prediction, and Grad-CAM arrays by deterministic waveform
matching against the retained SVG. The recovered prediction agrees with the SVG
to a maximum absolute error of $5.87\times10^{-9}$.