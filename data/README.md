# Data

The raw synchronized acoustic and ECG recordings are available from Figshare:

<https://doi.org/10.6084/m9.figshare.31855450>

Version 2 contains 13 MIT-licensed text files totaling approximately 2.96 GB.
`figshare_manifest.csv` records the P1-P13 mapping, byte sizes, MD5 checksums,
and stable direct download URLs obtained from the Figshare API.

Each Biopac text file has 15 header lines followed by three columns:

1. Time
2. ECG
3. Heart sound/acoustic cavity signal

The acquisition sample rate is 4 kHz. The supplied MATLAB scripts resample to
2 kHz and write paired four-second windows under `[input]` and
`[wide_gaussian]`.

Recommended local layout:

```text
data/
|-- raw/                # downloaded Figshare .txt files
`-- processed/
    |-- [input]/        # acoustic .wav windows
    `-- [wide_gaussian]/# ECG .csv windows
```

Verify a downloaded file in PowerShell with:

```powershell
Get-FileHash -Algorithm MD5 'data\raw\[2025_10_06][13h51min][ECG+HS].txt'
```

Raw recordings are intentionally not copied into this Git release.