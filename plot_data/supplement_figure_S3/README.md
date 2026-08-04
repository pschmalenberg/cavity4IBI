# Supplement Figure S3

## Authoritative paper figure

`figure_S3.png` is the exact 5307 by 1284 pixel image object embedded on page 10
of the supplementary PDF. Use this file when comparing with or redisplaying the
paper. Its SHA-256 is
`e317f5d0803d467a39f00595822a4eec63da794215d41d07d2b5cd88bd05ea43`.

## Provenance folders

- `retained_source_export/` contains the March 9 two-panel PNG/SVG working
	export whose upper panel supplied S3 before PDF placement, plus machine-readable
	data recovered from its vector geometry.
- `original_data/` contains the recovered P13 input, prediction, raw Grad-CAM,
  upsampled Grad-CAM, full window bundles, hashes, and regeneration code.
- `candidate_recomputation_not_paper/` contains a later March 10 trace bundle,
        checkpoint, script, numerical arrays, and generated plot. It is a
        different, non-paper run and is not committed to the Git repository.
activation windows. It is retained only as method-development evidence.

The original segment was identified as P13 recording
`[B2][2025-10-07][13h48min]`, samples 230,000--237,999 of the reconstructed
trace (115--119 s). The recovered prediction matches the retained SVG at
$r=1.0$ with maximum absolute error $5.87\times10^{-9}$; the recovered Grad-CAM
matches the SVG color-derived importance at $r=0.999955$. See
`original_data/recovery_metadata.json` for hashes and validation details.

The retained SVG tables remain useful independent render-derived evidence; see
`retained_source_export/README.md`.
The audited supplementary PDF has SHA-256
`2906bd2d218fc94c26241050f83eb27551d7eea8f865cd2975f86feb1c7608c7`.