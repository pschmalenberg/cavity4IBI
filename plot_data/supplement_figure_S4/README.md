# Supplement Figure S4

## Authoritative paper figure

`figure_S4.png` is the exact 5932 by 1288 pixel image object embedded on page 11
of the supplementary PDF. Use this file when comparing with or redisplaying the
paper. Its SHA-256 is
`42f9eae57bced39127e66b68cfa507ac3195134aec91daf7973cc53b78721db3`.

## Provenance folders

- `retained_source_export/` contains the retained 5932 by 1397 PNG/SVG source
	export used before PDF placement, plus machine-readable data recovered from
	its vector geometry.
- `original_data/` contains the recovered P13 input, prediction, raw Grad-CAM,
  upsampled Grad-CAM, full window bundles, hashes, and regeneration code.
- `candidate_recomputation_not_paper/` contains a later trace bundle,
        checkpoint, script, numerical arrays, and generated plot. It is a
        different, non-paper run and is not committed to the Git repository.
attribution. It is retained only as method-development evidence.

The original segment was identified as P13 recording
`[B2][2025-10-07][13h48min]`, samples 230,000--237,999 of the reconstructed
trace (115--119 s). The recovered Grad-CAM matches the 8,000 retained S4 bar
heights at $r=0.999996$, with 99.925% agreement at the original 0.3 threshold.
See `original_data/recovery_metadata.json` for hashes and validation details.

The retained SVG tables remain useful independent render-derived evidence; see
`retained_source_export/README.md`.
The audited supplementary PDF has SHA-256
`2906bd2d218fc94c26241050f83eb27551d7eea8f865cd2975f86feb1c7608c7`.