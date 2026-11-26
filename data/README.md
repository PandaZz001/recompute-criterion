# Data

This directory is intentionally empty. Research data are **not bundled**:

## SWellEx-96 (environmental)

- CTD casts (52 casts, `.prn`): SIO archive, `swellex96.ucsd.edu` →
  `env/ctds/ctds`. Configure with `SWELLEX96_CTD_DIR`.
- Official SSP (`ssp_S5.txt`) and optimized geoacoustic environment
  (`sw96_optimzed.env`, 4-layer bottom): parsed by
  `criterion/sw3_common.py`.
- Bathymetry track, range metadata: `env/bachman/…`, `env/range/…`
  (configured via `SWELLEX96_DIR`).

## SWellEx-96 (acoustic)

Event S5 VLA/TLA `.sio` files (~75 min × 21 channels). Distributed under
the SIO archive terms. Read with `io/sio_read.py` (differential decoding
applied automatically; see
`docs/technical_note_swelex96_differential_encoding.md`).

## Argo / WOA23

Second-environment validation currently uses synthetic literature
waveguides (W1). Real Argo/WOA ingestion is pending a reachable channel
(see derivation manuscript ch. 19.5 for the mirror checklist).

## Acoustics Toolbox

Kraken + KrakenField executables: self-compile from the official
repository or use the official release; set `KRAKEN_EXE` /
`KRAKEN_FIELD_EXE`.
