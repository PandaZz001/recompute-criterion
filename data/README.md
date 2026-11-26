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

- WOA23 seasonal climatological SSP (quarterly `woa23_ssp_s13..16.nc`,
  0.25° global, 102 standard levels, m/s) + ETOPO2022 bathymetry:
  real second-environment validation (experiments/W2_woa_env.py).
  Configure `WOA23_DIR` to the folder containing the SSP files and
  `Etopo2022_Little/ETOPO_2022_v1_60s.nc`.
- Argo float profiles: still pending a reachable GDAC channel (see
  derivation manuscript ch. 19.5 mirror checklist).

## Acoustics Toolbox

Kraken + KrakenField executables: self-compile from the official
repository or use the official release; set `KRAKEN_EXE` /
`KRAKEN_FIELD_EXE`.
