# Recompute Criterion: A First-Principles Scalar Test for Sound-Field Recomputation Decisions

Reference implementation of the sound-field recomputation criterion **Θ**:
a scalar, microseconds-cheap, cross-perturbation measure that decides
whether an acoustic field must be recomputed after an environmental
change or may be reused. The decision thresholds are derived, not
calibrated.

## Core result (one sentence)

For a modal pair-differential phase perturbation Θ, the pattern
correlation obeys

    ρ(Θ) = ρ_f + (1 − ρ_f) · exp(−λ Θ^p),

with all parameters derived from first principles: the phase folding
factor c_ph = 0.51 (chain: c_G = 0.798 → spectral functional q →
structure factor S_dom), the shape truncation factor c_sh ≈ 1.4–1.5, the
main-modal-pair floor ρ_f = 0.4313, and the form exponent
p = 2 + C = 1.409 (coherent-truncation mechanism) — reproducing the
fitted p = 1.40 within 0.6%. The derived threshold
Θ*_ph = √(2A)/c_ph = 1.201 rad reproduces the historical calibration
1.2 rad to 0.1%.

## Repository layout

```
criterion/          core implementation (pure numpy + official Kraken)
  mode_sensitivity.py   ModeSensitivity + theta_phase (the criterion)
  sensitivity_modes.py  ensemble driver (reference implementation)
  at_core.py            Kraken/Field wrappers (ModResult, env, flp, shd)
  sw3_common.py         SWellEx-96 official environment parsing
io/
  sio_read.py        SWellEx-96 SIO reader with differential-decoding fix
experiments/         reproducible experiment scripts (one JSON per run)
  SW3e2_gray_zone.py   261-sample gray-zone family on measured CTD pairs
  AB1_ablation.py      9-variant component ablation
  F1_freq_scaling.py   29-pair frequency-scaling census (β distribution)
  W1_second_env.py     cross-waveguide portability (3 literature waveguides)
docs/
  technical_note_swelex96_differential_encoding.md
data/                empty by design — see data/README.md
```

## Dependencies

- Python 3.10+, numpy, matplotlib
- [Acoustics Toolbox](https://github.com/org-arl/acu-toolbox) Kraken +
  KrakenField executables (self-compiled or official release). Paths are
  configured via environment variables `KRAKEN_EXE`, `KRAKEN_FIELD_EXE`
  (defaults point to a local AT installation; override for your machine).
- SWellEx-96 environmental data (CTD casts) for SW3e2/AB1/F1 — see
  `data/README.md`. Not bundled.

## Reproducing the headline numbers

```bash
cd experiments
python SW3e2_gray_zone.py   # -> work/sw3e2_gray_zone.json
python AB1_ablation.py      # -> work/ab1_ablation.json
python F1_freq_scaling.py   # -> work/f1_freq_scaling.json
python W1_second_env.py     # -> work/w1_second_env.json
```

Each script is self-contained and writes its results to `work/*.json`.
All manuscript numbers trace to these JSON files.

## Honest-boundary notes

- The measured acoustic path (SWellEx-96 tonal SNR ~1) is unusable for
  validation after decoding; the validation narrative uses synthetic
  truth on measured environments (B′ line). Evidence chain in
  `docs/technical_note_swelex96_differential_encoding.md` and the
  derivation manuscript.
- The H3 double bounds are a bandwidth reference, not a decision rule;
  decision-level claims rest on the gray-zone family (91% in gray zone).
- Deep-water portability is conservative but requires per-waveguide
  threshold scaling (see W1).

## License

MIT. The SWellEx-96 data are distributed under the SIO archive terms
and are not part of this repository.

## Citation

Derivation manuscript (CC, in preparation, 2026). Full chain:
c_ph/c_sh folding → ρ_f floor → p = 2+C → threshold family.
