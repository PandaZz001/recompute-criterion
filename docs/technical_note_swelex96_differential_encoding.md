# Technical Note: Differential Encoding in the SWellEx-96 SIO Acoustic Archive

*Reader implementation, validation, and implications for re-analyses*
*(2026-11-25; findings incorporated in the derivation manuscript, ch. 16)*

## Abstract

The SWellEx-96 Event S5 acoustic files (SIO `.sio` format) store
**first differences** `d[n] = x[n] − x[n−1]` rather than raw pressure
samples, on both the VLA and TLA recordings. Reading the archive without
cumulative-sum (cumsum) recovery produces an apparent "flicker ceiling"
and spurious relative spectral peaks that have contaminated earlier
analyses. This note documents the three decoding requirements (differential
encoding, record addressing, in-block accumulation), the validation
evidence, and a standalone reader implementation (`io/sio_read.py`).

## 1. SIO header and record layout

The 132-byte header is big-endian int32: `ID, nr (nrecords), rl (record
byte length), nc (nchannels), sl (bytes/sample: 4 = float32, 2 = int16),
f0, np (points/channel), bs (byte-order constant 32677), fn (24 chars),
comment (72 chars)`.

Record addressing follows the official `sioread.m` convention:
`fseek(fid, rl * trec)` — record `k` lives at byte offset `rl × k`
(k is 1-based; the padding region after the 132-byte header does not
participate in addressing). Channel `j` of record `i` is record number
`(j−1) + i·nc`, with `ptrec = rl/sl` points per record.

## 2. Differential encoding

The stored values are **sample-to-sample differences within each channel**:

    d[n] = x[n] − x[n−1]

Recovery requires a per-channel cumulative sum, seeded by the
cross-record chain. A reader that skips this step returns the
differential signal, whose spectrum is a high-pass-filtered version of
the true spectrum — producing:

- a "flicker ceiling" in spectrograms (the dominant artifact before the
  fix), and
- a 9 dB spurious tonal in the differential domain (false relative peak).

Validation after recovery (Event S5 VLA, 21 channels × 75 min):

| Check | Value |
|-------|-------|
| lag-1 autocorrelation of recovered waveform | 0.998 |
| broadband correlation between adjacent channels | 0.933 |
| cross-block difference-chain continuity | 0.996 |

## 3. Random-access reader (block-cumulative algorithm)

Random reads cannot cumsum from the file start. The implementation builds
a per-channel table of block-head prefix sums (`offsets[ch, k]`), then
recovers any requested segment by: block-head offset + cumsum of the
in-block differences from the block start (including the partial first
block). This is the `pp1` in-block accumulation fix.

Reference implementation: `io/sio_read.py` (`SioFile` class),
pure standard library + numpy, no external dependencies.

```python
from sio_read import SioFile
f = SioFile("VLA_file.sio")
x = f.read_channel(ch=1, p1=0, npi=32768)   # 1-based channel, recovered
```

## 4. Implications for re-analyses

1. Any tonal-SNR, MFP, or CSDM result computed from the SWellEx-96
   public files **without differential recovery is invalid** (the
   "flicker ceiling" is a decoding artifact, not ocean physics).
2. After recovery, the tonal SNR of Event S5 is ~1.2× (13 tonals,
   0.7–1.5), and official-convention CSDM beamforming does not focus —
   the archive as distributed does not support the measured-pressure
   validation path (full evidence chain in manuscript ch. 16).
3. The decoder is deterministic and permanent; all downstream results in
   the associated manuscript use the recovered waveforms.

## 5. Availability

`io/sio_read.py` (MIT). SWellEx-96 data: SIO archive
(swellex96.ucsd.edu) — not redistributed here.
