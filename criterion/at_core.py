# -*- coding: utf-8 -*-
"""
Minimal self-contained driver for the official Acoustics Toolbox (Bellhop),
github.com/oalib-acoustics/Acoustics-Toolbox (compiled from source on this
machine).  Deliberately independent of any third-party wrapper: env files are
written per the AT input format, and .shd files are parsed per the AT output
format documented in the toolbox (doc/at.pdf).

.shd layout (all records fixed length, reclen = first int32, in bytes):
  rec 1 (idx 1): title (80 chars)
  rec 2 (idx 2): freq, Ntheta, NSx, NSy, NSz, NRz, NRr, ... (int32 each)
  rec 7 (idx 7): sz   (NSz  x float32)
  rec 8 (idx 8): rz   (NRz  x float32)
  rec 9 (idx 9): rr   (NRr  x float64)
  rec 10+     : pressure field, complex64, NRz records x NRr samples
"""

import os
import struct
import subprocess

import numpy as np


class ShdResult:
    def __init__(self, path):
        with open(path, "rb") as f:
            data = f.read()
        reclen = struct.unpack("I", data[:4])[0]          # bytes per record
        nrec = len(data) // (reclen * 4)

        def rd(rec, val):
            addr = 4 * (rec * reclen + val)
            return struct.unpack("I", data[addr:addr + 4])[0]

        def rdf(rec, n):
            out = []
            for i in range(n):
                addr = 4 * (rec * reclen + i)
                out.append(struct.unpack("f", data[addr:addr + 4])[0])
            return np.array(out)

        def rdd(rec, n):
            out = []
            for i in range(n):
                addr = 4 * (rec * reclen + 2 * i)
                out.append(struct.unpack("d", data[addr:addr + 8])[0])
            return np.array(out)

        self.title = data[4:84].decode("ascii", "replace").strip()
        # note: the .shd header does NOT carry the source frequency (it
        # lives in the .env); record 2 holds Ntheta, NSx, NSy, NSz, NRz, NRr
        self.NSz = rd(2, 4)
        self.NRz = rd(2, 5)
        self.NRr = rd(2, 6)
        self.sz = rdf(7, self.NSz)
        self.rz = rdf(8, self.NRz)
        self.rr = rdd(9, self.NRr)
        # AT stores rr in km for some versions, meters for others:
        # auto-detect by comparing the last point with the env's max range
        if self.rr[-1] > 1000.0:
            self.rr = self.rr / 1000.0                     # km
        field = []
        for rec in range(10, 10 + self.NRz):
            row = []
            for i in range(self.NRr):
                addr = 4 * (rec * reclen + 2 * i)
                row.append(complex(*struct.unpack("ff",
                                                  data[addr:addr + 8])))
            field.append(row)
        self.pressure = np.array(field)

    def tl(self):
        """TL (dB) = -20 log10 |p| (AT convention, no source-level offset)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return -20.0 * np.log10(np.abs(self.pressure) + 1e-300)


def run_bellhop(exe, env_path, workdir):
    """Run `bellhop.exe <stem>` in workdir; return ShdResult of <stem>.shd."""
    stem = os.path.splitext(os.path.basename(env_path))[0]
    res = subprocess.run([exe, stem], cwd=workdir, capture_output=True,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(f"bellhop failed ({res.returncode}): "
                           f"{res.stdout[-500:]}{res.stderr[-500:]}")
    return ShdResult(os.path.join(workdir, stem + ".shd"))


def write_env(path, freq, ssp, depth, sd, rd, rmax, nr, title="",
              bottom=(1600.0, 1.8, 0.8), nbeams=0, alpha=(-20.3, 20.3),
              step=0.0, run_type="CG"):
    """Write a Bellhop .env file.

    ssp: (n, 2) array [depth, c];  rd: array of receiver depths (m);
    sd: scalar source depth (m);  rmax: km;  nr: range points.
    bottom: (cb, rho, atten) acousto-elastic halfspace below `depth`.
    """
    lines = [
        f"'{title or 'env'}'",
        f"{freq}          ! FREQ (Hz)",
        "1              ! NMEDIA",
        "'CVW'          ! SSPOPT",
        f"{len(ssp)}  0.0  {depth:.1f}   ! DEPTH of bottom (m)",
    ]
    for z, c in ssp:
        lines.append(f"  {z:9.1f}  {c:9.2f}  /")
    lines.append("'A' 0.0")
    lines.append(f" {depth:.1f}  {bottom[0]:.2f} 0.0 {bottom[1]:.1f} "
                 f"{bottom[2]:.1f} /")
    lines += [
        "1              ! NSD",
        f"  {sd:.1f} /     ! SD(1:NSD) (m)",
        f"{len(rd)}              ! NRD",
        "  " + " ".join(f"{z:.1f}" for z in rd) + " /   ! RD(1:NRD) (m)",
        f"{nr}            ! NR",
        f"  0.0  {rmax:.1f} /   ! R(1:NR) (km)",
        f"'{run_type}'       ! 'R/C/I/S' run type",
        f"{nbeams}             ! NBEAMS (0 = auto)",
        f" {alpha[0]:.1f} {alpha[1]:.1f} /      ! ALPHA1, 2 (degrees)",
        f"{step:.1f}  {depth + 1.0:.1f}  {rmax + 1.0:.1f}   "
        "! STEP (m), ZBOX (m), RBOX (km)",
    ]
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")


def write_env_kraken(path, freq, ssp, depth, sd, rd, rmax, nr, title="",
                     bottom=(1600.0, 1.8, 0.8), c_lim=(1400.0, 1700.0)):
    """Write a Kraken .env file (normal-mode kernel, same toolbox).

    Pekeris-style: vacuum top, acousto-elastic bottom halfspace.
    c_lim: (cLow, cHigh) phase-speed bracket for the mode search (m/s);
    for a Pekeris waveguide use cLow = min(c) and cHigh = cb (slightly
    padded) so that all trapped modes fall inside.
    """
    lines = [
        f"'{title or 'env'}'",
        f"{freq}          ! FREQ (Hz)",
        "1              ! NMEDIA",
        "'NVW'          ! SSPOPT",
        f"0  0.0  {depth:.1f}   ! NSSP(auto), sigma, DEPTH of bottom (m)",
    ]
    for i, (z, c) in enumerate(ssp):
        if i == 0:
            lines.append(f"  {z:9.1f}  {c:9.2f}  0.0  1.0  0.0  0.0")
        else:
            lines.append(f"  {z:9.1f}  {c:9.2f} /")
    lines += [
        "'A' 0.0",
        f" {depth:.1f}  {bottom[0]:.2f} 0.0 {bottom[1]:.2f} {bottom[2]:.2f} /",
        f"{c_lim[0]:.0f}  {c_lim[1]:.0f}     ! cLow cHigh (m/s)",
        f"{rmax:.1f}        ! RMAX (km)",
        "1              ! NSD",
        f"  {sd:.1f} /     ! SD(1:NSD) (m)",
        f"{len(rd)}              ! NRD",
        "  " + " ".join(f"{z:.1f}" for z in rd) + " /   ! RD(1:NRD) (m)",
    ]
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")


def write_flp(path, sd, rd, rmax, nr, title="/", opt="'R'", m=9999,
              r_only=None):
    """Write a KrakenField .flp (field parameters) file.

    r_only: 单个接收距离（km）——NRr=1 时 Field 不输出场，必须写
    两个点（0.0 与目标距离），场取第 2 列（pressure[:, 1]）。
    """
    if r_only is not None:
        nr, rr = 2, f" 0.0  {r_only:.3f} /"
    else:
        rr = f" 0.0  {rmax:.1f} /"
    lines = [
        f"{title},             ! TITLE",
        f"{opt}             ! OPT (R=point source, A=adiabatic)",
        f"{m}              ! M (number of modes to include)",
        "1              ! NProf",
        f"     0.0 {rmax:.1f} /   ! RProf(1:NProf) (km)",
        f"{nr}             ! NRr",
        f"{rr}   ! Rr(1:NRr) (km)",
        "1              ! NSz",
        f"   {sd:.1f} /    ! Sz(1:NSz) (m)",
        f"{len(rd)}              ! NRz",
        "  " + " ".join(f"{z:.1f}" for z in rd) + " /   ! Rz(1:NRz) (m)",
        f"{len(rd)}              ! NRo (must equal NRz)",
        "     " + " ".join("0.0" for _ in rd) + " /   ! Ro(1:NRo) (m)",
    ]
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")


class ModResult:
    """Parser for Kraken's .mod file (mode eigenvalues + eigenfunctions).

    Direct-access binary: record length = 4*L longwords, L = first int32.
    rec 1: L, title(80B), Nfreq, NMedia, NzTab, NzTab
    rec 5: zTab (NzTab x float32)
    rec 7: halfspace info (BC, cP, cS, rho, depth x2)
    rec 8..7+M: eigenfunction PhiTab (NzTab x complex64) per mode
    rec 8+M: eigenvalues k (M x complex64)
    """

    def __init__(self, path):
        with open(path, "rb") as f:
            data = f.read()
        L = struct.unpack("I", data[:4])[0]
        rlen = 4 * L

        def rec(n):                       # 1-based record index
            return data[(n - 1) * rlen:n * rlen]

        self.title = rec(1)[4:84].decode("ascii", "replace").strip()
        hdr = struct.unpack("III", rec(1)[84:96])
        self.nfreq, self.nmedia, self.nztab = hdr
        self.ztab = np.frombuffer(rec(5)[:4 * self.nztab],
                                  dtype="<f4").copy()
        # halfspace record: BC(8B) cP cS rho depth | BC(8B) cP cS rho depth
        hs = rec(7)
        self.bot_rho = struct.unpack("f", hs[20:24])[0]
        self.bot_cp = struct.unpack("f", hs[12:16])[0]
        self.bot_d = struct.unpack("f", hs[24:28])[0]
        # record layout (single-frequency case):
        #   rec 6: mode count M; rec 8..7+M: eigenfunctions (NzTab x c64);
        #   rec 8+M: eigenvalues k (M x c128, chunked in L/2 per record)
        nrec = len(data) // rlen
        M = nrec - 8
        self.M = M
        self.phi = np.frombuffer(
            b"".join(rec(7 + m) for m in range(1, M + 1)),
            dtype="<c8").reshape(M, self.nztab).T
        # k record: empirically complex64 despite the complex(kind=8) arg
        self.k = np.frombuffer(rec(8 + M)[:8 * M], dtype="<c8").copy()


def run_kraken(exe, env_path, workdir):
    """Run `kraken.exe <stem>` in workdir; return ModResult.

    Note: kraken.exe computes modes only (.mod); the pressure field
    comes from the KrakenField post-processor (run_kraken_field).
    """
    stem = os.path.splitext(os.path.basename(env_path))[0]
    res = subprocess.run([exe, stem], cwd=workdir, capture_output=True,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(f"kraken failed ({res.returncode}): "
                           f"{res.stdout[-500:]}{res.stderr[-500:]}")
    return ModResult(os.path.join(workdir, stem + ".mod"))


def run_kraken_field(exe, flp_path, workdir):
    """Run `field.exe <stem>` (KrakenField post-processor) in workdir."""
    stem = os.path.splitext(os.path.basename(flp_path))[0]
    res = subprocess.run([exe, stem], cwd=workdir, capture_output=True,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(f"field failed ({res.returncode}): "
                           f"{res.stdout[-500:]}{res.stderr[-500:]}")
    return ShdResult(os.path.join(workdir, stem + ".shd"))
