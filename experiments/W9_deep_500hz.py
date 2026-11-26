# -*- coding: utf-8 -*-
"""W9 深海 500 Hz 复验（2026-11-26，迭代任务）。

W3 深海标定在 100 Hz。本实验复验西太深海点在 500 Hz（模态数
~3100，Kraken 极限）下的判据行为，检验深海保守操作点 0.667 rad
经 F1 标度律换算到 500 Hz（0.667×5 ≈ 3.33 rad）的适用性。

若 Kraken 在 3100 模态下崩溃/超时 → 诚实降级为 200 Hz 并记录。

输出：work/w9_deep_500hz.json + figures/w9_deep_500hz.png
"""

import json
import os
import subprocess
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "criterion"))
from at_core import (ModResult, write_env_kraken, write_flp,  # noqa: E402
                     run_kraken_field)
from mode_sensitivity import ModeSensitivity, theta_phase  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
W9W = os.path.join(WORK, "w9")
FIG = os.path.join(HERE, "figures")
KEXE = os.environ.get(
    "KRAKEN_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\Kraken\kraken.exe")
FEXE = os.environ.get(
    "KRAKEN_FIELD_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\KrakenField\field.exe")
NAVY, TEAL, ORANGE, RED = "#1F3A5F", "#2A9D8F", "#E86A17", "#C12C2C"
os.makedirs(W9W, exist_ok=True)

WOA_DIR = os.environ.get("WOA23_DIR", r"E:\AI_Codes\水声")
LON, LAT = 135.0, 15.0          # 西太
SD, RD = 1000.0, 1000.0
CB, RHOB, ALPHAB = 1550.0, 1.5, 0.5
ALPHAS = [0.5, 1.0]
TH_STAR_500 = 0.667 * 5.0       # F1 标度律换算（β=1 口径）
FREQ = 500.0
TRIAL_FREQS = [500.0, 200.0]    # 500 失败时降级 200


def load_ssp(season):
    fn = os.path.join(WOA_DIR, f"woa23_ssp_s{season}.nc")
    with xr.open_dataset(fn, engine="h5netcdf") as ds:
        p = ds["ssp"].sel(lon=LON, lat=LAT, method="nearest")
        dep = ds["depth"].values
        vals = p.values
    valid = ~np.isnan(vals)
    dep, vals = dep[valid], vals[valid]
    dz = 1.0
    z = np.arange(0.0, dep[-1] + dz / 2, dz)
    c = np.interp(z, dep, vals, left=vals[0], right=vals[-1])
    return z, c


def run_tl(tag, freq, D, ssp_list, timeout_s=600):
    stem = f"w9_{tag}"
    for ext in (".env", ".mod", ".flp", ".shd", ".prt"):
        p = os.path.join(W9W, stem + ext)
        if os.path.exists(p):
            os.remove(p)
    zgrid = [float(z) for z, _ in ssp_list]
    write_env_kraken(os.path.join(W9W, stem + ".env"), freq, ssp_list, D,
                     SD, zgrid, 50.0, 401, title="W9",
                     bottom=(CB, RHOB, ALPHAB), c_lim=(1400.0, 1650.0))
    t0 = time.time()
    try:
        res = subprocess.run([KEXE, stem], cwd=W9W, capture_output=True,
                             text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"kraken {freq} Hz 超时 {timeout_s}s")
    dt_k = time.time() - t0
    if res.returncode != 0:
        raise RuntimeError(f"kraken {freq} Hz failed: "
                           f"{res.stdout[-300:]}")
    write_flp(os.path.join(W9W, stem + ".flp"), sd=SD, rd=[RD],
              rmax=50.0, nr=401)
    shd = run_kraken_field(FEXE, os.path.join(W9W, stem + ".flp"), W9W)
    import struct
    raw = open(os.path.join(W9W, stem + ".shd"), "rb").read()
    rl = struct.unpack("I", raw[:4])[0] * 4
    P = np.frombuffer(raw[10 * rl:], dtype="<c8").copy()[:401]
    return (-20.0 * np.log10(np.abs(P) + 1e-300)), dt_k


def criterion(S, dc, zs, c_base):
    dc_s = np.interp(S.z, zs, dc)
    cz = np.interp(S.z, zs, c_base)
    integ = np.trapezoid(dc_s[:, None] * S.phi.real ** 2
                         / cz[:, None] ** 3, S.z, axis=0)
    dk = -(S.w ** 2 / S.k) * integ
    P = S.phi.real
    dcw = dc_s * (-2.0 / cz ** 3)
    # I[m,n] = ∫dcw·P_m·P_n dz：矩阵乘 + 梯形首尾修正（M=1282 时
    # 外积式需 61.8 GB，此式仅 M² = 13 MB）
    dz = float(np.mean(np.diff(S.z)))
    W = P * dcw[:, None]                    # (Nz, M)
    I = W.T @ P                             # (M, M) 矩形和
    I -= 0.5 * (np.outer(W[0], P[0]) + np.outer(W[-1], P[-1]))
    I *= dz
    with np.errstate(divide="ignore", invalid="ignore"):
        Cm = S.w ** 2 * I / (S.k[:, None] ** 2 - S.k[None, :] ** 2)
    Cm[np.diag_indices(S.M)] = 0.0
    dph = S.phi @ Cm.T
    phiv = S._phi_at(RD)
    phis = S._phi_at(SD)
    dr = dph[int(np.argmin(np.abs(S.z - RD)))]
    ds = dph[int(np.argmin(np.abs(S.z - SD)))]
    dw = ds * phiv + phis * dr
    return theta_phase(S, dk, dw=dw, zs=SD, z=RD, r_eval=50.0)


print("== W9 深海 500 Hz 复验（西太）==\n")
z_w, c_winter = load_ssp(13)
z_s, c_summer = load_ssp(15)
D = float(z_w[-1])
dc0 = np.interp(z_w, z_s, c_summer) - c_winter
ssp_base = [(float(z), float(c)) for z, c in zip(z_w, c_winter)]

freq_used = None
rows = []
for freq_try in TRIAL_FREQS:
    try:
        print(f"尝试 {freq_try:.0f} Hz …")
        tl0, dt_k = run_tl("base", freq_try, D, ssp_base)
        freq_used = freq_try
        print(f"  Kraken {freq_try:.0f} Hz 基线完成：{dt_k:.1f} s")
        break
    except RuntimeError as e:
        print(f"  {freq_try:.0f} Hz 失败：{str(e)[:100]}")

if freq_used is None:
    raise SystemExit("500/200 Hz 均失败，W9 不可行（诚实降级记录）")

mod = ModResult(os.path.join(W9W, "w9_base.mod"))
cbar = float(np.mean(c_winter))
S = ModeSensitivity(mod, freq_used, cbar, D, CB, RHOB, ALPHAB)
print(f"  M={mod.M}")
th_star = 0.667 * (freq_used / 100.0)  # F1 标度律换算

for al in ALPHAS:
    dc = dc0 * al
    th = criterion(S, dc, z_w, c_winter)
    ssp_p = [(float(z), float(c + d))
             for z, c, d in zip(z_w, c_winter, dc)]
    tl, dt_k = run_tl(f"a{al}", freq_used, D, ssp_p)
    rho = float(np.corrcoef(tl0, tl)[0, 1])
    rows.append(dict(alpha=al, th=float(th), rho=rho, freq=freq_used))
    print(f"  α={al:4.2f}  Θ={th:7.3f}  ρ={rho:.4f}（{dt_k:.0f} s）")

ok = sum(bool(r["th"] <= th_star) == bool(r["rho"] >= 0.9) for r in rows)
fr = sum(bool(r["th"] <= th_star) and not bool(r["rho"] >= 0.9)
         for r in rows)
fm = sum(not bool(r["th"] <= th_star) and bool(r["rho"] >= 0.9)
         for r in rows)
print(f"\n复验（f={freq_used:.0f} Hz，Θ*={th_star:.2f} 按 F1 换算）："
      f"ok={ok}/{len(rows)}  fr={fr}  fm={fm}")

out = dict(freq=float(freq_used), n=len(rows), ok=ok, fr=fr, fm=fm,
           th_star=float(th_star), rows=rows,
           degraded=bool(freq_used != 500.0),
           tried=[float(f) for f in TRIAL_FREQS])
with open(os.path.join(WORK, "w9_deep_500hz.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

fig, ax = plt.subplots(figsize=(7.5, 5.0))
tc = np.linspace(0.02, max(r["th"] for r in rows) * 1.2, 100)
ax.plot(tc, 0.665 + 0.335 * np.exp(-0.344 * tc ** 1.444),
        color=NAVY, lw=1.8, ls="--", label="W7 深海扩展拟合（n=62）")
for r in rows:
    ax.plot(r["th"], r["rho"], "o", ms=8, color=ORANGE)
ax.axvline(th_star, color=RED, ls=":", lw=1.4,
           label=f"Θ*={th_star:.2f}（F1 换算）")
ax.axhline(0.9, color="#8A8F98", ls="-.", lw=1)
ax.set_xscale("log")
ax.set_xlabel("Θ (rad)")
ax.set_ylabel("真值 ρ")
ax.set_title(f"W9 西太 {freq_used:.0f} Hz 复验 vs 深海标定曲线")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "w9_deep_500hz.png"), dpi=110)
print("\n[Done] ->", os.path.join(WORK, "w9_deep_500hz.json"))
