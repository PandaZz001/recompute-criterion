# -*- coding: utf-8 -*-
"""W5 全球浅海点位普查（2026-11-26，迭代任务）。

浅海判据零假复用性质的全球大样本检验：8 个新陆架点（不同海区，
ETOPO 定深 15–170 m）× 夏−冬季节差 × α ∈ {0.5, 1} 两档 = 16 新
样本，与 W2 浅海三站点 12 样本合并为 28 样本 11 站点。

底质统一用砂泥典型 1600/1.7/0.5（W4 已证明底质不影响零假复用
安全性质）。500 Hz，Θ*=0.8，源 20/收 50 m，窗 1–20 km。

输出：work/w5_global_shelf.json + figures/w5_global_shelf.png
"""

import json
import os
import subprocess
import sys

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
W5W = os.path.join(WORK, "w5")
FIG = os.path.join(HERE, "figures")
KEXE = os.environ.get(
    "KRAKEN_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\Kraken\kraken.exe")
FEXE = os.environ.get(
    "KRAKEN_FIELD_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\KrakenField\field.exe")
NAVY, TEAL, ORANGE, RED = "#1F3A5F", "#2A9D8F", "#E86A17", "#C12C2C"
os.makedirs(W5W, exist_ok=True)

WOA_DIR = os.environ.get("WOA23_DIR", r"E:\AI_Codes\水声")
CB, RHOB, ALPHAB = 1600.0, 1.7, 0.5
ALPHAS = [0.5, 1.0]
FREQ, SD, RD = 500.0, 20.0, 50.0

SITES = [
    dict(name="北海", lon=3.0, lat=56.0),
    dict(name="波斯湾", lon=51.0, lat=26.5),
    dict(name="白令海陆架", lon=190.0, lat=62.0),
    dict(name="鄂霍次克海", lon=144.0, lat=53.0),
    dict(name="英吉利海峡", lon=0.5, lat=50.5),
    dict(name="阿拉弗拉海", lon=135.0, lat=-10.0),
    dict(name="孟加拉湾北部", lon=89.0, lat=21.0),
    dict(name="巽他陆架", lon=108.0, lat=-2.0),
]


def load_ssp(lon, lat, season):
    fn = os.path.join(WOA_DIR, f"woa23_ssp_s{season}.nc")
    with xr.open_dataset(fn, engine="h5netcdf") as ds:
        p = ds["ssp"].sel(lon=lon, lat=lat, method="nearest")
        dep = ds["depth"].values
        vals = p.values
    valid = ~np.isnan(vals)
    dep, vals = dep[valid], vals[valid]
    dz = 0.5
    z = np.arange(0.0, dep[-1] + dz / 2, dz)
    c = np.interp(z, dep, vals, left=vals[0], right=vals[-1])
    return z, c


def run_tl(tag, D, ssp_list, sd, rd, rmax=20.0, nr=401):
    stem = f"w5_{tag}"
    for ext in (".env", ".mod", ".flp", ".shd", ".prt"):
        p = os.path.join(W5W, stem + ext)
        if os.path.exists(p):
            os.remove(p)
    zgrid = [float(z) for z, _ in ssp_list]
    write_env_kraken(os.path.join(W5W, stem + ".env"), FREQ, ssp_list, D,
                     sd, zgrid, rmax, nr, title="W5",
                     bottom=(CB, RHOB, ALPHAB), c_lim=(1400.0, 1650.0))
    res = subprocess.run([KEXE, stem], cwd=W5W, capture_output=True,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(f"kraken {tag} failed: {res.stdout[-300:]}")
    write_flp(os.path.join(W5W, stem + ".flp"), sd=sd, rd=[rd],
              rmax=rmax, nr=nr)
    shd = run_kraken_field(FEXE, os.path.join(W5W, stem + ".flp"), W5W)
    import struct
    raw = open(os.path.join(W5W, stem + ".shd"), "rb").read()
    rl = struct.unpack("I", raw[:4])[0] * 4
    P = np.frombuffer(raw[10 * rl:], dtype="<c8").copy()[:nr]
    return -20.0 * np.log10(np.abs(P) + 1e-300)


def criterion(S, dc, zs, c_base, sd, rd, r_eval=25.0):
    dc_s = np.interp(S.z, zs, dc)
    cz = np.interp(S.z, zs, c_base)
    integ = np.trapezoid(dc_s[:, None] * S.phi.real ** 2
                         / cz[:, None] ** 3, S.z, axis=0)
    dk = -(S.w ** 2 / S.k) * integ
    P = S.phi.real
    dcw = dc_s * (-2.0 / cz ** 3)
    I = np.trapezoid(dcw[:, None, None] * P[:, None, :] * P[:, :, None],
                     S.z, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        Cm = S.w ** 2 * I / (S.k[:, None] ** 2 - S.k[None, :] ** 2)
    Cm[np.diag_indices(S.M)] = 0.0
    dph = S.phi @ Cm.T
    phiv = S._phi_at(rd)
    phis = S._phi_at(sd)
    dr = dph[int(np.argmin(np.abs(S.z - rd)))]
    ds = dph[int(np.argmin(np.abs(S.z - sd)))]
    dw = ds * phiv + phis * dr
    return theta_phase(S, dk, dw=dw, zs=sd, z=rd, r_eval=r_eval)


print("== W5 全球浅海点位普查（8 新点 × 2 档 = 16 新样本）==\n")
rows = []
for site in SITES:
    z_w, c_winter = load_ssp(site["lon"], site["lat"], 13)
    z_s, c_summer = load_ssp(site["lon"], site["lat"], 15)
    D = float(z_w[-1])
    dc0 = np.interp(z_w, z_s, c_summer) - c_winter
    ssp_base = [(float(z), float(c)) for z, c in zip(z_w, c_winter)]
    tl0 = run_tl(f"{site['name'][:2]}_base", D, ssp_base, SD, RD)
    stem = f"w5_{site['name'][:2]}_base"
    mod = ModResult(os.path.join(W5W, stem + ".mod"))
    cbar = float(np.mean(c_winter))
    S = ModeSensitivity(mod, FREQ, cbar, D, CB, RHOB, ALPHAB)
    print(f"[{site['name']}] D={D:.0f} m  Δc 峰峰 "
          f"{np.max(np.abs(dc0)):.1f} m/s  M={mod.M}")
    for al in ALPHAS:
        dc = dc0 * al
        th = criterion(S, dc, z_w, c_winter, SD, RD)
        ssp_p = [(float(z), float(c + d))
                 for z, c, d in zip(z_w, c_winter, dc)]
        tl = run_tl(f"{site['name'][:2]}_{al}", D, ssp_p, SD, RD)
        rho = float(np.corrcoef(tl0, tl)[0, 1])
        rows.append(dict(site=site["name"], alpha=al, th=float(th),
                         rho=rho, D=float(D)))
        print(f"    α={al:4.2f}  Θ={th:7.3f}  ρ={rho:.4f}")

# 并入 W2 浅海三站点
with open(os.path.join(WORK, "w2_woa_env.json"), encoding="utf-8") as f:
    w2 = json.load(f)
w2s = [r for r in w2["rows"] if r["D"] < 300.0]
rows.extend(w2s)

ok = sum(bool(r["th"] <= 0.8) == bool(r["rho"] >= 0.9) for r in rows)
fr = sum(bool(r["th"] <= 0.8) and not bool(r["rho"] >= 0.9) for r in rows)
fm = sum(not bool(r["th"] <= 0.8) and bool(r["rho"] >= 0.9) for r in rows)
print(f"\n合并（n={len(rows)}，11 站点）：ok={ok}  fr={fr}  fm={fm}")
sites_all = sorted({r["site"] for r in rows})
print(f"站点：{'、'.join(sites_all)}")

out = dict(rows=rows, n=len(rows), ok=ok, fr=fr, fm=fm,
           sites=sites_all)
with open(os.path.join(WORK, "w5_global_shelf.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# ---- 图 ----
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2))
ax = axs[0]
cmap = plt.get_cmap("tab20")
for k, sname in enumerate(sites_all):
    sub = [r for r in rows if r["site"] == sname]
    ax.plot([r["th"] for r in sub], [r["rho"] for r in sub], "o", ms=5,
            color=cmap(k % 20), alpha=0.8, label=sname)
ax.axvline(0.8, color=RED, ls=":", lw=1.2, label="Θ* = 0.8")
ax.axhline(0.9, color=RED, ls="-.", lw=1.2, label="ρ* = 0.9")
ax.set_xscale("log")
ax.set_xlabel("Θ (rad)")
ax.set_ylabel("真值 ρ")
ax.set_title("11 站点浅海 ρ(Θ)（夏−冬季节差）")
ax.legend(fontsize=6.5, ncol=2)
ax.grid(alpha=0.3, which="both")
ax = axs[1]
sub_fr = [r for r in rows if bool(r["th"] <= 0.8)
          and not bool(r["rho"] >= 0.9)]
sub_ok = [r for r in rows if bool(r["th"] <= 0.8) == bool(r["rho"] >= 0.9)]
ax.bar([0], [len(sub_ok) / len(rows)], 0.3, color=TEAL, label="正确率")
ax.bar([1], [len(sub_fr) / len(rows)], 0.3, color=RED, label="假复用率")
ax.bar([2], [fm / len(rows)], 0.3, color="#8A8F98", label="假重算率")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels([f"正确率 {ok}/{len(rows)}",
                    f"假复用 {fr}", f"假重算 {fm}"], fontsize=9)
ax.set_ylabel("比率")
ax.set_title("全球浅海 28 样本混淆矩阵")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.suptitle("W5 全球浅海点位普查：判据零假复用性质的全球检验", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG, "w5_global_shelf.png"), dpi=110)
print("\n[Done] ->", os.path.join(WORK, "w5_global_shelf.json"))
