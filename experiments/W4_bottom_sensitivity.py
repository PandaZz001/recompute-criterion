# -*- coding: utf-8 -*-
"""W4 底质敏感性扫描（2026-11-26，迭代任务）。

W2/W2b 的底质是文献典型值（诚实边界：WOA 网格点无实测地声参数）。
本实验对 5 个站点各加两套底质变体（软泥 / 硬砂），与 W2 原假设
三套对比：混淆矩阵结论（方向正确性、零假复用）是否对底质参数稳健。

- 软泥：cb=1520, ρ=1.4, α=0.3
- 原假设（W2）：见 SITES
- 硬砂：cb=1800, ρ=2.0, α=0.8
季节对：夏−冬（s15−s13），α ∈ {0.25, 0.5, 1, 2} 四档。

输出：work/w4_bottom_sensitivity.json + figures/w4_bottom_sensitivity.png
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
W4W = os.path.join(WORK, "w4")
FIG = os.path.join(HERE, "figures")
KEXE = os.environ.get(
    "KRAKEN_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\Kraken\kraken.exe")
FEXE = os.environ.get(
    "KRAKEN_FIELD_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\KrakenField\field.exe")
NAVY, TEAL, ORANGE, RED = "#1F3A5F", "#2A9D8F", "#E86A17", "#C12C2C"
os.makedirs(W4W, exist_ok=True)

WOA_DIR = os.environ.get("WOA23_DIR", r"E:\AI_Codes\水声")
ALPHAS = [0.25, 0.5, 1.0, 2.0]
VARIANTS = [("软泥", 1520.0, 1.4, 0.3), ("硬砂", 1800.0, 2.0, 0.8)]

SITES = [
    dict(name="南海北部陆架", lon=113.0, lat=20.0, cb=1600.0, rho=1.7,
         alpha=0.5, freq=500.0, sd=20.0, rd=50.0, r_lo=1.0, r_hi=20.0,
         r_eval=25.0),
    dict(name="黄海", lon=123.0, lat=35.0, cb=1650.0, rho=1.9,
         alpha=0.6, freq=500.0, sd=20.0, rd=50.0, r_lo=1.0, r_hi=20.0,
         r_eval=25.0),
    dict(name="东海陆架", lon=124.0, lat=30.0, cb=1650.0, rho=1.9,
         alpha=0.6, freq=500.0, sd=20.0, rd=50.0, r_lo=1.0, r_hi=20.0,
         r_eval=25.0),
    dict(name="西太", lon=135.0, lat=15.0, cb=1550.0, rho=1.5,
         alpha=0.5, freq=100.0, sd=1000.0, rd=1000.0, r_lo=10.0,
         r_hi=50.0, r_eval=50.0),
    dict(name="北大西洋", lon=330.0, lat=40.0, cb=1550.0, rho=1.5,
         alpha=0.5, freq=100.0, sd=1000.0, rd=1000.0, r_lo=10.0,
         r_hi=50.0, r_eval=50.0),
]


def load_ssp(lon, lat, season):
    fn = os.path.join(WOA_DIR, f"woa23_ssp_s{season}.nc")
    with xr.open_dataset(fn, engine="h5netcdf") as ds:
        p = ds["ssp"].sel(lon=lon, lat=lat, method="nearest")
        dep = ds["depth"].values
        vals = p.values
    valid = ~np.isnan(vals)
    dep, vals = dep[valid], vals[valid]
    dz = 0.5 if vals.size < 300 else 1.0
    z = np.arange(0.0, dep[-1] + dz / 2, dz)
    c = np.interp(z, dep, vals, left=vals[0], right=vals[-1])
    return z, c


def run_tl(tag, freq, D, ssp_list, sd, rd, rmax, nr, cb, rho, alpha):
    stem = f"w4_{tag}"
    for ext in (".env", ".mod", ".flp", ".shd", ".prt"):
        p = os.path.join(W4W, stem + ext)
        if os.path.exists(p):
            os.remove(p)
    zgrid = [float(z) for z, _ in ssp_list]
    write_env_kraken(os.path.join(W4W, stem + ".env"), freq, ssp_list, D,
                     sd, zgrid, rmax, nr, title="W4",
                     bottom=(cb, rho, alpha), c_lim=(1400.0, 1650.0))
    res = subprocess.run([KEXE, stem], cwd=W4W, capture_output=True,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(f"kraken {tag} failed: {res.stdout[-300:]}")
    write_flp(os.path.join(W4W, stem + ".flp"), sd=sd, rd=[rd],
              rmax=rmax, nr=nr)
    shd = run_kraken_field(FEXE, os.path.join(W4W, stem + ".flp"), W4W)
    import struct
    raw = open(os.path.join(W4W, stem + ".shd"), "rb").read()
    rl = struct.unpack("I", raw[:4])[0] * 4
    P = np.frombuffer(raw[10 * rl:], dtype="<c8").copy()[:nr]
    return -20.0 * np.log10(np.abs(P) + 1e-300)


def criterion(S, dc, zs, c_base, r_eval, sd, rd):
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


print("== W4 底质敏感性扫描（软泥/硬砂 vs W2 原假设）==\n")
rows = []
for site in SITES:
    z_w, c_winter = load_ssp(site["lon"], site["lat"], 13)
    z_s, c_summer = load_ssp(site["lon"], site["lat"], 15)
    D = float(z_w[-1])
    dc0 = np.interp(z_w, z_s, c_summer) - c_winter
    ssp_base = [(float(z), float(c)) for z, c in zip(z_w, c_winter)]
    s = site
    for vlabel, cb, rho_b, alpha_b in VARIANTS:
        tag = f"{s['name'][:2]}_{vlabel}"
        tl0 = run_tl(f"{tag}_base", s["freq"], D, ssp_base, s["sd"],
                     s["rd"], s["r_hi"], 401, cb, rho_b, alpha_b)
        stem = f"w4_{tag}_base"
        mod = ModResult(os.path.join(W4W, stem + ".mod"))
        cbar = float(np.mean(c_winter))
        S = ModeSensitivity(mod, s["freq"], cbar, D, cb, rho_b, alpha_b)
        for al in ALPHAS:
            dc = dc0 * al
            th = criterion(S, dc, z_w, c_winter, s["r_eval"], s["sd"],
                           s["rd"])
            ssp_p = [(float(z), float(c + d))
                     for z, c, d in zip(z_w, c_winter, dc)]
            tl = run_tl(f"{tag}_{al}", s["freq"], D, ssp_p, s["sd"],
                        s["rd"], s["r_hi"], 401, cb, rho_b, alpha_b)
            rho = float(np.corrcoef(tl0, tl)[0, 1])
            rows.append(dict(site=s["name"], bottom=vlabel, alpha=al,
                             th=float(th), rho=rho, D=float(D)))
        print(f"  [{s['name']}/{vlabel}] 完成（D={D:.0f} m）")

# 并入 W2 原假设数据
with open(os.path.join(WORK, "w2_woa_env.json"), encoding="utf-8") as f:
    w2 = json.load(f)
for r in w2["rows"]:
    r["bottom"] = "原假设"
rows.extend(w2["rows"])

th_star = lambda freq: 0.8 * (freq / 500.0)
print("\n三底质混淆矩阵对比（ρ_req=0.9）：")
summary = {}
for blabel in ("软泥", "原假设", "硬砂"):
    sub = [r for r in rows if r["bottom"] == blabel]
    ts_all = [th_star(500.0) if r["D"] < 300 else th_star(100.0)
              for r in sub]
    ok = sum(1 for r, ts in zip(sub, ts_all)
             if bool(r["th"] <= ts) == bool(r["rho"] >= 0.9))
    fr = sum(1 for r, ts in zip(sub, ts_all)
             if bool(r["th"] <= ts) and not bool(r["rho"] >= 0.9))
    fm = sum(1 for r, ts in zip(sub, ts_all)
             if not bool(r["th"] <= ts) and bool(r["rho"] >= 0.9))
    summary[blabel] = dict(n=len(sub), ok=ok, fr=fr, fm=fm)
    print(f"  {blabel:<4} n={len(sub):>2}  ok={ok}  fr={fr}  fm={fm}")

# 底质对 Θ 与 ρ 的直接扰动幅度
base = {r["site"] + str(r["alpha"]): r for r in rows
        if r["bottom"] == "原假设"}
dth, drho = [], []
for r in rows:
    if r["bottom"] == "原假设":
        continue
    key = r["site"] + str(r["alpha"])
    b = base.get(key)
    if b is None:
        continue
    dth.append(abs(r["th"] - b["th"]) / max(b["th"], 1e-9))
    drho.append(abs(r["rho"] - b["rho"]))
print(f"\n底质扰动幅度：Θ 相对变化中位 {np.median(dth):.2%} | "
      f"ρ 绝对变化中位 {np.median(drho):.3f}")

out = dict(rows=rows, summary=summary, dth_median=float(np.median(dth)),
           drho_median=float(np.median(drho)))
with open(os.path.join(WORK, "w4_bottom_sensitivity.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# ---- 图 ----
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2))
ax = axs[0]
bcolors = {"软泥": TEAL, "原假设": NAVY, "硬砂": ORANGE}
for blabel, col in bcolors.items():
    sub = [r for r in rows if r["bottom"] == blabel]
    ax.plot([r["th"] for r in sub], [r["rho"] for r in sub], "o", ms=5,
            color=col, alpha=0.7, label=blabel)
ax.set_xscale("log")
ax.set_xlabel("Θ (rad)")
ax.set_ylabel("真值 ρ")
ax.set_title("ρ(Θ) 散点的底质敏感性（三底质叠加）")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")
ax = axs[1]
xs = np.arange(3)
for k, (blabel, v) in enumerate(summary.items()):
    n = v["n"]
    ax.bar(xs[k] - 0.2, v["ok"] / n, 0.2, color=TEAL,
           label="正确率" if not k else None)
    ax.bar(xs[k], v["fr"] / n, 0.2, color=RED,
           label="假复用率" if not k else None)
    ax.bar(xs[k] + 0.2, v["fm"] / n, 0.2, color="#8A8F98",
           label="假重算率" if not k else None)
ax.set_xticks(xs)
ax.set_xticklabels(["软泥\n1520/1.4/0.3", "原假设", "硬砂\n1800/2.0/0.8"],
                   fontsize=9)
ax.set_ylabel("比率")
ax.set_title(f"三底质混淆矩阵（n={summary['原假设']['n']}/组）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.suptitle("W4 底质敏感性：判据结论对地声参数的稳健性", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG, "w4_bottom_sensitivity.png"), dpi=110)
print("\n[Done] ->", os.path.join(WORK, "w4_bottom_sensitivity.json"))
