# -*- coding: utf-8 -*-
"""W7 全球深海点位扩展（2026-11-26，迭代任务）。

W3 深海标定 56 样本来自 W1 Munk + W2/W2b 西太/北大西洋。本实验加
3 个新深海点（印度洋 60E/10S、南大洋 160E/55S、东太 250E/10N），
夏−冬 × α ∈ {0.5, 1} 两档 = 6 新样本，扩展深海标定样本池。

100 Hz（与 W3 同口径，Θ* 用 W3 保守操作点 0.667 复验），源/收
1000 m，窗 10–50 km。底质淤泥 1550/1.5/0.5（深海典型）。

输出：work/w7_deep_global.json + figures/w7_deep_global.png
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
W7W = os.path.join(WORK, "w7")
FIG = os.path.join(HERE, "figures")
KEXE = os.environ.get(
    "KRAKEN_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\Kraken\kraken.exe")
FEXE = os.environ.get(
    "KRAKEN_FIELD_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\KrakenField\field.exe")
NAVY, TEAL, ORANGE, RED = "#1F3A5F", "#2A9D8F", "#E86A17", "#C12C2C"
os.makedirs(W7W, exist_ok=True)

WOA_DIR = os.environ.get("WOA23_DIR", r"E:\AI_Codes\水声")
FREQ, SD, RD = 100.0, 1000.0, 1000.0
CB, RHOB, ALPHAB = 1550.0, 1.5, 0.5
ALPHAS = [0.5, 1.0]
TH_STAR_DEEP = 0.667  # W3 保守操作点

SITES = [
    dict(name="印度洋", lon=60.0, lat=-10.0),
    dict(name="南大洋", lon=160.0, lat=-55.0),
    dict(name="东太", lon=250.0, lat=10.0),
]


def load_ssp(lon, lat, season):
    fn = os.path.join(WOA_DIR, f"woa23_ssp_s{season}.nc")
    with xr.open_dataset(fn, engine="h5netcdf") as ds:
        p = ds["ssp"].sel(lon=lon, lat=lat, method="nearest")
        dep = ds["depth"].values
        vals = p.values
    valid = ~np.isnan(vals)
    dep, vals = dep[valid], vals[valid]
    dz = 1.0
    z = np.arange(0.0, dep[-1] + dz / 2, dz)
    c = np.interp(z, dep, vals, left=vals[0], right=vals[-1])
    return z, c


def run_tl(tag, D, ssp_list):
    stem = f"w7_{tag}"
    for ext in (".env", ".mod", ".flp", ".shd", ".prt"):
        p = os.path.join(W7W, stem + ext)
        if os.path.exists(p):
            os.remove(p)
    zgrid = [float(z) for z, _ in ssp_list]
    write_env_kraken(os.path.join(W7W, stem + ".env"), FREQ, ssp_list, D,
                     SD, zgrid, 50.0, 401, title="W7",
                     bottom=(CB, RHOB, ALPHAB), c_lim=(1400.0, 1650.0))
    res = subprocess.run([KEXE, stem], cwd=W7W, capture_output=True,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(f"kraken {tag} failed: {res.stdout[-300:]}")
    write_flp(os.path.join(W7W, stem + ".flp"), sd=SD, rd=[RD],
              rmax=50.0, nr=401)
    shd = run_kraken_field(FEXE, os.path.join(W7W, stem + ".flp"), W7W)
    import struct
    raw = open(os.path.join(W7W, stem + ".shd"), "rb").read()
    rl = struct.unpack("I", raw[:4])[0] * 4
    P = np.frombuffer(raw[10 * rl:], dtype="<c8").copy()[:401]
    return -20.0 * np.log10(np.abs(P) + 1e-300)


def criterion(S, dc, zs, c_base):
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
    phiv = S._phi_at(RD)
    phis = S._phi_at(SD)
    dr = dph[int(np.argmin(np.abs(S.z - RD)))]
    ds = dph[int(np.argmin(np.abs(S.z - SD)))]
    dw = ds * phiv + phis * dr
    return theta_phase(S, dk, dw=dw, zs=SD, z=RD, r_eval=50.0)


print("== W7 全球深海点位扩展（3 新点 × 2 档 = 6 新样本）==\n")
rows = []
for site in SITES:
    z_w, c_winter = load_ssp(site["lon"], site["lat"], 13)
    z_s, c_summer = load_ssp(site["lon"], site["lat"], 15)
    D = float(z_w[-1])
    dc0 = np.interp(z_w, z_s, c_summer) - c_winter
    ssp_base = [(float(z), float(c)) for z, c in zip(z_w, c_winter)]
    tl0 = run_tl(f"{site['name'][:2]}_base", D, ssp_base)
    stem = f"w7_{site['name'][:2]}_base"
    mod = ModResult(os.path.join(W7W, stem + ".mod"))
    cbar = float(np.mean(c_winter))
    S = ModeSensitivity(mod, FREQ, cbar, D, CB, RHOB, ALPHAB)
    print(f"[{site['name']}] D={D:.0f} m  Δc 峰峰 "
          f"{np.max(np.abs(dc0)):.1f} m/s  M={mod.M}")
    for al in ALPHAS:
        dc = dc0 * al
        th = criterion(S, dc, z_w, c_winter)
        ssp_p = [(float(z), float(c + d))
                 for z, c, d in zip(z_w, c_winter, dc)]
        tl = run_tl(f"{site['name'][:2]}_{al}", D, ssp_p)
        rho = float(np.corrcoef(tl0, tl)[0, 1])
        rows.append(dict(site=site["name"], alpha=al, th=float(th),
                         rho=rho, D=float(D)))
        print(f"    α={al:4.2f}  Θ={th:7.3f}  ρ={rho:.4f}")

# W3 深海保守操作点复验
ok = sum(bool(r["th"] <= TH_STAR_DEEP) == bool(r["rho"] >= 0.9)
         for r in rows)
fr = sum(bool(r["th"] <= TH_STAR_DEEP) and not bool(r["rho"] >= 0.9)
         for r in rows)
fm = sum(not bool(r["th"] <= TH_STAR_DEEP) and bool(r["rho"] >= 0.9)
         for r in rows)
print(f"\n新点复验（Θ*={TH_STAR_DEEP}）：ok={ok}/{len(rows)}  fr={fr}  "
      f"fm={fm}")

# 并入 W3 深海样本池，重拟合检验稳健性
w3 = json.load(open(os.path.join(WORK, "w3_deep_threshold.json"),
                    encoding="utf-8"))
pts_old = [(p["th"], p["rho"]) for p in w3["pts"]]
pts_new = [(r["th"], r["rho"]) for r in rows]
ths_all = np.array([p[0] for p in pts_old + pts_new])
rhos_all = np.array([p[1] for p in pts_old + pts_new])
from scipy.optimize import curve_fit


def gen(th, lam, p, rf):
    return rf + (1 - rf) * np.exp(-lam * np.abs(th) ** p)


popts, _ = curve_fit(gen, ths_all, rhos_all,
                     p0=[w3["lam"], w3["p"], w3["rho_f"]],
                     bounds=([0.001, 0.5, 0.0], [5.0, 3.0, 0.98]),
                     maxfev=20000)
lam2, p2, rf2 = popts
th_deep2 = ((-np.log((0.9 - rf2) / (1 - rf2))) / lam2) ** (1 / p2)
print(f"\n扩展后深海拟合（n={len(pts_old) + len(pts_new)}）："
      f"λ={lam2:.3f} p={p2:.3f} ρf={rf2:.3f}")
print(f"  vs W3（n=56）：λ={w3['lam']:.3f} p={w3['p']:.3f} "
      f"ρf={w3['rho_f']:.3f}")
print(f"  扩展后 Θ*_deep={th_deep2:.3f}（W3 0.953）→ 保守 c=0.7 口径 "
      f"{0.7 * th_deep2:.3f}（W3 0.667）")

out = dict(rows=rows, n_new=len(rows), ok=ok, fr=fr, fm=fm,
           refit=dict(lam=float(lam2), p=float(p2), rho_f=float(rf2),
                      th_deep=float(th_deep2), n=len(pts_old) + len(pts_new)))
with open(os.path.join(WORK, "w7_deep_global.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# ---- 图 ----
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2))
ax = axs[0]
for r in rows:
    ax.plot(r["th"], r["rho"], "o", ms=7, color=ORANGE, alpha=0.85)
for p in pts_old:
    ax.plot(p[0], p[1], "o", ms=3.5, color=NAVY, alpha=0.45)
tc = np.linspace(0.02, 9, 200)
ax.plot(tc, gen(tc, w3["lam"], w3["p"], w3["rho_f"]), color=TEAL, lw=2,
        label=f"W3 拟合（n=56，λ={w3['lam']:.2f}/ρf={w3['rho_f']:.2f}）")
ax.plot(tc, gen(tc, lam2, p2, rf2), color=RED, lw=1.6, ls="--",
        label=f"扩展拟合（n={len(pts_old)+len(pts_new)}，λ={lam2:.2f}/ρf={rf2:.2f}）")
ax.axvline(TH_STAR_DEEP, color=RED, ls=":", lw=1.2, label="保守 Θ*=0.667")
ax.axhline(0.9, color="#8A8F98", ls="-.", lw=1)
ax.set_xscale("log")
ax.set_xlabel("Θ (rad)")
ax.set_ylabel("真值 ρ")
ax.set_title("深海样本池扩展：新点（橙）vs W3 池（蓝）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
ax = axs[2 - 1]
labels = [f"{r['site']} α={r['alpha']}" for r in rows]
xs = np.arange(len(rows))
okv = [bool(r["th"] <= TH_STAR_DEEP) == bool(r["rho"] >= 0.9)
       for r in rows]
colors = [TEAL if v else RED for v in okv]
ax.bar(xs, [r["th"] for r in rows], 0.5, color=colors, alpha=0.85)
ax.axhline(TH_STAR_DEEP, color=RED, ls=":", lw=1.2)
ax.set_xticks(xs)
ax.set_xticklabels([f"{r['site']}\nρ={r['rho']:.2f}" for r in rows],
                   fontsize=8)
ax.set_ylabel("Θ (rad)")
ax.set_title("新点 Θ vs 保守阈值（绿=判定正确）")
ax.grid(alpha=0.3, axis="y")
fig.suptitle("W7 全球深海点位扩展：W3 深海标定的样本池稳健性", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG, "w7_deep_global.png"), dpi=110)
print("\n[Done] ->", os.path.join(WORK, "w7_deep_global.json"))
