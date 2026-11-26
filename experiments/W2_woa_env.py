# -*- coding: utf-8 -*-
"""W2 真实 WOA23 第二环境实验（2026-11-26，用户提供数据，补强 W1）。

数据（用户提供，E:\\AI_Codes\\水声）：
- woa23_ssp_s13/s14/s15/s16.nc：WOA23 季度气候态声速（0.25° 全球，
  102 标准深度层 0–5500 m，ssp 单位 m/s）——s13 冬 / s14 春 /
  s15 夏 / s16 秋。
- Etopo2022_Little/ETOPO_2022_v1_60s.nc：ETOPO2022 地形（1 分）。

实验：5 个代表点（ETOPO 定深）——
  浅海组（判据主分支）：南海北部陆架 (113E,20N) 118 m、黄海 (123E,35N)
    71 m、东海陆架 (124E,30N) 56 m（500 Hz，Θ*=0.8）
  深海组（深海分支）：西太 (135E,15N) 4527 m、北大西洋 (330E,40N)
    5152 m（100 Hz，Θ*=0.16，F1 标度律换算）
环境对 = 真实季节差 δc = s15（夏）− s13（冬），α ∈ {0.25, 0.5, 1, 2}
四档缩放构造灰区。底质为文献典型值（诚实假设，无实测）。

真值：Kraken+Field 全重算 ρ；判据：一阶 Θ（与 SWellEx 同式）。
输出：work/w2_woa_env.json + figures/w2_woa_env.png
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
W2W = os.path.join(WORK, "w2")
FIG = os.path.join(HERE, "figures")
KEXE = os.environ.get(
    "KRAKEN_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\Kraken\kraken.exe")
FEXE = os.environ.get(
    "KRAKEN_FIELD_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\KrakenField\field.exe")
NAVY, TEAL, ORANGE, RED = "#1F3A5F", "#2A9D8F", "#E86A17", "#C12C2C"
os.makedirs(W2W, exist_ok=True)

WOA_DIR = os.environ.get("WOA23_DIR", r"E:\AI_Codes\水声")
ETOPO60 = os.path.join(WOA_DIR, "Etopo2022_Little", "ETOPO_2022_v1_60s.nc")
LAM, P_EXP, RHO_F = 0.285324, 1.39808, 0.477323
ALPHAS = [0.25, 0.5, 1.0, 2.0]

SITES = [
    # (名称, lon, lat, 底质 cb/rho/alpha, freq, sd, rd, 窗 r_lo/r_hi, r_eval)
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
    """WOA 季度 SSP → (深度网格 z, 声速 c)，只留有效层，插值到细网格。"""
    fn = os.path.join(WOA_DIR, f"woa23_ssp_s{season}.nc")
    with xr.open_dataset(fn, engine="h5netcdf") as ds:
        p = ds["ssp"].sel(lon=lon, lat=lat, method="nearest")
        dep = ds["depth"].values
        vals = p.values
    valid = ~np.isnan(vals)
    dep, vals = dep[valid], vals[valid]
    if len(dep) < 3:
        raise RuntimeError("SSP 有效层过少")
    dz = 0.5 if vals.size < 300 else 1.0
    z = np.arange(0.0, dep[-1] + dz / 2, dz)
    c = np.interp(z, dep, vals, left=vals[0], right=vals[-1])
    return z, c


def run_tl(tag, freq, D, ssp_list, sd, rd, rmax, nr, cb, rho, alpha):
    """Kraken+Field → TL（dB），与 W1 同式。"""
    stem = f"w2_{tag}"
    for ext in (".env", ".mod", ".flp", ".shd", ".prt"):
        p = os.path.join(W2W, stem + ext)
        if os.path.exists(p):
            os.remove(p)
    zgrid = [float(z) for z, _ in ssp_list]
    write_env_kraken(os.path.join(W2W, stem + ".env"), freq, ssp_list, D,
                     sd, zgrid, rmax, nr, title="W2",
                     bottom=(cb, rho, alpha), c_lim=(1400.0, 1650.0))
    res = subprocess.run([KEXE, stem], cwd=W2W, capture_output=True,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(f"kraken {tag} failed: {res.stdout[-300:]}")
    write_flp(os.path.join(W2W, stem + ".flp"), sd=sd, rd=[rd],
              rmax=rmax, nr=nr)
    shd = run_kraken_field(FEXE, os.path.join(W2W, stem + ".flp"), W2W)
    import struct
    raw = open(os.path.join(W2W, stem + ".shd"), "rb").read()
    rl = struct.unpack("I", raw[:4])[0] * 4
    P = np.frombuffer(raw[10 * rl:], dtype="<c8").copy()[:nr]
    return -20.0 * np.log10(np.abs(P) + 1e-300)


def criterion(S, dc, zs, c_base, r_eval, sd, rd):
    """一阶判据 Θ（相位+形状），SWellEx 同式。"""
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


print("== W2 真实 WOA23 第二环境实验 ==\n")
rows = []
for site in SITES:
    z_w, c_winter = load_ssp(site["lon"], site["lat"], 13)
    z_s, c_summer = load_ssp(site["lon"], site["lat"], 15)
    D = float(z_w[-1])
    site["D"] = D
    dc0 = np.interp(z_w, z_s, c_summer) - c_winter
    ssp_base = [(float(z), float(c)) for z, c in zip(z_w, c_winter)]
    s = site
    tl0 = run_tl(s["name"][:2], s["freq"], D, ssp_base, s["sd"], s["rd"],
                 s["r_hi"], 401, s["cb"], s["rho"], s["alpha"])
    stem = f"w2_{s['name'][:2]}"
    mod = ModResult(os.path.join(W2W, stem + ".mod"))
    cbar = float(np.mean(c_winter))
    S = ModeSensitivity(mod, s["freq"], cbar, D, s["cb"], s["rho"],
                        s["alpha"])
    print(f"[{s['name']}] D={D:.0f} m  M={mod.M}  季节差 Δc 峰峰 "
          f"{np.max(np.abs(dc0)):.1f} m/s")
    for al in ALPHAS:
        dc = dc0 * al
        th = criterion(S, dc, z_w, c_winter, s["r_eval"], s["sd"],
                       s["rd"])
        ssp_p = [(float(z), float(c + d))
                 for z, c, d in zip(z_w, c_winter, dc)]
        tl = run_tl(f"{s['name'][:2]}_a{al}", s["freq"], D, ssp_p,
                    s["sd"], s["rd"], s["r_hi"], 401, s["cb"], s["rho"],
                    s["alpha"])
        rho = float(np.corrcoef(tl0, tl)[0, 1])
        rows.append(dict(site=s["name"], alpha=al, th=float(th),
                         rho=rho, D=float(D)))
        print(f"    α={al:4.2f}  Θ={th:7.3f}  ρ={rho:.4f}")

# ---- 分析 ----
th_star = lambda freq: 0.8 * (freq / 500.0)
print("\n混淆矩阵（ρ_req=0.9，Θ* 按 F1 标度律）：")
by_site = {}
for site in SITES:
    ts = th_star(site["freq"])
    sub = [r for r in rows if r["site"] == site["name"]]
    ok = sum((r["th"] <= ts) == (r["rho"] >= 0.9) for r in sub)
    fr = sum((r["th"] <= ts) and r["rho"] < 0.9 for r in sub)
    fm = sum((r["th"] > ts) and r["rho"] >= 0.9 for r in sub)
    by_site[site["name"]] = dict(n=len(sub), ok=ok, fr=fr, fm=fm,
                                 th_star=float(ts))
    print(f"  {site['name']:<8} Θ*={ts:.2f}  ok={ok}/{len(sub)}  "
          f"fr={fr}  fm={fm}")
shallow = [r for r in rows if r["D"] < 300.0]
deep = [r for r in rows if r["D"] >= 300.0]
ok_s = sum((r["th"] <= 0.8) == (r["rho"] >= 0.9) for r in shallow)
ok_d = sum((r["th"] <= 0.16) == (r["rho"] >= 0.9) for r in deep)
print(f"浅海组 {ok_s}/{len(shallow)} | 深海组 {ok_d}/{len(deep)}")

ths = np.array([r["th"] for r in rows])
rhos = np.array([r["rho"] for r in rows])
rho_gen = RHO_F + (1 - RHO_F) * np.exp(-LAM * ths ** P_EXP)
rmse = float(np.sqrt(np.mean((rhos - rho_gen) ** 2)))
print(f"广义型理论 RMSE = {rmse:.3f}（W1 合成口径 0.145）")

out = dict(rows=rows, by_site=by_site, rmse=rmse, lam=LAM, p=P_EXP,
           rho_f=RHO_F)
with open(os.path.join(WORK, "w2_woa_env.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# ---- 图 ----
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2))
ax = axs[0]
colors = {s["name"]: c for s, c in zip(
    SITES, [TEAL, NAVY, ORANGE, RED, "#8A8F98"])}
for site in SITES:
    sub = [r for r in rows if r["site"] == site["name"]]
    mk = "o" if site["D"] < 300 else "^"
    ax.plot([r["th"] for r in sub], [r["rho"] for r in sub], mk, ms=6,
            color=colors[site["name"]], alpha=0.8,
            label=f"{site['name']}（{site['D']:.0f} m）")
tc = np.linspace(0.02, 8, 200)
ax.plot(tc, RHO_F + (1 - RHO_F) * np.exp(-LAM * tc ** P_EXP),
        color=NAVY, lw=1.8, ls="--", label="广义型（SWellEx 标定）")
ax.set_xscale("log")
ax.set_xlabel("Θ (rad)")
ax.set_ylabel("真值 ρ")
ax.set_title("真实季节变化（夏−冬）ρ(Θ) vs SWellEx 理论曲线")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
ax = axs[1]
xs = np.arange(len(SITES))
okv = [by_site[s["name"]]["ok"] / by_site[s["name"]]["n"] for s in SITES]
frv = [by_site[s["name"]]["fr"] / by_site[s["name"]]["n"] for s in SITES]
fmv = [by_site[s["name"]]["fm"] / by_site[s["name"]]["n"] for s in SITES]
ax.bar(xs - 0.22, okv, 0.22, color=TEAL, label="正确率")
ax.bar(xs, frv, 0.22, color=RED, label="假复用率")
ax.bar(xs + 0.22, fmv, 0.22, color="#8A8F98", label="假重算率")
ax.set_xticks(xs)
ax.set_xticklabels([s["name"] for s in SITES], fontsize=9)
ax.set_ylabel("比率")
ax.set_title("阈值方向正确性（Θ* 按 F1 标度律换算）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.suptitle("W2 真实 WOA23 第二环境：季节变化下的判据可移植性",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG, "w2_woa_env.png"), dpi=110)
print("\n[Done] ->", os.path.join(WORK, "w2_woa_env.json"))
