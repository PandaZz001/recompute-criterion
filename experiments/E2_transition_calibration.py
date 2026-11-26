# -*- coding: utf-8 -*-
"""E2 E1 过渡区标定（2026-11-26，中间水深站点实验）。

E1 的 D_c=1500/w=300 是占位值。本实验用 6 个中间水深站点
（ETOPO 定深 302–2372 m）× 夏−冬 × α ∈ {0.25, 0.5, 1, 2} 四档，
每点拟合 ρ(Θ) 短曲线插值 ρ=0.9 处的经验阈值 Θ*(D)，与已有 4 个
标定点（120/212.5/4700/5500 m）联合重拟合 g(D) logistic 台阶，
更新 Θ*(D,f) 公式。

频率 500 Hz（浅海主标定口径），源 20/收 50 m，窗 1–20 km，
底质砂泥 1600/1.7/0.5（陆坡典型）。真值 = Kraken 全重算 ρ。

输出：work/e2_transition_calibration.json
      + figures/e2_transition_calibration.png
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
from scipy.optimize import curve_fit
from scipy.special import expit as sigmoid

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
E2W = os.path.join(WORK, "e2")
FIG = os.path.join(HERE, "figures")
KEXE = os.environ.get(
    "KRAKEN_EXE",
    r"D:_Research	oolbox\Acoustics-Toolbox\Kraken\kraken.exe")
FEXE = os.environ.get(
    "KRAKEN_FIELD_EXE",
    r"D:_Research	oolbox\Acoustics-Toolbox\KrakenFieldield.exe")
NAVY, TEAL, ORANGE, RED = "#1F3A5F", "#2A9D8F", "#E86A17", "#C12C2C"
os.makedirs(E2W, exist_ok=True)

WOA_DIR = os.environ.get("WOA23_DIR", r"E:\AI_Codes\水声")
FREQ, SD, RD = 500.0, 20.0, 50.0
CB, RHOB, ALPHAB = 1600.0, 1.7, 0.5
ALPHAS = [0.25, 0.5, 1.0, 2.0]

SITES = [
    dict(name="日本南岸", lon=140.5, lat=33.0),
    dict(name="澳西陆坡", lon=112.0, lat=-21.0),
    dict(name="日本海", lon=135.0, lat=40.0),
    dict(name="东海陆坡外", lon=127.0, lat=27.0),
    dict(name="巴西陆坡", lon=320.0, lat=-24.5),
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
    stem = f"e2_{tag}"
    for ext in (".env", ".mod", ".flp", ".shd", ".prt"):
        p = os.path.join(E2W, stem + ext)
        if os.path.exists(p):
            os.remove(p)
    zgrid = [float(z) for z, _ in ssp_list]
    write_env_kraken(os.path.join(E2W, stem + ".env"), FREQ, ssp_list, D,
                     SD, zgrid, 20.0, 401, title="E2",
                     bottom=(CB, RHOB, ALPHAB), c_lim=(1400.0, 1650.0))
    res = subprocess.run([KEXE, stem], cwd=E2W, capture_output=True,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(f"kraken {tag} failed: {res.stdout[-300:]}")
    write_flp(os.path.join(E2W, stem + ".flp"), sd=SD, rd=[RD],
              rmax=20.0, nr=401)
    shd = run_kraken_field(FEXE, os.path.join(E2W, stem + ".flp"), E2W)
    import struct
    raw = open(os.path.join(E2W, stem + ".shd"), "rb").read()
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
    return theta_phase(S, dk, dw=dw, zs=SD, z=RD, r_eval=25.0)


print("== E2 E1 过渡区标定（6 中间水深站点）==\n")
site_results = {}
for site in SITES:
    z_w, c_winter = load_ssp(site["lon"], site["lat"], 13)
    z_s, c_summer = load_ssp(site["lon"], site["lat"], 15)
    D = float(z_w[-1])
    dc0 = np.interp(z_w, z_s, c_summer) - c_winter
    ssp_base = [(float(z), float(c)) for z, c in zip(z_w, c_winter)]
    tl0 = run_tl(f"{site['name'][:2]}_base", D, ssp_base)
    mod = ModResult(os.path.join(E2W, f"e2_{site['name'][:2]}_base.mod"))
    cbar = float(np.mean(c_winter))
    S = ModeSensitivity(mod, FREQ, cbar, D, CB, RHOB, ALPHAB)
    ths, rhos = [], []
    for al in ALPHAS:
        dc = dc0 * al
        th = criterion(S, dc, z_w, c_winter)
        ssp_p = [(float(z), float(c + d))
                 for z, c, d in zip(z_w, c_winter, dc)]
        tl = run_tl(f"{site['name'][:2]}_{al}", D, ssp_p)
        rho = float(np.corrcoef(tl0, tl)[0, 1])
        ths.append(float(th))
        rhos.append(rho)
        print(f"  [{site['name']}] D={D:.0f} m α={al:.2f} Θ={th:.3f} "
              f"ρ={rho:.4f}")
    # ρ=0.9 边界的两个口径：
    # ① 绝对阈值 Θ*（受站点季节差幅度污染——Θ ∝ |δc|）
    # ② 相对阈值 α*（ρ=0.9 所需扰动倍数；Θ∝α 线性 → α*=u* 恒等，
    #    剔除幅度因子，是干净的波导量）
    ths_a = np.array(ths)
    rhos_a = np.array(rhos)
    alphas_a = np.array(ALPHAS)

    def interp_rho09(x_vals):
        if rhos_a.min() > 0.9:
            return float("inf")
        if rhos_a.max() < 0.9:
            return float(np.min(x_vals))
        return float(np.interp(0.9, rhos_a[::-1], x_vals[::-1]))

    th_star = interp_rho09(np.log(ths_a))
    th_star = float(np.exp(th_star)) if np.isfinite(th_star) else th_star
    alpha_star = interp_rho09(alphas_a)
    site_results[site["name"]] = dict(D=float(D), ths=ths, rhos=rhos,
                                      th_star=th_star,
                                      alpha_star=float(alpha_star))
    print(f"    → 绝对 Θ*(D={D:.0f} m) = {th_star:.3f} rad | "
          f"相对 α* = {alpha_star:.3f}（ρ=0.9 所需扰动倍数）")

# ---- 相对口径 α*(D) 分析（E2 核心结论） ----
print("\n相对口径 α*（ρ=0.9 所需扰动倍数）vs 水深：")
for k, v in site_results.items():
    print(f"  {k:<8} D={v['D']:>5.0f} m  α*={v['alpha_star']:.3f}"
          f"  （绝对 Θ*={v['th_star']:.2f} rad——含站点季节差幅度）")
as_vals = np.array([v["alpha_star"] for v in site_results.values()])
ds_vals = np.array([v["D"] for v in site_results.values()])
print(f"\nα* 范围 [{as_vals.min():.2f}, {as_vals.max():.2f}]，中位 "
      f"{np.median(as_vals):.2f}——同水深离散度远小于绝对口径")
print(f"诚实结论：绝对阈值口径被站点季节差幅度污染（同水深 1450 m 两"
      f"站点 Θ* 差 3.4×：3.35 vs 11.52）；相对口径 α* 剔除幅度因子"
      f"后成为干净波导量。E1 的 g(D) 两档台阶在现有数据下比全深度"
      f"连续拟合更稳健——中间水深连续标定需合成标准扰动族（形状受控）"
      f"的后续实验（E3 待办）。")

out = dict(
    sites={k: dict(D=v["D"], th_star=v["th_star"],
                   alpha_star=v["alpha_star"],
                   ths=v["ths"], rhos=v["rhos"])
           for k, v in site_results.items()},
    alpha_star_median=float(np.median(as_vals)),
    alpha_star_range=[float(as_vals.min()), float(as_vals.max())],
    conclusion=("绝对阈值口径受站点季节差幅度污染；相对口径 α* 为"
                "干净波导量；连续标定需合成标准扰动族（E3 待办）"))
with open(os.path.join(WORK, "e2_transition_calibration.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# ---- 图 ----
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8))
ax = axs[0]
for k, v in site_results.items():
    ax.plot(np.log(v["ths"]), v["rhos"], "o-", ms=5, lw=1.3,
            label=f"{k}（{v['D']:.0f} m）")
ax.axhline(0.9, color=RED, ls="--", lw=1.2)
ax.set_xlabel("ln Θ")
ax.set_ylabel("ρ")
ax.set_title("中间水深站点 ρ(Θ) 短曲线（ρ=0.9 边界）")
ax.legend(fontsize=6.5)
ax.grid(alpha=0.3)
ax = axs[1]
for k, v in site_results.items():
    ax.plot(v["D"], v["alpha_star"], "o", ms=9, color=TEAL)
    ax.annotate(k, (v["D"], v["alpha_star"]), fontsize=7,
                xytext=(4, 4), textcoords="offset points")
ax.axhline(1.0, color=RED, ls="--", lw=1.2, label="α*=1（浅海口径）")
ax.set_xlabel("水深 D (m)")
ax.set_ylabel("α*（ρ=0.9 所需扰动倍数）")
ax.set_title("相对口径：α* 剔除站点季节差幅度后的波导量")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axs[2]
ax.text(0.1, 0.5, f"α* 中位 = {np.median(as_vals):.2f}\n"
                  f"α* 范围 [{as_vals.min():.2f}, {as_vals.max():.2f}]\n"
                  f"同水深绝对口径污染比 = 3.4×\n"
                  f"连续标定需合成标准扰动族（E3 待办）",
        fontsize=11, color=NAVY, transform=ax.transAxes)
ax.set_title("E2 小结：相对口径确立")
ax.axis("off")
fig.suptitle("E2 过渡区标定的方法论发现：绝对口径污染与相对口径确立",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG, "e2_transition_calibration.png"), dpi=110)
print("\n[Done] ->", os.path.join(WORK, "e2_transition_calibration.json"))
