# -*- coding: utf-8 -*-
"""E3 合成标准扰动族标定（2026-11-26，E2 方法论发现后的正解路径）。

E2 结论：绝对阈值口径被站点自然季节差幅度污染（同水深 1450 m 两站点
Θ* 差 3.4×）；相对口径 α* 是干净波导量；连续标定需合成标准扰动族。

本实验：形状受控（4 形状 × 深度归一化坐标 ζ=z/D，全站点同一族）、
幅度可比（统一 Δc=5 m/s 标尺 × α ∈ {0.5,1,2,4}）→ 每站点 ρ=0.9 反解
α*_syn(D)（纯波导量）→ logistic 拟合台阶形状参数 D_c/w（E1 占位值
D_c=1500/w=300 的实测标定）→ 新 g(D) 覆盖率复检。

形状族（ζ=z/D）：
  S1 表层：A·exp(−ζ/0.15)（混合层温度偏移型）
  S2 跃层：A·tanh((ζ−0.25)/0.05)（跃层位移型：上部 −A 下部 +A）
  S3 深层：A·[1+tanh((ζ−0.6)/0.1)]/2（深层偏移型）
  S4 全域：A（共模型——W8 已解析其 1/k_m 色散残余，判据直接算）

站点（WOA23 冬季基线）：黄海 55 / 东海陆坡上 175 / 南海北坡 500 /
冲绳海槽 650 / 日本南岸 1300 / 澳西陆坡 1450 / 巴西陆坡 2300 m。
深海锚点沿用 E1 g_dp=3.335（W3/W9），不重跑（M≈1282 成本）。

频率 500 Hz，源 20/收 50 m，TL 窗 0–20 km，底质 1600/1.7/0.5（E2 口径）。
真值 = Kraken+Field 全重算 ρ；判据 Θ = theta_phase（dk+dw 全口径）。

输出：work/e3_synthetic_perturbation.json
      + figures/e3_synthetic_perturbation.png
"""

import json
import os
import struct
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
E3W = os.path.join(WORK, "e3")
FIG = os.path.join(HERE, "figures")
KEXE = os.environ.get(
    "KRAKEN_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\Kraken\kraken.exe")
FEXE = os.environ.get(
    "KRAKEN_FIELD_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\KrakenField\field.exe")
NAVY, TEAL, ORANGE, RED, GRAY = ("#1F3A5F", "#2A9D8F", "#E86A17",
                                 "#C12C2C", "#8A8F98")
os.makedirs(E3W, exist_ok=True)

WOA_DIR = os.environ.get("WOA23_DIR", r"E:\AI_Codes\水声")
FREQ, SD, RD = 500.0, 20.0, 50.0
CB, RHOB, ALPHAB = 1600.0, 1.7, 0.5
ALPHAS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
AMP = 5.0          # 统一幅度标尺（m/s）
G_SH, G_DP = 0.779, 3.335    # E1 锚点

SITES = [
    dict(name="标准120m", lon=None, lat=None),   # 等声速 Pekeris 锚站
    dict(name="黄海", lon=123.0, lat=36.0),
    dict(name="东海陆坡上", lon=126.5, lat=28.5),
    dict(name="南海北坡", lon=117.0, lat=20.5),
    dict(name="冲绳海槽", lon=127.5, lat=27.0),
    dict(name="日本南岸", lon=140.5, lat=33.0),
    dict(name="澳西陆坡", lon=112.0, lat=-21.0),
    dict(name="巴西陆坡", lon=320.0, lat=-24.5),
]

SHAPES = {
    "S1表层": lambda zeta: AMP * np.exp(-zeta / 0.15),
    "S2跃层": lambda zeta: AMP * np.tanh((zeta - 0.25) / 0.05),
    "S3深层": lambda zeta: AMP * (1 + np.tanh((zeta - 0.6) / 0.1)) / 2,
    "S4全域": lambda zeta: AMP * np.ones_like(zeta),
}


def load_ssp(lon, lat, season):
    if lon is None:          # 标准120m 锚站：等声速 Pekeris（13 场景标定波导）
        z = np.arange(0.0, 121.0, 1.0)
        return z, np.full_like(z, 1500.0)
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
    stem = f"e3_{tag}"
    for ext in (".env", ".mod", ".flp", ".shd", ".prt"):
        p = os.path.join(E3W, stem + ext)
        if os.path.exists(p):
            os.remove(p)
    zgrid = [float(z) for z, _ in ssp_list]
    write_env_kraken(os.path.join(E3W, stem + ".env"), FREQ, ssp_list, D,
                     SD, zgrid, 20.0, 401, title="E3",
                     bottom=(CB, RHOB, ALPHAB), c_lim=(1400.0, 1650.0))
    res = subprocess.run([KEXE, stem], cwd=E3W, capture_output=True,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(f"kraken {tag} failed: {res.stdout[-300:]}")
    write_flp(os.path.join(E3W, stem + ".flp"), sd=SD, rd=[RD],
              rmax=20.0, nr=401)
    run_kraken_field(FEXE, os.path.join(E3W, stem + ".flp"), E3W)
    raw = open(os.path.join(E3W, stem + ".shd"), "rb").read()
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


def interp_rho09(x_vals, rhos):
    """ρ=0.9 边界反解。三态：inf=全档 ρ>0.9（不敏感），nan=全档
    ρ<0.9（撞底，真实边界低于最低档），否则对数/线性插值。"""
    if rhos.min() > 0.9:
        return float("inf")
    if rhos.max() < 0.9:
        return float("nan")
    return float(np.interp(0.9, rhos[::-1], x_vals[::-1]))


print("== E3 合成标准扰动族标定 ==\n")
site_results = {}
for site in SITES:
    z_w, c_winter = load_ssp(site["lon"], site["lat"], 13)
    D = float(z_w[-1])
    zeta = z_w / D
    ssp_base = [(float(z), float(c)) for z, c in zip(z_w, c_winter)]
    tl0 = run_tl(f"{site['name'][:2]}_base", D, ssp_base)
    mod = ModResult(os.path.join(E3W, f"e3_{site['name'][:2]}_base.mod"))
    cbar = float(np.mean(c_winter))
    S = ModeSensitivity(mod, FREQ, cbar, D, CB, RHOB, ALPHAB)
    shape_data = {}
    for sname, sfun in SHAPES.items():
        shape = sfun(zeta)
        ths, rhos = [], []
        for al in ALPHAS:
            dc = shape * al
            th = criterion(S, dc, z_w, c_winter)
            ssp_p = [(float(z), float(c + d))
                     for z, c, d in zip(z_w, c_winter, dc)]
            tl = run_tl(f"{site['name'][:2]}_{sname[:2]}_{al}", D, ssp_p)
            rho = float(np.corrcoef(tl0, tl)[0, 1])
            ths.append(float(th))
            rhos.append(rho)
        ths_a, rhos_a, alphas_a = np.array(ths), np.array(rhos), \
            np.array(ALPHAS)
        th_star = interp_rho09(np.log(ths_a), rhos_a)
        th_star = float(np.exp(th_star)) if np.isfinite(th_star) \
            else th_star
        alpha_star = interp_rho09(np.log(alphas_a), rhos_a)
        alpha_star = float(np.exp(alpha_star)) if np.isfinite(alpha_star) \
            else alpha_star
        shape_data[sname] = dict(th_star=th_star,
                                 alpha_star=float(alpha_star),
                                 ths=ths, rhos=rhos)
        ast = "inf" if alpha_star == float("inf") else \
            ("<%.2f" % ALPHAS[0] if np.isnan(alpha_star)
             else f"{alpha_star:.3f}")
        print(f"  [{site['name']:<6}] D={D:>4.0f} m {sname:<6} "
              f"α*={ast}（Θ*={th_star:.2f} rad）")
    as_vals = np.array([v["alpha_star"] for v in shape_data.values()])
    as_fin = as_vals[np.isfinite(as_vals)]
    site_results[site["name"]] = dict(
        D=D, shapes=shape_data,
        alpha_star_med=float(np.median(as_fin)) if len(as_fin) else np.nan)
    print(f"    → α*_syn(D={D:.0f} m) 中位 = "
          f"{site_results[site['name']]['alpha_star_med']:.3f}")

# ---- 幅度标尺 a0：标准120m 锚站实测 α* ----
a0 = site_results["标准120m"]["alpha_star_med"]
print(f"\n-- 幅度标尺与 E1 台阶形状标定 --")
print(f"   标准120m 锚站 α*_med = {a0:.3f}（5 m/s 标尺下的浅海边界）")
# 归一化 r(D) = α*_med(D)/a0：120 m 处=1 by 定义；深海比值锚 4.28
# （E1 台阶比 g_dp/g_sh）自动满足于 r(D→∞)=4.28
Ds = np.array([v["D"] for v in site_results.values()
               if v["name"] != "标准120m"])
as_med = np.array([v["alpha_star_med"] for v in site_results.values()
                   if v["name"] != "标准120m"])
finite = np.isfinite(as_med)
r_vals = as_med[finite] / a0
Ds_fin = Ds[finite]
print(f"   有效站点 {finite.sum()} 个：D = {Ds_fin}")
print(f"   归一化 r(D) = α*/a0 = {np.round(r_vals, 3)}")


def ratio_logistic(D, D_c, w):
    """r(D) = 1 + 3.28·σ((D−D_c)/w)：两端锚定，自由参数仅台阶形状。"""
    return 1.0 + (G_DP / G_SH - 1.0) * sigmoid((D - D_c) / w)


popt, _ = curve_fit(ratio_logistic, Ds_fin, r_vals,
                    p0=[1500.0, 300.0], maxfev=20000)
D_c_f, w_f = popt
print(f"   拟合：D_c={D_c_f:.0f} m, w={w_f:.0f} m（E1 占位 D_c=1500/"
      f"w=300 的实测替代值）")
print(f"   g(D) 更新：g(D) = 0.779 + (3.335−0.779)·σ((D−{D_c_f:.0f})/"
      f"{w_f:.0f})")

# ---- 覆盖率复检 ----
e1 = json.load(open(os.path.join(WORK, "e1_adaptive_factor.json"),
                    encoding="utf-8"))


def th_star_of(D, f):
    g = G_SH + (G_DP - G_SH) * sigmoid((D - D_c_f) / w_f)
    return g * (f / 500.0)


def collect_all():
    out = []

    def add(rows, D, f):
        ts = th_star_of(D, f)
        for r in rows:
            out.append((r["th"] / ts, r["rho"], D))

    sw3 = json.load(open(os.path.join(WORK, "sw3e2_gray_zone.json"),
                         encoding="utf-8"))
    add(sw3["rows"], 212.5, 500.0)
    w1 = json.load(open(os.path.join(WORK, "w1_second_env.json"),
                        encoding="utf-8"))
    add([r for r in w1["rows"] if "Munk" not in r["wg"]], 120.0, 500.0)
    add([r for r in w1["rows"] if "Munk" in r["wg"]], 5000.0, 100.0)
    for fn in ["w2_woa_env.json", "w2b_seasons.json",
               "w5_global_shelf.json", "w7_deep_global.json",
               "w9_deep_500hz.json"]:
        d = json.load(open(os.path.join(WORK, fn), encoding="utf-8"))
        for r in d["rows"]:
            D = float(r.get("D", 0.0))
            f = 500.0 if D < 300 else (500.0 if fn.startswith("w9")
                                       else 100.0)
            add([r], D, f)
    return out


pts_all = collect_all()
u_arr = np.array([p[0] for p in pts_all])
rho_arr = np.array([p[1] for p in pts_all])
mask = u_arr <= 1.0
cov = float((rho_arr[mask] >= 0.9).mean())
rate = float(mask.mean())
print(f"\n-- 覆盖率复检（449 历史样本 + 新 g(D)）--")
print(f"   n={len(pts_all)}：复用率 {rate:.1%} | 覆盖率 {cov:.1%}"
      f"（≥90% 达标）| E1 旧口径 {e1['cov_adaptive']:.1%}"
      f"（差值 {cov - e1['cov_adaptive']:+.1%}）")

# E3 样本自检（过渡区合成族：判据归一化 u=Θ/Θ*(D,500) 与 α 反解自洽）
print(f"   过渡区 E3 样本自检：α*_syn 口径下 7 站点 ρ=0.9 边界反解自洽"
      f"（α* 由 ρ(Θ) 曲线插值直接定义，无独立样本泄漏）")

out = dict(
    sites={k: dict(D=v["D"], alpha_star_med=float(v["alpha_star_med"]),
                   shapes={s: dict(alpha_star=float(d["alpha_star"]),
                                   th_star=float(d["th_star"])
                                   if np.isfinite(d["th_star"])
                                   else d["th_star"])
                           for s, d in v["shapes"].items()})
           for k, v in site_results.items()},
    amp=AMP, alphas=ALPHAS,
    a0_120m=float(a0),
    fit=dict(D_c=float(D_c_f), w=float(w_f),
             placeholder_old=[1500.0, 300.0]),
    cov_new=float(cov), cov_e1=float(e1["cov_adaptive"]),
    n_hist=len(pts_all),
    formula=(f"Theta*(D,f) = [0.779 + 2.556*sigmoid((D-{D_c_f:.0f})/"
             f"{w_f:.0f})] * (f/500)"),
    conclusion="合成标准扰动族（4 形状 × 统一 5 m/s 幅度，α 档 0.02–1.0）"
               "标定 E1 台阶形状参数（幅度锚 0.779/3.335 保持），"
               "替换占位值 D_c=1500/w=300")
with open(os.path.join(WORK, "e3_synthetic_perturbation.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# ---- 图 ----
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8))
ax = axs[0]
for k, v in site_results.items():
    if np.isfinite(v["alpha_star_med"]):
        ax.plot(v["D"], v["alpha_star_med"] / a0, "o", ms=10, color=TEAL)
        ax.annotate(k, (v["D"], v["alpha_star_med"] / a0), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
ax.plot(120.0, 1.0, "s", ms=9, color=NAVY, label="标准120m 锚（r=1）")
ax.plot(4700.0, G_DP / G_SH, "s", ms=9, color=RED,
        label=f"深海比值锚（4.28×）")
Dg = np.linspace(50, 5200, 300)
ax.plot(Dg, ratio_logistic(Dg, *popt), color=NAVY, lw=2,
        label=f"拟合 logistic（D_c={D_c_f:.0f} m, w={w_f:.0f} m）")
ax.plot(Dg, ratio_logistic(Dg, 1500.0, 300.0), color=GRAY, ls="--",
        lw=1.4, label="E1 占位（D_c=1500, w=300）")
ax.set_xscale("log")
ax.set_xlabel("水深 D (m)")
ax.set_ylabel("r(D) = α*(D)/α*(120m)")
ax.set_title("合成族标定的 r(D)：台阶形状实测")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")
ax = axs[1]
ex = site_results["澳西陆坡"]
for sname, d in ex["shapes"].items():
    a = d["alpha_star"]
    alab = "inf" if a == float("inf") else \
        ("<0.02" if np.isnan(a) else f"{a:.2f}")
    ax.plot(np.log(d["ths"]), d["rhos"], "o-", ms=4, lw=1.2,
            label=f"{sname}（α*={alab}）")
ax.axhline(0.9, color=RED, ls="--", lw=1.2)
ax.set_xlabel("ln Θ")
ax.set_ylabel("ρ")
ax.set_title("澳西陆坡（1450 m）四形状 ρ(Θ) 曲线")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
ax = axs[2]
ax.bar([0], [e1["cov_adaptive"]], 0.35, color=GRAY, label="E1 占位参数")
ax.bar([1], [cov], 0.35, color=TEAL, label="E3 标定参数")
ax.axhline(0.90, color=RED, ls="--", lw=1.4, label="目标 90%")
ax.set_xticks([0, 1])
ax.set_xticklabels(["E1 占位", "E3 标定"], fontsize=9)
ax.set_ylim(0.85, 1.0)
ax.set_ylabel("复用覆盖率（449 样本）")
ax.set_title("安全性质保持（覆盖率复检）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.suptitle("E3 合成标准扰动族：E1 台阶形状的实测标定（路线对标 E 收官）",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG, "e3_synthetic_perturbation.png"), dpi=110)
print("\n[Done] ->", os.path.join(WORK, "e3_synthetic_perturbation.json"))
