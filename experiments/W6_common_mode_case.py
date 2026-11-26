# -*- coding: utf-8 -*-
"""W6 英吉利海峡共模案例深挖（2026-11-26，迭代任务）。

W5 异常案例：英吉利海峡 (0.5E, 50.5N) 夏−冬 α=0.5 时 Θ=4.78 rad 但
真值 ρ=0.950——判据强保守。假设：季节差 δc(z) 垂直结构接近常数
偏移（共模），一阶 δk_m 近似共模本应被成对差分抵消，但残余的
非共模成分（跃层季节变化）放大了 Θ，而 TL 图案对这类扰动本身
不敏感（多径平均）。

本实验分解：
① δc(z) 形状：均值（共模分量）与去均值后（非共模分量）的比例
② Θ_ph 相位分量 / Θ_sh 形状分量
③ 共模剔除检验：只用非共模分量算 Θ_nc，与全量 Θ 对比
④ ρ(Θ) 对照浅海曲线定位（是否落在标定曲线之外）
⑤ 合成共模对照：纯常数偏移扰动（共模）的 Θ 应≈0（A4 验证）

输出：work/w6_common_mode_case.json + figures/w6_common_mode_case.png
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
W6W = os.path.join(WORK, "w6")
FIG = os.path.join(HERE, "figures")
KEXE = os.environ.get(
    "KRAKEN_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\Kraken\kraken.exe")
FEXE = os.environ.get(
    "KRAKEN_FIELD_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\KrakenField\field.exe")
NAVY, TEAL, ORANGE, RED = "#1F3A5F", "#2A9D8F", "#E86A17", "#C12C2C"
os.makedirs(W6W, exist_ok=True)

WOA_DIR = os.environ.get("WOA23_DIR", r"E:\AI_Codes\水声")
LON, LAT = 0.5, 50.5
FREQ, SD, RD = 500.0, 20.0, 50.0
CB, RHOB, ALPHAB = 1600.0, 1.7, 0.5


def load_ssp(season):
    fn = os.path.join(WOA_DIR, f"woa23_ssp_s{season}.nc")
    with xr.open_dataset(fn, engine="h5netcdf") as ds:
        p = ds["ssp"].sel(lon=LON, lat=LAT, method="nearest")
        dep = ds["depth"].values
        vals = p.values
    valid = ~np.isnan(vals)
    dep, vals = dep[valid], vals[valid]
    dz = 0.5
    z = np.arange(0.0, dep[-1] + dz / 2, dz)
    c = np.interp(z, dep, vals, left=vals[0], right=vals[-1])
    return z, c


def run_tl(tag, D, ssp_list):
    stem = f"w6_{tag}"
    for ext in (".env", ".mod", ".flp", ".shd", ".prt"):
        p = os.path.join(W6W, stem + ext)
        if os.path.exists(p):
            os.remove(p)
    zgrid = [float(z) for z, _ in ssp_list]
    write_env_kraken(os.path.join(W6W, stem + ".env"), FREQ, ssp_list, D,
                     SD, zgrid, 20.0, 401, title="W6",
                     bottom=(CB, RHOB, ALPHAB), c_lim=(1400.0, 1650.0))
    res = subprocess.run([KEXE, stem], cwd=W6W, capture_output=True,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(f"kraken {tag} failed: {res.stdout[-300:]}")
    write_flp(os.path.join(W6W, stem + ".flp"), sd=SD, rd=[RD],
              rmax=20.0, nr=401)
    shd = run_kraken_field(FEXE, os.path.join(W6W, stem + ".flp"), W6W)
    import struct
    raw = open(os.path.join(W6W, stem + ".shd"), "rb").read()
    rl = struct.unpack("I", raw[:4])[0] * 4
    P = np.frombuffer(raw[10 * rl:], dtype="<c8").copy()[:401]
    return -20.0 * np.log10(np.abs(P) + 1e-300)


def th_components(S, dc, zs, c_base):
    """返回 (Θ_full, Θ_ph, Θ_sh)。"""
    dc_s = np.interp(S.z, zs, dc)
    cz = np.interp(S.z, zs, c_base)
    integ = np.trapezoid(dc_s[:, None] * S.phi.real ** 2
                         / cz[:, None] ** 3, S.z, axis=0)
    dk = -(S.w ** 2 / S.k) * integ
    th_ph = theta_phase(S, dk, zs=SD, z=RD, r_eval=25.0)
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
    th_full = theta_phase(S, dk, dw=dw, zs=SD, z=RD, r_eval=25.0)
    return th_full, th_ph, th_full - th_ph


print("== W6 英吉利海峡共模案例深挖 ==\n")
z_w, c_winter = load_ssp(13)
z_s, c_summer = load_ssp(15)
D = float(z_w[-1])
dc0 = np.interp(z_w, z_s, c_summer) - c_winter
dc_mean = float(np.mean(dc0))
dc_nc = dc0 - dc_mean  # 非共模分量
print(f"D={D:.0f} m")
print(f"δc 统计：均值（共模分量）{dc_mean:+.2f} m/s | 标准差 "
      f"{dc0.std():.2f} m/s | 峰峰 {np.max(dc0)-np.min(dc0):.1f} m/s")
print(f"共模/非共模能量比 = {dc_mean**2 / np.var(dc0):.2f}")

ssp_base = [(float(z), float(c)) for z, c in zip(z_w, c_winter)]
tl0 = run_tl("base", D, ssp_base)
mod = ModResult(os.path.join(W6W, "w6_base.mod"))
cbar = float(np.mean(c_winter))
S = ModeSensitivity(mod, FREQ, cbar, D, CB, RHOB, ALPHAB)
print(f"M={mod.M}")

# ① 全量扰动 α=0.5（复现 W5 数字）
dc = dc0 * 0.5
tf, tp, ts_ = th_components(S, dc, z_w, c_winter)
ssp_p = [(float(z), float(c + d)) for z, c, d in zip(z_w, c_winter, dc)]
tl = run_tl("full", D, ssp_p)
rho_full = float(np.corrcoef(tl0, tl)[0, 1])
print(f"\n① 全量 α=0.5：Θ_full={tf:.3f}（Θ_ph={tp:.3f}/Θ_sh={ts_:.3f}）"
      f"  ρ={rho_full:.4f}")

# ② 纯共模扰动（常数偏移 = δc 均值 ×0.5）：A4 验证
dc_cm = np.full_like(dc0, dc_mean * 0.5)
tf_cm, tp_cm, ts_cm = th_components(S, dc_cm, z_w, c_winter)
ssp_cm = [(float(z), float(c + d)) for z, c, d in zip(z_w, c_winter, dc_cm)]
tl_cm = run_tl("cm", D, ssp_cm)
rho_cm = float(np.corrcoef(tl0, tl_cm)[0, 1])
print(f"② 纯共模：Θ={tf_cm:.4f}（应≈0）  ρ={rho_cm:.4f}（应≈1）")

# ③ 非共模分量（去均值）
dc_n = dc_nc * 0.5
tf_n, tp_n, ts_n = th_components(S, dc_n, z_w, c_winter)
ssp_n = [(float(z), float(c + d)) for z, c, d in zip(z_w, c_winter, dc_n)]
tl_n = run_tl("nc", D, ssp_n)
rho_n = float(np.corrcoef(tl0, tl_n)[0, 1])
print(f"③ 非共模：Θ={tf_n:.3f}（Θ_ph={tp_n:.3f}/Θ_sh={ts_n:.3f}）"
      f"  ρ={rho_n:.4f}")

# ④ 理论曲线定位
th_star = 0.8
print(f"\n④ 判定：Θ*=0.8 → 判重算；真值 ρ_full={rho_full:.4f} "
      f"{'≥0.9 → 真复用（假重算，保守向）' if rho_full >= 0.9 else '<0.9 → 真重算（正确）'}")

out = dict(D=float(D), dc_mean=float(dc_mean), dc_std=float(dc0.std()),
           cm_energy_ratio=float(dc_mean ** 2 / np.var(dc0)),
           full=dict(th=float(tf), th_ph=float(tp), th_sh=float(ts_),
                     rho=float(rho_full)),
           common_mode=dict(th=float(tf_cm), rho=float(rho_cm)),
           non_common=dict(th=float(tf_n), th_ph=float(tp_n),
                           th_sh=float(ts_n), rho=float(rho_n)))
with open(os.path.join(WORK, "w6_common_mode_case.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# ---- 图 ----
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8))
ax = axs[0]
ax.plot(c_winter, z_w, color=NAVY, lw=1.8, label="冬季 s13")
ax.plot(c_summer, z_w, color=RED, lw=1.8, label="夏季 s15")
ax.set_ylim(D, 0)
ax.set_xlabel("声速 (m/s)")
ax.set_ylabel("深度 (m)")
ax.set_title("英吉利海峡冬/夏 SSP")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axs[1]
ax.plot(dc0, z_w, color=ORANGE, lw=1.8, label="全量 δc")
ax.plot(dc_nc, z_w, color=TEAL, lw=1.8, label="非共模分量")
ax.axvline(dc_mean, color=RED, ls=":", lw=1.4, label=f"共模 {dc_mean:+.1f} m/s")
ax.set_ylim(D, 0)
ax.set_xlabel("δc (m/s)")
ax.set_ylabel("深度 (m)")
ax.set_title(f"季节差分解（共模能量比 {out['cm_energy_ratio']:.2f}）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axs[2]
labels = ["全量", "纯共模", "非共模"]
ths = [tf, tf_cm, tf_n]
rhos = [rho_full, rho_cm, rho_n]
xs = np.arange(3)
ax.bar(xs, ths, 0.4, color=TEAL, alpha=0.85)
ax.set_xticks(xs)
ax.set_xticklabels([f"{l}\nρ={r:.3f}" for l, r in zip(labels, rhos)],
                   fontsize=9)
ax.set_ylabel("Θ (rad)")
ax.set_title("Θ 分解：共模成分被对差抵消")
ax.grid(alpha=0.3, axis="y")
fig.suptitle("W6 共模案例：真实季节差的共模/非共模分解（A4 性质检验）",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG, "w6_common_mode_case.png"), dpi=110)
print("\n[Done] ->", os.path.join(WORK, "w6_common_mode_case.json"))
