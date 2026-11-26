# -*- coding: utf-8 -*-
"""B′ 强化版 2：缩放灰区族（破解判定平凡的核心实验）。

29 对实测 CTD δc 乘缩放因子 α ∈ {0.03..1.0}，把 Θ 拉进灰区
（Θ∈[0.3, 1.5]、ρ∈[0.7, 0.99]），在官方口径（分层底质 + 官方 SSP
基线）下检验判据对"复用/重算"的方向正确性。

- 真值：Kraken 全重算 ρ(α·δc)（29 对 × 9 缩放 = 261 样本）
- 判据：一阶 Θ(α·δc)（线性微扰下 Θ ∝ α，作为自检一并验证）
- 灰区统计：Θ∈[0.4, 1.6] 子集的混淆矩阵 + ρ(Θ) 全区间曲线
  与理论预测对照（广义型 exp(−λΘ^p)，推导参数 λ=0.285、p=1.40）

输出：work/sw3e2_gray_zone.json + figures/sw3e2_gray_zone.png
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "criterion"))
from at_core import write_flp, run_kraken_field  # noqa: E402
from sw3_common import (parse_official_env, write_env_layered)  # noqa: E402
from mode_sensitivity import ModeSensitivity, theta_phase  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
SW3W = os.path.join(WORK, "sw3")
FIG = os.path.join(HERE, "figures")
KEXE = os.environ.get(
    "KRAKEN_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\Kraken\kraken.exe")
FEXE = os.environ.get(
    "KRAKEN_FIELD_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\KrakenField\field.exe")
NAVY, TEAL, ORANGE, RED = "#1F3A5F", "#2A9D8F", "#E86A17", "#C12C2C"

CTD_DIR = os.environ.get(
    "SWELLEX96_CTD_DIR",
    r"D:\0_Research\SWellEx96\env\ctds\ctds")
FREQ = 500.0
SD, RD0, RMAX, NR = 20.0, 50.0, 25.0, 501
TH_STAR, RHO_STAR = 1.2, 0.9
ALPHAS = [0.03, 0.06, 0.10, 0.15, 0.22, 0.32, 0.45, 0.65, 1.0]

ssp_o, layers, half = parse_official_env()
D = ssp_o[-1][0]
ssp_o = [(float(z), float(c)) for z, c in ssp_o]
c_off = np.array([c for _, c in ssp_o])
CW = float(np.mean(c_off))
D_J = 212.5
zgrid = np.arange(0.0, D_J + 0.5, 0.5)
os.makedirs(SW3W, exist_ok=True)


def read_ctd(path):
    zs, cs = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                zs.append(float(parts[0]))
                cs.append(float(parts[3]))
            except ValueError:
                continue
    return np.array(zs), np.array(cs)


def interp_to(z, c, zgrid):
    return np.interp(zgrid, z, c, left=np.nan, right=np.nan)


def kraken_tl(tag, dcp):
    """官方 4 层底 Kraken+Field → TL（RD=50，501 距离点）。"""
    import subprocess
    ssp = [(z, c + float(np.interp(z, zgrid, dcp))) for z, c in ssp_o]
    stem = f"sw3e2_{tag}"
    for ext in (".env", ".mod", ".flp", ".shd", ".prt"):
        p = os.path.join(SW3W, stem + ext)
        if os.path.exists(p):
            os.remove(p)
    write_env_layered(os.path.join(SW3W, stem + ".env"), FREQ, ssp,
                      layers, half, RMAX, SD, [RD0], title="SW3E2")
    res = subprocess.run([KEXE, stem], cwd=SW3W, capture_output=True,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(f"kraken failed: {res.stdout[-300:]}")
    write_flp(os.path.join(SW3W, stem + ".flp"), sd=SD, rd=[RD0],
              rmax=RMAX, nr=NR)
    shd = run_kraken_field(FEXE, os.path.join(SW3W, stem + ".flp"), SW3W)
    import struct
    raw = open(os.path.join(SW3W, stem + ".shd"), "rb").read()
    rl = struct.unpack("I", raw[:4])[0] * 4
    P = np.frombuffer(raw[10 * rl:], dtype="<c8").copy()[:NR]
    return -20.0 * np.log10(np.abs(P) + 1e-300)


# 基线 TL + 判据模态
tl0 = kraken_tl("base", np.zeros(len(zgrid)))
mask = (np.linspace(0.0, RMAX, NR) >= 1.0) & (np.linspace(0.0, RMAX, NR)
                                              <= 20.0)
from at_core import write_env_kraken
import subprocess
c_uni = np.interp(zgrid, [z for z, _ in ssp_o], [c for _, c in ssp_o])
ssp_uni = [(float(z), float(c)) for z, c in zip(zgrid, c_uni)]
write_env_kraken(os.path.join(SW3W, "sw3e2_j.env"), FREQ, ssp_uni, D_J,
                 SD, list(zgrid), RMAX, NR, title="SW3E2",
                 bottom=(1572.4, 1.76, 0.2), c_lim=(1400.0, 1650.0))
res = subprocess.run([KEXE, "sw3e2_j"], cwd=SW3W, capture_output=True,
                     text=True)
if res.returncode != 0:
    raise RuntimeError(res.stdout[-300:])
from at_core import ModResult
mod_j = ModResult(os.path.join(SW3W, "sw3e2_j.mod"))
S = ModeSensitivity(mod_j, FREQ, CW, D_J, 1572.4, 1.76, 0.2)
cz_grid = np.interp(S.z, zgrid, c_uni)
print(f"== B′ 强化版 2：缩放灰区族（判据模态 M={mod_j.M}）==\n")


def criterion(dc):
    dc = np.interp(S.z, zgrid, dc)
    integ = np.trapezoid(dc[:, None] * S.phi.real ** 2
                         / cz_grid[:, None] ** 3, S.z, axis=0)
    dk = -(S.w ** 2 / S.k) * integ
    th_ph = theta_phase(S, dk)
    P = S.phi.real
    dcw = dc * (-2.0 / cz_grid ** 3)
    I = np.trapezoid(dcw[:, None, None] * P[:, None, :] * P[:, :, None],
                     S.z, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        Cm = S.w ** 2 * I / (S.k[:, None] ** 2 - S.k[None, :] ** 2)
    Cm[np.diag_indices(S.M)] = 0.0
    dph = S.phi @ Cm.T
    phiv = S._phi_at(RD0)
    phis = S._phi_at(SD)
    dr = dph[int(np.argmin(np.abs(S.z - RD0)))]
    ds = dph[int(np.argmin(np.abs(S.z - SD)))]
    dw = ds * phiv + phis * dr
    return theta_phase(S, dk, dw=dw)


files = sorted(f for f in os.listdir(CTD_DIR) if f.endswith(".prn"))
casts = {}
for fn in files:
    z, c = read_ctd(os.path.join(CTD_DIR, fn))
    casts[fn] = (z, c)

rows = []
for i in range(len(files) - 1):
    ca = interp_to(casts[files[i]][0], casts[files[i]][1], zgrid)
    cb = interp_to(casts[files[i + 1]][0], casts[files[i + 1]][1], zgrid)
    valid = (~np.isnan(ca)) & (~np.isnan(cb))
    if valid.sum() < 200:
        continue
    ca = np.where(np.isnan(ca), CW, ca)
    cb = np.where(np.isnan(cb), CW, cb)
    dc0 = np.where(valid, cb - ca, 0.0)
    for a in ALPHAS:
        dc = dc0 * a
        th = criterion(dc)
        tl = kraken_tl(f"p{i}_a{a}", dc)
        rho = float(np.corrcoef(tl0[mask], tl[mask])[0, 1])
        rows.append(dict(i=i, alpha=a, th=float(th), rho=rho,
                         dec=bool(th < TH_STAR),
                         truth=bool(rho >= RHO_STAR)))
    print(f"  对 {i:2d}（{files[i][:8]}）9 缩放完成")

n = len(rows)
ok = sum(r["dec"] == r["truth"] for r in rows)
fr = sum(r["dec"] and not r["truth"] for r in rows)
fm = sum(not r["dec"] and r["truth"] for r in rows)
gray = [r for r in rows if 0.4 <= r["th"] <= 1.6]
ok_g = sum(r["dec"] == r["truth"] for r in gray)
fr_g = sum(r["dec"] and not r["truth"] for r in gray)
fm_g = sum(not r["dec"] and r["truth"] for r in gray)
print(f"\n1. 全样本（n={n}）：正确 {ok} | 假复用 {fr} | 假重算 {fm}")
print(f"2. 灰区 Θ∈[0.4,1.6]（n={len(gray)}）：正确 {ok_g} | 假复用 "
      f"{fr_g} | 假重算 {fm_g}")
print(f"   灰区 ρ 范围 [{min(r['rho'] for r in gray):.3f}, "
      f"{max(r['rho'] for r in gray):.3f}]")

# 理论对照：广义型 exp(-λΘ^p)（E2g 推导参数）与椭圆模型
th_arr = np.array([r["th"] for r in rows])
rho_arr = np.array([r["rho"] for r in rows])
lam, p = 0.285, 1.40
rho_gen = np.exp(-lam * th_arr ** p)
rho_ell = 0.4313 + 0.5687 * np.exp(-(0.518 * th_arr) ** 2 / 2)
rmse_gen = float(np.sqrt(np.mean((rho_arr - rho_gen) ** 2)))
rmse_ell = float(np.sqrt(np.mean((rho_arr - rho_ell) ** 2)))
print(f"\n3. 理论对照 RMSE：广义型 {rmse_gen:.3f} | 椭圆模型 {rmse_ell:.3f}")

out = dict(n=n, alphas=ALPHAS, ok=ok, fr=fr, fm=fm,
           n_gray=len(gray), ok_gray=ok_g, fr_gray=fr_g, fm_gray=fm_g,
           rmse_generalized=rmse_gen, rmse_ellipse=rmse_ell,
           rows=rows)
with open(os.path.join(WORK, "sw3e2_gray_zone.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

fig, axs = plt.subplots(1, 3, figsize=(16.0, 4.8))
ax = axs[0]
for r in rows:
    c = TEAL if r["truth"] else RED
    ax.plot(r["th"], r["rho"], "o", ms=4, color=c, alpha=0.55)
th_curve = np.linspace(0.02, 14, 200)
ax.plot(th_curve, np.exp(-lam * th_curve ** p), color=NAVY, lw=1.8,
        label="广义型 exp(−λΘ^p)（推导参数）")
ax.plot(th_curve, 0.4313 + 0.5687 * np.exp(-(0.518 * th_curve) ** 2 / 2),
        color=ORANGE, lw=1.5, ls="--", label="椭圆模型（推导参数）")
ax.axvline(TH_STAR, color=RED, ls=":", lw=1.2, label="Θ* = 1.2")
ax.axhline(RHO_STAR, color=RED, ls="-.", lw=1.2, label="ρ* = 0.9")
ax.set_xscale("log")
ax.set_xlabel("判据 Θ (rad)")
ax.set_ylabel("真值 ρ")
ax.set_title("缩放灰区族 ρ-Θ 全区间（绿=复用，红=重算）")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")
ax = axs[1]
gs = [r for r in rows if 0.4 <= r["th"] <= 1.6]
ths = sorted(set(r["th"] for r in gs))
fr_bin = []
for lo in np.linspace(0.4, 1.6, 7)[:-1]:
    hi = lo + 0.2
    seg = [r for r in gs if lo <= r["th"] < hi]
    fr_bin.append(sum(r["dec"] and not r["truth"] for r in seg) /
                  max(len(seg), 1))
ax.bar(np.linspace(0.5, 1.5, 6), fr_bin, width=0.17, color=RED, alpha=0.8)
ax.set_xlabel("Θ 分箱")
ax.set_ylabel("假复用率")
ax.set_title("灰区分箱假复用率（保守性分布）")
ax.grid(alpha=0.3, axis="y")
ax = axs[2]
for i in range(min(6, len(files))):
    seg = [r for r in rows if r["i"] == i]
    seg = sorted(seg, key=lambda r: r["alpha"])
    ax.plot([r["th"] for r in seg], [r["rho"] for r in seg], "o-",
            ms=3, lw=1, alpha=0.8)
ax.set_xlabel("Θ (rad)")
ax.set_ylabel("ρ")
ax.set_title("前 6 对的 ρ(Θ) 缩放轨迹")
ax.grid(alpha=0.3)
fig.suptitle("SW3e2 缩放灰区族：官方口径下的判据区分能力检验",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG, "sw3e2_gray_zone.png"), dpi=110)
print("\n[Done] ->", os.path.join(WORK, "sw3e2_gray_zone.json"))
