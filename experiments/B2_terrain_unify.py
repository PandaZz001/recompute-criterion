# -*- coding: utf-8 -*-
"""B2 路线 B 第二步：地形判据的统一口径 + 方向不对称 + 共振解析
（2026-11-26）。

B2a 统一口径：地形扰动的一阶等效模态扰动分两部分——
  adiabatic 部分：Δ_m^adia = ∫δD(x)·(∂k_m/∂D)dx（局部模态敏感性沿程
    积分，与主判据同量纲 rad）
  拍频耦合部分：B1 的 Θ_terr^cpl 换算 rad 口径（共振增强修正）
统一判据 Θ_terr^rad = 对差口径，阈值复用主判据 Θ*=0.8。

B2b 方向不对称：ditch50 vs bump50 同幅度反向，ρ 0.938 vs 0.983。
  调查 δTL 空间分布差异 + 非对称修正项。

B2c 共振解析：resonance（λ=拍频波长正弦）耦合幅度的定量预测
  vs 数值。

输出：work/b2_terrain_unify.json + figures/b2_terrain_unify.png
"""

import json
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "criterion"))
from at_core import (ModResult, ShdResult, write_env_kraken,  # noqa: E402
                     run_bellhop)
from mode_sensitivity import ModeSensitivity, theta_phase  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
FIG = os.path.join(HERE, "figures")
KEXE = os.environ.get(
    "KRAKEN_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\Kraken\kraken.exe")
BEXE = os.environ.get(
    "BELLHOP_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\Bellhop\bellhop.exe")
NAVY, TEAL, ORANGE, RED = "#1F3A5F", "#2A9D8F", "#E86A17", "#C12C2C"

FREQ, D_W, SD, RD = 500.0, 120.0, 20.0, 50.0
CB, RHOB, ALPHAB = 1600.0, 1.8, 0.8
RMAX, R_EVAL = 25.0, 25.0
TH_STAR = 0.8  # 主判据阈值


def kraken_modes(D, tag):
    stem = f"b2_{tag}"
    for ext in (".env", ".mod", ".flp", ".shd", ".prt"):
        p = os.path.join(WORK, stem + ext)
        if os.path.exists(p):
            os.remove(p)
    ssp = [(0.0, 1500.0), (D, 1500.0)]
    zgrid = list(np.arange(0.0, D + 0.5, 0.5))
    write_env_kraken(os.path.join(WORK, stem + ".env"), FREQ, ssp, D,
                     SD, zgrid, RMAX, 501, title="B2",
                     bottom=(CB, RHOB, ALPHAB), c_lim=(1400.0, 1650.0))
    res = subprocess.run([KEXE, stem], cwd=WORK, capture_output=True,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stdout[-300:])
    mod = ModResult(os.path.join(WORK, stem + ".mod"))
    return ModeSensitivity(mod, FREQ, 1500.0, D, CB, RHOB, ALPHAB)


def weights_of(S):
    w = np.abs(S._phi_at(SD) * S._phi_at(RD)) * \
        np.exp(S.k.imag * R_EVAL * 1000.0)
    return w / w.sum()


print("== B2 地形判据统一口径（路线 B 第二步）==\n")

# ---- B2a: ∂k_m/∂D 数值灵敏度（D=120 vs 121） ----
S120 = kraken_modes(120.0, "d120")
S121 = kraken_modes(121.0, "d121")
M_min = min(S120.M, S121.M)
dkdD = (S121.k[:M_min] - S120.k[:M_min]) / 1.0  # 复灵敏度（截断对齐）
kr = np.abs(np.real(S120.k))
w = weights_of(S120)
print(f"① ∂k_m/∂D 数值灵敏度（M={S120.M}）：|∂k/∂D| 中位 "
      f"{np.median(np.abs(dkdD)):.2e}（1/m 每 m）")
# 理想波导解析对照：∂k_m/∂D = γ_m²/(k_m·D) = m²π²/(k_m·D³)
k0 = 2.0 * np.pi * FREQ / 1500.0
ms = np.arange(1, S120.M + 1)
gamma = ms * np.pi / 120.0
km_pek = np.sqrt(np.maximum(k0 ** 2 - gamma ** 2, 1e-12))
dkdD_pek = gamma ** 2 / (km_pek * 120.0)
print(f"    Pekeris 解析对照中位 {np.median(dkdD_pek):.2e} | "
      f"数值/解析比 {np.median(np.abs(dkdD)) / np.median(dkdD_pek):.2f}")

# ---- 地形场景（同 B1） ----
def bty_pts(kind):
    if kind == "bump50":
        return [(0, 120), (9.90, 120), (9.95, 121), (10.05, 121),
                (10.10, 120), (25, 120)]
    if kind == "ditch50":
        return [(0, 120), (9.90, 120), (9.95, 119), (10.05, 119),
                (10.10, 120), (25, 120)]
    if kind == "bump1k":
        return [(0, 120), (9.0, 120), (9.5, 121), (10.5, 121),
                (11.0, 120), (25, 120)]
    if kind == "slope":
        return [(0, 120), (25, 121)]
    if kind == "resonance":
        lam = 5.7
        pts = [(0.0, 120.0)]
        for r in np.arange(lam / 4, RMAX, lam / 4):
            pts.append((float(r), float(120.0 + np.sin(2 * np.pi * r / lam))))
        pts.append((25.0, 120.0))
        return pts
    if kind == "shortwave":
        lam = 0.1
        pts = [(0.0, 120.0)]
        for r in np.arange(lam / 4, RMAX, lam / 4):
            pts.append((float(r), float(120.0 + np.sin(2 * np.pi * r / lam))))
        pts.append((25.0, 120.0))
        return pts
    raise ValueError(kind)


def terrain_profile(kind, nr=5001):
    pts = bty_pts(kind)
    r = np.linspace(0.0, RMAX, nr)
    return r, np.interp(r, [p[0] for p in pts],
                        [p[1] - 120.0 for p in pts])


def theta_adia(kind):
    """adiabatic 口径：Δ_m = ∫δD·∂k_m/∂D·dx → 对差加权（rad）。"""
    r, dd = terrain_profile(kind)
    integ = np.trapezoid(dd, r) * 1000.0     # ∫δD dx（m·km→m·m）
    delta = dkdD * integ                     # 各模态相位扰动
    dmat = np.abs(delta[:, None] - delta[None, :])
    ww = w[:, None] * w[None, :]
    return float(np.sum(ww * dmat))


def theta_cpl(kind):
    """拍频耦合口径（B1 拍频项，谱在拍频带的加权分量）。"""
    r, dd = terrain_profile(kind)
    dr = r[1] - r[0]
    kappa = 2.0 * np.pi * np.fft.rfftfreq(len(r), dr)
    spec = np.abs(np.fft.rfft(dd)) * dr
    beat = np.abs(kr[:, None] - kr[None, :])
    g = 1.0 / (np.abs(kr[:, None] ** 2 - kr[None, :] ** 2) + 1e-12)
    np.fill_diagonal(g, 0.0)
    ww = w[:, None] * w[None, :]
    val = np.zeros_like(beat)
    for i in range(S120.M):
        for j in range(S120.M):
            if i == j:
                continue
            val[i, j] = np.interp(beat[i, j], kappa, spec)
    # 谱加权值（rad 换算需完整耦合幅度推导——路线 B 核心理论缺口，
    # 留待 B3；此处保留相对口径）
    return float(np.sum(ww * g * val))


def trip(kind):
    r, dd = terrain_profile(kind)
    dr = r[1] - r[0]
    kappa = 2.0 * np.pi * np.fft.rfftfreq(len(r), dr)
    spec = np.abs(np.fft.rfft(dd)) * dr
    k_peak = kappa[int(np.argmax(spec))]
    beat = np.abs(kr[:, None] - kr[None, :])
    k_beat_hi = float(np.percentile(beat[np.triu_indices(S120.M, 1)], 95))
    return bool(k_peak > 10.0 * k_beat_hi)


# 真值（B1 已存）
b1 = json.load(open(os.path.join(WORK, "b1_terrain_coupling.json"),
                    encoding="utf-8"))
print("\n② 统一口径判定（adiabatic rad + 挡板；Θ*=0.8）：")
print(f"   {'场景':<12}{'Θ_adia':>9}{'Θ_cpl(相对)':>13}{'挡板':>5}"
      f"{'ρ':>8}{'判定':>7}{'方向':>5}")
results = {}
for kind in b1["results"]:
    ta = theta_adia(kind)
    tc = theta_cpl(kind)
    tr = trip(kind)
    rho = b1["results"][kind]["rho"]
    dec = "强制重算" if tr else ("判重算" if ta > TH_STAR else "判复用")
    ok = ("重算" in dec) == (rho < 0.9)
    results[kind] = dict(adia=ta, cpl=tc, trip=bool(tr), rho=rho)
    print(f"   {kind:<12}{ta:>9.3f}{tc:>13.4g}{str(tr):>5}"
          f"{rho:>8.3f}{dec:>7}{'✓' if ok else '✗':>5}")

n_ok = sum(1 for v in results.values()
           if ("重算" in ("强制重算" if v["trip"] else
               ("判重算" if v["adia"] > TH_STAR else "判复用")))
           == (v["rho"] < 0.9))
print(f"   方向正确 {n_ok}/6（Θ_cpl 的 rad 换算待 B3 耦合幅度推导）")

# ---- B2b: 方向不对称 δTL 分布 ----
print("\n③ 方向不对称（ditch50 vs bump50）δTL 分布：")
dtl = {}
for kind in ("bump50", "ditch50"):
    shd = ShdResult(os.path.join(WORK, kind + ".shd"))
    tl = shd.tl()[0]
    flat = ShdResult(os.path.join(WORK, "shallow.shd"))
    iz50 = int(np.argmin(np.abs(flat.rz - 50.0)))
    tl0 = flat.tl()[iz50]
    dtl[kind] = tl - tl0
rr = flat.rr
for kind in ("bump50", "ditch50"):
    seg = (rr >= 9.0) & (rr <= 22.0)
    print(f"   {kind}: 特征后区 r∈[9,22] km δTL 中位 "
          f"{np.median(dtl[kind][seg]):+.2f} dB | 峰值区 r∈[9.5,12] "
          f"{np.max(np.abs(dtl[kind][(rr >= 9.5) & (rr <= 12)])):.2f} dB")

# ---- B2c: 共振幅度解析 ----
print("\n④ 共振解析（resonance vs bump1k 同幅度）：")
r, dd_r = terrain_profile("resonance")
r, dd_b = terrain_profile("bump1k")
# 正弦 1 m 的沿程谱在拍频波数处 = ε·L/2（共振），bump1k 矩形谱分量为
# ε·L·sinc
lam_beat = 5.7
amp_theory = (1.0 * RMAX / 2.0) / (1.0 * 1.0)  # 共振正弦 vs 1 m 特征
print(f"   共振增强预测：正弦(λ=拍频)谱分量 = εL/2 = "
      f"{1.0 * RMAX / 2:.1f} m·km vs bump1k 矩形谱分量 ε·ΔL≈1 m·km "
      f"→ 增强 ~{RMAX / 2:.1f}×")
th_r, th_b = results["resonance"]["cpl"], results["bump1k"]["cpl"]
print(f"   数值 Θ_cpl：resonance {th_r:.3g} vs bump1k {th_b:.3g} "
      f"→ 数值增强 {th_r / max(th_b, 1e-12):.1f}×")

out = dict(results=results, n_ok=n_ok, th_star=TH_STAR,
           dkdD_med=float(np.median(np.abs(dkdD))),
           dkdD_pek_med=float(np.median(dkdD_pek)))
with open(os.path.join(WORK, "b2_terrain_unify.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# ---- 图 ----
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8))
ax = axs[0]
kinds = list(results)
xs = np.arange(len(kinds))
ax.bar(xs - 0.2, [results[k]["adia"] for k in kinds], 0.2, color=TEAL,
       label="adiabatic")
ax.bar(xs, [results[k]["cpl"] for k in kinds], 0.2, color=ORANGE,
       label="拍频耦合")
ax.axhline(TH_STAR, color=RED, ls=":", lw=1.2, label=f"Θ*={TH_STAR}")
ax.set_yscale("log")
ax.set_xticks(xs)
ax.set_xticklabels(kinds, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Θ (rad)")
ax.set_title("统一口径分解（对数轴）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
ax = axs[1]
for kind in ("bump50", "ditch50"):
    ax.plot(rr, dtl[kind], lw=1.4,
            label=f"{kind}（ρ={b1['results'][kind]['rho']:.3f}）")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("距离 (km)")
ax.set_ylabel("δTL (dB)")
ax.set_title("方向不对称：凹坑损伤大于凸起")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axs[2]
for kind, v in results.items():
    mk = "s" if v["trip"] else "o"
    ax.plot(v["adia"], v["rho"], mk, ms=7,
            label=f"{kind}（Θ={v['adia']:.2f}）")
ax.axvline(TH_STAR, color=RED, ls=":", lw=1.2)
ax.axhline(0.9, color=RED, ls="-.", lw=1.2)
ax.set_xscale("log")
ax.set_xlabel("Θ_adia (rad)")
ax.set_ylabel("ρ")
ax.set_title(f"adiabatic 口径判定（方向正确 {n_ok}/6）")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")
fig.suptitle("B2 地形判据统一口径：rad 口径 + 主阈值复用（路线 B 第二步）",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG, "b2_terrain_unify.png"), dpi=110)
print("\n[Done] ->", os.path.join(WORK, "b2_terrain_unify.json"))
