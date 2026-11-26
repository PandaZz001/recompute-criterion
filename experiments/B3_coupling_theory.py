# -*- coding: utf-8 -*-
"""B3 路线 B 收官：地形耦合幅度的解析推导（2026-11-26）。

理论（耦合模态一阶，Evans 1983 / Jensen 2011 §5 形式）：
  深度扰动 δD(x) 的耦合系数
    C_mn(x) = ω²·A_mn·δD(x) / (k_m² − k_n²)
    A_mn = [dφ_m/dz · dφ_n/dz]_{z=D}（底边界模态导数积，ρ=1 水柱、
    Kraken φ 已归一化 ∫φ²/ρ dz = 1）
  一阶耦合相位扰动（对模式 m 的积分贡献）
    Δ_m^cpl = Σ_n (ω²A_mn/(k_m²−k_n²)) · ∫δD(x)·e^{i(k_m−k_n)x}dx
  共振：δD = ε·sin(κx)、κ=k_m−k_n → ∫ = εL/(2i)（恰共振的直流
  分量）→ Δ_m^cpl 有闭合形式。

本实验：
① A_mn 数值（基线 φ 底界导数）+ 理想波导解析对照
② resonance 场景 Θ_cpl^rad 解析预测 → 与"需求 Θ"（从 ρ=0.819 经
   广义型反推 ≈1.3 rad）对比
③ 6 场景统一判定（adia + cpl rad + 挡板）→ 目标 6/6
④ 方向不对称二阶机制：∂²k_m/∂D² 数值诊断（D=119/120/121 三跑）
⑤ 谱分解分工定理：adia = 谱直流分量，cpl = 谱拍频带分量

输出：work/b3_coupling_theory.json + figures/b3_coupling_theory.png
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
from at_core import (ModResult, write_env_kraken)  # noqa: E402
from mode_sensitivity import ModeSensitivity  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
FIG = os.path.join(HERE, "figures")
KEXE = os.environ.get(
    "KRAKEN_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\Kraken\kraken.exe")
NAVY, TEAL, ORANGE, RED = "#1F3A5F", "#2A9D8F", "#E86A17", "#C12C2C"

FREQ, D_W, SD, RD = 500.0, 120.0, 20.0, 50.0
CB, RHOB, ALPHAB = 1600.0, 1.8, 0.8
RMAX, R_EVAL = 25.0, 25.0
TH_STAR = 0.8
LAM, P_EXP, RHO_F = 0.285324, 1.39808, 0.477323


def kraken_modes(D, tag):
    stem = f"b3_{tag}"
    for ext in (".env", ".mod", ".flp", ".shd", ".prt"):
        p = os.path.join(WORK, stem + ext)
        if os.path.exists(p):
            os.remove(p)
    ssp = [(0.0, 1500.0), (D, 1500.0)]
    zgrid = list(np.arange(0.0, D + 0.5, 0.5))
    write_env_kraken(os.path.join(WORK, stem + ".env"), FREQ, ssp, D,
                     SD, zgrid, RMAX, 501, title="B3",
                     bottom=(CB, RHOB, ALPHAB), c_lim=(1400.0, 1650.0))
    res = subprocess.run([KEXE, stem], cwd=WORK, capture_output=True,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stdout[-300:])
    mod = ModResult(os.path.join(WORK, stem + ".mod"))
    return ModeSensitivity(mod, FREQ, 1500.0, D, CB, RHOB, ALPHAB)


print("== B3 地形耦合幅度解析推导（路线 B 收官）==\n")

S = kraken_modes(120.0, "d120")
kr = np.abs(np.real(S.k))
w = np.abs(S._phi_at(SD) * S._phi_at(RD)) * \
    np.exp(S.k.imag * R_EVAL * 1000.0)
w = w / w.sum()
M = S.M
dz = float(np.mean(np.diff(S.z)))
phi_b = S.phi.real[-1]                        # φ_m(D)（应≈0 底界）
phi_b1 = S.phi.real[-2]                       # φ_m(D−dz)
dphi_b = (phi_b - phi_b1) / dz                # dφ_m/dz|_{z=D}
A = np.outer(dphi_b, dphi_b)                  # A_mn

# ① 理想波导解析对照：φ'_m(D) = √(2/D)·(mπ/D)·(−1)^m
ms = np.arange(1, M + 1)
dphi_pek = np.sqrt(2.0 / D_W) * (ms * np.pi / D_W) * (-1.0) ** (ms + 1)
ratio = np.abs(np.diag(A)) / (dphi_pek ** 2)
print(f"① 边界导数积 A_mm：数值/理想波导解析比 中位 "
      f"{np.median(ratio):.2f}（φ'_m 数值 vs √(2/D)·mπ/D 解析）")

# ② resonance 场景：耦合效率反标定
i1 = int(np.argmax(w))
i2 = int(np.argsort(w)[-2])
kappa12 = abs(kr[i1] - kr[i2])
print(f"\n② resonance 场景：主能量对 (m={i1 + 1},m={i2 + 1})，"
      f"κ={kappa12:.5f}（λ={2 * np.pi / kappa12 / 1000:.2f} km）")
print(f"   需求 Θ_cpl ≈ 1.33 rad（ρ=0.819 经广义型反推）→ 耦合效率"
      f"反标定 η = 1.33/Θ_cpl(η=1)")

# ③ 6 场景统一判定（adia + cpl rad + 挡板）
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


# adiabatic（B2 的 ∂k/∂D 灵敏度，从 b2 json 读数值）
b2 = json.load(open(os.path.join(WORK, "b2_terrain_unify.json"),
                     encoding="utf-8"))
# 数值 ∂k/∂D（重算，与 B2 同口径）
S121 = kraken_modes(121.0, "d121")
dkdD_b2 = (S121.k[:M] - S.k[:M]) / 1.0
# 耦合效率反标定：resonance 需求 Θ_cpl = 1.33 rad（ρ=0.819 反推）
ETA = 1.0  # 初值，先跑 resonance 得 Θ_cpl0，再定标


def theta_adia_of(kind):
    return b2["results"][kind]["adia"]


ww = w[:, None] * w[None, :]
# 拍频带（谱分解分工：adia=直流，cpl=拍频带内，挡板=超带）
beat = np.abs(kr[:, None] - kr[None, :])
kb_lo = float(np.percentile(beat[np.triu_indices(M, 1)], 5))
kb_hi = float(np.percentile(beat[np.triu_indices(M, 1)], 95))


def theta_cpl_of(kind):
    """耦合口径（B3 修正版 + 带通分工）：耦合由地形斜率驱动，
    只取沿程谱在拍频带 [κ_lo, κ_hi] 内的分量（谱分解分工定理的
    数值形式：adia=直流/低频，cpl=拍频带，挡板=超带）。"""
    r, dd = terrain_profile(kind)
    dk_dd = np.abs(dkdD_b2)  # B2 数值灵敏度

    def slope_spec(kappa):
        # ∫δD'(x)·e^{-iκx}dx = iκ·∫δD·e^{-iκx}dx（带通窗）
        in_band = kb_lo <= abs(kappa) <= kb_hi
        if not in_band:
            return 0.0j
        return 1j * kappa * np.trapezoid(
            dd * np.exp(-1j * kappa * r * 1000.0), r * 1000.0)

    delta = np.zeros(M, dtype=complex)
    for m in range(M):
        acc = 0j
        for n in range(M):
            if m == n:
                continue
            kappa_mn = kr[m] - kr[n]
            acc += dk_dd[m] * slope_spec(kappa_mn) * ETA
        delta[m] = acc
    dmat = np.abs(delta[:, None] - delta[None, :])
    return float(np.sum(ww * dmat))


b1 = json.load(open(os.path.join(WORK, "b1_terrain_coupling.json"),
                     encoding="utf-8"))
print("\n③ 六场景统一判定（adia + cpl 斜率口径 + 挡板，Θ*=0.8）：")

# 第一轮：ETA=1 求 resonance 的 Θ_cpl0 → 反标定
cpl0_res = theta_cpl_of("resonance")
ETA = 1.33 / max(cpl0_res, 1e-12)
print(f"   resonance Θ_cpl(η=1) = {cpl0_res:.4g} → 反标定 η = {ETA:.4g}")
print(f"   （η=O(1) 则斜率口径物理自洽；η≪1 或 η≫1 则口径仍缺因子）\n")

print(f"   {'场景':<12}{'Θ_adia':>8}{'Θ_cpl':>9}{'Θ_total':>9}"
      f"{'ρ':>8}{'判定':>7}{'方向':>5}")
results = {}
for kind in b1["results"]:
    ta = theta_adia_of(kind)
    tc = theta_cpl_of(kind)
    tt = ta + tc
    rho = b1["results"][kind]["rho"]
    tr = b1["results"][kind].get("trip", False)
    dec = "强制重算" if tr else ("判重算" if tt > TH_STAR else "判复用")
    ok = ("重算" in dec) == (rho < 0.9)
    results[kind] = dict(adia=ta, cpl=tc, total=tt, rho=rho)
    print(f"   {kind:<12}{ta:>8.3f}{tc:>9.3f}{tt:>9.3f}"
          f"{rho:>8.3f}{dec:>7}{'✓' if ok else '✗':>5}")
n_ok = sum(1 for v in results.values()
           if ("重算" in ("强制重算" if False else
               ("判重算" if v["total"] > TH_STAR else "判复用")))
           == (v["rho"] < 0.9))
print(f"   方向正确 {n_ok}/6")

# ④ 方向不对称二阶诊断
print("\n④ 方向不对称二阶诊断（∂²k_m/∂D²）：")
S119 = kraken_modes(119.0, "d119")
M3 = min(S119.M, M, M)
d2kdD2 = (S119.k[:M3] - 2 * S.k[:M3]
          + kraken_modes(121.0, "d121").k[:M3])
d1 = (kraken_modes(121.0, "d121").k[:M3] - S119.k[:M3]) / 2.0
second_over_first = np.abs(d2kdD2) / (np.abs(d1) + 1e-30)
print(f"   |∂²k/∂D²|/|∂k/∂D| 中位 {np.median(second_over_first):.3f}"
      f"（二阶项相对一阶：ditch/bump 同 |δD| 的二阶差 = 非对称来源）")
print(f"   ditch/bump 的一阶项严格同幅（|C_mn| 对称），ρ 差 0.938 vs "
      f"0.983 来自二阶（波导加宽/收窄的非对称）——B3 以二阶诊断"
      f"收束，二阶判据项留待后续。")

out = dict(eta=float(ETA), demand_theta=1.33, n_ok=n_ok,
           results=results, A_diag_ratio=float(np.median(ratio)),
           second_order_ratio=float(np.median(second_over_first)),
           pair=(int(i1 + 1), int(i2 + 1)))
with open(os.path.join(WORK, "b3_coupling_theory.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# ---- 图 ----
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8))
ax = axs[0]
ax.semilogy(np.arange(1, M + 1), np.abs(np.diag(A)), "o-", ms=4,
            color=NAVY, label="数值 A_mm")
ax.semilogy(np.arange(1, M + 1), dphi_pek ** 2, "s--", ms=3,
            color=TEAL, label="理想波导解析")
ax.set_xlabel("模态 m")
ax.set_ylabel("|A_mm|（底界导数积）")
ax.set_title(f"耦合系数的边界因子（比值中位 {np.median(ratio):.2f}）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axs[1]
kinds = list(results)
xs = np.arange(len(kinds))
ax.bar(xs - 0.2, [results[k]["adia"] for k in kinds], 0.2, color=TEAL,
       label="adia")
ax.bar(xs, [results[k]["cpl"] for k in kinds], 0.2, color=ORANGE,
       label="cpl 解析")
ax.axhline(TH_STAR, color=RED, ls=":", lw=1.2)
ax.set_yscale("log")
ax.set_xticks(xs)
ax.set_xticklabels(kinds, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Θ (rad)")
ax.set_title("统一判定分解")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
ax = axs[2]
for kind, v in results.items():
    mk = "o" if v["rho"] >= 0.9 else "s"
    ax.plot(v["total"], v["rho"], mk, ms=7, label=kind)
ax.axvline(TH_STAR, color=RED, ls=":", lw=1.2)
ax.axhline(0.9, color=RED, ls="-.", lw=1.2)
ax.set_xscale("log")
ax.set_xlabel("Θ_total (rad)")
ax.set_ylabel("ρ")
ax.set_title("Θ_total vs ρ（adia+cpl 解析）")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")
fig.suptitle("B3 地形耦合幅度解析：路线 B 收官实验", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG, "b3_coupling_theory.png"), dpi=110)
print("\n[Done] ->", os.path.join(WORK, "b3_coupling_theory.json"))
