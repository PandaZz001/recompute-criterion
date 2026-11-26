# -*- coding: utf-8 -*-
"""B1 路线 B 第一步：地形沿程扰动的统一拍频判据（2026-11-26）。

把 N4 拍频规则从经验升级为判据内构件：
  模态耦合由地形沿程谱 S_δD(κ) 在拍频波数 |k_m−k_n| 处的分量驱动
  → 地形判据 Θ_terr = Σ_{m,n} w_m w_n · |S_δD(|k_m−k_n|)| · g_mn
  （g_mn = 耦合几何因子，一阶近似取 1/|k_m²−k_n²| 尺度）。

实验：
① Kraken 基线模态（500 Hz、120 m 等声速、底 1600/1.8/0.8）→
   拍频波数集与能量权重
② 4 旧场景回测（bump50/ditch50/bump1k/slope，真值 ρ 已有）+
   2 新场景（共振 λ=1.6 km 正弦 / 非共振 λ=100 m 正弦）Bellhop 真值
③ Θ_terr vs ρ 真值的判定方向对比

输出：work/b1_terrain_coupling.json + figures/b1_terrain_coupling.png
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
from mode_sensitivity import ModeSensitivity  # noqa: E402

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
RMAX = 25.0


# ---- ① Kraken 基线模态 ----
def base_modes():
    stem = "b1_base"
    for ext in (".env", ".mod", ".flp", ".shd", ".prt"):
        p = os.path.join(WORK, stem + ext)
        if os.path.exists(p):
            os.remove(p)
    ssp = [(0.0, 1500.0), (120.0, 1500.0)]  # SSP 只到底部（同 bq1_base）
    zgrid = list(np.arange(0.0, 120.5, 0.5))  # 密集 RD（SW3 经验）
    write_env_kraken(os.path.join(WORK, stem + ".env"), FREQ, ssp, D_W,
                     SD, zgrid, RMAX, 501, title="B1",
                     bottom=(CB, RHOB, ALPHAB), c_lim=(1400.0, 1650.0))
    res = subprocess.run([KEXE, stem], cwd=WORK, capture_output=True,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stdout[-300:])
    mod = ModResult(os.path.join(WORK, stem + ".mod"))
    S = ModeSensitivity(mod, FREQ, 1500.0, D_W, CB, RHOB, ALPHAB)
    w = np.abs(S._phi_at(SD) * S._phi_at(RD)) * \
        np.exp(S.k.imag * 25.0 * 1000.0)
    w = w / w.sum()
    return S, w


print("== B1 地形拍频判据（路线 B 第一步）==\n")
S, w = base_modes()
kr = np.abs(np.real(S.k))
beat = np.abs(kr[:, None] - kr[None, :])
# 拍频波长表（主能量模态对）
i1 = int(np.argmax(w))
i2 = int(np.argsort(w)[-2])
lam12 = 2.0 * np.pi / beat[i1, i2]
print(f"① 基线模态 M={S.M}：主能量模态对 (m={i1 + 1},{i2 + 1}) 拍频波长 "
      f"λ_beat={lam12 / 1000.0:.2f} km")
print(f"   拍频波长范围 "
      f"{2 * np.pi / beat.max() / 1000.0:.3f}–"
      f"{2 * np.pi / (beat.min() + 1e-12) / 1000.0:.1f} km")


# ---- ② 地形场景定义（4 旧 + 2 新） ----
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
        lam = lam12 / 1000.0
        pts = [(0.0, 120.0)]
        for r in np.arange(lam / 4, RMAX, lam / 4):
            d = 120.0 + np.sin(2 * np.pi * r / lam)
            pts.append((float(r), float(d)))
        pts.append((25.0, 120.0))
        return pts
    if kind == "shortwave":
        lam = 0.1
        pts = [(0.0, 120.0)]
        for r in np.arange(lam / 4, RMAX, lam / 4):
            d = 120.0 + np.sin(2 * np.pi * r / lam)
            pts.append((float(r), float(d)))
        pts.append((25.0, 120.0))
        return pts
    raise ValueError(kind)


def terrain_profile(kind, nr=5001):
    """δD(r) 采样剖面（0–25 km）。"""
    pts = bty_pts(kind)
    r = np.linspace(0.0, RMAX, nr)
    return r, np.interp(r, [p[0] for p in pts], [p[1] - 120.0
                                                for p in pts])


def write_env_bty(tag):
    lines = [
        f"'{tag}'", "500.0          ! FREQ (Hz)", "1              ! NMEDIA",
        "'CVW'          ! SSPOPT", "3  0.0  130.0   ! DEPTH of bottom (m)",
        "    0.0  1500.00  /", "  120.0  1500.00  /", "  130.0  1500.00  /",
        "'A~' 0.0", " 130.0  1600.00 0.0 1.8 0.8 /",
        "1              ! NSD", "  20.0 /       ! SD",
        "1              ! NRD", "  50.0 /       ! RD",
        "1251           ! NR", "  0.0  25.0 /  ! R (km)",
        "'CG'           ! run type", "501            ! NBEAMS",
        " -89.0 89.0 /  ! ALPHA1,2", "0.0  140.0  26.0   ! STEP, ZBOX, RBOX",
    ]
    with open(os.path.join(WORK, tag + ".env"), "w",
              encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")


def write_bty(tag, pts):
    with open(os.path.join(WORK, tag + ".bty"), "w",
              encoding="ascii") as f:
        f.write("'L'\n%d\n" % len(pts))
        for r, d in pts:
            f.write(f" {r:g} {d:g}\n")


# 平坦基线（Bellhop）
write_env_bty("b1_flat")
write_bty("b1_flat", [(0, 120), (25, 120)])
shd0 = run_bellhop(BEXE, os.path.join(WORK, "b1_flat.env"), WORK)
iz50 = int(np.argmin(np.abs(shd0.rz - 50.0)))
tl_flat = shd0.tl()[iz50]
rr = shd0.rr
mask = (rr >= 1.0) & (rr <= 24.0)

# 旧场景真值（terrain_scale 档案数值）
OLD_TRUTH = {"bump50": 0.983, "ditch50": 0.997, "bump1k": 0.877,
             "slope": 0.586}
# 用 work 里已有的 .shd（terrain_scale 曾跑过）
OLD_FILES = {"bump50": "bump50.shd", "ditch50": "ditch50.shd",
             "bump1k": "bump1k.shd", "slope": "slope.shd"}


def truth_of(tag):
    """真值 ρ：旧场景读存档 .shd，新场景现跑 Bellhop。"""
    if tag in OLD_FILES:
        p = os.path.join(WORK, OLD_FILES[tag])
        if os.path.exists(p):
            shd = ShdResult(p)
            tl = shd.tl()[0]
            return float(np.corrcoef(tl_flat[mask], tl[mask])[0, 1])
    write_env_bty(f"b1_{tag}")
    write_bty(f"b1_{tag}", bty_pts(tag))
    shd = run_bellhop(BEXE, os.path.join(WORK, f"b1_{tag}.env"), WORK)
    tl = shd.tl()[0]
    return float(np.corrcoef(tl_flat[mask], tl[mask])[0, 1])


# ---- ③ Θ_terr：拍频谱加权判据 + 粗糙度规则 ----
def theta_terr(kind):
    r, dd = terrain_profile(kind)
    dr = r[1] - r[0]
    kappa = 2.0 * np.pi * np.fft.rfftfreq(len(r), dr)
    spec = np.abs(np.fft.rfft(dd)) * dr  # 沿程谱模
    # 在拍频波数处取谱（插值），耦合几何因子 1/|k_m²−k_n²|
    ww = w[:, None] * w[None, :]
    g = 1.0 / (np.abs(kr[:, None] ** 2 - kr[None, :] ** 2) + 1e-12)
    np.fill_diagonal(g, 0.0)
    val = np.zeros_like(beat)
    for i in range(S.M):
        for j in range(S.M):
            if i == j:
                continue
            val[i, j] = np.interp(beat[i, j], kappa, spec)
    th_beat = float(np.sum(ww * g * val))
    # 粗糙度规则（Rayleigh 参数 2kσ_rms）
    k0 = 2.0 * np.pi * FREQ / 1500.0
    th_rough = 2.0 * k0 * float(np.std(dd))
    # 谱峰挡板：周期粗糙地形（窄带谱峰远高于拍频带）走射线轨道
    # 失稳机制（多反弹累积），与拍频耦合不同源——强制重算
    peak_idx = int(np.argmax(spec))
    k_peak = kappa[peak_idx]
    k_beat_hi = float(np.percentile(beat[np.triu_indices(S.M, 1)], 95))
    return th_beat + th_rough, bool(k_peak > 10.0 * k_beat_hi)


print("\n② 六场景 Θ_terr vs 真值 ρ（含谱峰挡板）：")
print(f"   {'场景':<12}{'Θ_terr':>10}{'ρ 真值':>9}{'挡板':>6}{'判定':>8}")
results = {}
for kind in ["bump50", "ditch50", "bump1k", "slope", "resonance",
             "shortwave"]:
    th, trip = theta_terr(kind)
    rho = truth_of(kind)
    results[kind] = dict(theta=th, rho=rho, trip=bool(trip))
    dec = "强制重算" if trip else ("判重算" if th > 100 else "判复用")
    truth_dec = "真重算" if rho < 0.9 else "真复用"
    ok = "✓" if (("重算" in dec) == (rho < 0.9)) else "✗"
    print(f"   {kind:<12}{th:>10.4g}{rho:>9.3f}{str(trip):>6}"
          f"{dec:>8}{truth_dec + ok:>8}")

# 判定方向检验：Θ_terr 与 (1−ρ) 的单调性（Spearman）
ths = np.array([results[k]["theta"] for k in results])
rhos = np.array([results[k]["rho"] for k in results])
from scipy.stats import spearmanr
sp, _ = spearmanr(ths, -rhos)
print(f"\nΘ_terr 与 (1−ρ) 的 Spearman 相关 = {sp:+.2f}"
      f"（应 >0：Θ 大 ↔ ρ 低）")

out = dict(results=results, spearman=float(sp),
           lam_beat_main=float(lam12), M=S.M)
with open(os.path.join(WORK, "b1_terrain_coupling.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# ---- 图 ----
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8))
ax = axs[0]
for kind in results:
    r, dd = terrain_profile(kind)
    ax.plot(r / 1000.0, dd, lw=1.3, label=kind)
ax.set_xlabel("距离 (km)")
ax.set_ylabel("δD (m)")
ax.set_title("六种地形剖面")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)
ax = axs[1]
for kind in results:
    r, dd = terrain_profile(kind)
    dr = r[1] - r[0]
    kappa = 2.0 * np.pi * np.fft.rfftfreq(len(r), dr)
    spec = np.abs(np.fft.rfft(dd)) * dr
    ax.plot(kappa, spec, lw=1.3, label=kind)
beat_all = beat[np.triu_indices(S.M, 1)]
for b in [lam12, 2 * np.pi / beat[i1, i2]]:
    ax.axvline(b, color=RED, ls=":", lw=1)
ax.axvspan(np.percentile(beat_all, 5), np.percentile(beat_all, 95),
           color=RED, alpha=0.1, label="拍频波数带（5–95%）")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("波数 κ (rad/m)")
ax.set_ylabel("|S_δD|")
ax.set_title("沿程谱与拍频波数带（红=耦合敏感带）")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3, which="both")
ax = axs[2]
for kind, v in results.items():
    ax.plot(v["theta"], v["rho"], "o", ms=8,
            label=f"{kind}（Θ={v['theta']:.2g}）")
ax.set_xscale("log")
ax.set_xlabel("Θ_terr（拍频判据）")
ax.set_ylabel("真值 ρ")
ax.set_title(f"Θ_terr vs ρ（Spearman {sp:+.2f}）")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")
fig.suptitle("B1 地形拍频判据：N4 规则升级为判据内构件（路线 B 第一步）",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG, "b1_terrain_coupling.png"), dpi=110)
print("\n[Done] ->", os.path.join(WORK, "b1_terrain_coupling.json"))
