# -*- coding: utf-8 -*-
"""F1 频率标度律多频验证（路线 D，2026-11-26 今日接续第②项）。

判据阈值 Θ*=0.8 rad 是在 500 Hz 标定的。本实验回答"只测 500 Hz"的
批评：Θ 对频率的标度律 Θ(f) ∝ f^β 的数值测量 + Pekeris 理想波导解析
对照（低模态 β=1 预测、高模态 1/k_m 膨胀修正）。

- 数值口径：SWellEx-96 官方环境（ssp_o + 4 层底质，SW3e2 同口径），
  3 对实测 CTD δc（Θ 低/中/高各一），f ∈ {50..500} Hz 扫 7 频点，
  每频点 Kraken 基线模态 + 一阶微扰 Θ(f)。
- 解析口径：Pekeris 理想波导 δk_m = ω²·I_m/k_m，
  I_m = (2/D)∫δc·c⁻³·sin²(mπz/D)dz，k_m = √(k²−(mπ/D)²)。
- 输出：work/f1_freq_scaling.json + figures/f1_freq_scaling.png
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
from sw3_common import parse_official_env  # noqa: E402
from mode_sensitivity import ModeSensitivity, theta_phase  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
SW3W = os.path.join(WORK, "sw3")
FIG = os.path.join(HERE, "figures")
KEXE = r"D:\0_Research\toolbox\Acoustics-Toolbox\Kraken\kraken.exe"
NAVY, TEAL, ORANGE, RED = "#1F3A5F", "#2A9D8F", "#E86A17", "#C12C2C"

CTD_DIR = os.environ.get(
    "SWELLEX96_CTD_DIR",
    r"D:\0_Research\SWellEx96\env\ctds\ctds")
SD, RD0 = 20.0, 50.0
RMAX, NR = 25.0, 501
R_EVAL = 25.0
FREQS = [50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0]
CB, RHOB, ALPHAB = 1572.4, 1.76, 0.2

# ---- 官方环境（SW3e2 同口径） ----
ssp_o, layers, half = parse_official_env()
ssp_o = [(float(z), float(c)) for z, c in ssp_o]
c_off = np.array([c for _, c in ssp_o])
CW = float(np.mean(c_off))
D_J = 212.5
zgrid = np.arange(0.0, D_J + 0.5, 0.5)
c_uni = np.interp(zgrid, [z for z, _ in ssp_o], [c for _, c in ssp_o])
ssp_uni = [(float(z), float(c)) for z, c in zip(zgrid, c_uni)]
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


def interp_to(z, c):
    return np.interp(zgrid, z, c, left=np.nan, right=np.nan)


def make_sens(freq):
    """Kraken 基线模态（官方 SSP 无扰动）→ ModeSensitivity。"""
    stem = f"f1_j{int(freq)}"
    for ext in (".env", ".mod", ".flp", ".shd", ".prt"):
        p = os.path.join(SW3W, stem + ext)
        if os.path.exists(p):
            os.remove(p)
    write_env_kraken(os.path.join(SW3W, stem + ".env"), freq, ssp_uni, D_J,
                     SD, list(zgrid), RMAX, NR, title="F1",
                     bottom=(CB, RHOB, ALPHAB), c_lim=(1400.0, 1650.0))
    res = subprocess.run([KEXE, stem], cwd=SW3W, capture_output=True,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(f"kraken {freq} Hz failed: {res.stdout[-300:]}")
    mod = ModResult(os.path.join(SW3W, stem + ".mod"))
    return ModeSensitivity(mod, freq, CW, D_J, CB, RHOB, ALPHAB)


def criterion(S, dc):
    """Θ 全量（相位+形状），SW3e2/AB1 同式。"""
    dc = np.interp(S.z, zgrid, dc)
    cz = np.interp(S.z, zgrid, c_uni)
    integ = np.trapezoid(dc[:, None] * S.phi.real ** 2
                         / cz[:, None] ** 3, S.z, axis=0)
    dk = -(S.w ** 2 / S.k) * integ
    P = S.phi.real
    dcw = dc * (-2.0 / cz ** 3)
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


def theta_pek(freq, dc, c0=1500.0):
    """Pekeris 理想波导解析 Θ(f)：δk_m = ω²I_m/k_m。

    仅传播模态（γ_m ≤ k，严格截止，不 pad——消逝模 k_m 虚数会被
    1e-12 地板放大到物理上错误的发散值）。
    """
    om = 2.0 * np.pi * freq
    k = om / c0
    m_cut = int(np.floor(k * D_J / np.pi))
    if m_cut < 2:
        return 0.0
    ms = np.arange(1, m_cut + 1)
    gamma = ms * np.pi / D_J
    # 严格传播条件：γ_m < k·(1−1e-6)。2Df/c 恰为整数时（如 300 Hz=85.0）
    # 浮点会混入一个消逝模 → sqrt 负数 → nan，必须剔除。
    keep = gamma < k * (1.0 - 1e-6)
    ms, gamma = ms[keep], gamma[keep]
    km = np.sqrt(k ** 2 - gamma ** 2)
    dz = zgrid[1] - zgrid[0]
    sinm = np.sin(ms[:, None] * np.pi * zgrid[None, :] / D_J)
    # I_m = (2/D)∫δc·c⁻³·sin² dz（c 用基线 c_uni）
    integ = np.sum(sinm ** 2 * dc[None, :]
                   / (c_uni[None, :] ** 3), axis=1) * dz * (2.0 / D_J)
    dk = -(om ** 2 / km) * integ
    w = np.abs(np.sin(ms * np.pi * SD / D_J)
               * np.sin(ms * np.pi * RD0 / D_J))
    w = w / w.sum()
    delta = dk * (R_EVAL * 1000.0)
    dmat = np.abs(delta[:, None] - delta[None, :])
    ww = w[:, None] * w[None, :]
    return float(np.sum(ww * dmat))


# ---- 全部 29 对（α=1 原始 δc0）----
files = sorted(f for f in os.listdir(CTD_DIR) if f.endswith(".prn"))
casts = {fn: read_ctd(os.path.join(CTD_DIR, fn)) for fn in files}
pairs = []
for i in range(len(files) - 1):
    ca = interp_to(*casts[files[i]])
    cb = interp_to(*casts[files[i + 1]])
    valid = (~np.isnan(ca)) & (~np.isnan(cb))
    if valid.sum() < 200:
        continue
    dc0 = np.where(valid, np.where(np.isnan(ca), CW, ca)
                   - np.where(np.isnan(cb), CW, cb), 0.0)
    pairs.append((i, dc0))
sens500 = make_sens(500.0)
ths500 = {i: criterion(sens500, dc0) for i, dc0 in pairs}
print(f"== F1 频率标度律（{len(pairs)} 对全扫 + Pekeris 对照）==\n")

# ---- 主实验：全部对的数值 Θ(f)（每频一次 Kraken，criterion 复用） ----
sens_cache = {500.0: sens500}
num = {i: [] for i, _ in pairs}
for freq in FREQS:
    sens_cache[freq] = make_sens(freq) if freq != 500.0 else sens500
    for i, dc0 in pairs:
        num[i].append(criterion(sens_cache[freq], dc0))
    print(f"  {freq:5.0f} Hz (M={sens_cache[freq].M}): "
          f"{len(pairs)} 对完成")

# Pekeris 对照（3 对代表：Θ500 低/中/高）
ths_sorted = sorted(pairs, key=lambda t: ths500[t[0]])
sel_idx = [ths_sorted[0][0], ths_sorted[len(ths_sorted) // 2][0],
           ths_sorted[-1][0]]
pek = {i: [theta_pek(f, dict(pairs)[i]) for f in FREQS] for i in sel_idx}

# ---- β 拟合（log-log 线性回归） ----
def fit_beta(freqs, ths):
    x = np.log(freqs)
    y = np.log(np.array(ths))
    b = np.polyfit(x, y, 1)
    return b[0], b[1]


beta_num = {i: fit_beta(FREQS, num[i])[0] for i, _ in pairs}
beta_pek = {i: fit_beta(FREQS, pek[i])[0] for i in sel_idx}
bs = np.array(list(beta_num.values()))
print(f"\nβ 分布（{len(pairs)} 对）：中位 {np.median(bs):+.3f} | "
      f"均 {bs.mean():+.3f} | 范围 [{bs.min():+.3f}, {bs.max():+.3f}] | "
      f"σ={bs.std():.3f}")
for i in sel_idx:
    print(f"  代表对 {i:2d}: 数值 β={beta_num[i]:+.3f} | "
          f"Pekeris β={beta_pek[i]:+.3f}")

# ---- 阈值移植安全性：灰区贴边对（Θ500 ∈ [0.4,1.6]）的 β ----
gray_pairs = {i: beta_num[i] for i, _ in pairs if 0.4 <= ths500[i] <= 1.6}
bg = np.array(list(gray_pairs.values()))
print(f"灰区贴边对（Θ500∈[0.4,1.6]，n={len(gray_pairs)}）："
      f"β 中位 {np.median(bg):+.3f} | 最大 {bg.max():+.3f} | "
      f"β>1 占比 {np.mean(bg > 1.0):.0%}")


out = dict(freqs=FREQS, n_pairs=len(pairs), sel_idx=sel_idx,
           num=num, pek=pek, ths500=ths500,
           beta_num=beta_num, beta_pek=beta_pek,
           gray_beta=gray_pairs,
           beta_stats=dict(median=float(np.median(bs)),
                           mean=float(bs.mean()),
                           lo=float(bs.min()), hi=float(bs.max()),
                           std=float(bs.std())),
           gray_beta_stats=dict(median=float(np.median(bg)),
                                hi=float(bg.max()),
                                frac_gt1=float(np.mean(bg > 1.0))))
with open(os.path.join(WORK, "f1_freq_scaling.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# ---- 图 ----
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2))
ax = axs[0]
colors = [TEAL, NAVY, ORANGE]
for j, i in enumerate(sorted(sel_idx)):
    ax.plot(FREQS, num[i], "o-", color=colors[j], ms=5, lw=1.6,
            label=f"数值 对{i}（β={beta_num[i]:+.2f}）")
    ax.plot(FREQS, pek[i], "s--", color=colors[j], ms=4, lw=1.2,
            alpha=0.7, label=f"Pekeris 对{i}（β={beta_pek[i]:+.2f}）")
f500 = np.array(FREQS) / 500.0
for i in sel_idx:
    ax.plot(FREQS, num[i][-1] * f500, ":", color=RED, lw=1.2, alpha=0.6)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("频率 f (Hz)")
ax.set_ylabel("Θ (rad)")
ax.set_title("Θ(f) 标度：数值 vs Pekeris 解析（红点线=β=1 外推）")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")
ax = axs[1]
for i, _ in pairs:
    in_gray = 0.4 <= ths500[i] <= 1.6
    ax.plot(ths500[i], beta_num[i], "o", ms=6,
            color=TEAL if in_gray else RED, alpha=0.75)
ax.axhline(1.0, color=NAVY, ls="--", lw=1.4, label="β=1（线性标度）")
ax.axvspan(0.4, 1.6, color=TEAL, alpha=0.12,
           label="灰区 Θ500∈[0.4,1.6]")
ax.set_xscale("log")
ax.set_xlabel("Θ(500 Hz) (rad)")
ax.set_ylabel("β（Θ ∝ f^β 拟合）")
ax.set_title(f"β 普查（{len(pairs)} 对）：绿=灰区贴边对"
             f"（β 中位 {np.median(bg):+.2f}）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
fig.suptitle("F1 频率标度律：路线 D 多频验证（SWellEx-96 官方环境）",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG, "f1_freq_scaling.png"), dpi=110)
print("\n[Done] ->", os.path.join(WORK, "f1_freq_scaling.json"))
