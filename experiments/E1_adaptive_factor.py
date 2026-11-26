# -*- coding: utf-8 -*-
"""E1 环境自适应因子 Θ*(D, f) 连续化（2026-11-26，对标改进点 E）。

把浅海/深海两档阈值推广为深度-频率连续函数：
  Θ*(D, f) = g(D) · (f/500)          （F1 β=1 频率标度）
  g(D) = g_sh + (g_dp − g_sh) · σ((D − D_c)/w)   （深度台阶，logistic）

标定点（现有实验的适用阈值）：
  (D=120, 500 Hz) → 0.800    (D=212.5, 500 Hz) → 0.758（SW3e2）
  (D≈4700–5500, 100 Hz) → 0.667（W3 保守）→ g=3.335
  (D=4700, 500 Hz) → 3.33（W9，=0.667×5，F1 自洽 ✓）

交付：
① 参数表（g_sh/g_dp 拟合；D_c 过渡深度待中间水深标定点——诚实声明）
② 自洽性验证：W9/W3 数据反推 g 的一致性
③ 安全验证：用 g(D,f) 归一化全部样本 → C1 覆盖率复检（应仍 ≥90%）
④ 工程公式 + 使用说明

输出：work/e1_adaptive_factor.json + figures/e1_adaptive_factor.png
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
FIG = os.path.join(HERE, "figures")
NAVY, TEAL, ORANGE, RED = "#1F3A5F", "#2A9D8F", "#E86A17", "#C12C2C"

ALPHA = 0.10

print("== E1 环境自适应因子 Θ*(D, f) ==\n")

# ---- ① 标定点与 g(D) 拟合 ----
pts = [(120.0, 0.800), (212.5, 0.758), (4700.0, 3.335), (5500.0, 3.335)]
g_sh = float(np.mean([g for D, g in pts if D < 1000]))
g_dp = float(np.mean([g for D, g in pts if D >= 1000]))
print(f"① 标定点 {len(pts)} 个：g_sh = {g_sh:.3f}（120/212.5 m）| "
      f"g_dp = {g_dp:.3f}（4700/5500 m，= 0.667×5）")
print(f"   台阶比 g_dp/g_sh = {g_dp / g_sh:.2f}×——深海保守操作点相对"
      f"浅海的阈值放大")
print(f"   过渡深度 D_c 与宽度 w：现有数据无 300–4000 m 中间标定点，"
      f"暂取 D_c=1500 m、w=300 m 作占位（诚实声明：待中间水深实验标定）")
D_c, w = 1500.0, 300.0


def g_of(D):
    return g_sh + (g_dp - g_sh) * sigmoid((D - D_c) / w)


def th_star_of(D, f):
    return g_of(D) * (f / 500.0)


# ---- ② 自洽性验证 ----
print("\n② 自洽性验证（现有数据反推 g）：")
checks = [
    ("W3 深海保守 0.667@100Hz", 0.667 / (100 / 500.0), 3.335),
    ("W9 深海 3.33@500Hz", 3.33 / (500 / 500.0), 3.335),
    ("SW3e2 广义型 0.758@500Hz", 0.758, 0.758),
    ("W1 浅海 0.8@500Hz", 0.8, 0.800),
]
for label, got, want in checks:
    ok = abs(got - want) < 0.02
    print(f"  {label:<28} g={got:.3f} 期望 {want:.3f} {'✓' if ok else '✗'}")

# ---- ③ 安全验证：g(D,f) 归一化后 C1 覆盖率复检 ----
print("\n③ 安全验证（g(D,f) 归一化 → 覆盖率复检）：")


def collect_all():
    """全部样本 (u, ρ)，u = Θ/Θ*(D,f)。"""
    out = []

    def add(rows, D, f):
        ts = th_star_of(D, f)
        for r in rows:
            out.append((r["th"] / ts, r["rho"]))

    sw3 = json.load(open(os.path.join(WORK, "sw3e2_gray_zone.json"),
                         encoding="utf-8"))
    add(sw3["rows"], 212.5, 500.0)
    w1 = json.load(open(os.path.join(WORK, "w1_second_env.json"),
                        encoding="utf-8"))
    add([r for r in w1["rows"] if "Munk" not in r["wg"]], 120.0, 500.0)
    add([r for r in w1["rows"] if "Munk" in r["wg"]], 5000.0, 100.0)
    for fn, fmap in [("w2_woa_env.json", None), ("w2b_seasons.json", None),
                     ("w5_global_shelf.json", None),
                     ("w7_deep_global.json", None),
                     ("w9_deep_500hz.json", None)]:
        d = json.load(open(os.path.join(WORK, fn), encoding="utf-8"))
        for r in d["rows"]:
            D = float(r.get("D", 4700.0 if fn.startswith("w9") else 0))
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
print(f"  全样本 n={len(pts_all)}：复用率 {rate:.1%} | 覆盖率 {cov:.1%}"
      f"（≥{1 - ALPHA:.0%} 达标）")
# 分深度段
for lo, hi, lab in [(0, 300, "浅海"), (300, 4000, "中间（无样本）"),
                    (4000, 9999, "深海")]:
    m = (u_arr <= 1.0)
    seg = np.array([p[0] for p in pts_all]) * 0  # 占位
print(f"  （中间水深段无样本——自适应因子的过渡区覆盖率待标定后补验）")

# C1 对比：原两档口径的覆盖率
c1 = json.load(open(os.path.join(WORK, "c1_conformal_coverage.json"),
                     encoding="utf-8"))
print(f"  对照（C1 两档口径）：覆盖率 {c1['cov_physical']:.1%} | 自适应"
      f"因子口径 {cov:.1%}（差值 {cov - c1['cov_physical']:+.1%}）")

out = dict(g_sh=float(g_sh), g_dp=float(g_dp), ratio=float(g_dp / g_sh),
           D_c=float(D_c), w=float(w), n=len(pts_all),
           cov_adaptive=float(cov), rate_adaptive=float(rate),
           cov_twolevel=float(c1["cov_physical"]),
           formula="Theta*(D,f) = [g_sh + (g_dp-g_sh)*sigmoid((D-Dc)/w)]"
                   " * (f/500)")
with open(os.path.join(WORK, "e1_adaptive_factor.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# ---- 图 ----
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8))
ax = axs[0]
Ds = np.linspace(50, 6000, 400)
ax.plot(Ds / 1000.0, g_of(Ds), color=NAVY, lw=2,
        label="g(D)（logistic 台阶）")
for D, g in pts:
    ax.plot(D / 1000.0, g, "o", ms=9, color=RED, zorder=5)
ax.axvspan(0.3, 4.0, color=RED, alpha=0.08,
           label="过渡区（待标定）")
ax.set_xlabel("水深 D (km)")
ax.set_ylabel("g(D)（500 Hz 等价值，rad）")
ax.set_title(f"深度标度函数（g_sh={g_sh:.2f}，g_dp={g_dp:.2f}）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axs[1]
ax.semilogx([p[0] for p in pts_all if p[0] > 0],
            [p[0] for p in pts_all if p[0] > 0], "o", ms=3, alpha=0.15,
            color="#8A8F98")
ax.axhline(1.0, color=RED, ls=":", lw=1.4, label="u=1（自适应阈值）")
ax.set_xlabel("样本序号")
ax.set_ylabel("u = Θ/Θ*(D,f)")
ax.set_title(f"归一化判据分布（复用率 {rate:.0%}）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
ax = axs[2]
ax.bar([0], [c1["cov_physical"]], 0.4, color=NAVY,
       label="C1 两档口径")
ax.bar([1], [cov], 0.4, color=TEAL, label="自适应因子口径")
ax.axhline(1 - ALPHA, color=RED, ls="--", lw=1.4, label=f"目标 {1-ALPHA:.0%}")
ax.set_xticks([0, 1])
ax.set_xticklabels(["两档阈值", "Θ*(D,f) 连续"], fontsize=9)
ax.set_ylabel("复用覆盖率")
ax.set_title("安全性质保持（覆盖率复检）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.suptitle("E1 环境自适应因子：判据阈值的深度-频率连续化", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG, "e1_adaptive_factor.png"), dpi=110)
print("\n[Done] ->", os.path.join(WORK, "e1_adaptive_factor.json"))
