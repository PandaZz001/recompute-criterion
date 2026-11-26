# -*- coding: utf-8 -*-
"""W3 深海分支阈值独立标定（2026-11-26，迭代任务）。

W1/W2/W2b 的共同诚实缺口：深海 ρ(Θ) 曲线高于浅海标定，F1 频率换算
阈值 0.16 rad（100 Hz）在深海过度保守（西太 12 样本 4 假重算）。
本实验汇集全部深海样本（W1 Munk 24 + W2 西太/北大西洋 8 + W2b 深海
16 = 48，全部 100 Hz、r_eval=50 km 同口径），自由拟合深海专属
广义型参数（λ/p/ρ_f），反解深海阈值 Θ*_deep，并复验决策。

输出：work/w3_deep_threshold.json + figures/w3_deep_threshold.png
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
FIG = os.path.join(HERE, "figures")
NAVY, TEAL, ORANGE, RED = "#1F3A5F", "#2A9D8F", "#E86A17", "#C12C2C"

LAM, P_EXP, RHO_F = 0.285324, 1.39808, 0.477323  # 浅海标定（E2g）


def load_deep():
    """汇集全部深海 (Θ, ρ) 样本。"""
    pts = []
    with open(os.path.join(WORK, "w1_second_env.json"),
              encoding="utf-8") as f:
        w1 = json.load(f)
    pts += [(r["th"], r["rho"], "W1 Munk")
            for r in w1["rows"] if "Munk" in r["wg"]]
    with open(os.path.join(WORK, "w2_woa_env.json"), encoding="utf-8") as f:
        w2 = json.load(f)
    pts += [(r["th"], r["rho"], f"W2 {r['site']}")
            for r in w2["rows"] if r["D"] >= 300.0]
    with open(os.path.join(WORK, "w2b_seasons.json"),
              encoding="utf-8") as f:
        w2b = json.load(f)
    pts += [(r["th"], r["rho"], f"W2b {r['site']}·{r['season']}")
            for r in w2b["rows"] if r["D"] >= 300.0]
    return pts


def gen_model(th, lam, p, rho_f):
    return rho_f + (1.0 - rho_f) * np.exp(-lam * np.abs(th) ** p)


print("== W3 深海分支阈值独立标定 ==\n")
pts = load_deep()
ths = np.array([p[0] for p in pts])
rhos = np.array([p[1] for p in pts])
print(f"深海样本 n={len(pts)}：Θ 范围 [{ths.min():.3f}, {ths.max():.3f}]，"
      f"ρ 范围 [{rhos.min():.3f}, {rhos.max():.3f}]")

# ---- 深海广义型自由拟合 ----
p0 = [LAM, P_EXP, RHO_F]
popts, _ = curve_fit(gen_model, ths, rhos, p0=p0,
                     bounds=([0.001, 0.5, 0.0], [5.0, 3.0, 0.98]),
                     maxfev=20000)
lam_d, p_d, rf_d = popts
rho_pred = gen_model(ths, *popts)
rmse = float(np.sqrt(np.mean((rhos - rho_pred) ** 2)))
rmse_shallow = float(np.sqrt(np.mean(
    (rhos - gen_model(ths, LAM, P_EXP, RHO_F)) ** 2)))
print(f"深海拟合：λ={lam_d:.3f}  p={p_d:.3f}  ρ_f={rf_d:.3f}  "
      f"RMSE={rmse:.3f}")
print(f"对照（浅海标定参数在深海样本上）：RMSE={rmse_shallow:.3f}")

# ---- 反解深海阈值 ----
th_deep = ((-np.log((0.9 - rf_d) / (1.0 - rf_d))) / lam_d) ** (1.0 / p_d)
print(f"\nΘ*_deep(ρ_req=0.9) = {th_deep:.3f} rad（100 Hz）")
print(f"对照：F1 频率换算 0.16 | 浅海 500 Hz 标定 0.8")

# ---- 决策复验 ----
def conf(th_star):
    ok = sum(1 for t, r, _ in pts
             if bool(t <= th_star) == bool(r >= 0.9))
    fr = sum(1 for t, r, _ in pts
             if bool(t <= th_star) and not bool(r >= 0.9))
    fm = sum(1 for t, r, _ in pts
             if not bool(t <= th_star) and bool(r >= 0.9))
    return ok, fr, fm


for label, ts in [("F1 换算 0.16", 0.16), ("深海标定 Θ*_deep", th_deep),
                  ("浅海参数反解", ((-np.log((0.9 - RHO_F)
                                    / (1 - RHO_F))) / LAM)
                   ** (1 / P_EXP))]:
    ok, fr, fm = conf(ts)
    print(f"  {label:<18} Θ*={ts:6.3f}  ok={ok:2d}/{len(pts)}  fr={fr}  "
          f"fm={fm}")

# 安全系数扫描：Θ* = c·Θ*_deep
print("\n安全系数扫描（Θ* = c·Θ*_deep）：")
sweep = {}
for c in [0.2, 0.3, 0.5, 0.7, 1.0]:
    ts = c * th_deep
    ok, fr, fm = conf(ts)
    sweep[str(c)] = dict(ts=float(ts), ok=ok, fr=fr, fm=fm)
    print(f"  c={c:.1f}  Θ*={ts:6.3f}  ok={ok:2d}/{len(pts)}  fr={fr}  "
          f"fm={fm}")

out = dict(n=len(pts), lam=float(lam_d), p=float(p_d),
           rho_f=float(rf_d), rmse=rmse, rmse_shallow=rmse_shallow,
           th_star_deep=float(th_deep),
           conf_016=dict(zip(("ok", "fr", "fm"), conf(0.16))),
           conf_deep=dict(zip(("ok", "fr", "fm"), conf(th_deep))),
           sweep=sweep,
           pts=[dict(th=p[0], rho=p[1], src=p[2]) for p in pts])
with open(os.path.join(WORK, "w3_deep_threshold.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# ---- 图 ----
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2))
ax = axs[0]
src_colors = {"W1 Munk": ORANGE, "W2 西太": RED, "W2 北大西洋": NAVY,
              "W2b 西太·春−冬": RED, "W2b 西太·秋−冬": RED,
              "W2b 北大西洋·春−冬": NAVY, "W2b 北大西洋·秋−冬": NAVY}
for t, r, src in pts:
    ax.plot(t, r, "o", ms=5, color=src_colors.get(src, GRAY := "#8A8F98"),
            alpha=0.65)
tc = np.linspace(0.02, max(ths.max(), 8), 200)
ax.plot(tc, gen_model(tc, LAM, P_EXP, RHO_F), color=GRAY, lw=1.6, ls="--",
        label=f"浅海标定（λ={LAM:.2f}/p={P_EXP:.2f}/ρf={RHO_F:.2f}）")
ax.plot(tc, gen_model(tc, *popts), color=TEAL, lw=2,
        label=f"深海拟合（λ={lam_d:.2f}/p={p_d:.2f}/ρf={rf_d:.2f}）")
ax.axvline(0.16, color=RED, ls=":", lw=1.2, label="F1 换算 Θ*=0.16")
ax.axvline(th_deep, color=TEAL, ls=":", lw=1.4,
           label=f"深海标定 Θ*={th_deep:.2f}")
ax.axhline(0.9, color="#8A8F98", ls="-.", lw=1, label="ρ*=0.9")
ax.set_xscale("log")
ax.set_xlabel("Θ (rad)")
ax.set_ylabel("真值 ρ")
ax.set_title("深海 48 样本 ρ(Θ)：浅海标定 vs 深海专属拟合")
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3, which="both")
ax = axs[1]
cs = [0.2, 0.3, 0.5, 0.7, 1.0]
xs = np.arange(len(cs))
for k, c in enumerate(cs):
    v = sweep[str(c)]
    n = len(pts)
    ax.bar(xs[k] - 0.2, v["ok"] / n, 0.2, color=TEAL,
           label="正确率" if not k else None)
    ax.bar(xs[k], v["fr"] / n, 0.2, color=RED,
           label="假复用率" if not k else None)
    ax.bar(xs[k] + 0.2, v["fm"] / n, 0.2, color="#8A8F98",
           label="假重算率" if not k else None)
ax.axhline(0, color=RED, ls=":", lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels([f"c={c:.1f}\nΘ*={c * th_deep:.2f}" for c in cs],
                   fontsize=9)
ax.set_ylabel("比率")
ax.set_title(f"安全系数扫描（n={len(pts)}）：c 越大正确率越高但假复用出现")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.suptitle("W3 深海分支阈值独立标定：把保守缺口变成标定结果", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG, "w3_deep_threshold.png"), dpi=110)
print("\n[Done] ->", os.path.join(WORK, "w3_deep_threshold.json"))
