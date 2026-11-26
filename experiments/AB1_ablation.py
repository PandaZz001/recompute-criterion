# -*- coding: utf-8 -*-
"""AB1 判据构件消融实验（2026-11-26，今日接续第①项）。

在 SW3e2 缩放灰区族（29 对实测 CTD × 9 缩放 = 261 样本）上逐一移除
判据构件，量化各构件对决策性能的贡献：

- 特征层：Θ_full（相位+形状）→ 去形状（Θ_ph）/ 去相位（Θ_sh）
- 映射层：广义型 exp(−λΘ^p)（λ=0.285/p=1.40）→ 高斯型 p=2 / 指数型 p=1
- 地板层：ρ_f 有无
- 判据家族：一维广义型 vs 二维椭圆模型（推导/拟合两组参数）

统一决策目标：ρ̂ ≥ ρ_req ⇔ 复用（ρ_req 默认 0.9，横扫做等假复用比较）。
真值：Kraken 全重算 ρ（work/sw3e2_gray_zone.json 既有结果，不重跑）。

分量来源：Θ 是 δc 的线性泛函，29 对原始 δc0 各算一次 Θ_ph/Θ_full 后
按 α 线性缩放（自检：重构 th vs json 原 th 相对误差 < 1e-9）。

输出：work/ab1_ablation.json + figures/ab1_ablation.png
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
from at_core import ModResult  # noqa: E402
from sw3_common import parse_official_env  # noqa: E402
from mode_sensitivity import ModeSensitivity, theta_phase  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
SW3W = os.path.join(WORK, "sw3")
FIG = os.path.join(HERE, "figures")
NAVY, TEAL, ORANGE, RED, GRAY = "#1F3A5F", "#2A9D8F", "#E86A17", "#C12C2C", "#8A8F98"

CTD_DIR = os.environ.get(
    "SWELLEX96_CTD_DIR",
    r"D:\0_Research\SWellEx96\env\ctds\ctds")
FREQ, SD, RD0 = 500.0, 20.0, 50.0
ALPHAS = [0.03, 0.06, 0.10, 0.15, 0.22, 0.32, 0.45, 0.65, 1.0]
GRAY_LO, GRAY_HI = 0.4, 1.6
RHO_REQ_MAIN = 0.9

# ---- 复用 SW3e2 环境装配（保证同口径） ----
ssp_o, layers, half = parse_official_env()
ssp_o = [(float(z), float(c)) for z, c in ssp_o]
c_off = np.array([c for _, c in ssp_o])
CW = float(np.mean(c_off))
D_J = 212.5
zgrid = np.arange(0.0, D_J + 0.5, 0.5)
c_uni = np.interp(zgrid, [z for z, _ in ssp_o], [c for _, c in ssp_o])

mod_j = ModResult(os.path.join(SW3W, "sw3e2_j.mod"))
S = ModeSensitivity(mod_j, FREQ, CW, D_J, 1572.4, 1.76, 0.2)
cz_grid = np.interp(S.z, zgrid, c_uni)
print(f"== AB1 判据构件消融（基线模态 M={mod_j.M}）==\n")


def dk_of(dc):
    """一阶模态波数微扰 δk_m（与 SW3e2 同式）。"""
    dc = np.interp(S.z, zgrid, dc)
    integ = np.trapezoid(dc[:, None] * S.phi.real ** 2
                         / cz_grid[:, None] ** 3, S.z, axis=0)
    return -(S.w ** 2 / S.k) * integ


def dw_of(dc):
    """形状微扰 dw（模态耦合矩阵），与 SW3e2 同式。"""
    dc = np.interp(S.z, zgrid, dc)
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
    return ds * phiv + phis * dr


def th_full_of(dc):
    """Θ 全量（相位 + 形状），即 SW3e2 criterion。"""
    return theta_phase(S, dk_of(dc), dw=dw_of(dc))


def th_ph_of(dc):
    """Θ 相位分量（仅模态波数对差）。"""
    return theta_phase(S, dk_of(dc))


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


# ---- 29 对原始 δc0 的分量计算（线性缩放源） ----
files = sorted(f for f in os.listdir(CTD_DIR) if f.endswith(".prn"))
casts = {fn: read_ctd(os.path.join(CTD_DIR, fn)) for fn in files}
pair_comp = {}  # i -> (th_full0, th_ph0)
n_neg_sh = 0
for i in range(len(files) - 1):
    ca = interp_to(*casts[files[i]])
    cb = interp_to(*casts[files[i + 1]])
    valid = (~np.isnan(ca)) & (~np.isnan(cb))
    if valid.sum() < 200:
        continue
    ca = np.where(np.isnan(ca), CW, ca)
    cb = np.where(np.isnan(cb), CW, cb)
    dc0 = np.where(valid, cb - ca, 0.0)
    tf = th_full_of(dc0)
    tp = th_ph_of(dc0)
    pair_comp[i] = (tf, tp)
    if tf - tp < 0:
        n_neg_sh += 1

# ---- 读 261 样本真值，自检线性性 ----
with open(os.path.join(WORK, "sw3e2_gray_zone.json"), encoding="utf-8") as f:
    rows = json.load(f)["rows"]
max_rel = 0.0
for r in rows:
    tf, tp = pair_comp[r["i"]]
    r["th_ph"] = float(tp * r["alpha"])
    r["th_full"] = float(tf * r["alpha"])
    r["th_sh"] = r["th_full"] - r["th_ph"]
    err = abs(r["th_full"] - r["th"]) / max(r["th"], 1e-12)
    max_rel = max(max_rel, err)
print(f"1. 分量重算：{len(pair_comp)} 对 | Θ_sh 负值对 {n_neg_sh} "
      f"| 线性缩放重构相对误差 {max_rel:.2e}")
n_sh_neg = sum(1 for r in rows if r["th_sh"] < 0)
print(f"   261 样本中 Θ_sh < 0 占比 {n_sh_neg}/261")

# ---- 消融变体定义 ----
# kind="gen"：ρ̂ = ρ_f + (1−ρ_f)exp(−λ·Θ^p)；kind="ell"：二维椭圆
VARIANTS = {
    "M0 广义型+地板（E2g 拟合）": dict(kind="gen", lam=0.285324, p=1.39808,
                                    rho_f=0.477323, feat="full"),
    "M0b 广义型+地板（椭圆ρf）": dict(kind="gen", lam=0.285324, p=1.39808,
                                     rho_f=0.431261, feat="full"),
    "M1 去地板（ρf=0）": dict(kind="gen", lam=0.285324, p=1.39808,
                            rho_f=0.0, feat="full"),
    "M2 高斯型（p=2）": dict(kind="gen", lam=0.750491, p=2.0,
                           rho_f=0.538349, feat="full"),
    "M3 指数型（p=1）": dict(kind="gen", lam=0.277431, p=1.0,
                           rho_f=0.414440, feat="full"),
    "M4 去形状项（Θph）": dict(kind="gen", lam=0.285324, p=1.39808,
                             rho_f=0.477323, feat="ph"),
    "M5 去相位项（Θsh）": dict(kind="gen", lam=0.285324, p=1.39808,
                             rho_f=0.477323, feat="sh"),
    "M6 椭圆推导（0.518/1.45）": dict(kind="ell", c_ph=0.518, c_sh=1.45,
                                     rho_f=0.431261),
    "M7 椭圆拟合（1.031/1.389）": dict(kind="ell", c_ph=1.031243,
                                      c_sh=1.388714, rho_f=0.431261),
}


def th_star_gen(v, rho_req):
    """一维广义族：ρ̂ ≥ ρ_req 反解阈值 Θ*。"""
    if rho_req <= v["rho_f"]:
        return np.inf
    return ((-np.log((rho_req - v["rho_f"]) / (1.0 - v["rho_f"]))
             / v["lam"]) ** (1.0 / v["p"]))


def dec_ell(v, r, rho_req):
    """椭圆族：域内 ⇔ 复用。"""
    lim = -np.log((rho_req - v["rho_f"]) / (1.0 - v["rho_f"]))
    return (v["c_ph"] * r["th_ph"]) ** 2 / 2.0 \
        + (v["c_sh"] * r["th_sh"]) ** 2 <= lim


def confusion(v, rows_sel, rho_req):
    """统一口径混淆矩阵：dec = ρ̂≥ρ_req，truth = ρ_true≥ρ_req。"""
    ok = fr = fm = 0
    for r in rows_sel:
        if v["kind"] == "gen":
            th = r["th_" + v["feat"]]
            if v["feat"] == "sh":
                th = max(th, 0.0)  # 截断负增量（E2a 口径）
            dec = bool(th <= th_star_gen(v, rho_req))
        else:
            dec = bool(dec_ell(v, r, rho_req))
        truth = bool(r["rho"] >= rho_req)
        ok += dec == truth
        fr += dec and not truth
        fm += not dec and truth
    return ok, fr, fm


gray_rows = [r for r in rows if GRAY_LO <= r["th"] <= GRAY_HI]
print(f"\n2. 灰区子集（Θ∈[{GRAY_LO},{GRAY_HI}]）：n={len(gray_rows)}")
print("\n3. 主口径 ρ_req=0.9 混淆矩阵：")
print(f"   {'变体':<26} {'全样本 ok/fr/fm':>22} {'灰区 ok/fr/fm':>20}")
main_tab = {}
for name, v in VARIANTS.items():
    ok, fr, fm = confusion(v, rows, RHO_REQ_MAIN)
    okg, frg, fmg = confusion(v, gray_rows, RHO_REQ_MAIN)
    main_tab[name] = dict(ok=ok, fr=fr, fm=fm,
                          gray=dict(ok=okg, fr=frg, fm=fmg))
    print(f"   {name:<26} {ok:>4}/{fr:>3}/{fm:>3} ({len(rows)})"
          f"      {okg:>4}/{frg:>3}/{fmg:>3} ({len(gray_rows)})")

# ---- 等假复用横扫：ρ_req ∈ [0.70, 0.99] ----
rho_sweep = np.arange(0.70, 0.995, 0.01)
sweep, sweep_gray = {}, {}
for name, v in VARIANTS.items():
    pts, pts_g = [], []
    for rr in rho_sweep:
        ok, fr, fm = confusion(v, rows, float(rr))
        pts.append((fr / len(rows), ok / len(rows)))
        okg, frg, fmg = confusion(v, gray_rows, float(rr))
        pts_g.append((frg / len(gray_rows), okg / len(gray_rows)))
    sweep[name] = dict(fr=[p[0] for p in pts], ok=[p[1] for p in pts])
    sweep_gray[name] = dict(fr=[p[0] for p in pts_g], ok=[p[1] for p in pts_g])


def ok_at_fr(curve, fr_target):
    """曲线插值：假复用率 fr_target 处的正确率（按 fr 排序后插值）。"""
    order = np.argsort(curve["fr"])
    frs = np.array(curve["fr"])[order]
    oks = np.array(curve["ok"])[order]
    return float(np.interp(fr_target, frs, oks))


print("\n4. 等假复用正确率（全样本 fr=0.05 / 灰区 fr=0.05 插值）：")
print(f"   {'变体':<26} {'全样本':>8} {'灰区':>8}")
eqfr = {}
for name in VARIANTS:
    o1 = ok_at_fr(sweep[name], 0.05)
    o2 = ok_at_fr(sweep_gray[name], 0.05)
    eqfr[name] = [o1, o2]
    print(f"   {name:<26} {o1:>7.3f} {o2:>7.3f}")

# ---- 输出 ----
out = dict(linearity_max_rel_err=max_rel, n_pairs=len(pair_comp),
           n_sh_neg=n_sh_neg, n=len(rows), n_gray=len(gray_rows),
           rho_req=RHO_REQ_MAIN, main=main_tab,
           sweep=sweep, sweep_gray=sweep_gray, eq_fr_005=eqfr,
           variants={k: v for k, v in VARIANTS.items()})
with open(os.path.join(WORK, "ab1_ablation.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# ---- 图 ----
fig, axs = plt.subplots(1, 3, figsize=(17.5, 5.2))
names = list(VARIANTS)
x = np.arange(len(names))
ax = axs[0]
okv = [main_tab[n]["gray"]["ok"] / len(gray_rows) for n in names]
frv = [main_tab[n]["gray"]["fr"] / len(gray_rows) for n in names]
fmv = [main_tab[n]["gray"]["fm"] / len(gray_rows) for n in names]
ax.bar(x - 0.25, okv, 0.25, color=TEAL, label="正确率")
ax.bar(x, frv, 0.25, color=RED, label="假复用率")
ax.bar(x + 0.25, fmv, 0.25, color=GRAY, label="假重算率")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=50, ha="right", fontsize=7.5)
ax.set_ylabel("比率")
ax.set_title(f"灰区（n={len(gray_rows)}）ρ_req=0.9 消融混淆矩阵")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
ax = axs[1]
for name in names:
    if name.startswith("M0 "):
        ax.plot(sweep[name]["fr"], sweep[name]["ok"], "o-", ms=4, lw=2,
                color=NAVY, label=name)
    else:
        ax.plot(sweep[name]["fr"], sweep[name]["ok"], "o-", ms=3, lw=1,
                alpha=0.75, label=name)
ax.axvline(0.05, color=RED, ls=":", lw=1.2)
ax.set_xlabel("假复用率")
ax.set_ylabel("正确率")
ax.set_title("全样本（n=261）等假复用曲线")
ax.legend(fontsize=6.5, ncol=2)
ax.grid(alpha=0.3)
ax = axs[2]
for name in names:
    if name.startswith("M0 "):
        ax.plot(sweep_gray[name]["fr"], sweep_gray[name]["ok"], "o-", ms=4,
                lw=2, color=NAVY, label=name)
    else:
        ax.plot(sweep_gray[name]["fr"], sweep_gray[name]["ok"], "o-", ms=3,
                lw=1, alpha=0.75, label=name)
ax.axvline(0.05, color=RED, ls=":", lw=1.2)
ax.set_xlabel("假复用率")
ax.set_ylabel("正确率")
ax.set_title(f"灰区（n={len(gray_rows)}）等假复用曲线")
ax.legend(fontsize=6.5, ncol=2)
ax.grid(alpha=0.3)
fig.suptitle("AB1 判据构件消融：各构件对决策性能的边际贡献", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG, "ab1_ablation.png"), dpi=110)
print("\n[Done] ->", os.path.join(WORK, "ab1_ablation.json"))
