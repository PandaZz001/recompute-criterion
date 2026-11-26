# -*- coding: utf-8 -*-
"""W1 第二环境可移植性检验（合成文献波导，2026-11-26 今日第③项）。

Argo/WOA 数据下载通道本机全部不可达（usgodae 索引为 1028 行测试文件、
ifremer 被墙、NCEI WOA 403、中国中心无公开 API），切换为合成第二环境
方案：三类文献典型波导 × 三类扰动 × 4 档缩放，检验 SWellEx-96 标定的
判据（广义型 λ=0.285/p=1.398/ρ_f=0.477 + 阈值 Θ*）在新波导下的
ρ(Θ) 映射可移植性与方向正确性。

波导：
  W1 理想 Pekeris（等声速 1500 m/s，D=100 m，砂底）——解析参照
  W2 浅海强分层（夏季型：混合层+跃层+底层，D=120 m，砂泥底）
  W3 Munk 深海剖面（D=5000 m，SOFAR 轴 1300 m，淤泥底）——深海分支

扰动（合成 δc，三类形状）：表层指数衰减 exp(−z/L)、梯度型（轴深漂移
等效）、均匀偏移。各取 4 档幅度覆盖灰区。

真值：Kraken+Field 全重算 ρ（基/扰 TL 相关）；判据：一阶 Θ。
频率：浅海 500 Hz（主标定），深海 100 Hz（模态数 ~670，阈值按 F1
标度律换算 Θ*(100 Hz)=0.8·(100/500)=0.16 rad）。

输出：work/w1_second_env.json + figures/w1_second_env.png
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
from at_core import (ModResult, write_env_kraken, write_flp,  # noqa: E402
                     run_kraken_field)
from mode_sensitivity import ModeSensitivity, theta_phase  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
W1W = os.path.join(WORK, "w1")
FIG = os.path.join(HERE, "figures")
KEXE = os.environ.get(
    "KRAKEN_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\Kraken\kraken.exe")
FEXE = os.environ.get(
    "KRAKEN_FIELD_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\KrakenField\field.exe")
NAVY, TEAL, ORANGE, RED = "#1F3A5F", "#2A9D8F", "#E86A17", "#C12C2C"
os.makedirs(W1W, exist_ok=True)

LAM, P_EXP, RHO_F = 0.285324, 1.39808, 0.477323
ALPHAS = [0.25, 0.5, 1.0, 2.0]


def munk_ssp(D=5000.0, z_axis=1300.0, dz_axis=0.0):
    """Munk 1974 北太平洋标准剖面（SOFAR 轴可偏移 dz_axis）。"""
    zs = np.arange(0.0, D + 0.5, 1.0)
    eta = 2.0 * (zs - (z_axis + dz_axis)) / z_axis
    c = 1500.0 * (1.0 + 0.00737 * (eta - 1.0 + np.exp(-eta)))
    return zs, c


def shelf_ssp(D=120.0, h_ml=20.0, h_thermo=30.0, c_surf=1530.0,
              c_bot=1480.0):
    """夏季浅海三层：混合层 + 线性跃层 + 底层近等温。"""
    zs = np.arange(0.0, D + 0.5, 0.5)
    c = np.where(zs <= h_ml, c_surf, c_bot)
    in_th = (zs > h_ml) & (zs <= h_ml + h_thermo)
    c[in_th] = c_surf + (c_bot - c_surf) * ((zs[in_th] - h_ml) / h_thermo)
    c[zs > h_ml + h_thermo] = c_bot
    return zs, c


WAVEGUIDES = {
    "W1 Pekeris-100m": dict(freq=500.0, D=100.0, dz=0.5, sd=20.0, rd=50.0,
                            r_lo=1.0, r_hi=20.0, r_eval=25.0, nr=401,
                            cb=1650.0, rho=1.9, alpha=0.6,
                            ssp=lambda: shelf_ssp(100.0, 100.0, 0.0, 1500.0,
                                                  1500.0)),
    "W2 浅海强分层-120m": dict(freq=500.0, D=120.0, dz=0.5, sd=20.0, rd=50.0,
                              r_lo=1.0, r_hi=20.0, r_eval=25.0, nr=401,
                              cb=1600.0, rho=1.7, alpha=0.5,
                              ssp=lambda: shelf_ssp()),
    "W3 Munk深海-5000m": dict(freq=100.0, D=5000.0, dz=1.0, sd=1000.0,
                              rd=1000.0, r_lo=10.0, r_hi=50.0, r_eval=50.0,
                              nr=401, cb=1550.0, rho=1.5, alpha=0.5,
                              ssp=lambda: munk_ssp()),
}


def perturb(zs, kind, amp, L=50.0):
    """三类合成 δc：表层指数 / 梯度型 / 均匀。"""
    if kind == "surface":
        return amp * np.exp(-zs / L)
    if kind == "gradient":
        return amp * (zs / zs.max() - 0.5) * 2.0
    return amp * np.ones_like(zs)


def run_tl(tag, freq, D, ssp_list, sd, rd, rmax, nr, cb, rho, alpha):
    """Kraken+Field → TL（dB）。"""
    stem = f"w1_{tag}"
    for ext in (".env", ".mod", ".flp", ".shd", ".prt"):
        p = os.path.join(W1W, stem + ext)
        if os.path.exists(p):
            os.remove(p)
    zgrid = [float(z) for z, _ in ssp_list]
    write_env_kraken(os.path.join(W1W, stem + ".env"), freq, ssp_list, D,
                     sd, zgrid, rmax, nr, title="W1",
                     bottom=(cb, rho, alpha), c_lim=(1400.0, 1650.0))
    res = subprocess.run([KEXE, stem], cwd=W1W, capture_output=True,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(f"kraken {tag} failed: {res.stdout[-300:]}")
    write_flp(os.path.join(W1W, stem + ".flp"), sd=sd, rd=[rd],
              rmax=rmax, nr=nr)
    shd = run_kraken_field(FEXE, os.path.join(W1W, stem + ".flp"), W1W)
    import struct
    raw = open(os.path.join(W1W, stem + ".shd"), "rb").read()
    rl = struct.unpack("I", raw[:4])[0] * 4
    P = np.frombuffer(raw[10 * rl:], dtype="<c8").copy()[:nr]
    return -20.0 * np.log10(np.abs(P) + 1e-300)


def criterion(S, dc, zs, c_base, r_eval, sd, rd):
    """一阶判据 Θ（相位+形状），SWellEx 同式（基线声速剖面 c_base）。"""
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
    phiv = S._phi_at(rd)
    phis = S._phi_at(sd)
    dr = dph[int(np.argmin(np.abs(S.z - rd)))]
    ds = dph[int(np.argmin(np.abs(S.z - sd)))]
    dw = ds * phiv + phis * dr
    return theta_phase(S, dk, dw=dw, zs=sd, z=rd, r_eval=r_eval)


rows = []
for wname, w in WAVEGUIDES.items():
    zs, c_base = w["ssp"]()
    ssp_base = [(float(z), float(c)) for z, c in zip(zs, c_base)]
    tl0 = run_tl(f"{wname.split()[1]}_base", w["freq"], w["D"], ssp_base,
                 w["sd"], w["rd"], w["r_hi"], w["nr"], w["cb"], w["rho"],
                 w["alpha"])
    rgrid = np.linspace(w["r_lo"], w["r_hi"], w["nr"])
    mask = np.ones_like(rgrid, dtype=bool)
    # 判据模态（基线环境）
    stem = f"w1_{wname.split()[1]}_base"
    mod = ModResult(os.path.join(W1W, stem + ".mod"))
    cbar = float(np.mean(c_base))
    S = ModeSensitivity(mod, w["freq"], cbar, w["D"], w["cb"], w["rho"],
                        w["alpha"])
    for kind in ("surface", "gradient", "uniform"):
        for amp_sign, a0 in (("+", 1.0), ("-", -1.0)):
            dc0 = perturb(zs, kind, a0)
            for al in ALPHAS:
                dc = dc0 * al
                th = criterion(S, dc, zs, c_base, w["r_eval"], w["sd"],
                               w["rd"])
                ssp_p = [(float(z), float(c + d))
                         for z, c, d in zip(zs, c_base, dc)]
                tl = run_tl(f"{wname.split()[1]}_{kind[0]}{'+' if a0 > 0 else 'm'}_{al}",
                            w["freq"], w["D"], ssp_p, w["sd"], w["rd"],
                            w["r_hi"], w["nr"], w["cb"], w["rho"],
                            w["alpha"])
                rho = float(np.corrcoef(tl0[mask], tl[mask])[0, 1])
                rows.append(dict(wg=wname, kind=kind, alpha=al,
                                 sign=int(np.sign(a0)), th=th, rho=rho))
                print(f"  {wname:<18} {kind:<8} a={al:4.2f} "
                      f"Θ={th:7.3f} ρ={rho:.4f}")

# ---- 分析与图 ----
th_star = lambda f500: 0.8 * (f500 / 500.0)
ok_all = 0
by_wg = {}
for wname, w in WAVEGUIDES.items():
    ts = th_star(w["freq"])
    sub = [r for r in rows if r["wg"] == wname]
    ok = sum((r["th"] <= ts) == (r["rho"] >= 0.9) for r in sub)
    fr = sum((r["th"] <= ts) and r["rho"] < 0.9 for r in sub)
    fm = sum((r["th"] > ts) and r["rho"] >= 0.9 for r in sub)
    by_wg[wname] = dict(n=len(sub), ok=ok, fr=fr, fm=fm,
                        th_star=float(ts))
    ok_all += ok
    print(f"{wname}: n={len(sub)} Θ*={ts:.3f} ok={ok} fr={fr} fm={fm}")
print(f"全部：ok={ok_all}/{len(rows)}")

# 理论曲线对照
ths = np.array([r["th"] for r in rows])
rhos = np.array([r["rho"] for r in rows])
rho_gen = RHO_F + (1 - RHO_F) * np.exp(-LAM * ths ** P_EXP)
rmse = float(np.sqrt(np.mean((rhos - rho_gen) ** 2)))
print(f"广义型理论 RMSE = {rmse:.3f}")

out = dict(rows=rows, by_wg=by_wg, rmse=rmse, lam=LAM, p=P_EXP,
           rho_f=RHO_F)
with open(os.path.join(WORK, "w1_second_env.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2))
ax = axs[0]
marks = {"W1 Pekeris-100m": ("o", TEAL), "W2 浅海强分层-120m": ("s", NAVY),
         "W3 Munk深海-5000m": ("^", ORANGE)}
for wname, (mk, col) in marks.items():
    sub = [r for r in rows if r["wg"] == wname]
    ax.plot([r["th"] for r in sub], [r["rho"] for r in sub], mk,
            ms=5, color=col, alpha=0.75, label=wname)
tc = np.linspace(0.02, 8, 200)
ax.plot(tc, RHO_F + (1 - RHO_F) * np.exp(-LAM * tc ** P_EXP),
        color=RED, lw=1.8, label="广义型（SWellEx 标定）")
ax.set_xscale("log")
ax.set_xlabel("Θ (rad)")
ax.set_ylabel("真值 ρ")
ax.set_title("第二环境 ρ(Θ)：三类文献波导 vs SWellEx 理论曲线")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
ax = axs[1]
xs = np.arange(len(by_wg))
for k, (wname, v) in enumerate(by_wg.items()):
    ax.bar(xs[k] - 0.22, v["ok"] / v["n"], 0.22, color=TEAL,
           label="正确率" if k == 0 else None)
    ax.bar(xs[k], v["fr"] / v["n"], 0.22, color=RED,
           label="假复用率" if k == 0 else None)
    ax.bar(xs[k] + 0.22, v["fm"] / v["n"], 0.22, color="#8A8F98",
           label="假重算率" if k == 0 else None)
ax.set_xticks(xs)
ax.set_xticklabels([f"{w.split()[1]}\nΘ*={by_wg[w]['th_star']:.2f}"
                    for w in by_wg], fontsize=9)
ax.set_ylabel("比率")
ax.set_title("阈值方向正确性（ρ_req=0.9）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.suptitle("W1 第二环境可移植性：合成文献波导检验", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG, "w1_second_env.png"), dpi=110)
print("\n[Done] ->", os.path.join(WORK, "w1_second_env.json"))
