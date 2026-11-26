# -*- coding: utf-8 -*-
"""B4 路线 B 第四步：η 闭合前置——耦合模态官方真值交叉验证（2026-11-26）。

背景：B3"斜率口径"估计 resonance 场景 Θ_cpl(η=1)=2.2e-4 rad，需 η≈6000
才能锚定 ρ=0.819（广义型反推 Θ≈1.33 rad）。本实验不拟合 η，把缺口逐段
钉死，三部分：

  A. 真值稳定性：resonance 场景 Bellhop ρ=0.819 对 bty 采样 / run type /
     波束数的敏感性（4 变体）——真值本身硬不硬；
  B. 官方耦合模态：kraken 多剖面 env（51 剖面 × 500 m，D(r)=120+
     sin(2πr/λ_beat)）+ field.exe OPT='C'（模态投影耦合）/ 'A'
     （adiabatic）/ 平坦 'R' → ρ_cpl / ρ_adia 与 Bellhop 对拍；
  C. 口径逐段定位：①界面投影系数 S_mn=∫φ_m(z;D)φ_n(z;D+1)dz 数值直测
     （b3_d119/120/121 三组 .mod）vs PR 解析 2mn(−1)^(m+n)/(D(m²−n²))；
     ②拍频带 [kb_lo,kb_hi] 是否把共振对 κ_43 滤掉（带通边界嫌疑）；
     ③逐模态边界因子分布（B3"0.25 归一化"误诊澄清）。

输出：work/b4_cm_truth.json + figures/b4_cm_truth.png
"""

import json
import os
import struct
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "criterion"))
from at_core import ModResult, write_env_kraken  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
FIG = os.path.join(HERE, "figures")
KEXE = os.environ.get(
    "KRAKEN_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\Kraken\kraken.exe")
FEXE = os.environ.get(
    "KRAKEN_FIELD_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\KrakenField\field.exe")
BEXE = os.environ.get(
    "BELLHOP_EXE",
    r"D:\0_Research\toolbox\Acoustics-Toolbox\Bellhop\bellhop.exe")
NAVY, TEAL, ORANGE, RED, GRAY = ("#1F3A5F", "#2A9D8F", "#E86A17",
                                 "#C12C2C", "#8A8F98")

FREQ, D_W, SD, RD = 500.0, 120.0, 20.0, 50.0
CB, RHOB, ALPHAB = 1600.0, 1.8, 0.8
RMAX = 25.0
LAM_BEAT = 5703.005684190558     # b1 json lam_beat_main（主对拍频波长, m）


# ================= A. Bellhop 真值稳定性 =================
def bty_pts_resonance(sample_frac):
    lam_km = LAM_BEAT / 1000.0
    pts = [(0.0, 120.0)]
    for r in np.arange(lam_km / sample_frac, RMAX, lam_km / sample_frac):
        pts.append((float(r), float(120.0 + np.sin(2 * np.pi * r / lam_km))))
    pts.append((25.0, 120.0))
    return pts


def write_env_bty(path, tag, rt, nb):
    lines = [
        f"'{tag}'", "500.0          ! FREQ (Hz)", "1              ! NMEDIA",
        "'CVW'          ! SSPOPT", "3  0.0  130.0   ! DEPTH of bottom (m)",
        "    0.0  1500.00  /", "  120.0  1500.00  /", "  130.0  1500.00  /",
        "'A~' 0.0", " 130.0  1600.00 0.0 1.8 0.8 /",
        "1              ! NSD", "  20.0 /       ! SD",
        "1              ! NRD", "  50.0 /       ! RD",
        "{NR}           ! NR", "  0.0  25.0 /  ! R (km)",
        "'{RT}'           ! run type", "{NB}            ! NBEAMS",
        " -89.0 89.0 /  ! ALPHA1,2", "0.0  140.0  26.0   ! STEP, ZBOX, RBOX",
    ]
    text = "\n".join(lines).format(NR=1251, RT=rt, NB=nb)
    with open(path, "w", encoding="ascii") as f:
        f.write(text + "\n")


def write_bty(path, pts):
    with open(path, "w", encoding="ascii") as f:
        f.write("'L'\n%d\n" % len(pts))
        for r, d in pts:
            f.write(f" {r:g} {d:g}\n")


def run_bellhop(tag, pts, rt, nb):
    env_p = os.path.join(WORK, tag + ".env")
    write_env_bty(env_p, tag, rt, nb)
    write_bty(os.path.join(WORK, tag + ".bty"), pts)
    stem = tag
    res = subprocess.run([BEXE, stem], cwd=WORK, capture_output=True,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stdout[-300:])
    from at_core import ShdResult
    shd = ShdResult(os.path.join(WORK, tag + ".shd"))
    return shd.rr, shd.tl()[0]


def rho_tl(rr, tl_flat, tl):
    mask = (rr >= 1.0) & (rr <= 24.0)
    return float(np.corrcoef(tl_flat[mask], tl[mask])[0, 1])


print("== B4 耦合模态官方真值交叉验证 ==\n")
print("-- A. Bellhop 真值稳定性（resonance λ=%.1f m，±1 m）--" % LAM_BEAT)
rr_b, tl_flat = run_bellhop("b4_flat", [(0, 120), (25, 120)], "CG", 501)
bell_variants = {}
for tag, sfrac, rt, nb in [
        ("b4_v1_q4", 4, "CG", 501),     # B1 原口径复现
        ("b4_v2_q8", 8, "CG", 501),     # 加密 bty 采样
        ("b4_v3_ray", 4, "C", 501),     # 纯射线（无高斯束）
        ("b4_v4_nb", 4, "CG", 2001)]:   # 波束加密
    rr, tl = run_bellhop(tag, bty_pts_resonance(sfrac), rt, nb)
    rho = rho_tl(rr, tl_flat, tl)
    bell_variants[tag] = dict(sampling=sfrac, run_type=rt, nbeams=nb,
                              rho=rho)
    print(f"   {tag:<12} 采样 λ/{sfrac:<3} run={rt:<3} beams={nb:<5} "
          f"ρ={rho:.4f}")

# ================= B. 官方耦合模态（kraken 多剖面 + field C/A） ======
NPROF, DR = 51, 500.0
r_prof = np.arange(NPROF) * DR
d_prof = 120.0 + np.sin(2 * np.pi * r_prof / LAM_BEAT)


def env_block(idx, D):
    # 模态深度网格由 RD 列表决定（SW3 踩坑）：必须写密集 RD 且
    # 首点 0、末点恰为 D（耦合投影检查 z(1)=depthT, z(NR)=depthB）。
    rd = np.linspace(0.0, D, 241)
    rd_s = " ".join(f"{v:.4f}" for v in rd)
    return [
        f"'B4 profile {idx}'",
        "500.0          ! FREQ (Hz)",
        "1              ! NMEDIA",
        "'NVW'          ! SSPOPT",
        f"0  0.0  {D:.4f}   ! NSSP sigma DEPTH",
        "    0.0  1500.00  0.0  1.0  0.0  0.0",
        f"  {D:.4f}  1500.00 /",
        "'A' 0.0",
        f" {D:.4f}  1600.00 0.0 1.80 0.80 /",
        "1400  1650     ! cLow cHigh (m/s)",
        "25.0        ! RMAX (km)",
        "1              ! NSD",
        "  0.0 /      ! SD(1:NSD) (m): SD=0 keeps NzTab constant "
        "across profiles; true source depth set in .flp Sz",
        "241            ! NRD",
        f"  {rd_s} /   ! RD(1:NRD) (m)",
    ]


def write_flp_nprof(path, tag, opt, nprof, rprof):
    rprof_s = " ".join(f"{r:.2f}" for r in rprof)
    lines = [
        f"{tag},             ! TITLE",
        f"{opt}             ! OPT (R=point source, 2nd char: C/A)",
        "33               ! M (modes at source)",
        f"{nprof}              ! NProf",
        f"     {rprof_s} /   ! RProf(1:NProf) (km)",
        "501             ! NRr",
        " 0.0 25.0 /   ! Rr(1:NRr) (km)",
        "1              ! NSz",
        "   20.0 /    ! Sz(1:NSz) (m)",
        "1              ! NRz",
        "   50.0 /    ! Rz(1:NRz) (m)",
        "1              ! NRo (must equal NRz)",
        "     0.0 /   ! Ro(1:NRo) (m)",
    ]
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")


def read_field_shd(path):
    """field.exe .shd（RWSHDFile 布局：rec1=LRecl+title80B；
    rec3=7×int32(Nfreq,Ntheta,NSx,NSy,NSz,NRz,NRr)+freq0(f64)+atten(f64)；
    rec8=Sz rec9=Rz rec10=Rr(f64) rec11+=复数场(NRz×NRr)。"""
    data = open(path, "rb").read()
    L = struct.unpack("I", data[:4])[0]
    rlen = 4 * L

    def rec(n):
        return data[(n - 1) * rlen:n * rlen]

    (nf, nt, nsx, nsy, nsz, nrz, nrr) = struct.unpack("7i", rec(3)[:28])
    freq0, atten = struct.unpack("dd", rec(3)[28:44])
    sz = np.frombuffer(rec(8)[:4 * nsz], dtype="<f4").copy()
    rz = np.frombuffer(rec(9)[:4 * nrz], dtype="<f4").copy()
    rr = np.frombuffer(rec(10)[:8 * nrr], dtype="<f8").copy()
    if rr[-1] > 1000.0:
        rr = rr / 1000.0
    p = np.zeros((nrz, nrr), dtype=complex)
    for i in range(nrz):
        p[i] = np.frombuffer(rec(11 + i)[:8 * nrr], dtype="<c8")
    return rr, rz, p


def tl_of(p):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -20.0 * np.log10(np.abs(p) + 1e-300)


print("\n-- B. 官方耦合模态（kraken %d 剖面 + field C/A）--" % NPROF)
# B1 多剖面 env（resonance）
stem_r = "b4_res"
with open(os.path.join(WORK, stem_r + ".env"), "w", encoding="ascii") as f:
    for i in range(NPROF):
        f.write("\n".join(env_block(i, float(d_prof[i]))) + "\n")
# B2 平坦单剖面 env
with open(os.path.join(WORK, "b4_flatm.env"), "w", encoding="ascii") as f:
    f.write("\n".join(env_block(0, 120.0)) + "\n")

for stem in (stem_r, "b4_flatm"):
    for ext in (".mod", ".prt"):
        p = os.path.join(WORK, stem + ext)
        if os.path.exists(p):
            os.remove(p)
    res = subprocess.run([KEXE, stem], cwd=WORK, capture_output=True,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(f"kraken {stem} failed:\n" + res.stdout[-500:])

cm_out = {}
for opt, tag in [("'RC'", "b4_res_c"), ("'RA'", "b4_res_a")]:
    stem_f = tag
    # field 以 FileRoot 为 .mod 前缀：kraken 多剖面输出是 b4_res.mod，
    # 为每个 OPT 变体复制一份同前缀 .mod
    import shutil
    shutil.copyfile(os.path.join(WORK, "b4_res.mod"),
                    os.path.join(WORK, stem_f + ".mod"))
    write_flp_nprof(os.path.join(WORK, stem_f + ".flp"), tag, opt,
                    NPROF, r_prof / 1000.0)
    for ext in (".shd", ".prt"):
        p = os.path.join(WORK, stem_f + ext)
        if os.path.exists(p):
            os.remove(p)
    res = subprocess.run([FEXE, stem_f], cwd=WORK, capture_output=True,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(f"field {tag} failed:\n" + res.stdout[-500:])
    rr, rz, p = read_field_shd(os.path.join(WORK, stem_f + ".shd"))
    cm_out[tag] = tl_of(p)[0]
    print(f"   {tag}（{opt.strip()}）跑通，TL 长度 {len(rr)}")

# B-2：λ/4 采样分段线性底（与 Bellhop bty 完全同底形）的耦合模态
# ——检验 Bellhop 与平滑正弦 CM 的差距是否来自 bty 离散化谐波
pts_pwl = bty_pts_resonance(4)
r_pwl = np.linspace(0.0, RMAX * 1000.0, 73)
d_pwl = np.interp(r_pwl, [p[0] * 1000.0 for p in pts_pwl],
                  [p[1] for p in pts_pwl])
with open(os.path.join(WORK, "b4_pwl.env"), "w", encoding="ascii") as f:
    for i in range(len(r_pwl)):
        f.write("\n".join(env_block(i, float(d_pwl[i]))) + "\n")
for ext in (".mod", ".prt"):
    p = os.path.join(WORK, "b4_pwl" + ext)
    if os.path.exists(p):
        os.remove(p)
res = subprocess.run([KEXE, "b4_pwl"], cwd=WORK, capture_output=True,
                     text=True)
if res.returncode != 0:
    raise RuntimeError(f"kraken b4_pwl failed:\n" + res.stdout[-500:])
import shutil as _sh
_sh.copyfile(os.path.join(WORK, "b4_pwl.mod"),
             os.path.join(WORK, "b4_pwl_c.mod"))
write_flp_nprof(os.path.join(WORK, "b4_pwl_c.flp"), "b4_pwl_c",
                "'RC'", len(r_pwl), r_pwl / 1000.0)
for ext in (".shd",):
    p = os.path.join(WORK, "b4_pwl_c" + ext)
    if os.path.exists(p):
        os.remove(p)
res = subprocess.run([FEXE, "b4_pwl_c"], cwd=WORK, capture_output=True,
                     text=True)
if res.returncode != 0:
    raise RuntimeError(f"field b4_pwl_c failed:\n" + res.stdout[-500:])
rr, rz, p = read_field_shd(os.path.join(WORK, "b4_pwl_c.shd"))
cm_out["b4_pwl_c"] = tl_of(p)[0]

# 平坦（range-independent）
write_flp_nprof(os.path.join(WORK, "b4_flatm.flp"), "b4_flatm",
                "'R '", 1, [0.0])
for ext in (".shd",):
    p = os.path.join(WORK, "b4_flatm" + ext)
    if os.path.exists(p):
        os.remove(p)
res = subprocess.run([FEXE, "b4_flatm"], cwd=WORK, capture_output=True,
                     text=True)
if res.returncode != 0:
    raise RuntimeError(res.stdout[-500:])
rr, rz, p = read_field_shd(os.path.join(WORK, "b4_flatm.shd"))
tl_flat_m = tl_of(p)[0]

mask = (rr >= 1.0) & (rr <= 24.0)
rho_cm = {}
for tag, tl in cm_out.items():
    rho_cm[tag] = float(np.corrcoef(tl_flat_m[mask], tl[mask])[0, 1])
    print(f"   {tag}: ρ（模态平坦基准，1–24 km）= {rho_cm[tag]:.4f}")
# 模态平坦 vs Bellhop 平坦互验（Bellhop 网格插值到 field 网格）
tl_flat_interp = np.interp(rr, rr_b, tl_flat)
rho_xcheck = float(np.corrcoef(tl_flat_interp[mask],
                               tl_flat_m[mask])[0, 1])
print(f"   [互验] Bellhop 平坦 TL vs field 平坦 TL 相关（1–24 km）"
      f" = {rho_xcheck:.4f}")

# ================= C. 口径逐段定位 =================
print("\n-- C. 口径逐段定位 --")
# C1 界面投影系数 S_mn = ∫φ_m(z;120)φ_n(z;121)dz
m120 = ModResult(os.path.join(WORK, "b3_d120.mod"))
m121 = ModResult(os.path.join(WORK, "b3_d121.mod"))


def renorm(mod):
    phi = mod.phi.astype(np.complex128)
    dz = np.gradient(mod.ztab)
    I = np.sum(phi.real ** 2 * dz[:, None], axis=0)
    return mod.ztab.copy(), phi / np.sqrt(I)


z120, phi120 = renorm(m120)
z121, phi121 = renorm(m121)
M = min(phi120.shape[1], phi121.shape[1])
S = np.zeros((M, M))
for n in range(M):
    interp = np.interp(z120, z121, phi121[:, n].real)
    for m in range(M):
        S[m, n] = float(np.trapezoid(phi120[:, m].real * interp, z120))
ms = np.arange(1, M + 1)
S_pr = np.zeros((M, M))
for m in range(M):
    for n in range(M):
        if m != n:
            S_pr[m, n] = 2.0 * (m + 1) * (n + 1) * (-1.0) ** (m + n + 2) \
                / (D_W * ((m + 1) ** 2 - (n + 1) ** 2))
print(f"   C1 界面投影 S_mn（δD=1 m）：对角中位 {np.median(np.diag(S)):.5f}"
      f"（应≈1，正交性自检）")
i1, i2 = 3, 2        # 模态 (4,3) 的 0-based 下标
print(f"   S(4,3) 数值 = {S[i1, i2]:+.5f} | PR 解析 = {S_pr[i1, i2]:+.5f}"
      f" | 比值 = {S[i1, i2] / S_pr[i1, i2]:+.3f}")

# C2 带通边界嫌疑：拍频带 5–95% 分位 vs 共振波数 κ_43
kr = np.abs(np.real(m120.k.astype(np.complex128)))
beat = np.abs(kr[:, None] - kr[None, :])
triu = beat[np.triu_indices(M, 1)]
kb_lo = float(np.percentile(triu, 5))
kb_hi = float(np.percentile(triu, 95))
kap43 = abs(kr[3] - kr[2])
kap_p = 2 * np.pi / LAM_BEAT
in_band = kb_lo <= kap43 <= kb_hi
print(f"   C2 拍频带 5–95% = [{kb_lo:.6f}, {kb_hi:.6f}] rad/m；"
      f"κ_43 = {kap43:.6f}（剖面波数 {kap_p:.6f}）"
      f"→ {'带内 ✓' if in_band else '被带通滤掉 ✗（η 缺口主因）'}")

# C3 逐模态边界因子分布（B3"0.25 归一化"误诊澄清）
phi_b = phi120[-1].real
phi_b1 = phi120[-2].real
dz = z120[-1] - z120[-2]
dphi_b = (phi_b - phi_b1) / dz
dphi_pek = np.sqrt(2.0 / D_W) * (ms * np.pi / D_W) * (-1.0) ** (ms + 1)
ratio_m = np.abs(dphi_b) / np.abs(dphi_pek)
print(f"   C3 边界因子比 |φ'_m|/|φ'_m^PR|：低模态(m≤8) "
      f"{np.median(ratio_m[:8]):.3f} | 中段(9–20) "
      f"{np.median(ratio_m[8:20]):.3f} | 近截止(21–33) "
      f"{np.median(ratio_m[20:]):.3f} | 全体中位 {np.median(ratio_m):.3f}"
      f"（B3 报告 0.25 = 混合中位，非归一化口径）")

# ---- 汇总与图 ----
out = dict(bell_variants=bell_variants, rho_cm=rho_cm,
           rho_xcheck=rho_xcheck, S43_num=float(S[i1, i2]),
           S43_pr=float(S_pr[i1, i2]), kb_lo=kb_lo, kb_hi=kb_hi,
           kap43=kap43, in_band=bool(in_band),
           ratio_low=float(np.median(ratio_m[:8])),
           ratio_mid=float(np.median(ratio_m[8:20])),
           ratio_high=float(np.median(ratio_m[20:])),
           lam_beat=LAM_BEAT, nprof=NPROF)
with open(os.path.join(WORK, "b4_cm_truth.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8))
ax = axs[0]
labels = [v for v in bell_variants] + ["cm_c", "cm_a"]
vals = [bell_variants[v]["rho"] for v in bell_variants] + \
    [rho_cm["b4_res_c"], rho_cm["b4_res_a"]]
colors = [GRAY] * len(bell_variants) + [TEAL, ORANGE]
ax.bar(np.arange(len(labels)), vals, color=colors)
ax.axhline(0.9, color=RED, ls="-.", lw=1.2)
ax.axhline(0.819, color=NAVY, ls=":", lw=1.2, label="B1 档案真值 0.819")
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("ρ（1–24 km 窗口）")
ax.set_title("resonance 场景：Bellhop 变体 vs 官方耦合模态")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, axis="y")
ax = axs[1]
ax.imshow(np.abs(S), cmap="viridis", aspect="auto",
          extent=[1, M, 1, M])
ax.set_xlabel("模态 n")
ax.set_ylabel("模态 m")
ax.set_title(f"S_mn=∫φ_m(D=120)φ_n(D=121)dz（S_43={S[i1, i2]:+.4f}）")
ax = axs[2]
ax.plot(ms, ratio_m, "o-", ms=4, color=NAVY)
ax.axhline(1.0, color=TEAL, ls="--", lw=1)
ax.axhline(0.2466, color=RED, ls=":", lw=1.2, label="B3 混合中位 0.2466")
ax.axvspan(kb_lo, kb_hi, color=ORANGE, alpha=0.06)
ax.set_xlabel("模态 m")
ax.set_ylabel("|φ'_m(D)| / |φ'_m^PR|")
ax.set_title("逐模态边界因子（低模态≈0.9，近截止→0）")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
fig.suptitle("B4 η 缺口定位：真值稳定性 + 官方耦合模态 + 口径逐段检查",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG, "b4_cm_truth.png"), dpi=110)
print("\n[Done] ->", os.path.join(WORK, "b4_cm_truth.json"))
