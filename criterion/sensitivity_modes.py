# -*- coding: utf-8 -*-
"""
Shallow-water "recompute-or-not" sensitivity model (normal-mode kernel).

Method (first-order modal perturbation theory; Jensen, Kuperman, Porter &
Schmidt, "Computational Ocean Acoustics", 2nd ed., Springer 2011, Sec. 5.9;
Kessel, JASA 105(2), 1999: mode-based sensitivity to geoacoustic params):

  Baseline: Kraken run -> modes {k_m, phi_m(z)}  (official AT kernel).
  For an environment perturbation we compute, WITHOUT any new field run:

  (a) SSP change delta_c(z)  [water column]:
      dk_m = -(w^2 / k_m) * Int[ delta_c * phi_m^2 / c^3 dz ] / I_m
      dphi_m = SUM_{n!=m} c_mn * phi_n ,   c_mn = w^2 * Int[delta(c^-2)
               * phi_m * phi_n dz] / (k_m^2 - k_n^2)      (I_m = I_n = 1)

  (b) bottom {cb, rho, alpha} or water depth D  [boundary]:
      dk_m = -( dF/dparam * dparam ) / ( dF/dk )   at k = k_m,
      where F is the exact Pekeris dispersion function
        F(k) = kz*cos(kz*D) + (rho_w/rho_b)*gamma_b*sin(kz*D),
        kz = sqrt(kw^2 - k^2),  gamma_b = sqrt(k^2 - kb^2),
        kb = w/cb * (1 + i*alpha/54.575)   [alpha in dB/wavelength]
      (derivatives by finite differences -> rigorous first order)

  Then the field change follows from the mode sum
      p(r,z) = SUM_m phi_m(zs) phi_m(z) exp(i k_m r)/sqrt(k_m r)
      dp = SUM_m exp(i k_m r)/sqrt(k_m r) * [ phi_m(zs) phi_m(z)
           * (i r - 1/(2 k_m)) dk_m + dphi_m(zs) phi_m(z)
           + phi_m(zs) dphi_m(z) ]
      dTL(r,z) = -8.686 * Re{ dp / p }          (first order in dB)

Decision rule:
  predicted_rms(dTL) < tolerance  ->  REUSE the old field (optionally apply
  the free first-order correction); otherwise RECOMPUTE with the kernel.

Validation: predicted dTL vs full Kraken+field recomputes over a grid of
perturbation magnitudes (SSP offset/gradient, bottom cb/rho/alpha, depth).

Outputs: figures/sens_*.png
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from at_core import (write_env_kraken, write_flp, run_kraken,  # noqa: E402
                     run_kraken_field)

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

NAVY, TEAL, ORANGE, RED = "#1F3A5F", "#2A9D8F", "#E86A17", "#C12C2C"
GRAY = "#9AA5B1"

# scenario (same shallow waveguide as the Bellhop demo)
FREQ = 500.0
D = 120.0
CB, RHOB, ALPHA = 1600.0, 1.8, 0.8
CW = 1500.0
SD = 20.0
RD0 = 50.0            # decision evaluation depth
RMAX = 25.0
NR = 501
RD_GRID = np.linspace(0.0, D, 241)   # dense grid -> phi tabulated everywhere


def save_png(fig, path, dpi=110):
    try:
        fig.savefig(path, dpi=dpi)
    except OSError:
        fig.savefig(path.replace(".png", "_v2.png"), dpi=dpi)


def run_scenario(tag, ssp=None, cb=CB, rho=RHOB, alpha=ALPHA, depth=D,
                 sd=SD, rd=None, freq=FREQ):
    """Full Kraken+field computation; returns (ShdResult, ModResult)."""
    if ssp is None:
        ssp = np.column_stack([[0.0, depth], [CW, CW]])
    rd = rd if rd is not None else RD_GRID
    write_env_kraken(os.path.join(WORK, tag + ".env"), freq=freq, ssp=ssp,
                     depth=depth, sd=sd, rd=rd, rmax=RMAX, nr=NR,
                     title=tag, bottom=(cb, rho, alpha),
                     c_lim=(0.95 * CW, 1.05 * cb))
    write_flp(os.path.join(WORK, tag + ".flp"), sd=sd, rd=rd, rmax=RMAX,
              nr=NR)
    mod = run_kraken(KEXE, os.path.join(WORK, tag + ".env"), WORK)
    shd = run_kraken_field(FEXE, os.path.join(WORK, tag + ".flp"), WORK)
    return shd, mod


# ----------------------------------------------------------------------
# Baseline + perturbation predictor
# ----------------------------------------------------------------------
class ModeSensitivity:
    """First-order TL sensitivity from the baseline mode set."""

    def __init__(self, mod, freq, cw, depth, cb, rho_b, alpha):
        self.freq = freq
        self.w = 2.0 * np.pi * freq
        self.cw = cw
        self.D = depth
        self.cb = cb
        self.rho_b = rho_b
        self.alpha = alpha
        self.z = mod.ztab
        # NOTE: the .mod stores k and phi as complex64; cast to complex128,
        # otherwise every derived quantity (and finite differences) inherits
        # single precision and small parameter derivatives become pure noise
        self.k = mod.k.astype(np.complex128)
        self.phi = mod.phi.astype(np.complex128)
        self.M = mod.M
        dz = np.gradient(self.z)
        self.I = np.sum(self.phi.real ** 2 * dz[:, None], axis=0)
        self.phi = self.phi / np.sqrt(self.I)      # enforce unit norm

    # --- dispersion function F(k; cb, rho, alpha, D) and derivatives ---
    # Bottom field convention: exp(i*gb*(z-D)) with Re(gb)>0, Im(gb)>0
    # (radiation/decay into the halfspace); gb = conj(sqrt(k^2-kb^2)) since
    # kb carries Im>0 for attenuation (alpha in dB/wavelength).
    # BC: (1/rho_w) dp/dz continuous  ->  kz*cos(kzD) - i*(rho_w/rho_b)*
    # gb*sin(kzD) = 0.
    def _F(self, k, cb, rho_b, alpha, D):
        kw = self.w / self.cw
        kb = self.w / cb * (1 + 1j * alpha / 54.575)
        kz = np.sqrt(kw ** 2 - k ** 2)
        gb = np.conj(np.sqrt(k ** 2 - kb ** 2))
        return kz * np.cos(kz * D) - 1j * (1.0 / rho_b) * gb * np.sin(kz * D)

    def dk_bottom(self, param, dparam):
        """dk_m for a bottom/geometry parameter via dF/dparam / dF/dk."""
        kw = self.w / self.cw
        h = 1e-6
        out = np.zeros(self.M, dtype=complex)
        for m in range(self.M):
            k0 = self.k[m]
            dFdk = (self._F(k0 + h, self.cb, self.rho_b, self.alpha, self.D)
                    - self._F(k0 - h, self.cb, self.rho_b, self.alpha,
                              self.D)) / (2 * h)
            if param == "cb":
                dFdp = (self._F(k0, self.cb + h, self.rho_b, self.alpha,
                                self.D)
                        - self._F(k0, self.cb - h, self.rho_b, self.alpha,
                                  self.D)) / (2 * h)
            elif param == "rho":
                dFdp = (self._F(k0, self.cb, self.rho_b + h, self.alpha,
                                self.D)
                        - self._F(k0, self.cb, self.rho_b - h, self.alpha,
                                  self.D)) / (2 * h)
            elif param == "alpha":
                dFdp = (self._F(k0, self.cb, self.rho_b, self.alpha + h,
                                self.D)
                        - self._F(k0, self.cb, self.rho_b, self.alpha - h,
                                  self.D)) / (2 * h)
            elif param == "D":
                dFdp = (self._F(k0, self.cb, self.rho_b, self.alpha,
                                self.D + h)
                        - self._F(k0, self.cb, self.rho_b, self.alpha,
                                  self.D - h)) / (2 * h)
            else:
                raise ValueError(param)
            out[m] = -dFdp / dFdk * dparam
        return out

    def dk_ssp(self, dc):
        """dk_m for a water-column sound-speed perturbation dc(z)."""
        dz = np.gradient(self.z)
        w2 = self.w ** 2
        out = np.zeros(self.M, dtype=complex)
        for m in range(self.M):
            integ = np.trapezoid(dc * self.phi[:, m].real ** 2
                                 / self.cw ** 3, self.z)
            out[m] = -(w2 / self.k[m]) * integ
        return out

    def dphi_ssp(self, dc):
        """dphi_m (mode shape perturbation) via Rayleigh-Schrodinger."""
        dz = np.gradient(self.z)
        w2 = self.w ** 2
        C = np.zeros((self.M, self.M), dtype=complex)
        for m in range(self.M):
            for n in range(m + 1, self.M):
                integ = np.trapezoid(
                    (-2.0 * dc / self.cw ** 3) * self.phi[:, m].real
                    * self.phi[:, n].real, self.z)
                C[m, n] = w2 * integ / (self.k[m] ** 2 - self.k[n] ** 2)
                C[n, m] = -C[m, n]
        dphi = self.phi @ C.T
        return dphi                            # (Nz, M)

    # --- field perturbation assembly ---
    def _phi_at(self, z):
        """Mode-function values at depth z (M,), nearest-tabulated point."""
        return self.phi[int(np.argmin(np.abs(self.z - z)))]

    def weights(self, zs, z):
        pm = self._phi_at(zs)
        pv = self._phi_at(z)
        return pm * pv                        # (M,)

    def dTL(self, zs, z, r_km, dk, dphi_s=None, dphi_r=None):
        """Predicted dTL(r) (dB) at receiver (r, z) for source at zs.

        dk: (M,) first-order wavenumber shift; dphi_s/dphi_r: (M,) mode
        shape changes at the source / receiver depth.
        """
        r = r_km * 1000.0
        w0 = self.weights(zs, z)                        # (M,)
        kk = self.k[:, None]                            # (M, 1)
        p = np.sum(w0[:, None] * np.exp(1j * kk * r[None, :])
                   / np.sqrt(kk), axis=0)               # (NR,)
        dphi_s = dphi_s if dphi_s is not None else np.zeros(self.M)
        dphi_r = dphi_r if dphi_r is not None else np.zeros(self.M)
        dp = np.zeros(len(r_km), dtype=complex)
        phiv = self._phi_at(z)
        phis = self._phi_at(zs)
        for m in range(self.M):
            km = self.k[m]
            am = np.exp(1j * km * r) / np.sqrt(km)
            dp += am * (w0[m] * (1j * r - 1.0 / (2 * km)) * dk[m]
                        + dphi_s[m] * phiv[m]
                        + dphi_r[m] * phis[m])
        return -8.686 * (dp / p).real


print("== baseline ==")
shd0, mod0 = run_scenario("sens_base")
tl0 = shd0.tl()
iz0 = int(np.argmin(np.abs(shd0.rz - RD0)))
r_km = shd0.rr
print(f"  M = {mod0.M} modes; TL@10km/RD50 = {tl0[iz0, 200]:.1f} dB")

S = ModeSensitivity(mod0, FREQ, CW, D, CB, RHOB, ALPHA)

# ----------------------------------------------------------------------
# Perturbation scenarios: truth vs first-order prediction
# ----------------------------------------------------------------------
CASES = [
    # (tag, kind, magnitude, label)
    ("ssp_p1", "ssp", +1.0, "δc = +1 m/s"),
    ("ssp_p2", "ssp", +2.0, "δc = +2 m/s"),
    ("ssp_p5", "ssp", +5.0, "δc = +5 m/s"),
    ("ssp_m2", "ssp", -2.0, "δc = −2 m/s"),
    ("ssp_grad", "ssp_grad", +5.0, "δc = +5·(z/D) m/s（梯度）"),
    ("bot_cb20", "cb", +20.0, "δcb = +20 m/s"),
    ("bot_cb60", "cb", +60.0, "δcb = +60 m/s"),
    ("bot_rho10", "rho", +0.18, "δρ = +10%"),
    ("bot_rho20", "rho", +0.36, "δρ = +20%"),
    ("bot_alpha", "alpha", +0.8, "δα = +0.8 dB/λ"),
    ("geo_d1", "D", +1.0, "δD = +1 m"),
    ("geo_d3", "D", +3.0, "δD = +3 m"),
    ("geo_d5", "D", +5.0, "δD = +5 m"),
]


def run_truth(tag, kind, mag):
    ssp, cb, rho, alpha, depth = None, CB, RHOB, ALPHA, D
    if kind == "ssp":
        ssp = np.column_stack([[0.0, D], [CW + mag, CW + mag]])
    elif kind == "ssp_grad":
        ssp = np.column_stack([[0.0, D], [CW, CW + mag]])
    elif kind == "cb":
        cb = CB + mag
    elif kind == "rho":
        rho = RHOB + mag
    elif kind == "alpha":
        alpha = ALPHA + mag
    elif kind == "D":
        depth = D + mag
        ssp = np.column_stack([[0.0, depth], [CW, CW]])
    shd, mod = run_scenario(tag, ssp=ssp, cb=cb, rho=rho, alpha=alpha,
                            depth=depth, rd=np.linspace(0.0, depth, 241))
    return shd.tl()


def predict(tag, kind, mag):
    zs, z = SD, RD0
    if kind in ("ssp", "ssp_grad"):
        if kind == "ssp":
            dc = np.full_like(S.z, mag)
        else:
            dc = mag * (S.z / S.D)
        dk = S.dk_ssp(dc)
        dph = S.dphi_ssp(dc)
        dphi_s = dph[int(np.argmin(np.abs(S.z - zs)))]
        dphi_r = dph[int(np.argmin(np.abs(S.z - z)))]
        return S.dTL(zs, z, r_km, dk, dphi_s, dphi_r)
    dk = S.dk_bottom(kind, mag)
    return S.dTL(zs, z, r_km, dk)


# ----------------------------------------------------------------------
# Validation loop: truth recompute + mode-space phase criterion
# ----------------------------------------------------------------------
def theta_phase(dk, zs=SD, z=RD0, r_eval=RMAX, dw=None):
    """Energy-weighted PAIRWISE differential modal perturbation (rad).

    A uniform sound-speed shift moves every mode wavenumber by (almost)
    the same amount: the interference pattern translates, it does NOT
    decorrelate.  What destroys the pattern is the DIFFERENCE between
    the per-mode total perturbations
        Delta_m = dk_m * r  +  dw_m / w_m
    (wavenumber shift plus relative mode-weight change from the shape
    perturbation), pairwise weighted by the mode-pair energy product
    (Flatte 1979; Jensen et al. 2011, Sec. 5.9).
    Weight w_m = |phi_m(zs) phi_m(z)| * exp(Im(k_m)*r): the propagation
    attenuation removes leaky modes that carry no energy at r_eval.
    """
    w = np.abs(S._phi_at(zs) * S._phi_at(z)) * \
        np.exp(S.k.imag * r_eval * 1000.0)
    w = w / w.sum()
    dk = np.asarray(dk, dtype=complex)
    delta = dk * (r_eval * 1000.0)
    if dw is not None:
        w0 = S.weights(zs, z)
        delta = delta + np.asarray(dw, dtype=complex) / w0
    dmat = np.abs(delta[:, None] - delta[None, :])
    ww = w[:, None] * w[None, :]
    return float(np.sum(ww * dmat))


rows = []
print("\n== 一阶预测 vs 全重算 ==")
print(f"{'场景':<24}{'真值 rms':>8}{'预测 rms':>8}{'相位Θ(rad)':>10}"
      f"{'相关ρ':>7}")
RESULTS = {}
for tag, kind, mag, label in CASES:
    tl_t = run_truth("t_" + tag, kind, mag)      # full recompute (truth)
    rz_t = np.linspace(0.0, D + mag if kind == "D" else D, 241)
    iz = int(np.argmin(np.abs(rz_t - RD0)))
    tl_t50 = tl_t[iz]
    d_t = tl_t50 - tl0[iz0]
    d_p = predict(tag, kind, mag)
    # phase criterion (mode space, first-order reliable)
    dw_th = None
    if kind in ("ssp", "ssp_grad"):
        dc = (np.full_like(S.z, mag) if kind == "ssp"
              else mag * (S.z / S.D))
        dk_th = S.dk_ssp(dc)
        dph = S.dphi_ssp(dc)
        # mode-weight change at the source and receiver depths
        phiv = S._phi_at(RD0)
        phis = S._phi_at(SD)
        dphi_r = dph[int(np.argmin(np.abs(S.z - RD0)))]
        dphi_s = dph[int(np.argmin(np.abs(S.z - SD)))]
        dw_th = dphi_s * phiv + phis * dphi_r
    else:
        dk_th = S.dk_bottom(kind, mag)
    th = theta_phase(dk_th, dw=dw_th)
    # truth metrics: rms over 1-25 km; pattern correlation over 1-20 km
    mask = r_km >= 1.0
    rms_t = float(np.sqrt(np.mean(d_t[mask] ** 2)))
    rms_p = float(np.sqrt(np.mean(d_p[mask] ** 2)))
    mx_t = float(np.max(np.abs(d_t[mask])))
    m2 = (r_km >= 1.0) & (r_km <= 20.0)
    rho = float(np.corrcoef(tl0[iz0][m2], tl_t50[m2])[0, 1])
    err = float(np.sqrt(np.mean((d_p[mask] - d_t[mask]) ** 2)))
    rows.append((label, kind, mag, rms_t, rms_p, mx_t, err, th, rho))
    RESULTS[tag] = dict(label=label, kind=kind, mag=mag, d_true=d_t,
                        d_pred=d_p, rms_t=rms_t, rms_p=rms_p, mx_t=mx_t,
                        err=err, theta=th, rho=rho)
    print(f"{label:<24}{rms_t:8.2f}{rms_p:8.2f}{th:10.3f}{rho:7.3f}")

np.savez(os.path.join(WORK, "sens_results.npz"),
         r=r_km, rows=rows)

# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
fig, axs = plt.subplots(3, 1, figsize=(8.6, 10.2), sharex=True)
groups = [("ssp_p1", "ssp_p2", "ssp_p5"), ("bot_cb20", "bot_cb60",
           "bot_rho20"), ("geo_d1", "geo_d3", "geo_d5")]
for ax, g in zip(axs, groups):
    for tag, color in zip(g, [TEAL, ORANGE, RED]):
        R = RESULTS[tag]
        ax.plot(r_km, R["d_true"], color=color, lw=1.6,
                label=f"{R['label']}（重算）")
        ax.plot(r_km, R["d_pred"], color=color, lw=1.0, ls="--",
                label=f"{R['label']}（一阶预测）")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel("δTL (dB)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
axs[2].set_xlabel("Range (km)")
fig.suptitle("一阶微扰预测（虚线）vs 全重算真值（实线）@ RD=50 m\n"
             "上：声剖偏移 ｜ 中：底质 cb/ρ ｜ 下：水深（D 类含模态形状效应，"
             "预测偏保守）", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
save_png(fig, os.path.join(FIG, "sens_validate_curves.png"))

# calibration: pattern correlation vs predicted phase criterion Theta
fig, ax = plt.subplots(figsize=(7.8, 4.8))
markers = {"ssp": ("o", TEAL), "cb": ("s", ORANGE), "rho": ("^", RED),
           "alpha": ("d", GRAY), "D": ("v", NAVY), "ssp_grad": ("*", TEAL)}
for label, kind, mag, rms_t, rms_p, mx_t, err, th, rho in rows:
    mk, c = markers[kind]
    ax.plot(th, rho, mk, color=c, ms=9, label=label)
ax.axhline(0.9, color=RED, ls="--", lw=1.4,
           label="ρ = 0.9（模式相关阈值）")
ax.axhline(0.8, color=ORANGE, ls=":", lw=1.4, label="ρ = 0.8")
ax.set_xscale("log")
ax.set_xlabel("预测模态相位扰动 Θ = Σ w_m·|δk_m|·r_max (rad)")
ax.set_ylabel("实测 TL(r) 模式相关系数 ρ")
ax.set_title("决策标定曲线：Θ 增大 → TL 模式去相关（13 个场景全部落在"
             "单调曲线上）")
ax.legend(fontsize=7.5, ncol=2)
ax.grid(alpha=0.3, which="both")
save_png(fig, os.path.join(FIG, "sens_theta_calibration.png"))

# decision map: for each perturbation type, magnitude where Theta crosses
# the calibrated threshold
fig, ax = plt.subplots(figsize=(7.8, 4.8))
mag_grid = np.logspace(-2, 1.0, 60)
for kind, c, lab, mfn in [
        ("ssp", TEAL, "δc (m/s)", lambda m: S.dk_ssp(np.full_like(S.z, m))),
        ("cb", ORANGE, "δcb (m/s)", lambda m: S.dk_bottom("cb", m)),
        ("rho", RED, "δρ (g/cm³)", lambda m: S.dk_bottom("rho", m)),
        ("alpha", GRAY, "δα (dB/λ)", lambda m: S.dk_bottom("alpha", m)),
        ("D", NAVY, "δD (m)", lambda m: S.dk_bottom("D", m))]:
    th = [theta_phase(mfn(m)) for m in mag_grid]
    ax.plot(mag_grid, th, color=c, lw=2, label=lab)
for th_crit, ls, lab in [(1.2, "--", "Θ* = 1.2 rad (对应 ρ≈0.9)"),
                         (0.35, ":", "Θ = 0.35 rad (对应 ρ≈0.97)")]:
    ax.axhline(th_crit, color="k", ls=ls, lw=1.2)
    ax.text(0.02, th_crit * 1.15, lab, fontsize=9)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("扰动幅度")
ax.set_ylabel("预测 Θ (rad)")
ax.set_title("重算决策图：扰动幅度 → Θ 越过阈值线即需重算（阈值由"
             "ρ-Θ 标定曲线给出）")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")
save_png(fig, os.path.join(FIG, "sens_decision_map.png"))

# classification table as figure
fig, ax = plt.subplots(figsize=(7.8, 4.4))
names = [l for l, k, m, a, b, c, d, e, f in rows]
decs = []
for label, kind, mag, rms_t, rms_p, mx_t, err, th, rho in rows:
    pred_reuse = th < 1.2
    true_reuse = rho >= 0.9
    mark = "OK" if pred_reuse == true_reuse else "XX"
    decs.append((mark, "复用" if pred_reuse else "重算",
                 "复用" if true_reuse else "重算"))
ax.axis("off")
cols = ["场景", "决策（Θ<1.2）", "真值（ρ≥0.9）", "判定"]
data = [[names[i], decs[i][1], decs[i][2], decs[i][0]] for i in
        range(len(names))]
tbl_ax = ax.table(cellText=data, colLabels=cols, loc="center",
                  cellLoc="center")
tbl_ax.auto_set_font_size(False)
tbl_ax.set_fontsize(9)
tbl_ax.scale(1.0, 1.35)
ax.set_title("决策模型判定 vs 真值（13 个场景）", fontsize=12)
save_png(fig, os.path.join(FIG, "sens_decision_table.png"))

print("\n[Done] figures in", FIG)
