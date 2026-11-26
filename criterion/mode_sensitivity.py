# -*- coding: utf-8 -*-
"""可导入的模态一阶微扰敏感度模块（供 DM/E 系列系综实验复用）。

权威实现是 sensitivity_modes.py（含 13 场景验证主流程，直接运行）；
本模块把其中的 ModeSensitivity 类与 theta_phase 判据函数原样提取出来，
避免 import sensitivity_modes 时把整个验证流程重跑一遍。

数值口径与 sensitivity_modes.py 逐点一致：
  - dk_ssp / dphi_ssp 循环版 = 原实现（原样复制）
  - dk_ssp_vec / dphi_ssp_vec = 同一公式的向量化写法，与循环版
    allclose(rtol=1e-12) 核验通过（BQ1_cost_benchmark.py 内完成核验）
  - dk_bottom 用 Pekeris 色散函数 F 的有限差分隐式微分（原实现）
"""

import numpy as np


class ModeSensitivity:
    """First-order TL sensitivity from the baseline mode set.

    与 sensitivity_modes.py 中同名类完全一致（含 complex64→complex128
    升精度、单位归一化等口径）。
    """

    def __init__(self, mod, freq, cw, depth, cb, rho_b, alpha):
        self.freq = freq
        self.w = 2.0 * np.pi * freq
        self.cw = cw
        self.D = depth
        self.cb = cb
        self.rho_b = rho_b
        self.alpha = alpha
        self.z = mod.ztab
        # .mod 的 k 与 phi 是 complex64：必须升 complex128，否则有限差分
        # 导数全是单精度噪声（模型文档踩坑 1）
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
                integ = np.trapezoid((-2.0 * dc / self.cw ** 3)
                                     * self.phi[:, m].real
                                     * self.phi[:, n].real, self.z)
                C[m, n] = w2 * integ / (self.k[m] ** 2 - self.k[n] ** 2)
                C[n, m] = -C[m, n]
        return self.phi @ C.T

    # --- 向量化实现（同一公式的矩阵写法，与循环版 1e-12 一致） ---
    def dk_ssp_vec(self, dc):
        integ = np.trapezoid(dc[:, None] * self.phi.real ** 2
                             / self.cw ** 3, self.z, axis=0)
        return -(self.w ** 2 / self.k) * integ

    def dphi_ssp_vec(self, dc):
        P = self.phi.real                                    # (Nz, M)
        dcw = dc * (-2.0 / self.cw ** 3)                     # 循环版同因子
        I = np.trapezoid(dcw[:, None, None] * P[:, None, :]
                         * P[:, :, None], self.z, axis=0)    # (M, M)
        with np.errstate(divide="ignore", invalid="ignore"):
            C = self.w ** 2 * I / (self.k[:, None] ** 2
                                   - self.k[None, :] ** 2)
        C[np.diag_indices(self.M)] = 0.0
        return self.phi @ C.T

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


def theta_phase(S, dk, zs=20.0, z=50.0, r_eval=25.0, dw=None):
    """Energy-weighted PAIRWISE differential modal perturbation (rad).

    与 sensitivity_modes.theta_phase 同构（权威实现所在处），此处把
    敏感度对象 S 作为参数传入以便系综复用。共模分量在对差 |Δ_m−Δ_n|
    中严格抵消（模型档案"共模相移陷阱"）。
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
