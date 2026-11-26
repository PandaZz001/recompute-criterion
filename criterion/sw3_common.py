# -*- coding: utf-8 -*-
"""SW3 系列公共模块：官方口径环境装配（SWellEx-96 Event S5）。

与 SW3a_env_assembly.py 同源（解析逻辑一致，已随 SW3a 验证），
供 SW3c/SW3d 复用：官方 env 解析、航迹序列解析、官方 4 层 env 写出。

只做数据装配，不做信号处理。
"""

import os
import re

import numpy as np

KRAKEN_EXAMPLE = (os.environ.get("SWELLEX96_DIR", r"D:\0_Research\SWellEx96") + r"\env\bach2geo\bach2geo"
                  r"\KRAKENexample")
SSP_OFFICIAL = os.path.join(KRAKEN_EXAMPLE, "ssp_S5.txt")
ENV_OFFICIAL = os.path.join(KRAKEN_EXAMPLE, "sw96_optimzed.env")
BATHY_TRACK = (os.environ.get("SWELLEX96_DIR", r"D:\0_Research\SWellEx96") + r"\env\bachman\bachmanDB\extras"
               r"\EventS5.Bathy.txt")
RANGE_VLA = (os.environ.get("SWELLEX96_DIR", r"D:\0_Research\SWellEx96") + r"\env\range\RangeEventS5"
             r"\SproulToVLA.S5.txt")
POS_VLA = os.environ.get("SWELLEX96_DIR", r"D:\0_Research\SWellEx96") + r"\acoustic\positions_vla.txt"

# 事件口径（官方 ReadMe / Event S5 定义）
T0_MIN = 0.0                 # 数据起点 = 23:15 GMT
SD_OFFICIAL = 62.0           # 官方例：23:55 声源 62 m 深
VLA_DEPTH = 216.5            # 官方优化 env 水层底（VLA 处 219.7 实测）


def parse_official_env():
    """解析 sw96_optimzed.env → (ssp_arr, layers, half)。

    ssp_arr: [(z, c), ...] 官方水层 SSP；layers: [dict(z,c,cs,rho,ap,as),...]
    沉积层点；half: dict 半空间。与 SW3a 验证口径一致。
    """
    env_txt = open(ENV_OFFICIAL, encoding="ascii").read()
    layers, half = [], None
    water_lines = []
    in_water = True
    n_mesh_headers = 0
    line_h = env_txt.splitlines()
    for i, ln in enumerate(line_h):
        s = ln.split("!")[0].strip()
        if s.startswith("'A'"):
            vals = re.findall(r"[\d.]+", line_h[i + 1])
            half = dict(z=float(vals[0]), c=float(vals[1]),
                        cs=float(vals[2]), rho=float(vals[3]),
                        ap=float(vals[4]))
            break
        if s.startswith("'") or not s:
            continue
        if in_water:
            if s.endswith("/"):              # 水层 SSP 点行
                vals = re.findall(r"[\d.]+", s[:-1])
                if len(vals) >= 2:
                    water_lines.append((float(vals[0]), float(vals[1])))
            elif len(re.findall(r"[\d.]+", s)) == 3:
                n_mesh_headers += 1
                if n_mesh_headers >= 2:      # 第 2 个层头=沉积层头
                    in_water = False
            continue
        vals = re.findall(r"[\d.]+", s)
        if len(vals) == 3:
            continue
        if len(vals) >= 6:
            layers.append(dict(z=float(vals[0]), c=float(vals[1]),
                               cs=float(vals[2]), rho=float(vals[3]),
                               ap=float(vals[4]), as_=float(vals[5])))
    ssp_arr = [(float(z), float(c)) for z, c in water_lines]
    return ssp_arr, layers, half


def load_track():
    """解析航迹序列 → (tmin, depth_t, range_t)。

    文件里 00:00–00:09 写成 "0: 0"（冒号后有空格），split 会错位，
    必须用正则按列位解析：jday HH:MM since depth。
    """
    re_tr = re.compile(r"\s*(\d+)\s+(\d+):\s*(\d+)\s+(-?\d+)\s+([\d.]+)")
    rows_b = []
    with open(BATHY_TRACK) as f:
        for ln in f:
            m = re_tr.match(ln)
            if m:
                rows_b.append((float(m.group(4)), float(m.group(5))))
    bt = np.array(rows_b)
    rows_r = []
    with open(RANGE_VLA) as f:
        for ln in f:
            m = re_tr.match(ln)
            if m:
                rows_r.append((float(m.group(4)), float(m.group(5))))
    rt = np.array(rows_r)
    return bt[:, 0], bt[:, 1], rt[:, 1]


def load_vla_depths():
    """VLA 21 通道深度（m，第 2 列）。"""
    vla = np.loadtxt(POS_VLA)
    return vla[:, 1]


def write_env_halfspace(path, freq, ssp, depth, sd, rd, rmax,
                        bottom=(1572.4, 1.76, 0.2), title="",
                        c_lim=(1400.0, 1650.0)):
    """单介质半空间 Kraken env（判据模态用；bottom 默认=官方层 2 顶
    参数 1572.4 m/s, ρ=1.76, 0.2 dB/λ，作为分层底的有效半空间口径）。"""
    lines = [
        f"'{title or 'env'}'",
        f"{freq:.0f}",
        "1",
        "'NVW'",
        f"0  0.0  {depth:.1f}",
    ]
    for i, (z, c) in enumerate(ssp):
        if i == 0:                       # 首点 6 值（z c cs rho ap as）
            lines.append(f"  {z:9.5f}  {c:9.2f}  0.0  1.0  0.0  0.0")
        else:
            lines.append(f"  {z:9.5f}  {c:9.2f} /")
    lines.append("'A' 0.0")
    lines.append(f" {depth:.1f}  {bottom[0]:.2f} 0.0 {bottom[1]:.2f} "
                 f"{bottom[2]:.2f} /")
    lines.append(f"{c_lim[0]:.0f}  {c_lim[1]:.0f}")
    lines.append(f"{rmax:.1f}")
    lines.append("1")
    lines.append(f" {sd:.1f} /")
    lines.append(f"{len(rd)}")
    lines.append("  " + " ".join(f"{z:.1f}" for z in rd) + " /")
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")


def write_env_layered(path, freq, ssp, layers, half, rmax, sd, rd,
                      title="", c_lim=(100.0, 1650.0)):
    """官方 4 层结构 Kraken env（层参数全部来自 sw96_optimzed.env）。

    layers: [dict(z,c,cs,rho,ap,as_), ...]（层 2/3 各两点）。
    """
    wd = ssp[-1][0]
    lines = [
        f"'{title or 'env'}'",
        f"{freq:.0f}",
        str(len(layers) // 2 + 1),          # NMEDIA = 水层 + 沉积层
        "'CVF'",
        f"1000 0 {wd:.1f}",
    ]
    for z, c in ssp:
        lines.append(f"  {z:9.5f}  {c:9.2f} /")
    mesh = [100, 1500]
    for li, grp in enumerate(zip(layers[::2], layers[1::2])):
        z0, z1 = grp[0]["z"], grp[1]["z"]
        lines.append(f"{mesh[li]} 0 {z1:.0f}")
        for p in grp:
            lines.append(f" {p['z']}  {p['c']} {p['cs']} {p['rho']} "
                         f"{p['ap']} {p['as_']}")
    lines.append("'A', 0")
    lines.append(f" {half['z']} {half['c']} {half['cs']} {half['rho']} "
                 f"{half['ap']} 0.")
    lines.append(f"{c_lim[0]:.1f} {c_lim[1]:.1f}")
    lines.append(f"{rmax:.2f}")
    lines.append("1")
    lines.append(f" {sd:.1f} /")
    lines.append(f"{len(rd)}")
    lines.append("  " + " ".join(f"{z:.1f}" for z in rd) + " /")
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")
