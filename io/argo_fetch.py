# -*- coding: utf-8 -*-
"""Argo GDAC 剖面便携下载器（在任何能连 GDAC 的机器上运行）。

背景（2026-11-26）：开发机网络环境下 GDAC 全部镜像不可达
（ifremer 被墙、usgodae 索引损坏、AWS 沙盒桶空壳、中国中心无公开
API）。本脚本在可达机器（单位服务器/家庭宽带）运行一次，把目标
海区的剖面清单与数据下载到本地，拷贝回开发机即可。

用法：
    python argo_fetch.py scs            # 南海 (10-25N, 105-125E)
    python argo_fetch.py list --box 10 25 105 125
    python argo_fetch.py fetch --wmo 2902746 --nprof 4

依赖：requests（pip install requests）。源：usgodae 优先，
失败自动回落 ifremer。
"""

import argparse
import io
import os
import sys

import requests

BASE_URLS = [
    "https://usgodae.org/pub/outgoing/argo",
    "https://data-argo.ifremer.fr",
]
INDEX_REL = "ar_index_global_prof.txt"
TIMEOUT = 60


def get(url, stream=False):
    for base in BASE_URLS:
        try:
            r = requests.get(base + url, timeout=TIMEOUT, stream=stream)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            print(f"  [{base}] 失败：{e}", file=sys.stderr)
    raise RuntimeError("所有 GDAC 源均不可达")


def load_index():
    """下载并解压全局剖面索引（返回行列表）。"""
    print("下载 ar_index_global_prof.txt（可能数百 MB，耐心等待）…")
    r = get("/" + INDEX_REL, stream=True)
    raw = io.BytesIO()
    for chunk in r.iter_content(chunk_size=1 << 20):
        raw.write(chunk)
    data = raw.getvalue()
    if data[:2] == b"\x1f\x8b":  # gzip
        import gzip
        data = gzip.decompress(data)
    text = data.decode("ascii", "replace")
    return [ln for ln in text.splitlines() if ln and not ln.startswith("#")]


def norm(v):
    v = float(v)
    return v / 1e6 if abs(v) > 90 else v


def filter_box(lines, lat_lo, lat_hi, lon_lo, lon_hi):
    """索引行过滤（格式: file,date,lat,lon,ocean,type,institution,
    date_update；坐标为小数度或百万分度）。"""
    out = []
    for ln in lines:
        p = ln.split(",")
        if len(p) < 8:
            continue
        lat, lon = norm(p[2]), norm(p[3])
        if lat_lo <= lat <= lat_hi and lon_lo <= lon <= lon_hi:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scs")
    lp = sub.add_parser("list")
    lp.add_argument("--box", nargs=4, type=float, required=True)
    fp = sub.add_parser("fetch")
    fp.add_argument("--wmo", required=True)
    fp.add_argument("--nprof", type=int, default=4)
    args = ap.parse_args()

    if args.cmd == "scs":
        lines = load_index()
        hits = filter_box(lines, 10, 25, 105, 125)
        wmos = sorted({p[0].split("/")[1] for p in hits})
        print(f"南海剖面 {len(hits)} 条，浮标 {len(wmos)} 个（前 20）：")
        for w in wmos[:20]:
            print(" ", w)
        with open("argo_scs_index.csv", "w", encoding="ascii") as f:
            f.write("\n".join(",".join(p) for p in hits))
        print("已写 argo_scs_index.csv")
    elif args.cmd == "list":
        lines = load_index()
        lat_lo, lat_hi, lon_lo, lon_hi = args.box
        hits = filter_box(lines, lat_lo, lat_hi, lon_lo, lon_hi)
        for p in hits:
            print(",".join(p))
    elif args.cmd == "fetch":
        lines = load_index()
        hits = [p for p in lines if p[0].split("/")[1] == args.wmo]
        if not hits:
            raise SystemExit(f"浮标 {args.wmo} 不在索引中")
        hits = hits[-args.nprof:]
        os.makedirs("argo_data", exist_ok=True)
        for p in hits:
            rel = p[0]
            fn = os.path.basename(rel)
            r = get("/" + rel)
            with open(os.path.join("argo_data", fn), "wb") as f:
                f.write(r.content)
            print(f"已下载 {fn}（{p[2]},{p[3]} @ {p[1]}）")


if __name__ == "__main__":
    main()
