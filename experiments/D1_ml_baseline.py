# -*- coding: utf-8 -*-
"""D1 纯 ML 判据 vs 物理判据的跨波导对比（2026-11-26，对标改进点 D）。

问题：既然有 500+ 对拍样本，训练一个 ML 判据（GBDT/RF/MLP）能
不能替代物理判据？实验设计（诚实版）：
- 数据：W1 合成 72 样本（三类波导 × 三类扰动 × 4 档），特征 =
  [log D, 扰动类型 one-hot(3), 幅度 α]——ML 可用的原始环境特征。
- 训练/测试划分体现跨波导泛化：
    S1 分布内：随机 70/30 划分（乐观上界）
    S2 跨波导：浅海两波导（W1+W2 各 24）训练 → Munk 深海 24 测试
- 物理判据：u = Θ/Θ* ≤ 1 判复用（浅海 0.8、深海 100 Hz 0.667）
- 度量：正确率 + 假复用率（安全关键：ML 的假复用是危险方向）

输出：work/d1_ml_baseline.json + figures/d1_ml_baseline.png
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
FIG = os.path.join(HERE, "figures")
NAVY, TEAL, ORANGE, RED = "#1F3A5F", "#2A9D8F", "#E86A17", "#C12C2C"

RHO_REQ = 0.9
SEED = 42


def build_dataset():
    """W1 合成 72 样本：特征 [logD, kind one-hot, alpha]，标签=真决策。"""
    w1 = json.load(open(os.path.join(WORK, "w1_second_env.json"),
                        encoding="utf-8"))
    kind_map = {"surface": 0, "gradient": 1, "uniform": 2}
    X, y, u_phys, grp = [], [], [], []
    for r in w1["rows"]:
        deep = "Munk" in r["wg"]
        D = 5000.0 if deep else (100.0 if "Pekeris" in r["wg"] else 120.0)
        onehot = np.zeros(3)
        onehot[kind_map[r["kind"]]] = 1.0
        feat = [np.log10(D), r["alpha"], r["sign"]]
        feat.extend(onehot)
        X.append(feat)
        y.append(int(r["rho"] < RHO_REQ))          # 1=该重算
        th_star = 0.667 if deep else 0.8
        u_phys.append(r["th"] / th_star)
        grp.append("Munk深海" if deep else "浅海")
    return np.array(X), np.array(y), np.array(u_phys), np.array(grp)


def metrics(y_true, y_pred):
    ok = int((y_true == y_pred).sum())
    fr = int(((y_pred == 0) & (y_true == 1)).sum())  # 判复用但真重算
    fm = int(((y_pred == 1) & (y_true == 0)).sum())
    return ok, fr, fm


print("== D1 纯 ML 判据 vs 物理判据（跨波导对比）==\n")
X, y, u_phys, grp = build_dataset()
print(f"数据集 n={len(y)}：浅海 {int((grp == '浅海').sum())} | "
      f"Munk 深海 {int((grp == 'Munk深海').sum())} | "
      f"真重算样本 {int(y.sum())}")

models = {
    "GBDT": GradientBoostingClassifier(random_state=SEED),
    "随机森林": RandomForestClassifier(random_state=SEED, n_estimators=300),
    "MLP": make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(16, 8),
                                       max_iter=3000, random_state=SEED)),
}

# ---- S1 分布内（随机 70/30，重复 30 次取中位） ----
print("\nS1 分布内（随机 70/30）：")
s1 = {}
rng = np.random.default_rng(SEED)
for name, mdl in models.items():
    oks, frs, fms = [], [], []
    for _ in range(30):
        idx = rng.permutation(len(y))
        ntr = int(len(y) * 0.7)
        mdl.fit(X[idx[:ntr]], y[idx[:ntr]])
        ok, fr, fm = metrics(y[idx[ntr:]], mdl.predict(X[idx[ntr:]]))
        oks.append(ok / (len(y) - ntr))
        frs.append(fr)
        fms.append(fm)
    s1[name] = dict(ok=float(np.median(oks)), fr=int(np.median(frs)),
                    fm=int(np.median(fms)))
    print(f"  {name:<6} 正确率 {np.median(oks):.1%} | 假复用中位 "
          f"{int(np.median(frs))} | 假重算中位 {int(np.median(fms))}")
# 物理判据同口径（S1 测试集上的物理）
ok_p, fr_p, fm_p = [], [], []
for _ in range(30):
    idx = rng.permutation(len(y))
    ntr = int(len(y) * 0.7)
    ok, fr, fm = metrics(y[idx[ntr:]], (u_phys[idx[ntr:]] > 1.0).astype(int))
    ok_p.append(ok / (len(y) - ntr))
    fr_p.append(fr)
    fm_p.append(fm)
s1["物理判据"] = dict(ok=float(np.median(ok_p)), fr=int(np.median(fr_p)),
                      fm=int(np.median(fm_p)))
print(f"  {'物理判据':<6} 正确率 {np.median(ok_p):.1%} | 假复用中位 "
      f"{int(np.median(fr_p))} | 假重算中位 {int(np.median(fm_p))}")

# ---- S2 跨波导（浅海 48 训练 → Munk 24 测试） ----
print("\nS2 跨波导（浅海训练 → Munk 深海测试）：")
mask_tr = grp == "浅海"
mask_te = grp == "Munk深海"
s2 = {}
for name, mdl in models.items():
    mdl.fit(X[mask_tr], y[mask_tr])
    ok, fr, fm = metrics(y[mask_te], mdl.predict(X[mask_te]))
    s2[name] = dict(ok=ok / mask_te.sum(), fr=fr, fm=fm)
    print(f"  {name:<6} 正确率 {ok / mask_te.sum():.1%} | 假复用 {fr} | "
          f"假重算 {fm}")
ok_p2, fr_p2, fm_p2 = metrics(y[mask_te],
                              (u_phys[mask_te] > 1.0).astype(int))
s2["物理判据"] = dict(ok=ok_p2 / mask_te.sum(), fr=fr_p2, fm=fm_p2)
print(f"  {'物理判据':<6} 正确率 {ok_p2 / mask_te.sum():.1%} | 假复用 "
      f"{fr_p2} | 假重算 {fm_p2}")

# ---- S3 硬外推：W1 合成训练 → 全球真实 WOA 测试 ----
# 统一特征 = δc 统计矩（mean/std/skew/kurt/峰峰）+ logD + logf。
# δc 重算（不重跑 Kraken，只读 WOA + 重算统计量）。
import xarray as xr
WOA_DIR = os.environ.get("WOA23_DIR", r"E:\AI_Codes\水声")


def load_ssp_stat(lon, lat, season, D_cap=None):
    fn = os.path.join(WOA_DIR, f"woa23_ssp_s{season}.nc")
    with xr.open_dataset(fn, engine="h5netcdf") as ds:
        p = ds["ssp"].sel(lon=lon, lat=lat, method="nearest")
        dep = ds["depth"].values
        vals = p.values
    valid = ~np.isnan(vals)
    return dep[valid], vals[valid]


def stats_of(dc):
    dc = np.asarray(dc, dtype=float)
    st = __import__("scipy.stats", fromlist=["skew", "kurtosis"])
    sk = 0.0 if np.std(dc) < 1e-9 else float(st.skew(dc))
    ku = 0.0 if np.std(dc) < 1e-9 else float(st.kurtosis(dc))
    return [np.mean(dc), np.std(dc), sk, ku,
            float(np.max(dc) - np.min(dc))]


def synth_dc(kind, amp, D):
    zs = np.arange(0.0, D + 0.5, 1.0)
    if kind == "surface":
        dc = amp * np.exp(-zs / 50.0)
    elif kind == "gradient":
        dc = amp * (zs / zs.max() - 0.5) * 2.0
    else:
        dc = amp * np.ones_like(zs)
    return dc


def build_s3():
    """训练：W1 合成（δc 统计矩特征）；测试：W2/W2b/W5/W7/W9 全球真实。"""
    w1 = json.load(open(os.path.join(WORK, "w1_second_env.json"),
                        encoding="utf-8"))
    Xtr, ytr, utr = [], [], []
    for r in w1["rows"]:
        deep = "Munk" in r["wg"]
        D = 5000.0 if deep else (100.0 if "Pekeris" in r["wg"] else 120.0)
        f = 100.0 if deep else 500.0
        dc = synth_dc(r["kind"], r["alpha"] * r["sign"], D)
        Xtr.append([np.log10(D), np.log10(f)] + stats_of(dc))
        ytr.append(int(r["rho"] < RHO_REQ))
        utr.append(r["th"] / (0.667 if deep else 0.8))
    sites = [
        ("W2", "w2_woa_env.json", 0.8, 0.667),
        ("W2b", "w2b_seasons.json", 0.8, 0.667),
        ("W5", "w5_global_shelf.json", 0.8, 0.8),
        ("W7", "w7_deep_global.json", 0.667, 0.667),
        ("W9", "w9_deep_500hz.json", 3.33, 3.33),
    ]
    # 站点坐标表（dc 重算所需）
    lonlat = {
        "南海北部陆架": (113.0, 20.0), "黄海": (123.0, 35.0),
        "东海陆架": (124.0, 30.0), "西太": (135.0, 15.0),
        "北大西洋": (330.0, 40.0), "北海": (3.0, 56.0),
        "波斯湾": (51.0, 26.5), "白令海陆架": (190.0, 62.0),
        "鄂霍次克海": (144.0, 53.0), "英吉利海峡": (0.5, 50.5),
        "阿拉弗拉海": (135.0, -10.0), "孟加拉湾北部": (89.0, 21.0),
        "巽他陆架": (108.0, -2.0), "印度洋": (60.0, -10.0),
        "南大洋": (160.0, -55.0), "东太": (250.0, 10.0),
    }
    Xte, yte, ute, te_names = [], [], [], []
    cache = {}
    for tag, fn, ts_sh, ts_dp in sites:
        d = json.load(open(os.path.join(WORK, fn), encoding="utf-8"))
        for r in d["rows"]:
            if tag == "W9":          # W9 rows 无 site/D（固定西太 4700 m）
                r = dict(r, site="西太", D=4700.0)
            D = float(r["D"])
            f = 500.0 if D < 300 else (500.0 if tag == "W9" else 100.0)
            key = (r["site"], r["alpha"], r.get("season", ""))
            if key not in cache:
                lo, la = lonlat[r["site"]]
                dep_w, c_w = load_ssp_stat(lo, la, 13)
                dep_s, c_s = load_ssp_stat(lo, la, 15)
                dc0 = np.interp(dep_w, dep_s, c_s) - c_w
                cache[key] = stats_of(dc0 * r["alpha"])
            Xte.append([np.log10(D), np.log10(f)] + cache[key])
            yte.append(int(r["rho"] < RHO_REQ))
            th_star = ts_dp if D >= 300 else ts_sh
            ute.append(r["th"] / th_star)
            te_names.append(tag)
    return (np.array(Xtr), np.array(ytr), np.array(utr),
            np.array(Xte), np.array(yte), np.array(ute), te_names)


Xtr3, ytr3, utr3, Xte3, yte3, ute3, te_names = build_s3()
print(f"\nS3 硬外推（W1 合成 {len(ytr3)} 训练 → 全球真实 {len(yte3)} 测试，"
      f"特征=δc 统计矩+logD+logf）：")
s3 = {}
for name, mdl in models.items():
    mdl.fit(Xtr3, ytr3)
    ok, fr, fm = metrics(yte3, mdl.predict(Xte3))
    s3[name] = dict(ok=ok / len(yte3), fr=fr, fm=fm)
    print(f"  {name:<6} 正确率 {ok / len(yte3):.1%} | 假复用 {fr} | "
          f"假重算 {fm}")
ok3, fr3, fm3 = metrics(yte3, (ute3 > 1.0).astype(int))
s3["物理判据"] = dict(ok=ok3 / len(yte3), fr=fr3, fm=fm3)
print(f"  {'物理判据':<6} 正确率 {ok3 / len(yte3):.1%} | 假复用 {fr3} | "
      f"假重算 {fm3}")

out = dict(s1=s1, s2=s2, s3=s3, n=len(y), rho_req=RHO_REQ)
with open(os.path.join(WORK, "d1_ml_baseline.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# ---- 图 ----
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.0))
ax = axs[0]
names = list(models) + ["物理判据"]
xs = np.arange(len(names))
for k, name in enumerate(names):
    ax.bar(xs[k] - 0.2, s1[name]["ok"], 0.2, color=TEAL,
           label="正确率" if k == 0 else None)
    fr_ratio = s1[name]["fr"] / max(s1[name]["fr"] + s1[name]["fm"], 1)
    ax.bar(xs[k], fr_ratio, 0.2, color=RED, label="假复用占比" if k == 0 else None)
    ax.bar(xs[k] + 0.2, s1[name]["fm"] / 22, 0.2, color="#8A8F98",
           label="假重算/22" if k == 0 else None)
ax.set_xticks(xs)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("比率")
ax.set_title("S1 分布内（乐观上界）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
ax = axs[1]
for k, name in enumerate(names):
    ax.bar(xs[k] - 0.2, s2[name]["ok"], 0.2, color=TEAL,
           label="正确率" if k == 0 else None)
    ax.bar(xs[k], s2[name]["fr"] / 24, 0.2, color=RED,
           label="假复用率" if k == 0 else None)
    ax.bar(xs[k] + 0.2, s2[name]["fm"] / 24, 0.2, color="#8A8F98",
           label="假重算率" if k == 0 else None)
ax.set_xticks(xs)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("比率")
ax.set_title("S2 跨波导（浅海→Munk 深海，24 测试）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.suptitle("D1 纯 ML 判据 vs 物理判据：跨波导泛化的假复用风险",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIG, "d1_ml_baseline.png"), dpi=110)
print("\n[Done] ->", os.path.join(WORK, "d1_ml_baseline.json"))
