# -*- coding: utf-8 -*-
"""诊断：切片亏损估计量的**总体值**是多少？有限样本估计是否偏低？

两个问题：
  Q1 理论天花板。1-D 两点分布 {±c} 对 N(0,c^2) 的亏损 = E[(Z-1)^2]，
     Z 为半正态，= 1 - 2*sqrt(2/pi) + 1 = 0.40423。切片估计量必须
     在两点分布上重现这个值（若分离度足够大）。
  Q2 two_modes 实测 0.2976 与手算 0.40 的 25% 缺口从哪来？

做法：用**超大样本**（n=200000）逼近总体值，再与有限样本估计对比。
同时比较四种聚合方式：mean / ratio-first / q90 / max。
"""
import os
import sys


def _find_root(d):
    d = os.path.abspath(d)
    while True:
        if os.path.isdir(os.path.join(d, "code")) and os.path.isdir(os.path.join(d, "paper")):
            return d
        p = os.path.dirname(d)
        if p == d:
            raise RuntimeError("root not found")
        d = p


ROOT = _find_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code", "src"))

import numpy as np  # noqa: E402

from deficit import gaussian_deficit, w2_to_gaussian_1d, _norm_ppf  # noqa: E402


def deficit_variants(Z, n_proj=256, seed=0):
    """四种聚合方式，全部**不做**有限样本偏差校正（总体值不需要）。"""
    Z = np.asarray(Z, dtype=np.float64)
    n, d = Z.shape
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(n_proj, d))
    G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-300
    P = Z @ G.T
    w2 = np.empty(n_proj)
    var = np.empty(n_proj)
    for j in range(n_proj):
        w2[j], var[j] = w2_to_gaussian_1d(P[:, j])
    vbar = var.mean()
    ratio = w2 / np.maximum(var, 1e-300)
    return dict(mean=float(w2.mean() / vbar),
                ratio_first=float(ratio.mean()),
                q90=float(np.quantile(ratio, 0.90)),
                max=float(ratio.max()))


def main():
    print("=" * 76)
    print("Q1 理论天花板：1-D 两点分布 {±c} 对 N(0,c^2) 的亏损")
    print("   解析值 E[(Z-1)^2] = 1 - 2*sqrt(2/pi) + 1 = %.5f" % (1 - 2 * np.sqrt(2 / np.pi) + 1))
    for c in (1.0, 3.0, 10.0):
        for n in (200, 5000, 200000):
            rng = np.random.default_rng(11)
            sgn = rng.choice([-1.0, 1.0], size=n)
            Z = (c * sgn).reshape(-1, 1)
            w2, v = w2_to_gaussian_1d(Z.ravel())
            print("   c=%-5.1f n=%-7d  W2^2/Var = %.5f" % (c, n, w2 / v))
    print()

    print("=" * 76)
    print("Q2 two_modes（中心 ±2.5 各坐标，分量 sd=0.25）的总体值 vs 有限样本")
    rows = []
    for n in (200, 400, 2000, 20000, 200000):
        rng = np.random.default_rng(2024)
        half = n // 2
        Z = np.r_[rng.normal(-2.5, 0.25, size=(half, 2)),
                  rng.normal(2.5, 0.25, size=(n - half, 2))]
        v = deficit_variants(Z, n_proj=256, seed=1)
        lam, raw, _ = gaussian_deficit(Z, n_proj=256, seed=1)
        rows.append((n, v, lam, raw))
        print("   n=%-7d  mean=%.4f  ratio_first=%.4f  q90=%.4f  max=%.4f   "
              "(带偏差校正 lam=%.4f, 未校正 %.4f)"
              % (n, v["mean"], v["ratio_first"], v["q90"], v["max"], lam, raw))
    print()

    print("=" * 76)
    print("Q3 各形状的总体值（n=200000，四种聚合）")
    print("   %-14s %-9s %-9s %-9s %-9s" % ("shape", "mean", "ratio1st", "q90", "max"))
    for name in ["gauss", "t_df3", "banana", "lognorm", "exp_skew",
                 "uniform_ball", "hetero", "laplace", "two_modes", "ring8"]:
        rng = np.random.default_rng(7)
        n = 200000
        if name == "gauss":
            Z = rng.normal(size=(n, 2))
        elif name == "t_df3":
            Z = rng.standard_t(3.0, size=(n, 2)) * 0.7
        elif name == "banana":
            x1 = rng.normal(scale=1.0, size=(n, 1))
            Z = np.hstack([x1, x1 ** 2 + rng.normal(scale=0.30, size=(n, 1)) - 1.0])
        elif name == "lognorm":
            Z = np.exp(0.6 * rng.normal(size=(n, 2))) - np.exp(0.18)
        elif name == "exp_skew":
            Z = rng.exponential(size=(n, 2)) - 1.0
        elif name == "uniform_ball":
            dd = rng.normal(size=(n, 2))
            dd /= np.linalg.norm(dd, axis=1, keepdims=True) + 1e-12
            Z = dd * np.sqrt(rng.uniform(0, 1, size=(n, 1))) * 2.0
        elif name == "hetero":
            x1 = rng.normal(size=(n, 1))
            Z = np.hstack([x1, rng.normal(scale=0.20 + 0.80 * np.abs(x1))])
        elif name == "laplace":
            Z = rng.laplace(size=(n, 2))
        elif name == "two_modes":
            Z = np.r_[rng.normal(-2.5, 0.25, size=(n // 2, 2)),
                      rng.normal(2.5, 0.25, size=(n - n // 2, 2))]
        elif name == "ring8":
            ang = rng.integers(0, 8, size=n)
            th = ang * (2.0 * np.pi / 8.0)
            Z = np.c_[np.cos(th), np.sin(th)] * 3.0 + rng.normal(scale=0.25, size=(n, 2))
        v = deficit_variants(Z, n_proj=256, seed=3)
        print("   %-14s %-9.4f %-9.4f %-9.4f %-9.4f"
              % (name, v["mean"], v["ratio_first"], v["q90"], v["max"]))


if __name__ == "__main__":
    main()
