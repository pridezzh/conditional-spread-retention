# -*- coding: utf-8 -*-
"""_norm_ppf 的精度自检：与 scipy 对照，并顺带检查 W2 基线的量级。"""
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
from scipy.stats import norm  # noqa: E402

from deficit import _norm_ppf, _baseline, w2_to_gaussian_1d  # noqa: E402

u = np.r_[1e-6, 1e-4, 0.001, 0.01, 0.02425, 0.1, 0.25, 0.5,
          0.75, 0.9, 0.97575, 0.99, 0.999, 0.9999, 0.999999]
mine = _norm_ppf(u)
ref = norm.ppf(u)
err = np.abs(mine - ref)
worst = err.max()
print("u            mine          scipy         abs_err")
for a, b, c, e in zip(u, mine, ref, err):
    print("%-12.6g %-13.8f %-13.8f %.3e" % (a, b, c, e))
print("最大绝对误差 = %.3e" % worst)
print("判定: %s" % ("OK (<1e-6)" if worst < 1e-6 else "FAIL"))

print("\nW2 有限样本基线 b(n)（单位方差高斯的 W2^2 正偏）")
for n in (50, 100, 200, 400, 800):
    print("  n=%-5d b(n)=%.5f" % (n, _baseline(n)))

print("\n单样本自洽: 标准正态 n=400 的未校正比值应≈b(400)")
rng = np.random.default_rng(0)
vals = [w2_to_gaussian_1d(rng.normal(size=400))[0] for _ in range(30)]
print("  实测均值 %.5f" % np.mean(vals))
