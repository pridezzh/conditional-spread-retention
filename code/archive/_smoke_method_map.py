# -*- coding: utf-8 -*-
"""run_method_map 的冒烟测试：极小步数跑通全部六个方法族，只查不崩。"""
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
sys.path.insert(0, os.path.join(ROOT, "code", "experiments"))
sys.path.insert(0, os.path.join(ROOT, "code", "src"))

import run_method_map as M  # noqa: E402

M.STEPS = 60
M.N_PAIR = 400
M.N_EVAL = 200
M.SEEDS = (0,)

print("== 几何 ==")
print("  闭式 tr Var(x1|j) =", M.TR_VAR_COND.tolist())
x0, x1, c = M.sample_cond(8, __import__("numpy").random.default_rng(0))
print("  sample_cond 形状:", x0.shape, x1.dtype, c.dtype)

print("\n== ot_reorder ==")
x0, x1, c = M.sample_cond(64, __import__("numpy").random.default_rng(1))
x1o = M.ot_reorder(x0, x1, c)
d_before = ((x0 - x1) ** 2).sum(-1).mean()
d_after = ((x0 - x1o) ** 2).sum(-1).mean()
print("  配对前平均距离 %.3f -> 配对后 %.3f（应下降）" % (d_before, d_after))
assert d_after < d_before, "OT 配对没有降低距离"

print("\n== 六个方法族（60 步）==")
r = M.run_seed(0)
for k, v in r.items():
    for nfe, e in v.items():
        print("  %-14s %-6s rho=%.4f  cov=%.2f" % (k, nfe, e["rho"], e["coverage"]))

print("\n== make_pair_set / evaluate 路径 ==")
net = M.train_cfm(lambda bs, rng: M.sample_cond(bs, rng), 0, tag="smoke")
X0, Y, Cc = M.make_pair_set(net, 0, n=M.N_PAIR, N=8)
print("  pair set:", X0.shape, Y.shape, Cc.shape, Y.dtype)
print("\nSMOKE OK")
