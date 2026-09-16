# -*- coding: utf-8 -*-
"""冒烟：把 STEPS 调小，确认 run_loss_ladder 三条路径都能跑通。"""
import os
import sys


def _find_root(d):
    d = os.path.abspath(d)
    while True:
        if os.path.isdir(os.path.join(d, "code")) and os.path.isdir(os.path.join(d, "paper")):
            return d
        p = os.path.dirname(d)
        if p == d:
            raise RuntimeError("project root not found")
        d = p


ROOT = _find_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code", "experiments"))

import run_method_map as M  # noqa: E402
import run_loss_ladder as L  # noqa: E402

M.STEPS = 40
M.N_EVAL = 400
L.N_PROJ = 32
M.BS = 256

for name, fn in (("l2", L.train_l2), ("chamfer", L.train_chamfer), ("balanced", L.train_balanced)):
    net = fn(0)
    r = L.evaluate_full(net, 0)
    print("%-9s rho=%.4f cov=%.2f disp=%.3f cSW=%.4f rho_j=%s"
          % (name, r["rho"], r["coverage"], r["rho_disp"], r["csw"],
             [round(v, 3) for v in r["rho_per_cond"]]), flush=True)
print("SMOKE OK")
