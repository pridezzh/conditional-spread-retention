# -*- coding: utf-8 -*-
"""消融实验（论文 §5.6 / 表 5）。固定单元 K=8, sep=6, eta=0.5，每次只改一个因素。

  A. 判别量估计器（不训练，直接对比闭式真值 D）
  B. 网络容量（hidden / layers）
  C. 训练预算（steps）
  D. 时间调度（uniform / 高噪声偏移 / logit-normal）
  E. 重流 2-RF
  F. 求解器（Euler / Heun）
  G. 耦合（独立 / mini-batch OT）
"""
import argparse
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_root(d):
    """向上找到同时含 code/ 与 paper/ 的一级；与脚本自身深度无关。"""
    while os.path.dirname(d) != d:
        if os.path.isdir(os.path.join(d, "code")) and os.path.isdir(os.path.join(d, "paper")):
            return d
        d = os.path.dirname(d)
    return d


_ROOT_ = _find_root(_HERE)
_CODE_ = os.path.join(_ROOT_, "code")
sys.path.insert(0, os.path.join(_CODE_, "src"))
sys.path.insert(0, _HERE)
from common import Timer, save_result  # noqa: E402
from discriminant import (d_hat_pair, fit_conditional_mean,  # noqa: E402
                          d_hat_from_predictor)
import run_toy  # noqa: E402
from tasks import RingMixture  # noqa: E402

BASE = dict(K=8, sep=6.0, eta=0.5, d=2, tag="abl")


def ablation_estimators(seed=0):
    """A. 估计器对比：与闭式真值 D 比。"""
    cell = BASE
    task = RingMixture(K=cell["K"], sep_ratio=cell["sep"], sigma=0.12,
                       eta=cell["eta"], d=cell["d"], seed=seed, n_train=20000)
    (Ctr, Xtr), (Cte, Xte) = task.arrays(n_train=20000, n_test=4000)
    D_true = task.analytic_D()
    var_total = float(Xte.var(axis=0).sum() + 1e-12)
    res = dict(kind="estimator", D_true=D_true, seed=seed, items={})
    specs = [("mean", dict(estimator="mean")),
             ("linear", dict(estimator="linear")),
             ("rff256", dict(estimator="rff", n_rff=256)),
             ("knn5", dict(estimator="knn", k=5)),
             ("knn10", dict(estimator="knn", k=10)),
             ("knn20", dict(estimator="knn", k=20)),
             ("knn50", dict(estimator="knn", k=50)),
             ("knn100", dict(estimator="knn", k=100))]
    for name, kw in specs:
        pred = fit_conditional_mean(Ctr[:15000], Xtr[:15000], **kw)
        d = d_hat_from_predictor(pred, Cte[:1500], Xte[:1500], var_total)
        res["items"][name] = float(d)
    res["items"]["pair1nn"] = d_hat_pair(Cte[:1500], Xte[:1500])
    save_result("abl_estimator_s%d" % seed, res)
    print("A. estimators: D_true=%.4f | " % D_true +
          " ".join("%s=%.3f" % (k, v) for k, v in res["items"].items()), flush=True)


def run_one(name, seed, steps=5000, threads=4, **kw):
    cell = dict(BASE)
    cell["tag"] = name
    t = Timer()
    res = run_toy.run_cell(cell, seed, train_steps=steps, threads=threads, **kw)
    res["ablation"] = name
    res["kwargs"] = {k: str(v) for k, v in kw.items()}
    save_result("abl_%s_s%d" % (name, seed), res)
    print("%-22s seed=%d | csw1=%.4f csw64=%.4f relgap=%.2f spread1=%.3f "
          "off1=%.2f off64=%.2f | %.0fs"
          % (name, seed, res["nfe"]["1"]["csw"], res["nfe"]["64"]["csw"],
             res["rel_gap"], res["spread_ratio_1"],
             res["nfe"]["1"]["off_frac"], res["nfe"]["64"]["off_frac"],
             t.elapsed()), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="B", choices=list("ABCDEFG"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    if args.part == "A":
        for s in args.seeds:
            ablation_estimators(s)
        return
    plan = {
        "B": [("cap_h64", dict(hidden=64)), ("cap_h256", dict(hidden=256)),
              ("cap_h512", dict(hidden=512)), ("cap_L3", dict(n_layers=3)),
              ("cap_L6", dict(n_layers=6))],
        "C": [("budget1k", dict()), ("budget2k5", dict()), ("budget5k", dict()),
              ("budget10k", dict())],
        "D": [("time_uniform", dict(time_mode="uniform")),
              ("time_power2", dict(time_mode="power", time_alpha=2.0)),
              ("time_power3", dict(time_mode="power", time_alpha=3.0)),
              ("time_logit", dict(time_mode="logitnormal", time_alpha=1.0))],
        "E": [("noreflow", dict(reflow=False)), ("reflow2RF", dict(reflow=True))],
        "F": [("euler", dict(solver="euler")), ("heun", dict(solver="heun"))],
        "G": [("cpl_independent", dict(coupling="independent")),
              ("cpl_ot", dict(coupling="ot"))],
    }[args.part]
    budget = {"B": args.steps, "C": 1000, "D": args.steps, "E": args.steps,
              "F": args.steps, "G": args.steps}[args.part]
    for name, kw in plan:
        for s in args.seeds:
            st = budget
            if args.part == "C":
                st = {"budget1k": 1000, "budget2k5": 2500,
                      "budget5k": 5000, "budget10k": 10000}[name]
            run_one(name, s, steps=st, threads=args.threads, **kw)


if __name__ == "__main__":
    main()
