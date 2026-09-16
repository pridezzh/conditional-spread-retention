# -*- coding: utf-8 -*-
"""损失阶梯图（论文图 2）。

左面板是这张图的全部要点：**生成的逐条件展布 vs 真实的逐条件展布**。
  * 落在对角线上  = 恢复了条件律；
  * 压成一条**水平线** = 只恢复了支撑集、跨条件权重是错的；
  * 缩到原点      = 塌缩。

修正后（2026-09-16）这里有**四条**曲线，不是三条。多出来的一条是把非平衡 Chamfer
改成**逐条件**匹配之后的版本，与 Thm 4(L2) 的表述严格对齐。实测结论因此被改写：

  * 池化版（L2a，实践里通常的写法）确实压成水平线；
  * 逐条件版（L2b）基本回到对角线；
  * 平衡版（L3）在对角线附近且保真度（条件 sliced W2）更好。

也就是说「支撑恢复、权重自由」这个指纹的来源是**池化**，不是非平衡性——这正是把
混淆因子消掉之后才能看见的东西。

数据来源（只读 JSON，不重跑实验）：
  results/loss_ladder.json                                    → L1 / L2b / L3
  results/chamfer_pooled.json                                 → L2a
"""
import json
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

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SRC = os.path.join(ROOT, "results", "loss_ladder.json")
SRC_POOLED = os.path.join(ROOT, "results", "chamfer_pooled.json")
OUT = os.path.join(ROOT, "figures", "fig_loss_ladder")

# (来源, JSON 里的键, 图例, 颜色, 标记)
METHODS = [
    ("ladder", "onestep_l2", r"L1  pointwise $L^2$", "#999999", "o"),
    ("pooled", None, "L2a  unbalanced, POOLED over conditions", "#d1495b", "s"),
    ("ladder", "onestep_chamfer", r"L2b  unbalanced, within condition", "#e0a04a", "D"),
    ("ladder", "onestep_balanced", r"L3  balanced, within condition", "#2a6f97", "^"),
]
TICK_LABELS = ["L1", "L2a", "L2b", "L3"]


def _per_cond(src, key, sd):
    """取某个方法在某个种子下的逐条件 rho_j。"""
    if src == "pooled":
        return np.array(POOLED["per_seed"][sd]["rho_per_cond"], dtype=np.float64)
    return np.array(LADDER["per_seed"][sd][key]["rho_per_cond"], dtype=np.float64)


def _summary(src, key):
    if src == "pooled":
        return POOLED["summary"]
    return LADDER["summary"][key]


def main():
    global LADDER, POOLED
    with open(SRC, encoding="utf-8") as f:
        LADDER = json.load(f)
    POOLED = None
    if os.path.exists(SRC_POOLED):
        with open(SRC_POOLED, encoding="utf-8") as f:
            POOLED = json.load(f)

    usable = [m for m in METHODS if m[0] != "pooled" or POOLED is not None]
    tv = np.array(LADDER["tr_var_cond"], dtype=np.float64)
    seeds = sorted(LADDER["per_seed"].keys(), key=int)

    fig, axes = plt.subplots(1, 2, figsize=(9.9, 3.4))

    # ---------------- 左：生成的 vs 真实的逐条件展布 ----------------
    ax = axes[0]
    lo = 0.0
    hi = float(tv.max()) * 1.25
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, zorder=1,
            label="correct conditional law")
    for src, key, label, col, mk in usable:
        gen = [_per_cond(src, key, sd) * tv for sd in seeds]
        gen = np.array(gen)                       # (n_seed, C)
        mu = gen.mean(0)
        se = gen.std(0, ddof=1) / np.sqrt(len(seeds))
        ax.errorbar(tv, mu, yerr=se, fmt=mk + "-", color=col, ms=6, lw=1.6,
                    capsize=3, label=label, zorder=3)
    ax.set_xlabel(r"true conditional spread  $\mathrm{tr}\,\mathrm{Var}(x_1\mid c_j)$")
    ax.set_ylabel(r"generated  $\mathrm{tr}\,\mathrm{Var}(f(x_0,c_j)\mid c_j)$")
    ax.set_title(r"(a)  per-condition spread:  where the matching happens decides",
                 fontsize=9.5)
    ax.legend(fontsize=7.0, loc="upper left", framealpha=0.9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)

    # ---------------- 右：保真度与展布误差 ----------------
    ax = axes[1]
    csw = [_summary(s, k)["csw"] for s, k, _, _, _ in usable]
    csw_e = [_summary(s, k)["csw_std"] for s, k, _, _, _ in usable]
    err = [_summary(s, k)["rho_abs_err"] for s, k, _, _, _ in usable]
    cols = [m[3] for m in usable]
    x = np.arange(len(usable))
    w = 0.36
    b1 = ax.bar(x - w / 2, csw, w, yerr=csw_e, color=cols, alpha=0.95,
                capsize=3, label=r"cSW (conditional sliced $W_2$, $\downarrow$)")
    b2 = ax.bar(x + w / 2, err, w, color=cols, alpha=0.45, hatch="//",
                label=r"mean $|\rho_j-1|$ ($\downarrow$)")
    for b, v in zip(b1, csw):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, "%.3f" % v,
                ha="center", fontsize=7.5)
    for b, v in zip(b2, err):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, "%.3f" % v,
                ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([TICK_LABELS[i] for i in range(len(usable))], fontsize=9)
    ax.set_ylabel("error (lower is better)")
    ax.set_title("(b)  fidelity and per-condition spread error", fontsize=9.5)
    ax.legend(fontsize=7.0, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.tick_params(labelsize=8)
    ax.set_ylim(0, max(max(csw), max(err)) * 1.35)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig("%s.%s" % (OUT, ext), dpi=200, bbox_inches="tight")
        print("wrote %s.%s" % (OUT, ext))
    plt.close(fig)


if __name__ == "__main__":
    main()
