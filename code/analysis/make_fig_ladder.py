# -*- coding: utf-8 -*-
"""Loss-ladder figure (paper Figure 2).

The left panel is the whole point of this figure: **generated per-condition
spread vs true per-condition spread**.
  * on the diagonal      = recovered per-condition total variance (not the same as
                           recovering the full conditional law);
  * flattened to a **horizontal line** = only the support was recovered; the
                           cross-condition weights are wrong;
  * shrunk to the origin = collapse.

After revision (2026-09-16) there are **four** curves here, not three. The extra
one is the version of unbalanced Chamfer changed to **per-condition** matching,
strictly aligned with the statement of Thm 4 (L2). The empirical conclusion is
therefore rewritten:

  * the pooled version (L2a, the usual practical implementation) does flatten to a
    horizontal line;
  * the per-condition version (L2b) essentially returns to the diagonal;
  * the balanced version (L3) sits near the diagonal and has better fidelity
    (conditional sliced W2).

In other words, the signature "support recovered, weights free" comes from
**pooling**, not from unbalance -- this is exactly what becomes visible only after
removing the confound.

Data sources (read-only JSON, no experiment re-run):
  results/loss_ladder.json                                    -> L1 / L2b / L3
  results/chamfer_pooled.json                                 -> L2a
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

# (source, JSON key, legend, color, marker)
METHODS = [
    ("ladder", "onestep_l2", r"L1  pointwise $L^2$", "#999999", "o"),
    ("pooled", None, "L2a  unbalanced, POOLED over conditions", "#d1495b", "s"),
    ("ladder", "onestep_chamfer", r"L2b  unbalanced, within condition", "#e0a04a", "D"),
    ("ladder", "onestep_balanced", r"L3  balanced, within condition", "#2a6f97", "^"),
]
TICK_LABELS = ["L1", "L2a", "L2b", "L3"]


def _per_cond(src, key, sd):
    """Get a method's per-condition rho_j for a given seed."""
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

    # ---------------- left: generated vs true per-condition spread ----------------
    ax = axes[0]
    lo = 0.0
    hi = float(tv.max()) * 1.25
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, zorder=1,
            label="matched conditional total variance")
    for src, key, label, col, mk in usable:
        gen = [_per_cond(src, key, sd) * tv for sd in seeds]
        gen = np.array(gen)                       # (n_seed, C)
        mu = gen.mean(0)
        sd = gen.std(0, ddof=1)
        ax.errorbar(tv, mu, yerr=sd, fmt=mk + "-", color=col, ms=6, lw=1.6,
                    capsize=3, label=label, zorder=3)
    ax.set_xlabel(r"true conditional spread  $\mathrm{tr}\,\mathrm{Var}(x_1\mid c_j)$")
    ax.set_ylabel(r"generated  $\mathrm{tr}\,\mathrm{Var}(f(x_0,c_j)\mid c_j)$")
    ax.set_title(r"(a)  per-condition spread in the controlled ring task",
                 fontsize=9.5)
    ax.legend(fontsize=7.0, loc="upper left", framealpha=0.9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)

    # ---------------- right: fidelity and spread error ----------------
    ax = axes[1]
    csw = [_summary(s, k)["csw"] for s, k, _, _, _ in usable]
    csw_e = [_summary(s, k)["csw_std"] for s, k, _, _, _ in usable]
    err = [_summary(s, k)["rho_abs_err"] for s, k, _, _, _ in usable]
    err_e = []
    for src, key, _, _, _ in usable:
        vals = []
        for sd in seeds:
            rj = _per_cond(src, key, sd)
            vals.append(float(np.abs(rj - 1.0).mean()))
        err_e.append(float(np.std(vals, ddof=1)))
    cols = [m[3] for m in usable]
    x = np.arange(len(usable))
    w = 0.36
    b1 = ax.bar(x - w / 2, csw, w, yerr=csw_e, color=cols, alpha=0.95,
                capsize=3, label=r"cSW (conditional sliced $W_2$, $\downarrow$)")
    b2 = ax.bar(x + w / 2, err, w, yerr=err_e, color=cols, alpha=0.45,
                hatch="//", capsize=3, label=r"mean $|\rho_j-1|$ ($\downarrow$)")
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
