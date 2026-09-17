# -*- coding: utf-8 -*-
"""论文的**主图**：不可能性定理的 α-扫描 + 跨方法族失效地图。

图的结构（两块面板，讲同一件事的两面）

  (a) **α-扫描**：让 ρ 精确扫描 [0,1]，而 D̂ 与 Λ̂ 全程不动。
      —— 这是"任何只看数据的统计量都判定不了一步"的正面证据。
  (b) **跨方法族对照**：报告各受控实现的 ρ 及跨种子标准差。
      —— 这些点是有限的方法样本，不支持“没有中间地带”的普遍结论。

数据来源（全部读 JSON，不重跑实验）
  logs/verify_impossibility.json
  results/method_map.json
  results/method_map_chamfer.json
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

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.4, "lines.linewidth": 1.6,
})

BLUE, CORAL, GRAY, GREEN = "#378ADD", "#D85A30", "#888780", "#639922"


def main():
    with open(os.path.join(ROOT, "logs", "verify_impossibility.json"),
              encoding="utf-8") as f:
        imp = json.load(f)
    with open(os.path.join(ROOT, "results", "method_map.json"),
              encoding="utf-8") as f:
        mm = json.load(f)
    chamfer_path = os.path.join(ROOT, "results", "method_map_chamfer.json")
    chamfer = None
    if os.path.exists(chamfer_path):
        with open(chamfer_path, encoding="utf-8") as f:
            chamfer = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))

    # ---------------- (a) α-扫描 ----------------
    ax = axes[0]
    cv = imp["curve"]
    a = np.array(cv["alpha"], dtype=float)
    rho = np.array(cv["rho_hat"], dtype=float)
    pred = np.array(cv["knn_prediction"], dtype=float)
    D = np.array(cv["D"], dtype=float)
    lam = np.array(cv["Lambda"], dtype=float)

    rho_by_alpha = []
    k0 = str(imp["params"]["KNN_K"][0])
    for alpha in a:
        vals = []
        for rows in imp.get("per_seed", {}).values():
            row = next(r for r in rows if abs(float(r["alpha"]) - alpha) < 1e-12)
            vals.append(float(row["rho"][k0]))
        rho_by_alpha.append(vals)
    rho_sd = np.array([np.std(v, ddof=1) if len(v) > 1 else 0.0
                       for v in rho_by_alpha])
    ax.errorbar(a, rho, yerr=rho_sd, fmt="o-", capsize=2.5, color=BLUE,
                label=r"measured $\hat\rho$ (mean $\pm$ SD)", zorder=3)
    ax.plot(a, pred, "--", color=CORAL, linewidth=1.2,
            label=r"analytic $\alpha^2+(1-\alpha^2)/k$", zorder=2)
    ax.plot(a, a ** 2, ":", color=GRAY, linewidth=1.2,
            label=r"population $\rho^*=\alpha^2$", zorder=2)
    ax.set_xlabel(r"coupling interpolation $\alpha$  (0 = independent, 1 = transport)")
    ax.set_ylabel(r"spread retained  /  data-side statistic")
    ax.set_ylim(-0.05, 1.10)

    ax2 = ax.twinx()
    ax2.plot(a, D, "s-", color=GREEN, linewidth=1.3, markersize=3.5,
             label=r"$\hat D$ (data)", zorder=3)
    ax2.plot(a, lam, "^-", color="#993C1D", linewidth=1.3, markersize=3.5,
             label=r"$\hat\Lambda$ (data)", zorder=3)
    ax2.set_ylabel(r"$\hat D$, $\hat\Lambda$  (data-side)", color="#3B6D11")
    ax2.set_ylim(-0.05, 1.10)
    ax2.grid(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.9)
    ax.set_title(r"(a)  $\rho$ sweeps $[0,1]$ while the data does not move", fontsize=9)

    # ---------------- (b) 跨方法族 ----------------
    ax = axes[1]
    order = [("cfm_indep@1", "indep\nNFE=1"), ("onestep_l2@1", "1-step\n$L^2$"),
             ("onestep_minM@1", "1-step\nmin-of-8"), ("cfm_ot@1", "OT-FM\nNFE=1"),
             ("reflow@1", "reflow\nNFE=1"), ("distill@1", "consistency\nNFE=1"),
             ("cfm_indep@32", "indep\nNFE=32")]
    labels, vals, errs, cols = [], [], [], []
    for key, lab in order:
        s = mm["summary"][key]
        labels.append(lab)
        vals.append(s["rho"])
        errs.append(s.get("rho_std", 0.0))
        cols.append(CORAL if s["rho"] < 0.3 else BLUE)
    if chamfer is not None:
        labels.append("1-step\nChamfer")
        vals.append(chamfer["summary"]["rho"])
        errs.append(chamfer["summary"].get("rho_std", 0.0))
        cols.append(BLUE)
    y = np.arange(len(labels))
    ax.barh(y, vals, xerr=errs, color=cols, height=0.62, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel(r"realized conditional spread ratio  $\hat\rho$")
    ax.set_xlim(0, max(1.08, max(v + e for v, e in zip(vals, errs)) + 0.10))
    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, "%.3f" % v, va="center", fontsize=6.5)
    ax.set_title(r"(b)  controlled method comparison", fontsize=9)

    fig.tight_layout()
    outdir = os.path.join(ROOT, "figures")
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, "fig_impossibility.%s" % ext)
        fig.savefig(p)
        print("wrote %s" % p)
    plt.close(fig)


if __name__ == "__main__":
    main()
