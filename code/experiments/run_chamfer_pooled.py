# -*- coding: utf-8 -*-
"""对照实验：**池化**（跨条件）的非平衡 Chamfer —— 即"实践里真正会这么写"的那一版。

为什么单列一个脚本
------------------
`run_loss_ladder.py` 的 `train_chamfer` 已修正为**逐条件**匹配，与 Thm 4(L2) 的表述
（$p_c$、$q_c$ 逐条件）严格对齐。但 Chamfer / IMLE 在实践里通常是在**整个 batch** 上
算的，而池化会引入一条额外的失效通道：

    条件 c' 的生成样本可以充当条件 c 的目标的"最近点" ⟹
    模型可以在损失很小的情况下**跨条件共用样本** ⟹
    每个条件的生成集合变成全集的子集 ⟹ 条件展布被拉平。

这恰好就是"支撑恢复、权重自由"的观测量。所以论文需要**两行**：
  (a) 池化版（实践写法）：展布被拉平；
  (b) 逐条件版（Thm 4 的设定）：展布基本恢复。
两者之差把"池化"这一个变量单独隔离出来。

本脚本只跑 (a)，写 `results/chamfer_pooled.json`；其余三行（L1 / L2 逐条件 / L3 平衡）
由 `run_loss_ladder.py` 写 `results/loss_ladder.json`。聚合口径与 `run_loss_ladder.py`
的 `agg()` 完全一致（rho 取种子均值，disp/abs_err/csw 取种子均值）。

注：本脚本同时是一次**复现性自检**——若 `results/loss_ladder_pooled_20260916.json`
（修正前的原始产物）在位，脚本会把新旧数字逐字段比对并打印差异。
"""
import json
import os
import sys
import time

# Windows/Anaconda：numpy 与 torch 各带一份 libiomp5md.dll，重复初始化会让进程在
# 训练**中途**以 exit code 3 崩溃（OMP: Error #15）。必须在 import numpy/torch
# **之前**设置。此前只有 code/_run_pipeline.py 为子进程设过它，直接跑本脚本会崩。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


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
sys.path.insert(0, os.path.join(ROOT, "code", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_method_map as M  # noqa: E402
import run_loss_ladder as L  # noqa: E402

SEEDS = L.SEEDS
torch.set_num_threads(14)


def train_chamfer_pooled(seed, steps=None, tag="chamfer_pooled"):
    """非平衡双向 min-matching，**在整个 batch 上**取 argmin（修正前的实现）。

    保留它是为了做上面说的对照；它**不是** Thm 4(L2) 的设定。
    """
    steps = M.STEPS if steps is None else steps
    net = L._make_net(seed)
    opt = torch.optim.Adam(net.parameters(), lr=M.LR)
    rng = np.random.default_rng(seed + 505)          # 与 run_loss_ladder 同种子
    for s in range(steps):
        x0, y, c = M.sample_cond(M.BS, rng)
        tt = np.zeros(len(x0), dtype=np.float32)
        G = net(M.t_in(x0, tt, c))
        with torch.no_grad():
            Gn = G.detach().numpy().astype(np.float64)
            yn = y.astype(np.float64)
            d2 = ((yn[:, None, :] - Gn[None, :, :]) ** 2).sum(-1)
            a1 = d2.argmin(1)
            a2 = d2.argmin(0)
        l1 = ((G[a1] - torch.from_numpy(y)) ** 2).sum(-1).mean()
        l2 = ((G - torch.from_numpy(y)[a2]) ** 2).sum(-1).mean()
        loss = l1 + l2
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 4000 == 0:
            print("      [%s] step %d/%d  loss %.4f"
                  % (tag, s + 1, steps, float(loss)), flush=True)
    net.eval()
    return net


def main(new_seeds, merge):
    """`--seeds a,b --merge` 语义与 run_loss_ladder.py 相同（复用它的解析器）。

    目的：Table 1 里 L2a(pooled) 与 L2b/L3 必须**同种子数**对比，否则 ± 不可比。
    """
    t0 = time.time()
    out = os.path.join(ROOT, "results", "chamfer_pooled.json")
    out_all = {}
    if merge and os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            out_all = dict(json.load(f).get("per_seed", {}))
        print("merge: 载入既有种子 %s" % sorted(out_all, key=int), flush=True)
    for sd in new_seeds:
        if str(sd) in out_all:
            print("\nskip seed %d（结果已在报告中）" % sd, flush=True)
            continue
        print("\n" + "#" * 70 + "\n# seed %d\n" % sd + "#" * 70, flush=True)
        net = train_chamfer_pooled(sd)
        r = L.evaluate_full(net, sd)
        out_all[str(sd)] = r
        print("    rho=%.4f  cov=%.2f  disp=%.3f  |rho_j-1|=%.3f  cSW=%.4f"
              % (r["rho"], r["coverage"], r["rho_disp"], r["rho_abs_err"], r["csw"]),
              flush=True)
        print("    rho_j = %s" % np.round(r["rho_per_cond"], 3).tolist(), flush=True)

    seeds_all = sorted(int(k) for k in out_all)
    print("\n参与汇总的种子: %s" % seeds_all, flush=True)

    def _summ(seed_list):
        v = [out_all[str(s)]["onestep_chamfer"] if "onestep_chamfer" in out_all[str(s)]
             else out_all[str(s)] for s in seed_list]
        return dict(rho=float(np.mean([x["rho"] for x in v])),
                    rho_std=float(np.std([x["rho"] for x in v])),
                    coverage=float(np.mean([x["coverage"] for x in v])),
                    rho_disp=float(np.mean([x["rho_disp"] for x in v])),
                    rho_disp_std=float(np.std([x["rho_disp"] for x in v])),
                    rho_abs_err=float(np.mean([x["rho_abs_err"] for x in v])),
                    csw=float(np.mean([x["csw"] for x in v])),
                    csw_std=float(np.std([x["csw"] for x in v])))

    summary = _summ(seeds_all)

    print("\n" + "=" * 78)
    print("池化非平衡 Chamfer（%d 种子平均）" % len(seeds_all))
    print("  rho=%.4f±%.4f  cov=%.2f  disp=%.3f  |rho_j-1|=%.3f  cSW=%.4f±%.4f"
          % (summary["rho"], summary["rho_std"], summary["coverage"],
             summary["rho_disp"], summary["rho_abs_err"],
             summary["csw"], summary["csw_std"]))
    print("  用时 %.0fs" % (time.time() - t0))
    print("=" * 78)

    report = dict(variant="onestep_chamfer_pooled",
                  note="argmin taken over the whole batch (the practically used form); "
                       "run_loss_ladder.py's onestep_chamfer matches within condition",
                  params=dict(seeds=list(seeds_all), STEPS=M.STEPS, BS=M.BS, LR=M.LR,
                              C=M.C, K=M.K, sigma=M.SIGMA, N_EVAL=M.N_EVAL,
                              N_PROJ=L.N_PROJ),
                  tr_var_cond=M.TR_VAR_COND.tolist(),
                  summary=summary,
                  per_seed=out_all)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("报告已写入 %s" % out)

    # ---------------- 复现性自检：与修正前的原始产物逐字段比对 ----------------
    # 注意：自检必须在**共同种子**上做。原始产物只有种子 0/1/2，若直接拿 5 种子的
    # 汇总去比，差异里混进了"多算了两个种子"这一无关因素，会误报复现失败。
    old = os.path.join(ROOT, "results", "loss_ladder_pooled_20260916.json")
    if os.path.exists(old):
        with open(old, encoding="utf-8") as f:
            orep = json.load(f)
        o = orep["summary"]["onestep_chamfer"]
        o_seeds = [int(s) for s in orep.get("params", {}).get("seeds", seeds_all)]
        common = [s for s in seeds_all if s in set(o_seeds)]
        print("\n复现性自检（vs 修正前的原始产物，限共同种子 %s）:" % common)
        worst = 0.0
        for k, val in _summ(common).items():
            if k in o:
                d = abs(float(val) - float(o[k]))
                worst = max(worst, d)
                print("  %-14s new=%.10f  old=%.10f  |diff|=%.3e" % (k, val, o[k], d))
        print("  最大字段差 = %.3e" % worst)


if __name__ == "__main__":
    main(*L._parse_cli(sys.argv[1:]))
