# -*- coding: utf-8 -*-
"""补充第 7 个方法族：**双向 Chamfer（集合级）**损失的一步生成器。

为什么必须补这一个
------------------
`run_method_map.py` 里的 `onestep_minM`（min-of-M / IMLE）与本脚本的 Chamfer 机制不同，
不能互相替代：

    min-of-M：每个**目标**只在**它自己**的 M 个候选里挑最近的一个，
              M 条分支各自独立采样（original IMLE semantics）。
    双向 Chamfer：第一项在**整个 batch** 上为每个目标挑最近的生成样本，
              第二项反向。384 个样本、8 个模态，于是"最近的生成样本"多半
              来自正确模态，不同目标被指派到不同生成样本，网络被迫**铺开**。

注意（2026-09-16 修正）：min-of-M 早先的实现把同一个源噪声 repeat 了 M 次，
在确定性的网络下 M 个候选**恒等**，损失逐比特退化为点式 L2。因此早先
"minM 与 onestep_l2 的 loss 完全一致（4.5537 对 4.5537）"是**代码缺陷**的产物，
不是 IMLE 的性质。修正后 min-of-M 的 M 条候选各自独立采样，是真正的集合级目标，
其结果以 `run_method_map.py` 重跑后的 `results/method_map.json` 为准。

本脚本训练的是**整批池化**版 Chamfer（第一项的 argmin 覆盖整个 batch），
这正是外部校核 *From Flow to One Step* (2603.09415) 与多数实现里的写法。
它与"逐条件"版 Chamfer 的差别在论文 Sec. 5.4 中作为单独变量隔离讨论。

本脚本只训练这一个方法族，协议与 `run_method_map.py` **完全一致**
（同几何族、同架构、同 bs/步数/lr、同评估），结果可直接并入失效地图。
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_method_map as M  # noqa: E402
from provenance import (attach_provenance, protocol_fingerprint,  # noqa: E402
                        require_merge_compatible)

PROTOCOL_FILES = ["code/experiments/run_chamfer_supplement.py",
                  "code/experiments/run_method_map.py", "code/src/provenance.py"]

SEEDS = (0, 1, 2, 3, 4)


def _parse_cli(argv):
    """`--seeds a,b --merge`；语义与 run_method_map.py / run_loss_ladder.py 的同名函数一致。"""
    seeds = list(SEEDS)
    merge = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--merge":
            merge = True
        elif a == "--seeds":
            i += 1
            seeds = [int(x) for x in argv[i].replace(",", " ").split()]
        elif a.startswith("--seeds="):
            seeds = [int(x) for x in a.split("=", 1)[1].replace(",", " ").split()]
        else:
            raise SystemExit("unknown arg %r (use --seeds a,b --merge)" % a)
        i += 1
    return seeds, merge


torch.set_num_threads(14)


def train_chamfer(sample_fn, seed, steps=None, bs=None, lr=M.LR, tag="chamfer"):
    steps = M.STEPS if steps is None else steps
    bs = M.BS if bs is None else bs
    torch.manual_seed(seed)
    net = M.MLP(M.DIM + 1 + M.C, M.DIM)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 505)
    t0 = time.time()
    for s in range(steps):
        x0, y, c = sample_fn(bs, rng)
        tt = np.zeros(len(x0), dtype=np.float32)
        G = net(M.t_in(x0, tt, c))                       # (bs, d)，带梯度
        with torch.no_grad():
            Gn = G.detach().numpy()
            d2 = ((y[:, None, :] - Gn[None, :, :]) ** 2).sum(-1)   # (bs, bs)
            a1 = d2.argmin(1)                            # 目标 i -> 最近生成样本
            a2 = d2.argmin(0)                            # 生成样本 j -> 最近目标
        l1 = ((G[a1] - torch.from_numpy(y)) ** 2).sum(-1).mean()
        l2 = ((G - torch.from_numpy(y)[a2]) ** 2).sum(-1).mean()
        loss = l1 + l2
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % 4000 == 0:
            print("      [%s] step %d/%d  loss %.4f  (l1 %.4f / l2 %.4f)  (%.0fs)"
                  % (tag, s + 1, steps, float(loss), float(l1), float(l2),
                     time.time() - t0), flush=True)
    net.eval()
    return net


def main(new_seeds, merge):
    t0 = time.time()
    out = os.path.join(ROOT, "results", "method_map_chamfer.json")
    protocol_id, _ = protocol_fingerprint(ROOT, PROTOCOL_FILES)
    out_all = {}
    if merge and os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            existing = json.load(f)
        require_merge_compatible(existing, protocol_id, out)
        out_all = dict(existing.get("per_seed", {}))
        print("merge: 载入既有种子 %s" % sorted(out_all, key=int), flush=True)
    for sd in new_seeds:
        if str(sd) in out_all:
            print("\nskip seed %d（结果已在报告中）" % sd, flush=True)
            continue
        print("\n# seed %d" % sd, flush=True)
        net = train_chamfer(lambda bs, rng: M.sample_cond(bs, rng), sd,
                            tag="onestep_chamfer")
        res = M.evaluate(net, sd, 1, kind="one")
        out_all[str(sd)] = res
        print("  chamfer@1  rho=%.4f  cov=%.2f  rho_per_cond=%s  cov_per_cond=%s"
              % (res["rho"], res["coverage"],
                 np.round(res["rho_per_cond"], 4).tolist(), res["cov_per_cond"]),
              flush=True)

    seeds_all = sorted(int(k) for k in out_all)
    print("\n参与汇总的种子: %s" % seeds_all, flush=True)

    rhos = [out_all[str(s)]["rho"] for s in seeds_all]
    covs = [out_all[str(s)]["coverage"] for s in seeds_all]
    summary = dict(rho=float(np.mean(rhos)), rho_std=float(np.std(rhos)),
                   coverage=float(np.mean(covs)),
                   rho_per_seed=[float(r) for r in rhos])
    ok = summary["rho"] > M.CRIT["C7_MINM_N1_RHO_MIN"] and \
        summary["coverage"] >= M.CRIT["C7_MINM_N1_COV_MIN"]
    print("\n" + "=" * 70)
    print("onestep_chamfer@1  rho = %.4f ± %.4f   覆盖 = %.2f"
          % (summary["rho"], summary["rho_std"], summary["coverage"]))
    print("判据（沿用 C7）: rho > %.2f 且 覆盖 >= %d   -> %s"
          % (M.CRIT["C7_MINM_N1_RHO_MIN"], M.CRIT["C7_MINM_N1_COV_MIN"],
             "PASS" if ok else "FAIL"))
    print("用时 %.0fs" % (time.time() - t0))
    print("=" * 70)

    report = dict(method="onestep_chamfer",
                  params=dict(STEPS=M.STEPS, BS=M.BS, LR=M.LR, seeds=list(seeds_all)),
                  criteria={k: M.CRIT[k] for k in
                            ("C7_MINM_N1_RHO_MIN", "C7_MINM_N1_COV_MIN")},
                  summary=summary, per_seed=out_all,
                  passes_C7=bool(ok))
    attach_provenance(report, ROOT, "code/experiments/run_chamfer_supplement.py",
                      PROTOCOL_FILES, merged=merge)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("报告已写入 %s" % out)


if __name__ == "__main__":
    main(*_parse_cli(sys.argv[1:]))
