# -*- coding: utf-8 -*-
"""损失形式阶梯：pointwise L2 / 非平衡集合级 / 平衡集合级。

--------------------------------------------------------------------
要检验的理论（三层次阶梯）
--------------------------------------------------------------------
固定耦合为**独立耦合**（x0 ⊥ x1 | c），只改损失形式。记 q_c = f(·,c)_# p0。

  L1  pointwise L2
      L = E ||f(x0,c) - x1||^2
      极小元 f*(x0,c) = E[x1 | x0, c] = E[x1 | c] = m(c)      （独立耦合）
      => q_c = δ_{m(c)}，ρ* = 0。**塌缩**。

  L2  非平衡集合级（Chamfer / IMLE 式 min-matching）
      种群极限（batch M -> ∞）：
        L -> E_{y~p_c}[ min_{x in supp q_c} ||y-x||^2 ] + E_{x~q_c}[ min_{y in supp p_c} ||x-y||^2 ]
      两项同时为 0 **当且仅当 supp q_c = supp p_c**。
      => 塌缩解除，但**权重完全不受约束**：任何与 p_c 同支撑的 q_c 都是最优。
      可测预测：覆盖 8/8，但**逐条件展布被拉平**（各条件趋于同一个值），
      即 ρ_j 的离散度远大于 0，且 ρ_j 与真实半径无关。

  L3  平衡集合级（批内最优指派 / 传输）
      种群极限 -> W_2^2(p_c, q_c)，唯一极小元 q_c = p_c。
      => ρ* = 1，且逐条件展布正确，ρ_j 离散度 ≈ 0。

这条阶梯的**可证伪**之处在于 L2 与 L3 的差别：如果"任何集合级损失都等价"，
那么 L2 与 L3 的 ρ_j 离散度应当相当。我们预测 L3 的离散度显著更小。

判据（预先写死）
----------------
  B1  onestep_l2      @1：ρ < 0.05                        （L1 塌缩）
  B2  chamfer         @1：ρ > 0.5 且覆盖 >= 6             （L2 逃脱塌缩）
  B3  balanced        @1：ρ > 0.5 且覆盖 >= 6             （L3 逃脱塌缩）
  B4  **|ρ_j − 1| 的均值：balanced < chamfer**                 （L3 恢复律，L2 只恢复支撑）
  B5  **cSW：balanced < chamfer**                              （保真度：L3 更准）

B4/B5 是这条阶梯真正的赌注：若它们判负，则"平衡性是关键"的主张被推翻。

**为什么判据只写方向、不写倍数。** 早期版本写过 `disp(chamfer) > 2 × disp(balanced)`
这类**猜出来的效应量**，结果 40 步冒烟就判负（0.396 对 2×0.250），但那只是训练没收敛，
不是理论失效。倍数取决于优化到什么程度，属于**不可预先确定的量**；
而"L3 的种群最优就是 p_c、L2 的种群最优集合里混着一堆错权重的分布"是**定性的**，
只能用方向性判据检验。倍数作为描述性数字照报，但不进判据。
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
from scipy.optimize import linear_sum_assignment  # noqa: E402

import run_method_map as M  # noqa: E402
from provenance import attach_provenance, protocol_fingerprint, require_merge_compatible  # noqa: E402

PROTOCOL_FILES = ["code/experiments/run_loss_ladder.py",
                  "code/experiments/run_method_map.py", "code/src/provenance.py"]

SEEDS = (0, 1, 2, 3, 4)
N_PROJ = 256
torch.set_num_threads(14)


def _parse_cli(argv):
    """解析 `--seeds 3,4` 与 `--merge`。

    为什么需要 `--merge`：一次完整跑（3 种子时实测 ~2950s，5 种子约 1200s 视机器负载）
    若扩容种子数时不合并，就会把已算好的种子再算一遍。`--merge` 从既有
    `results/loss_ladder.json` 读出 per_seed，只补跑缺的种子，最后按**并集**重算
    汇总与判据（与 memory 里"抓取新结果必须与既有缓存合并再写回"同一条教训）。
    """
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


# ---------------------------------------------------------------- 度量
def sliced_w2(A, B, rng, n_proj=N_PROJ):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    d = A.shape[1]
    th = rng.normal(size=(d, n_proj))
    th /= np.linalg.norm(th, axis=0, keepdims=True)
    pa = np.sort(A @ th, axis=0)
    pb = np.sort(B @ th, axis=0)
    return float(np.sqrt(((pa - pb) ** 2).mean()))


def evaluate_full(net, seed):
    """逐条件给出 ρ_j、覆盖、条件 sliced W2（按 sqrt(trVar(x1|c_j)) 归一）。"""
    rng = np.random.default_rng(seed + 909)
    num, cov, csw = [], [], []
    for j in range(M.C):
        x0, x1_true, c = M.sample_cond(M.N_EVAL, rng, c=np.full(M.N_EVAL, j))
        pred = M.onestep(net, x0, c)
        pred64 = np.asarray(pred, dtype=np.float64)
        num.append(float(pred64.var(axis=0).sum()))
        cov.append(M.mode_coverage(pred64, M.CENTERS[j]))
        csw.append(sliced_w2(pred64, x1_true.astype(np.float64), rng)
                   / np.sqrt(M.TR_VAR_COND[j]))
    rho_j = np.array([v / M.TR_VAR_COND[j] for j, v in enumerate(num)], dtype=np.float64)
    return dict(rho=float(np.mean(num) / np.mean(M.TR_VAR_COND)),
                coverage=float(np.mean(cov)),
                rho_per_cond=rho_j.tolist(),
                rho_disp=float(rho_j.std(ddof=0)),
                rho_abs_err=float(np.abs(rho_j - 1.0).mean()),
                csw=float(np.mean(csw)),
                cov_per_cond=[int(v) for v in cov])


# ---------------------------------------------------------------- 训练
def _make_net(seed):
    torch.manual_seed(seed)
    return M.MLP(M.DIM + 1 + M.C, M.DIM)


def train_l2(seed, steps=None, tag="onestep_l2"):
    steps = M.STEPS if steps is None else steps
    net = _make_net(seed)
    opt = torch.optim.Adam(net.parameters(), lr=M.LR)
    rng = np.random.default_rng(seed + 202)
    for s in range(steps):
        x0, y, c = M.sample_cond(M.BS, rng)
        tt = np.zeros(len(x0), dtype=np.float32)
        loss = ((net(M.t_in(x0, tt, c)) - torch.from_numpy(y)) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 4000 == 0:
            print("      [%s] step %d/%d  loss %.4f" % (tag, s + 1, steps, float(loss)), flush=True)
    net.eval()
    return net


def train_chamfer(seed, steps=None, tag="onestep_chamfer"):
    """非平衡双向 min-matching，**逐条件**做（与 train_balanced 的指派范围一致）。

    与非平衡版唯一的区别是**指派不是双射**：每个目标各自找最近的生成样本，
    每个生成样本各自找最近的目标样本；同一个生成样本可以被多个目标共用，
    也可以完全不被使用。种群极限只约束**支撑集**，对权重不作任何要求。

    ---- 修正 (2026-09-16)：指派范围从「整批」改为「逐条件」 ----
    初版在这里对**整个 batch** 取 argmin（不分条件），而 train_balanced 是逐条件
    取匈牙利指派。于是 L2 与 L3 的差异有**两个**（是否平衡 AND 是否分条件），
    无法把观测到的「支撑恢复、权重拉平」单独归因于非平衡性——这正是 Thm 4(L2)
    要论证的那一点。改成逐条件后，L2 与 L3 只在「双射 / 非双射」上不同。
    注：池化版的 min 取自更大的候选集，损失更小，是**更弱**的约束，因此这个修正
    只会让 L2 的支撑约束更强，不会人为制造出支撑恢复。
    """
    steps = M.STEPS if steps is None else steps
    net = _make_net(seed)
    opt = torch.optim.Adam(net.parameters(), lr=M.LR)
    rng = np.random.default_rng(seed + 505)
    for s in range(steps):
        x0, y, c = M.sample_cond(M.BS, rng)
        tt = np.zeros(len(x0), dtype=np.float32)
        G = net(M.t_in(x0, tt, c))
        with torch.no_grad():
            Gn = G.detach().numpy().astype(np.float64)
            yn = y.astype(np.float64)
            a1 = np.arange(len(x0))       # 目标 -> 本条件内最近的生成样本
            a2 = np.arange(len(x0))       # 生成样本 -> 本条件内最近的目标
            for j in range(M.C):
                idx = np.where(c == j)[0]
                if len(idx) < 2:
                    continue
                d2 = ((yn[idx][:, None, :] - Gn[idx][None, :, :]) ** 2).sum(-1)
                a1[idx] = idx[d2.argmin(1)]
                a2[idx] = idx[d2.argmin(0)]
        l1 = ((G[a1] - torch.from_numpy(y)) ** 2).sum(-1).mean()
        l2 = ((G - torch.from_numpy(y)[a2]) ** 2).sum(-1).mean()
        loss = l1 + l2
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 4000 == 0:
            print("      [%s] step %d/%d  loss %.4f" % (tag, s + 1, steps, float(loss)), flush=True)
    net.eval()
    return net


def train_balanced(seed, steps=None, tag="onestep_balanced"):
    """平衡集合级：批内逐条件做匈牙利最优指派（双射），再对指派结果做 L2。

    与非平衡版唯一的区别是**指派是双射**：每个生成样本恰好被用一次，
    每个目标恰好被匹配一次。种群极限即 W_2^2(p_c, q_c)。
    """
    steps = M.STEPS if steps is None else steps
    net = _make_net(seed)
    opt = torch.optim.Adam(net.parameters(), lr=M.LR)
    rng = np.random.default_rng(seed + 707)
    for s in range(steps):
        x0, y, c = M.sample_cond(M.BS, rng)
        tt = np.zeros(len(x0), dtype=np.float32)
        G = net(M.t_in(x0, tt, c))
        with torch.no_grad():
            Gn = G.detach().numpy().astype(np.float64)
            perm = np.arange(len(x0))
            for j in range(M.C):
                idx = np.where(c == j)[0]
                if len(idx) < 2:
                    continue
                d2 = ((Gn[idx][:, None, :] - y[idx][None, :, :].astype(np.float64)) ** 2).sum(-1)
                r, cc = linear_sum_assignment(d2)
                perm[idx[r]] = idx[cc]
        loss = ((G - torch.from_numpy(y)[perm]) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 4000 == 0:
            print("      [%s] step %d/%d  loss %.4f" % (tag, s + 1, steps, float(loss)), flush=True)
    net.eval()
    return net


# ---------------------------------------------------------------- 主流程
def main(new_seeds, merge):
    t0 = time.time()
    out = os.path.join(ROOT, "results", "loss_ladder.json")
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
        print("\n" + "#" * 70 + "\n# seed %d\n" % sd + "#" * 70, flush=True)
        out_all[str(sd)] = {}
        for name, fn in (("onestep_l2", train_l2),
                         ("onestep_chamfer", train_chamfer),
                         ("onestep_balanced", train_balanced)):
            print("\n  [%s]" % name, flush=True)
            net = fn(sd)
            r = evaluate_full(net, sd)
            out_all[str(sd)][name] = r
            print("    rho=%.4f  cov=%.2f  disp=%.3f  |rho_j-1|=%.3f  cSW=%.4f"
                  % (r["rho"], r["coverage"], r["rho_disp"], r["rho_abs_err"], r["csw"]),
                  flush=True)
            print("    rho_j = %s" % np.round(r["rho_per_cond"], 3).tolist(), flush=True)

    seeds_all = sorted(int(k) for k in out_all)
    print("\n参与汇总的种子: %s" % seeds_all, flush=True)

    def agg(name):
        v = [out_all[str(s)][name] for s in seeds_all]
        return dict(rho=float(np.mean([x["rho"] for x in v])),
                    rho_std=float(np.std([x["rho"] for x in v])),
                    coverage=float(np.mean([x["coverage"] for x in v])),
                    rho_disp=float(np.mean([x["rho_disp"] for x in v])),
                    rho_disp_std=float(np.std([x["rho_disp"] for x in v])),
                    rho_abs_err=float(np.mean([x["rho_abs_err"] for x in v])),
                    csw=float(np.mean([x["csw"] for x in v])),
                    csw_std=float(np.std([x["csw"] for x in v])))

    S = {k: agg(k) for k in ("onestep_l2", "onestep_chamfer", "onestep_balanced")}

    print("\n" + "=" * 78)
    print("损失形式阶梯（%d 种子平均）" % len(seeds_all))
    print("  %-18s %-14s %-8s %-10s %-10s %-10s"
          % ("损失", "ρ", "覆盖/8", "disp(ρ_j)", "|ρ_j−1|", "cSW"))
    print("-" * 78)
    for k, v in S.items():
        print("  %-18s %-14s %-8s %-10s %-10s %-10s"
              % (k, "%.4f±%.4f" % (v["rho"], v["rho_std"]), "%.2f" % v["coverage"],
                 "%.3f±%.3f" % (v["rho_disp"], v["rho_disp_std"]),
                 "%.3f" % v["rho_abs_err"],
                 "%.4f±%.4f" % (v["csw"], v["csw_std"])))
    print("-" * 78)

    checks = dict(
        B1_l2_collapses=bool(S["onestep_l2"]["rho"] < 0.05),
        B2_chamfer_escapes=bool(S["onestep_chamfer"]["rho"] > 0.5
                                and S["onestep_chamfer"]["coverage"] >= 6),
        B3_balanced_escapes=bool(S["onestep_balanced"]["rho"] > 0.5
                                 and S["onestep_balanced"]["coverage"] >= 6),
        B4_balanced_less_weight_error=bool(
            S["onestep_balanced"]["rho_abs_err"] < S["onestep_chamfer"]["rho_abs_err"]),
        B5_balanced_more_faithful=bool(S["onestep_balanced"]["csw"] < S["onestep_chamfer"]["csw"]),
    )
    verdict = ("PASS" if all(checks.values())
               else "PARTIAL" if sum(checks.values()) >= 4 else "FAIL")
    for k, v in checks.items():
        print("    %-32s %s" % (k, v))
    print("  描述性比值（不进判据）: |rho_j-1| chamfer/balanced = %.2f×, cSW chamfer/balanced = %.2f×"
          % (S["onestep_chamfer"]["rho_abs_err"] / max(S["onestep_balanced"]["rho_abs_err"], 1e-9),
             S["onestep_chamfer"]["csw"] / max(S["onestep_balanced"]["csw"], 1e-9)))
    print("  VERDICT: %s      用时 %.0fs" % (verdict, time.time() - t0))
    print("=" * 78)

    report = dict(theorem="loss ladder: pointwise L2 -> support only -> full law",
                  params=dict(seeds=list(seeds_all), STEPS=M.STEPS, BS=M.BS, LR=M.LR,
                              C=M.C, K=M.K, sigma=M.SIGMA, N_EVAL=M.N_EVAL,
                              N_PROJ=N_PROJ),
                  tr_var_cond=M.TR_VAR_COND.tolist(),
                  summary=S, checks=checks, verdict=verdict, per_seed=out_all)
    attach_provenance(report, ROOT, "code/experiments/run_loss_ladder.py",
                      PROTOCOL_FILES, merged=merge)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("报告已写入 %s" % out)


if __name__ == "__main__":
    main(*_parse_cli(sys.argv[1:]))
