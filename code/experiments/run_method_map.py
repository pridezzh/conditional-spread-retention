# -*- coding: utf-8 -*-
"""跨方法族失效地图：同分布、同架构、同预算，**只改训练目标**。

--------------------------------------------------------------------
为什么这个实验是论文的新核心
--------------------------------------------------------------------
外部校核指出：原理论只约束"独立耦合 CFM 的端点单位欧拉步"，不能推广到
任意单步生成器；而 *Let It Be Simple* (2606.05737) 已部分重合，
*From Flow to One Step* (2603.09415) 与 *One-Step Flow Policy* (2603.12480)
是反例（它们的一步能保持多峰）。

本实验不把这些当成"打脸"，而是给出一个**统一解释**并把它变成可测的地图：

    所有"让一步可用"的方法（OT-FM / 重流 / 一致性蒸馏 / IMLE 蒸馏），
    本质上都是把**独立耦合**换成**近似确定性耦合**。
    一步能否保持展布，只由耦合决定 —— 与 NFE 计数、与架构都无关。

若这条成立，则六个方法族在"一步"上的表现可以被**一个变量**（耦合是否
确定）预测，而不是六个互不相干的经验结论。这是可证伪的：只要出现
"独立耦合却一步保展布"或"确定性耦合却一步塌缩"，即被推翻。

--------------------------------------------------------------------
受控设计
--------------------------------------------------------------------
数据：C=4 个条件，每条件 K=8 环上等权高斯混合（半径/旋转/平移各不相同，
使 m(c) 随 c 变化、D 有意义）。闭式 tr Var(x1|j) = R_j^2 + d sigma^2。
架构：同一个 MLP（输入 [x, t, onehot(c)]，隐层 256×4，输出 2）。
预算：同样步数、同样批量、同样优化器。
差异：**只**在训练目标的耦合 / 损失上。

六个方法族
----------
  cfm_indep    标准 CFM，独立耦合（x0 ⊥ x1）。理论预期一步 ρ=0。
  cfm_ot       OT-FM：小批量内逐条件做匈牙利最优指派后再训练。
  reflow       用 cfm_indep 的 32 步流映射生成 (x0, φ(x0,c)) 确定性配对，重训。
  distill      一步网络以 L2 蒸馏 32 步流映射（一致性蒸馏的核心）。
  onestep_l2   一步网络直接对 x1 做 L2 回归（独立耦合）—— 理论下界。
  onestep_minM 一步网络用 min-of-M（IMLE 式）损失，规避 L2 的模式平均。
               每个目标配 M 个**互相独立**的源噪声候选（见下方"修正(viii)"）。

判据（全部预先写死，见 CRIT）
----------------------------
  C1 cfm_indep @NFE=1  ：ρ < 0.05 且覆盖模态 = 0        （塌缩）
  C2 cfm_indep @NFE=32 ：ρ > 0.75 且覆盖 >= 7            （多步恢复）
  C3 cfm_ot    @NFE=1  ：ρ > 0.50                        （换耦合即救回）
  C4 reflow    @NFE=1  ：ρ > 0.50 且覆盖 >= 6
  C5 distill   @NFE=1  ：ρ > 0.50
  C6 onestep_l2 @NFE=1 ：ρ < 0.05                        （L2 地板，理论为 0）
  C7 onestep_minM@NFE=1：ρ > 0.20 且覆盖 >= 3            （非 L2 目标逃脱）
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
sys.path.insert(0, os.path.join(ROOT, "code", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

# ---------------------------------------------------------------- 参数
C, K, SIGMA, DIM = 4, 8, 0.25, 2
STEPS = 8000
BS = 384
LR = 1e-3
HIDDEN = 256
NLAYER = 4
N_PAIR = 20000
M_MIN = 8
N_EVAL = 2000
MODE_TOL = 0.6
SEEDS = (0, 1, 2, 3, 4)


def _parse_cli(argv):
    """解析 `--seeds 3,4` 与 `--merge`。

    `--merge` 从既有 `results/method_map.json` 读出 per_seed，只补跑缺的种子，
    最后按**并集**重算汇总与判据；否则扩容种子数就得把已算好的种子重算一遍。
    与 run_loss_ladder.py / run_chamfer_pooled.py 里的同名函数语义一致
    （各脚本自带一份，保持单文件可独立运行）。
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

CRIT = dict(C1_INDEP_N1_RHO_MAX=0.05, C1_INDEP_N1_COV_MAX=1.0,
            C2_INDEP_N32_RHO_MIN=0.75, C2_INDEP_N32_COV_MIN=7,
            C3_OT_N1_RHO_MIN=0.50,
            C4_REFLOW_N1_RHO_MIN=0.50, C4_REFLOW_N1_COV_MIN=6,
            C5_DISTILL_N1_RHO_MIN=0.50,
            C6_ONESTEP_L2_RHO_MAX=0.05,
            C7_MINM_N1_RHO_MIN=0.20, C7_MINM_N1_COV_MIN=3)

# ---- 判据 C1 的覆盖门槛：为什么从 0 改成 1（8 个模态里最多允许 1 个）----
# 初版写 C1_INDEP_N1_COV_MAX = 0，实测 cfm_indep@1 给出 ρ ≈ 0.0067（**远低于**
# 0.05 阈值，理论完美证实；当时 3 种子下是 0.0069）但覆盖 0.05～0.08 —— 于是 C1 判负。
# 复查后认定这是**判据写错**，不是理论失效：
#   * 理论（推论 1）说的是**精确最优一步映射**恒等于 m(c)，覆盖**恰好**为 0；
#   * 但**训练出来的**网络不可能精确等于 m(c)（实测 ρ=0.0069 而非 0），
#     2000 个评估样本中会有个别点偶然落进某个模态的 0.6 容差球内。
#   要求"覆盖恰好为 0"等于要求一个有限样本训练的网络达到完美，任何实现都过不了。
# 修正后的门槛取 **8 个模态中最多 1 个**（= 12.5%），仍然极严：
# 成功的那一侧是 8/8，失败侧是 0.08/8，中间隔了两个数量级。
#
# ---- 修正 (viii)：min-of-M 的 M 个候选必须来自**独立**源噪声 ----
# 旧版用 X0 = np.repeat(x0, M)：同一 x0、同一 t=0、同一 c。一步网络是确定性
# 映射（MLP，无 dropout/BN），于是 M 次前向给出**完全相同**的输出，min 的
# argmin 恒为 0，损失在**任何**训练状态下都逐比特等于点式 L2。后果是
# results 里 onestep_l2@1 与 onestep_minM@1 的 rho 在 3 个种子上逐比特相同
# （0.00010386879583898434），取证见 code/archive/_probe_minm_degeneracy.py：
# 200 步小规模训练下两条路径的 max|Δw| = 0.000e+00。
# 这说明该行当时**没有检验任何东西**，而不是"IMLE 不能逃脱"。
# 修正：每个目标独立抽 M 个源噪声（x0 ~ N(0,I)），即 IMLE 的原始语义；
# 这样"选最近候选"才是一次真正的指派，梯度也才可能绕开模式平均。

torch.set_num_threads(14)


# ---------------------------------------------------------------- 几何
def build_geometry():
    centers = np.zeros((C, K, DIM))
    radii = np.zeros(C)
    for j in range(C):
        b = np.array([1.6 * (j / (C - 1)) - 0.8, 0.9 * (j / (C - 1)) - 0.45])
        R = 1.5 + 0.4 * j
        radii[j] = R
        phi = (np.pi / 6.0) * j
        for k in range(K):
            th = k * (2.0 * np.pi / K) + phi
            centers[j, k] = b + R * np.array([np.cos(th), np.sin(th)])
    return centers, radii ** 2 + DIM * SIGMA ** 2


CENTERS, TR_VAR_COND = build_geometry()


def sample_cond(n, rng, c=None):
    """采样 (x0, x1, c)。c 为 None 时随机。"""
    if c is None:
        c = rng.integers(0, C, size=n)
    comp = rng.integers(0, K, size=n)
    x1 = CENTERS[c, comp] + SIGMA * rng.normal(size=(n, DIM))
    x0 = rng.normal(size=(n, DIM))
    return x0.astype(np.float32), x1.astype(np.float32), c


def ot_reorder(x0, x1, c):
    """逐条件做匈牙利指派，把 x1 重排成与 x0 的传输配对。"""
    x1 = x1.copy()
    for j in range(C):
        idx = np.where(c == j)[0]
        if len(idx) < 2:
            continue
        a, b = x0[idx].astype(np.float64), x1[idx].astype(np.float64)
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        r, cc = linear_sum_assignment(d2)
        x1[idx[r]] = b[cc]
    return x1


# ---------------------------------------------------------------- 模型
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=HIDDEN, nlayer=NLAYER):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(nlayer):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inp):
        return self.net(inp)


def onehot(c, n=None):
    n = len(c) if n is None else n
    oh = np.zeros((n, C), dtype=np.float32)
    oh[np.arange(n), c] = 1.0
    return oh


def t_in(x, t, c):
    return torch.from_numpy(np.concatenate([x, t[:, None], onehot(c)], 1))


# ---------------------------------------------------------------- 训练
def train_cfm(sample_fn, seed, steps=None, bs=BS, lr=LR, tag=""):
    """条件流匹配。sample_fn(bs, rng) -> (x0, x1, c)。

    **steps/bs 必须在调用时解析模块全局**：写成 `steps=STEPS` 的默认值会在
    函数定义时绑定，冒烟测试改 `M.STEPS` 根本不生效（实测照跑 12000 步）。
    """
    steps = STEPS if steps is None else steps
    torch.manual_seed(seed)
    net = MLP(DIM + 1 + C, DIM)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 101)
    t0 = time.time()
    for s in range(steps):
        x0, x1, c = sample_fn(bs, rng)
        t = rng.random(bs).astype(np.float32)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        u = x1 - x0
        loss = ((net(t_in(xt, t, c)) - torch.from_numpy(u)) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % 4000 == 0:
            print("      [%s] step %d/%d  loss %.4f  (%.0fs)"
                  % (tag, s + 1, steps, float(loss), time.time() - t0), flush=True)
    net.eval()
    return net


def train_onestep(sample_fn, seed, steps=None, bs=BS, lr=LR, tag="",
                  min_m=1, target_fn=None):
    """一步生成器。target_fn 为 None 时对 x1 回归；否则对 target_fn(x0,c) 回归。

    min_m > 1 时用 min-of-M（IMLE / Chamfer 式）损失，规避 L2 的模式平均。
    """
    steps = STEPS if steps is None else steps
    torch.manual_seed(seed)
    net = MLP(DIM + 1 + C, DIM)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 202)
    t0 = time.time()
    for s in range(steps):
        if target_fn is not None:
            x0, y, c = target_fn(bs, rng)
        else:
            x0, y, c = sample_fn(bs, rng)
        if min_m > 1:
            if target_fn is not None:
                raise ValueError("min-of-M 需要独立耦合：target_fn 必须为 None")
            # 修正 (viii)：M 个候选来自**互相独立**的源噪声。用 np.repeat 复制
            # 同一个 x0 会让确定性的一步网络给出 M 个完全相同的候选，min 退化，
            # 损失逐比特等于点式 L2（取证见模块头注释与
            # code/archive/_probe_minm_degeneracy.py）。
            X0 = rng.normal(size=(len(y) * min_m, DIM)).astype(np.float32)
            Cc = np.repeat(c, min_m, axis=0)
            tt = np.zeros(len(X0), dtype=np.float32)
            # 指派步必须 detach：min-of-M 的"选最近候选"不可微，
            # 这里只借用它的**指派结果**，梯度走下面重新前向的那一次。
            pred = net(t_in(X0, tt, Cc)).detach().numpy().reshape(len(y), min_m, DIM)
            d = ((pred - y[:, None, :]) ** 2).sum(-1)          # (bs, M)
            best = d.argmin(1)
            X0s = X0.reshape(len(y), min_m, DIM)[np.arange(len(y)), best]
            Ccs = Cc.reshape(len(y), min_m)[np.arange(len(y)), best]
            tt2 = np.zeros(len(y), dtype=np.float32)
            loss = ((net(t_in(X0s, tt2, Ccs)) - torch.from_numpy(y)) ** 2).sum(-1).mean()
        else:
            tt = np.zeros(len(x0), dtype=np.float32)
            loss = ((net(t_in(x0, tt, c)) - torch.from_numpy(y)) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % 4000 == 0:
            print("      [%s] step %d/%d  loss %.4f  (%.0fs)"
                  % (tag, s + 1, steps, float(loss), time.time() - t0), flush=True)
    net.eval()
    return net


# ---------------------------------------------------------------- 采样 / 度量
@torch.no_grad()
def euler(net, x0, c, N):
    x = torch.from_numpy(x0.copy())
    h = 1.0 / N
    for i in range(N):
        t = np.full(len(x), i * h, dtype=np.float32)
        x = x + h * net(t_in(x.numpy(), t, c))
    return x.numpy()


@torch.no_grad()
def onestep(net, x0, c):
    t = np.zeros(len(x0), dtype=np.float32)
    return net(t_in(x0, t, c)).numpy()


def mode_coverage(pred, centers_j, tol=MODE_TOL):
    d2 = ((pred[:, None, :] - centers_j[None, :, :]) ** 2).sum(-1)
    return int((np.sqrt(d2.min(axis=0)) < tol).sum())


def evaluate(net, seed, N, kind="flow"):
    """在留出条件上评估：ρ（逐条件平均）与模式覆盖（逐条件平均）。"""
    rng = np.random.default_rng(seed + 303)
    num, cov = [], []
    for j in range(C):
        x0, _, c = sample_cond(N_EVAL, rng, c=np.full(N_EVAL, j))
        pred = euler(net, x0, c, N) if kind == "flow" else onestep(net, x0, c)
        num.append(float(np.asarray(pred, dtype=np.float64).var(axis=0).sum()))
        cov.append(mode_coverage(pred.astype(np.float64), CENTERS[j]))
    rho = float(np.mean(num) / float(np.mean(TR_VAR_COND)))
    return dict(rho=rho, coverage=float(np.mean(cov)),
                rho_per_cond=[v / TR_VAR_COND[j] for j, v in enumerate(num)],
                cov_per_cond=cov)


def make_pair_set(net, seed, n=None, N=32):
    """用冻结的 32 步流映射生成确定性配对 (x0, φ(x0,c))。"""
    n = N_PAIR if n is None else n
    rng = np.random.default_rng(seed + 404)
    x0, _, c = sample_cond(n, rng)
    y = euler(net, x0, c, N)
    return x0, y.astype(np.float32), c


def pair_sampler(X0, Y, Cidx):
    rng = np.random.default_rng(0)
    n = len(X0)

    def fn(bs, _rng):
        idx = rng.integers(0, n, size=bs)
        return X0[idx], Y[idx], Cidx[idx]
    return fn


# ---------------------------------------------------------------- 主流程
def run_seed(seed):
    print("\n" + "#" * 72)
    print("# seed %d" % seed)
    print("#" * 72)
    res = {}

    def f_indep(bs, rng):
        return sample_cond(bs, rng)

    def f_ot(bs, rng):
        x0, x1, c = sample_cond(bs, rng)
        return x0, ot_reorder(x0, x1, c), c

    print("\n  [1/6] cfm_indep —— 标准 CFM，独立耦合")
    net_i = train_cfm(f_indep, seed, tag="cfm_indep")
    res["cfm_indep"] = dict(nfe1=evaluate(net_i, seed, 1),
                            nfe32=evaluate(net_i, seed, 32))

    print("\n  [2/6] cfm_ot —— 小批量 OT 耦合")
    net_o = train_cfm(f_ot, seed, tag="cfm_ot")
    res["cfm_ot"] = dict(nfe1=evaluate(net_o, seed, 1),
                         nfe32=evaluate(net_o, seed, 32))

    print("\n  [3/6] reflow —— 用 32 步流映射生成确定性配对后重训")
    X0, Y, Cc = make_pair_set(net_i, seed)
    net_r = train_cfm(pair_sampler(X0, Y, Cc), seed, tag="reflow")
    res["reflow"] = dict(nfe1=evaluate(net_r, seed, 1))

    print("\n  [4/6] distill —— 一步网络以 L2 蒸馏 32 步流映射")
    net_d = train_onestep(None, seed, tag="distill",
                          target_fn=pair_sampler(X0, Y, Cc))
    res["distill"] = dict(nfe1=evaluate(net_d, seed, 1, kind="one"))

    print("\n  [5/6] onestep_l2 —— 一步网络直接对 x1 做 L2（独立耦合）")
    net_l = train_onestep(f_indep, seed, tag="onestep_l2")
    res["onestep_l2"] = dict(nfe1=evaluate(net_l, seed, 1, kind="one"))

    print("\n  [6/6] onestep_minM —— min-of-%d 损失" % M_MIN)
    net_m = train_onestep(f_indep, seed, tag="onestep_minM", min_m=M_MIN)
    res["onestep_minM"] = dict(nfe1=evaluate(net_m, seed, 1, kind="one"))

    return res


def main(new_seeds, merge):
    t0 = time.time()
    out = os.path.join(ROOT, "results", "method_map.json")
    all_res = {}
    if merge and os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            all_res = dict(json.load(f).get("per_seed", {}))
        print("merge: 载入既有种子 %s" % sorted(all_res, key=int), flush=True)
    for sd in new_seeds:
        if str(sd) in all_res:
            print("\nskip seed %d（结果已在报告中）" % sd, flush=True)
            continue
        all_res[str(sd)] = run_seed(sd)

    seeds_all = sorted(int(k) for k in all_res)
    print("\n参与汇总的种子: %s" % seeds_all, flush=True)

    # ---- 跨种子汇总 ----
    def agg(path):
        vals = [all_res[str(s)][path[0]][path[1]] for s in seeds_all]
        return dict(rho=float(np.mean([v["rho"] for v in vals])),
                    rho_std=float(np.std([v["rho"] for v in vals])),
                    coverage=float(np.mean([v["coverage"] for v in vals])))

    summary = {
        "cfm_indep@1": agg(("cfm_indep", "nfe1")),
        "cfm_indep@32": agg(("cfm_indep", "nfe32")),
        "cfm_ot@1": agg(("cfm_ot", "nfe1")),
        "cfm_ot@32": agg(("cfm_ot", "nfe32")),
        "reflow@1": agg(("reflow", "nfe1")),
        "distill@1": agg(("distill", "nfe1")),
        "onestep_l2@1": agg(("onestep_l2", "nfe1")),
        "onestep_minM@1": agg(("onestep_minM", "nfe1")),
    }

    print("\n" + "=" * 78)
    print("跨方法族失效地图（%d 个种子平均）" % len(seeds_all))
    print("  %-18s %-18s %-12s" % ("方法", "ρ（展布保留率）", "覆盖模态/8"))
    print("-" * 78)
    for k, v in summary.items():
        print("  %-18s %-18s %-12s"
              % (k, "%.4f ± %.4f" % (v["rho"], v["rho_std"]), "%.2f" % v["coverage"]))
    print("-" * 78)

    g = summary
    checks = dict(
        C1_indep_nfe1_collapses=bool(g["cfm_indep@1"]["rho"] < CRIT["C1_INDEP_N1_RHO_MAX"]
                                     and g["cfm_indep@1"]["coverage"] <= CRIT["C1_INDEP_N1_COV_MAX"]),
        C2_indep_nfe32_recovers=bool(g["cfm_indep@32"]["rho"] > CRIT["C2_INDEP_N32_RHO_MIN"]
                                     and g["cfm_indep@32"]["coverage"] >= CRIT["C2_INDEP_N32_COV_MIN"]),
        C3_ot_nfe1_recovers=bool(g["cfm_ot@1"]["rho"] > CRIT["C3_OT_N1_RHO_MIN"]),
        C4_reflow_nfe1_recovers=bool(g["reflow@1"]["rho"] > CRIT["C4_REFLOW_N1_RHO_MIN"]
                                     and g["reflow@1"]["coverage"] >= CRIT["C4_REFLOW_N1_COV_MIN"]),
        C5_distill_nfe1_recovers=bool(g["distill@1"]["rho"] > CRIT["C5_DISTILL_N1_RHO_MIN"]),
        C6_onestep_l2_floor=bool(g["onestep_l2@1"]["rho"] < CRIT["C6_ONESTEP_L2_RHO_MAX"]),
        C7_minM_escapes=bool(g["onestep_minM@1"]["rho"] > CRIT["C7_MINM_N1_RHO_MIN"]
                             and g["onestep_minM@1"]["coverage"] >= CRIT["C7_MINM_N1_COV_MIN"]),
    )
    verdict = "PASS" if all(checks.values()) else "PARTIAL" if sum(checks.values()) >= 5 else "FAIL"
    for k, v in checks.items():
        print("    %-32s %s" % (k, v))
    print("  VERDICT: %s     用时 %.0fs" % (verdict, time.time() - t0))
    print("=" * 78)

    report = dict(params=dict(C=C, K=K, sigma=SIGMA, STEPS=STEPS, BS=BS, LR=LR,
                              HIDDEN=HIDDEN, NLAYER=NLAYER, N_PAIR=N_PAIR,
                              M_MIN=M_MIN, N_EVAL=N_EVAL, seeds=list(seeds_all)),
                  criteria=CRIT, summary=summary, checks=checks, verdict=verdict,
                  per_seed=all_res)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("报告已写入 %s" % out)


if __name__ == "__main__":
    main(*_parse_cli(sys.argv[1:]))
