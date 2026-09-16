# -*- coding: utf-8 -*-
"""条件情形下的耦合-确定性定理：受控几何族上的完整验证。

为什么必须做条件版
------------------
论文关心的是**策略** pi(a|s)，即条件生成。无条件版的所有结论都要在
"固定条件 c 之后"重新验证一遍，因为：

  * 判别量 D（条件未解释方差占比）是按 c 定义的；
  * 一步映射 f_1(., c) 的塌缩必须逐 c 成立才是真塌缩；
  * 最重要的：本实验要证明 **D 相同但 rho 天差地别**。

受控设计（本脚本的核心）
------------------------
固定**同一个**条件数据分布 p(x1|c)：
    条件 j 上，x1 服从 K 个等权高斯的环上混合，
    环心 = 平移 b_j + 半径 R_j 的旋转环，分量方差 sigma^2。
于是 tr Var(x1|j) = R_j^2 + d sigma^2（闭式），E[x1|j] = b_j。

在这个**完全相同的边缘分布**上，只改训练目标里的耦合：

  * 独立耦合（标准 CFM / DDPM 的做法）：x0 独立于 x1；
  * 传输耦合（reflow / OT-FM / 一致性蒸馏 / IMLE 的本质）：x0 与 x1
    按最优传输一一配对，几乎确定性地对应。

两者的数据边缘分布逐样本相同 ⇒ D 相同（判据 C5），
但一步映射的展布保留率 rho 应当差一个数量级以上（判据 C4）。

精确条件速度场
--------------
与无条件版同理（联合高斯的线性条件期望），对给定条件 j：

    u(x, t, j) = sum_k w_k(x,t,j) [ mu_{j,k} + ((t sigma^2 - (1-t)) / s_t^2) (x - t mu_{j,k}) ]
    w_k(x,t,j) ∝ N(x ; t mu_{j,k}, s_t^2 I),   s_t^2 = (1-t)^2 + t^2 sigma^2

t = 0 时 w_k 与 k 无关（N(x;0,I) 公共因子），括号化为 mu_{j,k} - x，故

    u(x, 0, j) = mean_k mu_{j,k} - x = b_j - x        ← 条件版端点恒等式
    f_1(x0, j) = x0 + u(x0, 0, j) = b_j  与 x0 无关 ⇒ rho_1(j) = 0

用闭式速度场可以彻底排除"网络拟合误差"这个混淆因素，
把"离散化 / 耦合"的机制单独隔离出来。
"""
import json
import os
import sys


def _find_root(d):
    d = os.path.abspath(d)
    while True:
        if os.path.isdir(os.path.join(d, "code")) and os.path.isdir(os.path.join(d, "paper")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("project root not found")
        d = parent


ROOT = _find_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code", "src"))

import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

# ---------------------------------------------------------------- 实验参数
C = 4               # 条件个数
K = 8               # 每个条件下的模态数
SIGMA = 0.25        # 各模态的内禀标准差
DIM = 2
N_EXACT = 3000      # 精确速度场实验每个条件的样本数
N_OT = 600          # 传输耦合实验每个条件的样本数（Hungarian 是 O(n^3)）
N_QUERY = 600       # 估计 rho 时用的查询点数（与训练集分开，避免过拟合）
K_LIST = (1, 2, 5, 10, 20, 30)     # kNN 的 k；有效区要求 k <= n/20 = 30
N_LIST = (1, 2, 4, 8, 16, 32, 64, 128, 256)
MODE_TOL = 0.6      # 判定"覆盖到某模态"的距离容差
SEED = 0


# ---------------------------------------------------------------- 几何族
def build_geometry():
    """返回每个条件的环心 (C, K, DIM)、条件均值 b_j、闭式 tr Var(x1|j)。"""
    centers = np.zeros((C, K, DIM))
    b = np.zeros((C, DIM))
    radii = np.zeros(C)
    for j in range(C):
        # 条件均值随 j 平移 —— 保证 D < 1（有条件可解释的部分）
        b[j] = np.array([1.6 * (j / (C - 1)) - 0.8, 0.9 * (j / (C - 1)) - 0.45])
        R = 1.5 + 0.4 * j
        radii[j] = R
        phi = (np.pi / 6.0) * j
        for k in range(K):
            th = k * (2.0 * np.pi / K) + phi
            centers[j, k] = b[j] + R * np.array([np.cos(th), np.sin(th)])
    # 闭式：等权环上混合的均值就是 b_j，故 tr Var(x1|j) = R_j^2 + d sigma^2
    tr_var_cond = radii ** 2 + DIM * SIGMA ** 2
    return centers, b, radii, tr_var_cond


def sample_x1(centers, n_per_cond, rng):
    """从条件混合中采样：条件 j 的第 i 个样本属于随机模态。"""
    out = np.zeros((C, n_per_cond, DIM))
    for j in range(C):
        comp = rng.integers(0, K, size=n_per_cond)
        out[j] = centers[j, comp] + SIGMA * rng.normal(size=(n_per_cond, DIM))
    return out


# ---------------------------------------------------------------- 精确速度场
class ConditionalExactVelocity:
    """独立耦合下、逐条件的**精确**边缘 CFM 速度场（闭式，无训练误差）。"""

    def __init__(self, centers):
        self.centers = np.asarray(centers, dtype=np.float64)   # (C,K,D)

    def __call__(self, x, t, j):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        mu = self.centers[j]                                    # (K,D)
        s2 = (1.0 - t) ** 2 + t ** 2 * SIGMA ** 2
        diff = x[:, None, :] - t * mu[None, :, :]               # (n,K,D)
        logp = -0.5 * (diff ** 2).sum(-1) / s2                  # 等权，无 log w
        logp -= logp.max(axis=1, keepdims=True)
        w = np.exp(logp)
        w /= w.sum(axis=1, keepdims=True)
        coef = (t * SIGMA ** 2 - (1.0 - t)) / s2
        vel = mu[None, :, :] + coef * diff
        return (w[:, :, None] * vel).sum(axis=1)


def euler_map_cond(vel, x0, j, N):
    """N 步显式欧拉，步长 1/N，返回 f_N(x0, j)。"""
    x = np.array(x0, dtype=np.float64, copy=True)
    h = 1.0 / N
    for i in range(N):
        x = x + h * vel(x, i * h, j)
    return x


def mode_coverage(pred, centers_j, tol=MODE_TOL):
    d2 = ((pred[:, None, :] - centers_j[None, :, :]) ** 2).sum(-1)
    return int((np.sqrt(d2.min(axis=0)) < tol).sum())


# ---------------------------------------------------------------- 估计器
def knn_uniform(X_train, Y_train, X_query, k):
    """等权 kNN 回归。用等权（而非距离加权）才能暴露 1/k 方差律。"""
    d2 = ((X_query[:, None, :] - X_train[None, :, :]) ** 2).sum(-1)
    idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    return Y_train[idx].mean(axis=1)


def ot_pairing(x0, x1):
    """最优传输配对：返回与 x0 一一对应的 x1（Hungarian 解 L2 传输问题）。"""
    d2 = ((x0[:, None, :] - x1[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(d2)
    return x1[c]


def tr_var(a):
    return float(np.asarray(a, dtype=np.float64).var(axis=0).sum())


def d_hat(X1_by_cond):
    """D = E_c[tr Var(x1|c)] / tr Var(x1)（条件未解释方差占比）。"""
    within = float(np.mean([tr_var(X1_by_cond[j]) for j in range(C)]))
    total = tr_var(X1_by_cond.reshape(-1, X1_by_cond.shape[-1]))
    return within / total


# ---------------------------------------------------------------- 主流程
def main():
    rng = np.random.default_rng(SEED)
    centers, b, radii, tr_var_cond = build_geometry()
    vel = ConditionalExactVelocity(centers)

    print("=" * 72)
    print("条件情形下的耦合-确定性定理验证（受控几何族）")
    print("  C=%d 个条件，每个条件 K=%d 环上高斯混合，sigma=%.2f" % (C, K, SIGMA))
    print("  逐条件闭式 tr Var(x1|j) = %s" % np.round(tr_var_cond, 4).tolist())
    print("=" * 72)

    # ================= 检验 1：条件版端点恒等式 =================
    print("\n[检验 1] 条件版端点恒等式  u(x,0,j) = b_j - x")
    errs = []
    for j in range(C):
        X0 = rng.normal(size=(N_EXACT, DIM))
        u0 = vel(X0, 0.0, j)
        errs.append(float(np.abs(u0 - (b[j] - X0)).max()))
        print("    条件 %d: max|u(x,0,j)-(b_j-x)| = %.3e" % (j, errs[-1]))
    endpoint_ok = max(errs) < 1e-9

    # ================= 检验 2：逐条件一步塌缩 =================
    print("\n[检验 2] 一步映射  f_1(x0,j) = b_j  ⇒ rho_1(j) = 0")
    X0_all = rng.normal(size=(C, N_EXACT, DIM))
    rho1 = []
    for j in range(C):
        f1 = euler_map_cond(vel, X0_all[j], j, 1)
        s = tr_var(f1)
        print("    条件 %d: tr Var(f_1) = %.3e   rho_1 = %.3e   覆盖模态 %d/%d"
              % (j, s, s / tr_var_cond[j], mode_coverage(f1, centers[j]), K))
        rho1.append(s / tr_var_cond[j])
    collapse_ok = max(rho1) < 1e-6

    # ================= 检验 3：逐条件 rho_N 单调上升到 1 =================
    print("\n[检验 3] 逐条件 rho_N 随步数上升")
    sweep = {j: [] for j in range(C)}
    for N in N_LIST:
        line = []
        for j in range(C):
            fN = euler_map_cond(vel, X0_all[j], j, N)
            r = tr_var(fN) / tr_var_cond[j]
            sweep[j].append(dict(N=N, rho=r, modes=mode_coverage(fN, centers[j])))
            line.append("j%d rho=%.4f (%d/%d)" % (j, r, mode_coverage(fN, centers[j]), K))
        print("    N=%-4d  %s" % (N, "   ".join(line)))

    mono_ok, reach_ok = True, True
    for j in range(C):
        rs = [e["rho"] for e in sweep[j]]
        if any(rs[i] > rs[i + 1] + 1e-6 for i in range(len(rs) - 1)):
            mono_ok = False
        if rs[-1] <= 0.90:
            reach_ok = False

    pooled = {}
    for N in N_LIST:
        num = np.mean([tr_var(euler_map_cond(vel, X0_all[j], j, N)) for j in range(C)])
        pooled[N] = num / float(np.mean(tr_var_cond))

    # ================= 检验 4/5：同 D 不同耦合 =================
    print("\n[检验 4] 同一条件分布上，只改耦合（数据边缘分布逐样本相同）")
    X0_ot = rng.normal(size=(C, N_OT, DIM))
    X1_ind = sample_x1(centers, N_OT, rng)          # 独立耦合：x1 与 x0 无关
    X1_det = np.zeros_like(X1_ind)                  # 传输耦合：逐条件 Hungarian 配对
    for j in range(C):
        X1_det[j] = ot_pairing(X0_ot[j], X1_ind[j])

    D_ind = d_hat(X1_ind)
    D_det = d_hat(X1_det)
    print("    D(独立耦合)   = %.4f" % D_ind)
    print("    D(传输耦合)   = %.4f" % D_det)
    print("    |D_ind-D_det| = %.4f   （应≈0：数据边缘分布相同）" % abs(D_ind - D_det))
    d_invariant_ok = abs(D_ind - D_det) < 0.03

    Xq = rng.normal(size=(C, N_QUERY, DIM))
    rows = []
    print("\n    等权 kNN 一步映射的展布保留率 rho_hat(k)（有效区 k <= n/20 = %d）" % (N_OT // 20))
    print("    %-5s %-14s %-14s %-10s %-10s" % ("k", "独立 rho", "传输 rho", "比值", "1/k 律校验"))
    for k in K_LIST:
        num_i = np.mean([tr_var(knn_uniform(X0_ot[j], X1_ind[j], Xq[j], k)) for j in range(C)])
        num_d = np.mean([tr_var(knn_uniform(X0_ot[j], X1_det[j], Xq[j], k)) for j in range(C)])
        den = float(np.mean(tr_var_cond))
        ri, rd = num_i / den, num_d / den
        rows.append(dict(k=k, rho_ind=ri, rho_det=rd, ratio=rd / max(ri, 1e-12),
                         k_times_rho_ind=k * ri))
        print("    %-5d %-14.4f %-14.4f %-10.1f %-10.3f"
              % (k, ri, rd, rd / max(ri, 1e-12), k * ri))

    # 在 kNN 一致的有效区（k <= n/20）判定对比度
    valid = [r for r in rows if r["k"] <= N_OT // 20]
    best = max(valid, key=lambda r: r["ratio"])
    contrast_ok = best["ratio"] > 10.0
    # 1/k 律：独立耦合下 k * rho_hat(k) 应≈1（允许估计波动）
    law_vals = [r["k_times_rho_ind"] for r in valid if r["k"] >= 2]
    law_ok = all(0.7 <= v <= 1.4 for v in law_vals)

    checks = dict(
        conditional_endpoint_identity=bool(endpoint_ok),
        per_condition_one_step_collapse=bool(collapse_ok),
        per_condition_monotone_in_N=bool(mono_ok),
        per_condition_reaches_one=bool(reach_ok),
        D_invariant_across_coupling=bool(d_invariant_ok),
        coupling_contrast_gt_10x=bool(contrast_ok),
        independent_knn_one_over_k_law=bool(law_ok),
    )
    verdict = "PASS" if all(checks.values()) else "FAIL"

    print("\n" + "=" * 72)
    print("判据（全部预先声明，未事后放宽）")
    for kk, vv in checks.items():
        print("    %-38s %s" % (kk, vv))
    print("  最大对比度 %.1fx  (k=%d)" % (best["ratio"], best["k"]))
    print("  VERDICT: %s" % verdict)
    print("=" * 72)

    report = dict(
        config=dict(C=C, K=K, sigma=SIGMA, dim=DIM, n_exact=N_EXACT,
                    n_ot=N_OT, n_query=N_QUERY, k_list=list(K_LIST),
                    n_list=list(N_LIST), mode_tol=MODE_TOL, seed=SEED),
        closed_form_tr_var_per_condition=tr_var_cond.tolist(),
        conditional_means=b.tolist(),
        endpoint_identity_max_err_per_condition=errs,
        one_step_rho_per_condition=rho1,
        multistep_sweep={str(j): sweep[j] for j in range(C)},
        pooled_rho_by_N={str(n): pooled[n] for n in N_LIST},
        D=dict(independent=D_ind, transport=D_det, abs_diff=abs(D_ind - D_det)),
        knn_sweep=rows,
        best_contrast=dict(k=best["k"], ratio=best["ratio"]),
        checks=checks,
        verdict=verdict,
    )
    logs = os.path.join(ROOT, "logs")
    os.makedirs(logs, exist_ok=True)
    out = os.path.join(logs, "verify_conditional_theory.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("报告已写入 %s" % out)


if __name__ == "__main__":
    main()
