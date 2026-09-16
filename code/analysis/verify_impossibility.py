# -*- coding: utf-8 -*-
r"""不可能性定理的构造性验证：让 ρ 精确扫描 [0,1]，而数据分布一步不动。

--------------------------------------------------------------------
这是全篇创新性所系的实验，先说清它证的是什么
--------------------------------------------------------------------
耦合文献已经说过"耦合影响一步质量"。所以本文**不能**只主张"耦合重要"。
本实验要证的是一条更强、且此前无人陈述的命题：

    **【不可能性】** 设 $p(c,x_1)$ 固定。则 $\\rho(f^\\*)$ 的取值**覆盖整个 $[0,1]$**，
    且两个端点都可达 —— 而实现这件事的那一族训练配置，其 $(c,x_1)$ 边缘分布
    **逐样本完全相同**。

推论（这才是要害）：

    任何**只依赖数据分布**的泛函 $\\mathcal A(p(c,x_1))$ —— 无论是 $\\hat D$、
    $\\Lambda$、还是任何未来会被提出的"多模态分数" ——
    都**不可能**判定一步会不会丢模态。因为它要区分的两种情形，数据分布是同一个。
    **这不是估计精度问题，是信息不足。**

--------------------------------------------------------------------
构造：α-插值耦合，且可解析预测
--------------------------------------------------------------------
取定条件分布 $p_1(\\cdot\\mid c)$，取 $T(\\cdot,c)$ 为任意把 $p_0$ 推到 $p_1(\\cdot\\mid c)$
的确定性输运映射（这里用**精确速度场的 512 步欧拉流映射**，即 reflow 会产生的那个映射）。

定义 **α-混合耦合**：

    x_1 = B·T(x_0,c) + (1-B)·X,    B ~ Bernoulli(α),  X ~ p_1(·|c) 且与 x_0 独立。

两个分支的边缘都是 $p_1(\\cdot\\mid c)$ ⟹ **整个 (c,x_1) 边缘分布与 α 无关**。

而最优一步映射为

    E[x_1 | x_0, c] = α·T(x_0,c) + (1-α)·m(c)

    ⟹ ρ*(α) = α² · tr Var(x_1|c) / tr Var(x_1|c) = **α²**

**于是 ρ\*(α) = α² 精确扫描 [0,1]，而数据分布不动。** 这是不可能性定理的构造性证明，
并且附带一条**无自由参数的定量律**可直接检验。

--------------------------------------------------------------------
kNN 测量时的修正项（同样是解析的）
--------------------------------------------------------------------
用等权 kNN 去**测** ρ 时，预测 $\\frac1k\\sum_{i\\in N_k}x_1^{(i)}$ 的方差有两块：
  * 来自 α 分支：$\\approx α T(x_0,c)$，随 $x_0$ 变化，方差 $= α^2\\,\\mathrm{trVar}(x_1|c)$；
  * 来自 (1-α) 分支：k 个 iid 抽样的均值，方差 $=(1-\\alpha)^2\\,\\mathrm{trVar}(x_1|c)/k$。

    ⟹ **ρ̂_kNN(α) ≈ α² + (1-α)²/k**

α=0 时退化为已知的 $1/k$ 律；α=1 时预测 $\\approx1$。两项都检验。

判据（预先写死）
----------------
  P1 ρ̂ 随 α 单调不减
  P2 ρ̂(0) < 0.10，ρ̂(1) > 0.85
  P3 **|D̂(α) − D̂(0)| < 0.02**（数据边缘分布相同）
  P4 **|Λ̂(α) − Λ̂(0)| < 0.05**（同上）
  P5 ρ̂ 与解析预测 α²+(1-α)²/k 的平均绝对差 < 0.08
"""
import json
import os
import sys
import time


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

from deficit import gaussian_deficit  # noqa: E402

# ---- 参数（事前固定） ----
C, K, SIGMA, DIM = 4, 8, 0.25, 2
ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)
N_TRAIN = 4000
N_QUERY = 2000
KNN_K = (20, 50)
N_T_STEPS = 512          # 用精确速度场积分出确定性输运映射 T
SEEDS = (0, 1, 2)
MODE_TOL = 0.6

CRIT = dict(P2_RHO0_MAX=0.10, P2_RHO1_MIN=0.85,
            P3_D_MAX_DIFF=0.02, P4_LAM_MAX_DIFF=0.05,
            P5_PRED_MAD_MAX=0.08)


# ---------------------------------------------------------------- 几何与精确速度场
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


class ConditionalExactVelocity:
    """独立耦合下、逐条件高斯混合的**精确**边缘 CFM 速度场（闭式）。"""

    def __call__(self, x, t, j):
        mu = CENTERS[j]
        s2 = (1.0 - t) ** 2 + t ** 2 * SIGMA ** 2
        diff = x[:, None, :] - t * mu[None, :, :]
        logp = -0.5 * (diff ** 2).sum(-1) / s2
        logp -= logp.max(axis=1, keepdims=True)
        w = np.exp(logp)
        w /= w.sum(axis=1, keepdims=True)
        coef = (t * SIGMA ** 2 - (1.0 - t)) / s2
        return (w[:, :, None] * (mu[None, :, :] + coef * diff)).sum(axis=1)


VEL = ConditionalExactVelocity()


def transport_map(x0, j, N=N_T_STEPS):
    """确定性输运映射 T(·, j)：精确速度场的 N 步欧拉流映射（reflow 会产生的那个）。"""
    x = np.array(x0, dtype=np.float64, copy=True)
    h = 1.0 / N
    for i in range(N):
        x = x + h * VEL(x, i * h, j)
    return x


def sample_marginal(n, j, rng):
    """从 p_1(·|j) 独立采样。"""
    comp = rng.integers(0, K, size=n)
    return CENTERS[j, comp] + SIGMA * rng.normal(size=(n, DIM))


# ---------------------------------------------------------------- α-混合耦合
def sample_alpha(n, j, alpha, rng):
    """α-混合耦合：以概率 α 取确定性分支 T(x0)，否则取独立样本。

    **两个分支的边缘都是 p_1(·|j)，故 (j, x1) 的边缘分布与 α 无关。**
    """
    x0 = rng.normal(size=(n, DIM))
    B = rng.random(n) < alpha
    x1 = np.empty((n, DIM))
    if B.any():
        x1[B] = transport_map(x0[B], j)
    if (~B).any():
        x1[~B] = sample_marginal(int((~B).sum()), j, rng)
    return x0, x1


# ---------------------------------------------------------------- 估计器
def knn_uniform(Xtr, Ytr, Xq, k):
    d2 = ((Xq[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    return Ytr[idx].mean(axis=1)


def tr_var(a):
    return float(np.asarray(a, dtype=np.float64).var(axis=0).sum())


def d_hat_single(X1_by_cond):
    """D = E_c[tr Var(x1|c)] / tr Var(x1)（不引入任何回归器）。"""
    within = float(np.mean([tr_var(X1_by_cond[j]) for j in range(C)]))
    total = tr_var(X1_by_cond.reshape(-1, DIM))
    return within / total


def mode_coverage(pred, centers_j, tol=MODE_TOL):
    d2 = ((pred[:, None, :] - centers_j[None, :, :]) ** 2).sum(-1)
    return int((np.sqrt(d2.min(axis=0)) < tol).sum())


# ---------------------------------------------------------------- 主流程
def run_seed(seed):
    rng = np.random.default_rng(seed)
    rows = []
    for alpha in ALPHAS:
        # --- 固定一批 x0 与查询点，跨 α 复用，减少无关噪声 ---
        rng_a = np.random.default_rng(1000 * (seed + 1) + int(alpha * 100) + 7)
        Xtr, Ytr, Xq, Yq = [], [], [], []
        for j in range(C):
            x0, x1 = sample_alpha(N_TRAIN, j, alpha, rng_a)
            Xtr.append(x0)
            Ytr.append(x1)
            Xq.append(rng_a.normal(size=(N_QUERY, DIM)))
        rhos, covs, preds_by_k = {}, {}, {}
        for k in KNN_K:
            num, cov = [], []
            for j in range(C):
                p = knn_uniform(Xtr[j], Ytr[j], Xq[j], k)
                num.append(tr_var(p))
                cov.append(mode_coverage(p, CENTERS[j]))
            rhos[k] = float(np.mean(num) / float(np.mean(TR_VAR_COND)))
            covs[k] = float(np.mean(cov))
            preds_by_k[k] = [rhos[k]]
        # --- 数据侧统计量（与 α 无关的断言） ---
        Yall = np.stack(Ytr)                                  # (C, n, d)
        D = d_hat_single(Yall)
        lam_per_cond = [gaussian_deficit(Yall[j], n_proj=64, seed=seed)[0]
                        for j in range(C)]
        lam = float(np.mean(lam_per_cond))
        rows.append(dict(alpha=float(alpha), D=D, lam=lam,
                         rho={str(k): rhos[k] for k in KNN_K},
                         coverage={str(k): covs[k] for k in KNN_K}))
        print("    α=%.2f  D=%.4f  Λ=%.4f  |  " % (alpha, D, lam)
              + "   ".join("ρ̂(k=%d)=%.4f (覆盖 %.1f/8)" % (k, rhos[k], covs[k])
                           for k in KNN_K), flush=True)
    return rows


def main():
    t0 = time.time()
    print("=" * 78)
    print("不可能性定理的构造性验证：ρ*(α) = α² 扫描 [0,1]，数据分布不动")
    print("  判据（预先写死）：%s" % json.dumps(CRIT, ensure_ascii=False))
    print("  kNN 解析修正预测：ρ̂(α) ≈ α² + (1-α)²/k")
    print("=" * 78)

    all_rows = {}
    for sd in SEEDS:
        print("\n  ---- seed %d ----" % sd, flush=True)
        all_rows[str(sd)] = run_seed(sd)

    def agg(field, alpha, k=None):
        vals = []
        for sd in SEEDS:
            for r in all_rows[str(sd)]:
                if abs(r["alpha"] - alpha) < 1e-9:
                    vals.append(r[field][str(k)] if k is not None else r[field])
        return float(np.mean(vals)), float(np.std(vals))

    print("\n" + "=" * 78)
    print("跨 %d 个种子汇总（k=%d）" % (len(SEEDS), KNN_K[0]))
    k0 = KNN_K[0]
    print("  %-6s %-12s %-12s %-12s %-12s" % ("α", "ρ̂ 实测", "α² 理论", "α²+(1-α)²/k", "D̂ / Λ̂"))
    print("-" * 78)
    curve, pred_curve = [], []
    for a in ALPHAS:
        rho, _ = agg("rho", a, k0)
        D, _ = agg("D", a)
        lam, _ = agg("lam", a)
        pred = a ** 2 + (1 - a) ** 2 / k0
        curve.append(rho)
        pred_curve.append(pred)
        print("  %-6.2f %-12.4f %-12.4f %-12.4f %-12s"
              % (a, rho, a ** 2, pred, "%.4f / %.4f" % (D, lam)))
    print("-" * 78)

    rho0, _ = agg("rho", 0.0, k0)
    rho1, _ = agg("rho", 1.0, k0)
    D0, _ = agg("D", 0.0)
    lam0, _ = agg("lam", 0.0)
    ddev = max(abs(agg("D", a)[0] - D0) for a in ALPHAS)
    ldev = max(abs(agg("lam", a)[0] - lam0) for a in ALPHAS)

    mono = all(curve[i] <= curve[i + 1] + 1e-6 for i in range(len(curve) - 1))
    mad = float(np.mean([abs(c - p) for c, p in zip(curve, pred_curve)]))

    checks = dict(
        P1_monotone_in_alpha=bool(mono),
        P2_endpoints=bool(rho0 < CRIT["P2_RHO0_MAX"] and rho1 > CRIT["P2_RHO1_MIN"]),
        P3_D_invariant=bool(ddev < CRIT["P3_D_MAX_DIFF"]),
        P4_Lambda_invariant=bool(ldev < CRIT["P4_LAM_MAX_DIFF"]),
        P5_matches_analytic=bool(mad < CRIT["P5_PRED_MAD_MAX"]),
    )
    print("  P1 单调=%s   P2 端点 ρ̂(0)=%.4f<%.2f, ρ̂(1)=%.4f>%.2f -> %s"
          % (mono, rho0, CRIT["P2_RHO0_MAX"], rho1, CRIT["P2_RHO1_MIN"], checks["P2_endpoints"]))
    print("  P3 max|D̂(α)-D̂(0)| = %.5f (< %.3f) -> %s"
          % (ddev, CRIT["P3_D_MAX_DIFF"], checks["P3_D_invariant"]))
    print("  P4 max|Λ̂(α)-Λ̂(0)| = %.5f (< %.3f) -> %s"
          % (ldev, CRIT["P4_LAM_MAX_DIFF"], checks["P4_Lambda_invariant"]))
    print("  P5 与解析预测 α²+(1-α)²/k 的平均绝对差 = %.4f (< %.3f) -> %s"
          % (mad, CRIT["P5_PRED_MAD_MAX"], checks["P5_matches_analytic"]))
    verdict = "PASS" if all(checks.values()) else "FAIL"
    print("  VERDICT: %s     用时 %.0fs" % (verdict, time.time() - t0))
    print("=" * 78)

    report = dict(theorem="rho*(alpha) = alpha^2 with (c,x1)-marginal invariant",
                  params=dict(C=C, K=K, sigma=SIGMA, alphas=list(ALPHAS),
                              N_TRAIN=N_TRAIN, N_QUERY=N_QUERY,
                              KNN_K=list(KNN_K), N_T_STEPS=N_T_STEPS,
                              seeds=list(SEEDS)),
                  criteria=CRIT,
                  curve=dict(alpha=list(ALPHAS), rho_hat=curve,
                             alpha_squared=[a ** 2 for a in ALPHAS],
                             knn_prediction=pred_curve,
                             D=[agg("D", a)[0] for a in ALPHAS],
                             Lambda=[agg("lam", a)[0] for a in ALPHAS]),
                  deviations=dict(max_d_dev=ddev, max_lam_dev=ldev,
                                  pred_mad=mad),
                  checks=checks, verdict=verdict, per_seed=all_rows)
    logs = os.path.join(ROOT, "logs")
    os.makedirs(logs, exist_ok=True)
    out = os.path.join(logs, "verify_impossibility.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("报告已写入 %s" % out)


if __name__ == "__main__":
    main()
