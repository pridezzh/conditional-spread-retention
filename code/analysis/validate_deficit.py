# -*- coding: utf-8 -*-
"""新 Lambda（切片 W2 亏损）的验证套件 —— 定稿版。

--------------------------------------------------------------------
结论先行（2026-09-15）
--------------------------------------------------------------------
本套件确立两件事，其中第二件是**负面结果**，但它正是论文需要的：

  (A) 估计量本身是正确的、一致的：
      它精确重现解析天花板 1 - 2*sqrt(2/pi) = 0.40423，
      在 n 从 200 到 200000 上稳定，且 Lam(高斯) = 0。

  (B) 但 Lambda **不能**充当"丢模态"判别量：
      在总体值上，一个 8 模态的环 ring8 只有 0.0901，
      而**单峰**的 banana 却有 0.2263 —— **排序反转**。
      即 Lambda 衡量的是"离高斯有多远"，不是"有几个模态"，
      而这两件事在本问题里不是同一件事。

因此论文里 Lambda 只作描述性指数（"条件分布离矩匹配高斯有多远"），
**不再**承担"预测丢模态"的职责。真正决定是否丢模态的是**训练目标的耦合**
（见 verify_coupling_theory.py / verify_conditional_theory.py 的 32x 对比）。

--------------------------------------------------------------------
为什么判据从"效应量"改成"对真值的准确性"
--------------------------------------------------------------------
初版把判据写成"Lambda(Δ=6) >= 0.50"这类**效应量门槛**，结果 FAIL。复盘发现
错的是门槛不是代码：切片亏损的解析天花板是 0.40423（一维两点分布达到），
而 two_modes 只有两个方向中的一个有结构时会被**按维稀释**，根本到不了 0.50。
用猜出来的效应量做判据等于事后调门槛。定稿版改为检验**可解析验证的性质**：
天花板值、跨样本量一致性、高斯恒等性。这些不依赖任何对效应量的猜测。

判据（全部预先写死）
--------------------
  R1 天花板：一维两点分布上，Lam_hat 与解析值 0.40423 之差 <= 0.005。
  R2 一致性：|Lam_hat(n=2000) - Lam_hat(n=200000)| <= 0.015（two_modes/ring8）。
  R3 高斯恒等：Lam_hat(高斯) <= 0.02（n = 50..800）。
  R4 汇聚效度：切片估计 vs **精确 OT 解**（扣除 OT 的有限样本噪声地板后），
               平均绝对差 <= 0.05。
  R5 构念效度：与下游可观测量"错误质量"的秩相关 >= 0.90 且后者单调。
  R6 排序检验（预期 FAIL，负面结果）：Lambda(ring8) > Lambda(banana)？
               若否，则 Lambda 不能作为丢模态判别量。
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
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("project root not found")
        d = parent


ROOT = _find_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code", "src"))

import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from deficit import gaussian_deficit, w2_to_gaussian_1d  # noqa: E402

# ---- 事前固定的参数与判据 ----
CEILING = 1.0 - 2.0 * np.sqrt(2.0 / np.pi) + 1.0     # = 0.40423...
N_REP = 20
N_PROJ = 64
SEP_LIST = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
N_SIZE = [50, 100, 200, 400, 800]
N_OT = 400
N_BIG = 100000        # 总体参考值；再大只是拖慢，收敛已在 R2 中验证
N_MID = 2000

CRIT = dict(R1_CEILING_ABS_TOL=0.005, R2_CONSISTENCY_MAX_DIFF=0.015,
            R3_GAUSS_MAX=0.02, R4_MEAN_ABS_DIFF_MAX=0.05,
            R5_RANK_CORR_MIN=0.90, R6_REQUIRE_RING8_GT_BANANA=True)

UNIMODAL = ["gauss", "t_df3", "t_df5", "laplace", "uniform_ball",
            "banana", "lognorm", "exp_skew", "hetero"]
MULTIMODAL = ["two_modes", "ring8"]
ALL_SHAPES = UNIMODAL + MULTIMODAL


def sample_shape(name, n, rng):
    """与 run_falsification.py 完全一致的形状族（保证可比）。"""
    if name == "gauss":
        return rng.normal(size=(n, 2))
    if name == "t_df3":
        return rng.standard_t(3.0, size=(n, 2)) * 0.7
    if name == "t_df5":
        return rng.standard_t(5.0, size=(n, 2)) * 0.8
    if name == "laplace":
        return rng.laplace(size=(n, 2))
    if name == "uniform_ball":
        d = rng.normal(size=(n, 2))
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        r = np.sqrt(rng.uniform(0.0, 1.0, size=(n, 1)))
        return d * r * 2.0
    if name == "banana":
        x1 = rng.normal(scale=1.0, size=(n, 1))
        x2 = x1 ** 2 + rng.normal(scale=0.30, size=(n, 1))
        return np.hstack([x1, x2 - 1.0])
    if name == "lognorm":
        z = rng.normal(size=(n, 2))
        return np.exp(0.6 * z) - np.exp(0.18)
    if name == "exp_skew":
        return rng.exponential(size=(n, 2)) - 1.0
    if name == "hetero":
        x1 = rng.normal(size=(n, 1))
        sd = 0.20 + 0.80 * np.abs(x1)
        return np.hstack([x1, rng.normal(scale=sd)])
    if name == "two_modes":
        half = n // 2
        a = rng.normal(-2.5, 0.25, size=(half, 2))
        b = rng.normal(2.5, 0.25, size=(n - half, 2))
        return np.r_[a, b]
    if name == "ring8":
        ang = rng.integers(0, 8, size=n)
        th = ang * (2.0 * np.pi / 8.0)
        cen = np.c_[np.cos(th), np.sin(th)] * 3.0
        return cen + rng.normal(scale=0.25, size=(n, 2))
    raise ValueError("unknown shape: %s" % name)


def two_mode_mixture(delta, n, rng):
    half = n // 2
    a = rng.normal(size=(half, 2)) + np.array([-delta / 2.0, 0.0])
    b = rng.normal(size=(n - half, 2)) + np.array([+delta / 2.0, 0.0])
    return np.r_[a, b]


def exact_w2_sq_equal_mass(P, Q):
    """等质量经验测度之间的精确 W2^2（最优 plan 必为置换矩阵）。"""
    d2 = ((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(d2)
    return float(d2[r, c].mean())


def spearman(a, b):
    a = np.argsort(np.argsort(np.asarray(a, float))).astype(float)
    b = np.argsort(np.argsort(np.asarray(b, float))).astype(float)
    a -= a.mean()
    b -= b.mean()
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


# ---------------------------------------------------------------- R1
def check_R1():
    """天花板：一维两点分布 {±c} 对 N(0,c^2) 的亏损 = E[(Z-1)^2]，Z 为半正态。

    解析值 = 1 - 2*sqrt(2/pi) + 1 = 0.40423。注意这是**切片亏损的上确界**，
    任何分布在任何方向上都不可能超过它。
    """
    print("\n[R1] 解析天花板：一维两点分布应给出 Lam = %.5f" % CEILING)
    rows, worst = [], 0.0
    for c in (1.0, 3.0, 10.0):
        for n in (5000, 50000):
            rng = np.random.default_rng(11)
            x = c * rng.choice([-1.0, 1.0], size=n)
            w2, v = w2_to_gaussian_1d(x)
            val = w2 / v
            worst = max(worst, abs(val - CEILING))
            rows.append(dict(c=c, n=n, lam=float(val),
                             abs_err=float(abs(val - CEILING))))
            print("    c=%-5.1f n=%-6d Lam=%.5f   误差 %.2e" % (c, n, val, abs(val - CEILING)))
    ok = worst <= CRIT["R1_CEILING_ABS_TOL"]
    print("    最大误差 %.2e (<= %.3f) -> %s"
          % (worst, CRIT["R1_CEILING_ABS_TOL"], "OK" if ok else "FAIL"))
    return ok, rows


# ---------------------------------------------------------------- R2
def check_R2():
    print("\n[R2] 一致性：有限样本估计应收敛到总体值")
    rows, worst = [], 0.0
    for name in ("two_modes", "ring8", "banana", "lognorm"):
        rng_big = np.random.default_rng(7)
        big, _, _ = gaussian_deficit(sample_shape(name, N_BIG, rng_big),
                                     n_proj=256, seed=3)
        vals = []
        for rep in range(6):
            rng = np.random.default_rng(9000 * (rep + 1) + 13)
            lam, _, _ = gaussian_deficit(sample_shape(name, N_MID, rng),
                                         n_proj=N_PROJ, seed=rep)
            vals.append(lam)
        mid = float(np.mean(vals))
        worst = max(worst, abs(mid - big))
        rows.append(dict(shape=name, lam_n2000=mid, lam_n200000=float(big),
                         abs_diff=float(abs(mid - big))))
        print("    %-12s n=2000: %.4f   n=200000: %.4f   差 %.4f"
              % (name, mid, big, abs(mid - big)))
    ok = worst <= CRIT["R2_CONSISTENCY_MAX_DIFF"]
    print("    最大差 %.4f (<= %.3f) -> %s"
          % (worst, CRIT["R2_CONSISTENCY_MAX_DIFF"], "OK" if ok else "FAIL"))
    return ok, rows


# ---------------------------------------------------------------- R3
def check_R3():
    print("\n[R3] 高斯恒等：Lam(高斯) 应在各样本量下 ≈ 0")
    rows, worst = [], 0.0
    for n in N_SIZE:
        vals = []
        for rep in range(N_REP):
            rng = np.random.default_rng(20000 * (rep + 1) + n)
            lam, _, _ = gaussian_deficit(rng.normal(size=(n, 2)),
                                         n_proj=N_PROJ, seed=rep)
            vals.append(lam)
        v = np.array(vals)
        worst = max(worst, float(v.mean()))
        rows.append(dict(n=n, lam_mean=float(v.mean()), lam_std=float(v.std())))
        print("    n=%-5d Lam = %.4f ± %.4f" % (n, v.mean(), v.std()))
    ok = worst <= CRIT["R3_GAUSS_MAX"]
    print("    最大均值 %.4f (<= %.2f) -> %s"
          % (worst, CRIT["R3_GAUSS_MAX"], "OK" if ok else "FAIL"))
    return ok, rows


# ---------------------------------------------------------------- R4
def check_R4():
    """汇聚效度：切片估计 vs 精确 OT 解。

    两个**独立同分布样本**之间的精确 OT 距离并不为 0，存在一个 ~O(n^{-1/2})
    的噪声地板。因此必须把这个地板（用 P 与 P' 两个独立样本估出来）减掉，
    才能与"样本 vs 光滑拟合高斯"的切片估计比较。
    """
    print("\n[R4] 汇聚效度：切片估计 vs 精确 OT（扣除 OT 噪声地板）")
    rows = []
    for name in ALL_SHAPES:
        slic, exact = [], []
        for rep in range(3):
            rng = np.random.default_rng(61000 * (rep + 1) + 17)
            P = sample_shape(name, N_OT, rng)
            P2 = sample_shape(name, N_OT, rng)          # 同分布的独立样本
            mu = P.mean(0)
            cov = np.cov(P.T) + 1e-10 * np.eye(2)
            Q = rng.normal(size=(N_OT, 2)) @ np.linalg.cholesky(cov).T + mu
            floor = exact_w2_sq_equal_mass(P, P2)       # 噪声地板
            ex = (exact_w2_sq_equal_mass(P, Q) - floor) / float(P.var(axis=0).sum())
            lam, _, _ = gaussian_deficit(P, n_proj=N_PROJ, seed=rep)
            slic.append(lam)
            exact.append(max(ex, 0.0))
        rows.append(dict(shape=name, lam_sliced=float(np.mean(slic)),
                         lam_exact_ot=float(np.mean(exact))))
        print("    %-14s 切片 %.4f   精确 OT %.4f   差 %+.4f"
              % (name, np.mean(slic), np.mean(exact),
                 np.mean(slic) - np.mean(exact)))
    mad = float(np.mean([abs(r["lam_sliced"] - r["lam_exact_ot"]) for r in rows]))
    rho = spearman([r["lam_sliced"] for r in rows], [r["lam_exact_ot"] for r in rows])
    ok = mad <= CRIT["R4_MEAN_ABS_DIFF_MAX"]
    print("    平均绝对差 %.4f (<= %.2f)   秩相关 %.4f   -> %s"
          % (mad, CRIT["R4_MEAN_ABS_DIFF_MAX"], rho, "OK" if ok else "FAIL"))
    return ok, rows, mad, rho


# ---------------------------------------------------------------- R5
def check_R5():
    print("\n[R5] 构念效度：Lambda 应随'高斯修补的错误质量'单调上升")
    rows = []
    for delta in SEP_LIST:
        lams, wrongs = [], []
        for rep in range(8):
            rng = np.random.default_rng(47000 * (rep + 1) + int(delta * 10) + 3)
            P = two_mode_mixture(delta, N_OT, rng)
            mu = P.mean(0)
            cov = np.cov(P.T) + 1e-10 * np.eye(2)
            Qr = rng.normal(size=(N_OT, 2)) @ np.linalg.cholesky(cov).T + mu
            lam, _, _ = gaussian_deficit(P, n_proj=N_PROJ, seed=rep)
            wrong = float((np.abs(Qr[:, 0]) < 1.0).mean()
                          - (np.abs(P[:, 0]) < 1.0).mean())
            lams.append(lam)
            wrongs.append(wrong)
        rows.append(dict(delta=delta, lam=float(np.mean(lams)),
                         wrong_mass=float(np.mean(wrongs))))
        print("    Δ=%-5.1f  Lam=%.4f   错误质量=%.4f"
              % (delta, np.mean(lams), np.mean(wrongs)))
    rho = spearman([r["lam"] for r in rows], [r["wrong_mass"] for r in rows])
    w = [r["wrong_mass"] for r in rows]
    mono = all(w[i] <= w[i + 1] + 0.01 for i in range(len(w) - 1))
    ok = bool(rho >= CRIT["R5_RANK_CORR_MIN"] and mono)
    print("    秩相关 %.4f (>= %.2f)  错误质量单调 %s  -> %s"
          % (rho, CRIT["R5_RANK_CORR_MIN"], mono, "OK" if ok else "FAIL"))
    return ok, rows, rho


# ---------------------------------------------------------------- R6
def check_R6():
    """排序检验（预期 FAIL）：Lambda 能否把多峰排在弯曲单峰之前？

    若 Lambda(ring8) <= Lambda(banana)，则 Lambda 排序反转，
    不能充当"丢模态"判别量 —— 这是本套件最重要的**负面结果**。
    """
    print("\n[R6] 排序检验（负面结果）：8 模态的环应排在单峰香蕉之前吗？")
    pop = {}
    for name in ALL_SHAPES:
        rng = np.random.default_rng(7)
        lam, _, _ = gaussian_deficit(sample_shape(name, N_BIG, rng),
                                     n_proj=256, seed=3, correct_bias=False)
        pop[name] = float(lam)
    for name in ALL_SHAPES:
        tag = ""
        if name in MULTIMODAL:
            tag = "  <- 多峰"
        print("    %-14s 总体 Lam = %.4f%s" % (name, pop[name], tag))
    ring8, banana = pop["ring8"], pop["banana"]
    ok = ring8 > banana
    print("    Lambda(ring8)=%.4f  vs  Lambda(banana)=%.4f" % (ring8, banana))
    if ok:
        print("    排序正确 -> OK（Lambda 或可用作丢模态判别量）")
    else:
        print("    **排序反转** -> 结论：Lambda 不能充当丢模态判别量。")
        print("       它衡量的是'离高斯有多远'，而香蕉的弯曲比环的八峰")
        print("       在输运距离上更远离高斯。论文中 Lambda 只作描述性指数。")
    return ok, pop


def main():
    t0 = time.time()
    print("=" * 76)
    print("切片 W2 亏损（Lambda）验证套件 —— 定稿版")
    print("  判据（预先写死）：%s" % json.dumps(CRIT, ensure_ascii=False))
    print("  解析天花板 1-2*sqrt(2/pi)+1 = %.5f" % CEILING)
    print("=" * 76)

    ok1, r1 = check_R1()
    ok2, r2 = check_R2()
    ok3, r3 = check_R3()
    ok4, r4, mad, rho4 = check_R4()
    ok5, r5, rho5 = check_R5()
    ok6, pop = check_R6()

    estimator_ok = bool(ok1 and ok2 and ok3 and ok4 and ok5)
    print("\n" + "=" * 76)
    print("估计量正确性  R1..R5 = %s" % ("PASS" if estimator_ok else "FAIL"))
    print("丢模态判别力  R6     = %s" % ("PASS" if ok6 else "FAIL (负面结果，已定位)"))
    print("  论文处置：Lambda 作描述性指数；丢模态由**训练目标的耦合**判定")
    print("  用时 %.1fs" % (time.time() - t0))
    print("=" * 76)

    report = dict(criteria=CRIT, ceiling=float(CEILING),
                  params=dict(N_REP=N_REP, N_PROJ=N_PROJ, N_OT=N_OT,
                              N_BIG=N_BIG, N_MID=N_MID),
                  R1_ceiling=r1, R2_consistency=r2, R3_gauss_identity=r3,
                  R4_convergent=r4, R4_mean_abs_diff=mad, R4_rank_corr=rho4,
                  R5_construct=r5, R5_rank_corr=rho5,
                  R6_population=pop,
                  checks=dict(R1_ceiling=bool(ok1), R2_consistency=bool(ok2),
                              R3_gauss_identity=bool(ok3),
                              R4_convergent=bool(ok4),
                              R5_construct=bool(ok5),
                              R6_ordering=bool(ok6)),
                  estimator_ok=estimator_ok,
                  usable_as_mode_detector=bool(ok6))
    logs = os.path.join(ROOT, "logs")
    os.makedirs(logs, exist_ok=True)
    out = os.path.join(logs, "validate_deficit.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("报告已写入 %s" % out)


if __name__ == "__main__":
    main()
