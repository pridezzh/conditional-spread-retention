# -*- coding: utf-8 -*-
"""多步欧拉的展布恢复：**精确速度场**下 rho_N 从 0 单调升到 1。

为什么用精确速度场而不训练网络
------------------------------
若用训练出来的速度场，"rho_N 随 N 上升"可能混入两个因素：
  1. 离散化误差随 N 减小（我们想验证的机制）；
  2. 网络对速度场的拟合误差（我们不想混入的混淆因素）。
用**闭式精确速度场**可以彻底消除 (2)，把机制单独隔离出来。

精确速度场的推导（各向同性高斯混合 + 独立耦合）
----------------------------------------------
设 x0 ~ N(0, I_d)，x1 ~ sum_k pi_k N(mu_k, sigma^2 I_d)，且 x0 _||_ x1。
在分量 k 内，x_t = (1-t) x0 + t x1 是高斯的：

    E[x_t | k] = t mu_k
    Var(x_t | k) = s_t^2 I_d,   s_t^2 = (1-t)^2 + t^2 sigma^2

且
    Cov(x1, x_t | k) = t sigma^2 I_d
    Cov(x0, x_t | k) = (1-t) I_d

于是（联合高斯的线性条件期望公式）

    E[x1 | x_t=x, k] = mu_k + (t sigma^2 / s_t^2) (x - t mu_k)
    E[x0 | x_t=x, k] =        ((1-t)   / s_t^2) (x - t mu_k)

边缘速度场 u(x,t) = E[x1 - x0 | x_t = x] 按分量后验 w_k(x,t) 加权：

    u(x,t) = sum_k w_k(x,t) [ mu_k + ((t sigma^2 - (1-t)) / s_t^2) (x - t mu_k) ]
    w_k(x,t) ∝ pi_k N(x ; t mu_k, s_t^2 I_d)

**t=0 的检验**：s_0^2 = 1，且 N(x;0,I) 对各分量相同，故 w_k = pi_k，
括号内化为 mu_k - x，于是

    u(x,0) = sum_k pi_k (mu_k - x) = m - x          ← 端点恒等式

故 f_1(x0) = x0 + u(x0,0) = m 与 x0 无关 ⇒ rho_1 = 0（推论 1）。

而 N 步欧拉是 N 个"近恒等映射"的复合：
    f_N = (I + u(., t_{N-1})/N) ∘ ... ∘ (I + u(., 0)/N)
每一步都是简单映射，但**复合之后是高度非线性的输运映射**，
且随 N 增大收敛到精确流映射 phi_{0->1}（它把 p0 推到 p1）。
因此预期 rho_N -> 1。这就是"多步如何逃出塌缩"的机制。

本脚本用闭式速度场数值验证：rho_1 = 0，rho_N 随 N 单调上升到接近 1。
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


class ExactGaussianMixtureVelocity:
    """独立耦合下高斯混合的**精确**边缘 CFM 速度场。"""

    def __init__(self, centers, weights, sigma):
        self.centers = np.asarray(centers, dtype=np.float64)
        self.logw = np.log(np.asarray(weights, dtype=np.float64))
        self.sigma = float(sigma)

    def posterior(self, x, t):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        s2 = (1.0 - t) ** 2 + t ** 2 * self.sigma ** 2
        diff = x[:, None, :] - t * self.centers[None, :, :]
        logp = -0.5 * (diff ** 2).sum(-1) / s2 + self.logw[None, :]
        logp -= logp.max(axis=1, keepdims=True)
        w = np.exp(logp)
        w /= w.sum(axis=1, keepdims=True)
        return w, diff, s2

    def __call__(self, x, t):
        w, diff, s2 = self.posterior(x, t)
        coef = (t * self.sigma ** 2 - (1.0 - t)) / s2
        vel = self.centers[None, :, :] + coef * diff
        return (w[:, :, None] * vel).sum(axis=1)


def euler_map(vel, x0, N):
    """N 步显式欧拉，步长 1/N，返回 f_N(x0)。"""
    x = np.array(x0, dtype=np.float64, copy=True)
    h = 1.0 / N
    for k in range(N):
        x = x + h * vel(x, k * h)
    return x


def realized_spread(pred, X_ref):
    pred = np.asarray(pred, dtype=np.float64)
    return float(np.clip(pred.var(axis=0).sum()
                         / (np.asarray(X_ref).var(axis=0).sum() + 1e-12), 0.0, None))


def mode_coverage(pred, centers, tol):
    d2 = ((pred[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    return int((np.sqrt(d2.min(axis=0)) < tol).sum()), int(len(centers))


def main():
    K, RADIUS, SIGMA = 8, 3.0, 0.25
    th = np.arange(K) * (2.0 * np.pi / K)
    C = np.c_[np.cos(th), np.sin(th)] * RADIUS
    W = np.full(K, 1.0 / K)
    vel = ExactGaussianMixtureVelocity(C, W, SIGMA)

    rng = np.random.default_rng(0)
    N_SAMP = 4000
    X0 = rng.normal(size=(N_SAMP, 2))
    # 目标边缘的真值（闭式：等权混合，中心在半径 R 的圆上）
    tr_var_x1 = float((C ** 2).sum(axis=1).mean() + 2 * SIGMA ** 2)
    m = (W[:, None] * C).sum(axis=0)

    print("=" * 68)
    print("多步欧拉的展布恢复（精确速度场，无训练误差）")
    print("  K=%d 环上等权高斯混合，半径 %.1f，sigma %.2f" % (K, RADIUS, SIGMA))
    print("  闭式 tr Var(x1) = %.4f" % tr_var_x1)
    print("=" * 68)

    # ---- 检验 1：端点恒等式 u(x,0) = m - x ----
    print("\n[检验 1] 端点恒等式  u(x,0) = m - x")
    u0 = vel(X0, 0.0)
    err = np.abs(u0 - (m - X0)).max()
    print("    max |u(x,0) - (m - x)| = %.3e" % err)
    endpoint_ok = err < 1e-9

    # ---- 检验 2：一步塌缩 rho_1 = 0 ----
    print("\n[检验 2] 一步欧拉  f_1(x0) = x0 + u(x0,0) = m")
    f1 = euler_map(vel, X0, 1)
    spread1 = float(f1.var(axis=0).sum())
    print("    tr Var(f_1) = %.3e   （应为 0）" % spread1)
    print("    rho_1 = %.6f" % realized_spread(f1, C * 0 + np.array([[0.0, 0.0]])))
    collapse_ok = spread1 < 1e-9

    # ---- 检验 3：rho_N 随 N 上升 ----
    print("\n[检验 3] rho_N 随步数上升")
    N_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    rows = []
    for N in N_LIST:
        fN = euler_map(vel, X0, N)
        rho = float(fN.var(axis=0).sum() / tr_var_x1)
        cov, tot = mode_coverage(fN, C, tol=0.6)
        rows.append(dict(N=N, rho=rho, modes_covered=cov, n_modes=tot))
        print("    N=%-4d rho=%.4f   覆盖模态 %d/%d" % (N, rho, cov, tot))

    rho = {r["N"]: r["rho"] for r in rows}
    mono = all(rho[N_LIST[i]] <= rho[N_LIST[i + 1]] + 1e-6
               for i in range(len(N_LIST) - 1))
    reaches = rho[256] > 0.90
    starts = rho[1] < 1e-6

    report = {
        "K": K, "radius": RADIUS, "sigma": SIGMA, "n_samples": N_SAMP,
        "closed_form_tr_var_x1": tr_var_x1,
        "endpoint_identity_max_err": float(err),
        "one_step_spread": spread1,
        "sweep": rows,
        "checks": dict(endpoint_identity=bool(endpoint_ok),
                       one_step_collapse=bool(collapse_ok),
                       monotone_in_N=bool(mono),
                       reaches_one=bool(reaches)),
        "verdict": "PASS" if (endpoint_ok and collapse_ok and mono and reaches) else "FAIL",
    }

    print("\n" + "=" * 68)
    print("判据：端点恒等式成立 & rho_1=0 & rho_N 单调 & rho_256 > 0.90")
    print("  " + "  ".join("%s=%s" % (k, v) for k, v in report["checks"].items()))
    print("  VERDICT: %s" % report["verdict"])
    print("=" * 68)

    logs = os.path.join(ROOT, "logs")
    os.makedirs(logs, exist_ok=True)
    out = os.path.join(logs, "verify_multistep_theory.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("报告已写入 %s" % out)


if __name__ == "__main__":
    main()
