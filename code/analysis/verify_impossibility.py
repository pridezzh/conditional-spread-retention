# -*- coding: utf-8 -*-
r"""Constructive validation of the impossibility theorem: let rho scan [0,1] exactly, while the data distribution stays fixed.

--------------------------------------------------------------------
This is the experiment carrying the paper's novelty; first clarify what it proves
--------------------------------------------------------------------
The coupling literature has already said "coupling affects one-step quality". So this
paper **cannot** merely claim "coupling matters". What this experiment proves is a
stronger claim that no one has stated before:

    **[Impossibility]** Fix $p(c,x_1)$. Then the values of $\rho(f^*)$ **cover the
    whole $[0,1]$**, and both endpoints are attainable -- yet the family of training
    configurations that achieves this has $(c,x_1)$ marginals that are **identical
    sample by sample**.

Corollary (this is the crux):

    Any functional $\mathcal A(p(c,x_1))$ that depends **only on the data distribution**
    -- whether $\hat D$, $\Lambda$, or any "multimodal score" proposed in the future --
    **cannot** determine whether one step loses modes. Because the two situations it
    must distinguish share the same data distribution.
    **This is not an estimation-accuracy problem; it is insufficient information.**

--------------------------------------------------------------------
Construction: alpha-interpolation coupling, with an analytic prediction
--------------------------------------------------------------------
Fix the conditional distribution $p_1(\cdot\mid c)$, and let $T(\cdot,c)$ be any
deterministic transport map pushing $p_0$ to $p_1(\cdot\mid c)$
(here we use the **512-step Euler flow map of the exact velocity field**, i.e. the map that reflow produces).

Define the **alpha-mixture coupling**:

    x_1 = B·T(x_0,c) + (1-B)·X,    B ~ Bernoulli(α),  X ~ p_1(·|c) independent of x_0.

Both branches have marginal $p_1(\cdot\mid c)$ ==> **the whole (c,x_1) marginal is independent of α**.

The optimal one-step map is

    E[x_1 | x_0, c] = α·T(x_0,c) + (1-α)·m(c)

    ==> ρ*(α) = α² · tr Var(x_1|c) / tr Var(x_1|c) = **α²**

**Thus ρ*(α) = α² scans [0,1] exactly, while the data distribution stays fixed.** This
is the constructive proof of the impossibility theorem, and it comes with a
**parameter-free quantitative law** that can be tested directly.

--------------------------------------------------------------------
Correction term when measuring with kNN (also analytic)
--------------------------------------------------------------------
When using uniform-weight kNN to **measure** ρ, the variance of the prediction
$\frac1k\sum_{i\in N_k}x_1^{(i)}$ has two parts:
  * from the alpha branch: $\approx α T(x_0,c)$, varying with $x_0$, variance
    $= α^2\,\mathrm{trVar}(x_1|c)$;
  * conditional residual: by the law of total variance, its averaged variance is
    $(1-α^2)\,\mathrm{trVar}(x_1|c)$; taking the mean of k local responses divides it by k.

    ==> **ρ̂_kNN(α) ≈ α² + (1-α²)/k**

At α=0 it reduces to the known $1/k$ law; at α=1 the prediction is $\approx1$. Both terms are tested.

Criteria (fixed up front)
-------------------------
  P1 rho_hat is non-decreasing in alpha
  P2 rho_hat(0) < 0.10, rho_hat(1) > 0.85
  P3 **|D_hat(α) - D_hat(0)| < 0.02** (data marginals identical)
  P4 **|Lambda_hat(α) - Lambda_hat(0)| < 0.05** (same reason)
  P5 mean absolute deviation of rho_hat from the analytic prediction alpha^2+(1-alpha^2)/k < 0.08
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

# ---- parameters (fixed up front) ----
C, K, SIGMA, DIM = 4, 8, 0.25, 2
ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)
N_TRAIN = 4000
N_QUERY = 2000
KNN_K = (20, 50)
N_T_STEPS = 512          # integrate the exact velocity field into the deterministic transport map T
SEEDS = (0, 1, 2)
MODE_TOL = 0.6

CRIT = dict(P2_RHO0_MAX=0.10, P2_RHO1_MIN=0.85,
            P3_D_MAX_DIFF=0.02, P4_LAM_MAX_DIFF=0.05,
            P5_PRED_MAD_MAX=0.08)


def knn_variance_ratio(alpha, k):
    """Variance ratio of the kNN conditional-mean estimator under an ideal local neighborhood.

    Var(E[Y|X]) / Var(Y) = alpha**2; the remaining conditional-variance fraction is
    1-alpha**2, not (1-alpha)**2.
    """
    return alpha ** 2 + (1.0 - alpha ** 2) / k


# ---------------------------------------------------------------- geometry and exact velocity field
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
    """Per-condition **exact** marginal CFM velocity field for Gaussian mixtures under
    independent coupling (closed form)."""

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
    """Deterministic transport map T(·, j): the N-step Euler flow map of the exact
    velocity field (the one reflow produces)."""
    x = np.array(x0, dtype=np.float64, copy=True)
    h = 1.0 / N
    for i in range(N):
        x = x + h * VEL(x, i * h, j)
    return x


def sample_marginal(n, j, rng):
    """Independent sampling from p_1(·|j)."""
    comp = rng.integers(0, K, size=n)
    return CENTERS[j, comp] + SIGMA * rng.normal(size=(n, DIM))


# ---------------------------------------------------------------- alpha-mixture coupling
def sample_alpha(n, j, alpha, rng):
    """Alpha-mixture coupling: with probability alpha take the deterministic branch
    T(x0), otherwise an independent sample.

    **Both branches have marginal p_1(·|j), so the (j, x1) marginal is independent of alpha.**
    """
    x0 = rng.normal(size=(n, DIM))
    B = rng.random(n) < alpha
    x1 = np.empty((n, DIM))
    if B.any():
        x1[B] = transport_map(x0[B], j)
    if (~B).any():
        x1[~B] = sample_marginal(int((~B).sum()), j, rng)
    return x0, x1


# ---------------------------------------------------------------- estimators
def knn_uniform(Xtr, Ytr, Xq, k):
    d2 = ((Xq[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    return Ytr[idx].mean(axis=1)


def tr_var(a):
    return float(np.asarray(a, dtype=np.float64).var(axis=0).sum())


def d_hat_single(X1_by_cond):
    """D = E_c[tr Var(x1|c)] / tr Var(x1) (without introducing any regressor)."""
    within = float(np.mean([tr_var(X1_by_cond[j]) for j in range(C)]))
    total = tr_var(X1_by_cond.reshape(-1, DIM))
    return within / total


def mode_coverage(pred, centers_j, tol=MODE_TOL):
    d2 = ((pred[:, None, :] - centers_j[None, :, :]) ** 2).sum(-1)
    return int((np.sqrt(d2.min(axis=0)) < tol).sum())


# ---------------------------------------------------------------- main flow
def run_seed(seed):
    rng = np.random.default_rng(seed)
    rows = []
    for alpha in ALPHAS:
        # --- fix a batch of x0 and query points, reused across alpha to reduce irrelevant noise ---
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
        # --- data-side statistics (assertions independent of alpha) ---
        Yall = np.stack(Ytr)                                  # (C, n, d)
        D = d_hat_single(Yall)
        lam_per_cond = [gaussian_deficit(Yall[j], n_proj=64, seed=seed)[0]
                        for j in range(C)]
        lam = float(np.mean(lam_per_cond))
        rows.append(dict(alpha=float(alpha), D=D, lam=lam,
                         rho={str(k): rhos[k] for k in KNN_K},
                         coverage={str(k): covs[k] for k in KNN_K}))
        print("    α=%.2f  D=%.4f  Λ=%.4f  |  " % (alpha, D, lam)
              + "   ".join("rho_hat(k=%d)=%.4f (cover %.1f/8)" % (k, rhos[k], covs[k])
                           for k in KNN_K), flush=True)
    return rows


def main():
    t0 = time.time()
    print("=" * 78)
    print("Constructive validation of the impossibility theorem: rho*(alpha) = alpha^2 scans [0,1], data distribution fixed")
    print("  criteria (fixed up front): %s" % json.dumps(CRIT, ensure_ascii=False))
    print("  kNN analytic correction: rho_hat(alpha) ~ alpha^2 + (1-alpha^2)/k")
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
    print("Aggregated over %d seeds (k=%d)" % (len(SEEDS), KNN_K[0]))
    k0 = KNN_K[0]
    print("  %-6s %-12s %-12s %-12s %-12s" % ("alpha", "rho_hat", "alpha^2", "alpha^2+(1-a^2)/k", "D_hat / Lambda_hat"))
    print("-" * 78)
    curve, pred_curve = [], []
    for a in ALPHAS:
        rho, _ = agg("rho", a, k0)
        D, _ = agg("D", a)
        lam, _ = agg("lam", a)
        pred = knn_variance_ratio(a, k0)
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
    print("  P1 mono=%s   P2 endpoints rho_hat(0)=%.4f<%.2f, rho_hat(1)=%.4f>%.2f -> %s"
          % (mono, rho0, CRIT["P2_RHO0_MAX"], rho1, CRIT["P2_RHO1_MIN"], checks["P2_endpoints"]))
    print("  P3 max|D_hat(alpha)-D_hat(0)| = %.5f (< %.3f) -> %s"
          % (ddev, CRIT["P3_D_MAX_DIFF"], checks["P3_D_invariant"]))
    print("  P4 max|Lambda_hat(alpha)-Lambda_hat(0)| = %.5f (< %.3f) -> %s"
          % (ldev, CRIT["P4_LAM_MAX_DIFF"], checks["P4_Lambda_invariant"]))
    print("  P5 mean abs diff from analytic alpha^2+(1-alpha^2)/k = %.4f (< %.3f) -> %s"
          % (mad, CRIT["P5_PRED_MAD_MAX"], checks["P5_matches_analytic"]))
    verdict = "PASS" if all(checks.values()) else "FAIL"
    print("  VERDICT: %s     elapsed %.0fs" % (verdict, time.time() - t0))
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
    print("Report written to %s" % out)


if __name__ == "__main__":
    main()
