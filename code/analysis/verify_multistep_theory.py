# -*- coding: utf-8 -*-
"""Spread recovery by multi-step Euler: rho_N rises monotonically from 0 to 1 under the
**exact velocity field**.

Why use the exact velocity field instead of a trained network
-----------------------------------------------------------
If a trained velocity field were used, "rho_N rising with N" could mix in two factors:
  1. discretization error shrinking as N grows (the mechanism we want to verify);
  2. the network's fitting error on the velocity field (a confounding factor we do not want).
Using the **closed-form exact velocity field** removes (2) entirely, isolating the mechanism.

Derivation of the exact velocity field (isotropic Gaussian mixture + independent coupling)
------------------------------------------------------------------------------------------
Let x0 ~ N(0, I_d), x1 ~ sum_k pi_k N(mu_k, sigma^2 I_d), with x0 _||_ x1.
Within component k, x_t = (1-t) x0 + t x1 is Gaussian:

    E[x_t | k] = t mu_k
    Var(x_t | k) = s_t^2 I_d,   s_t^2 = (1-t)^2 + t^2 sigma^2

and
    Cov(x1, x_t | k) = t sigma^2 I_d
    Cov(x0, x_t | k) = (1-t) I_d

Hence (linear conditional expectation of a joint Gaussian)

    E[x1 | x_t=x, k] = mu_k + (t sigma^2 / s_t^2) (x - t mu_k)
    E[x0 | x_t=x, k] =        ((1-t)   / s_t^2) (x - t mu_k)

The marginal velocity field u(x,t) = E[x1 - x0 | x_t = x] is weighted by the per-component
posterior w_k(x,t):

    u(x,t) = sum_k w_k(x,t) [ mu_k + ((t sigma^2 - (1-t)) / s_t^2) (x - t mu_k) ]
    w_k(x,t) ∝ pi_k N(x ; t mu_k, s_t^2 I_d)

**Check at t=0**: s_0^2 = 1, and N(x;0,I) is the same for every component, so w_k = pi_k,
the bracket reduces to mu_k - x, and therefore

    u(x,0) = sum_k pi_k (mu_k - x) = m - x          <- endpoint identity

Thus f_1(x0) = x0 + u(x0,0) = m is independent of x0 => rho_1 = 0 (Corollary 1).

And N-step Euler is the composition of N "near-identity maps":
    f_N = (I + u(., t_{N-1})/N) ∘ ... ∘ (I + u(., 0)/N)
Each step is a simple map, but **their composition is a highly nonlinear transport map**,
and as N grows it converges to the exact flow map phi_{0->1} (which pushes p0 to p1).
Hence rho_N -> 1 is expected. This is the mechanism of "how multi-step escapes collapse".

This script verifies numerically with the closed-form velocity field: rho_1 = 0, and
rho_N rises monotonically with N to approach 1.
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
    """Exact marginal CFM velocity field for a Gaussian mixture under independent coupling."""

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
    """N-step explicit Euler with step 1/N, returns f_N(x0)."""
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
    # ground-truth of the target marginal (closed form: equal-weight mixture, centers on a circle of radius R)
    tr_var_x1 = float((C ** 2).sum(axis=1).mean() + 2 * SIGMA ** 2)
    m = (W[:, None] * C).sum(axis=0)

    print("=" * 68)
    print("Spread recovery by multi-step Euler (exact velocity field, no training error)")
    print("  K=%d ring equal-weight Gaussian mixture, radius %.1f, sigma %.2f" % (K, RADIUS, SIGMA))
    print("  closed-form tr Var(x1) = %.4f" % tr_var_x1)
    print("=" * 68)

    # ---- Test 1: endpoint identity u(x,0) = m - x ----
    print("\n[Test 1] Endpoint identity  u(x,0) = m - x")
    u0 = vel(X0, 0.0)
    err = np.abs(u0 - (m - X0)).max()
    print("    max |u(x,0) - (m - x)| = %.3e" % err)
    endpoint_ok = err < 1e-9

    # ---- Test 2: one-step collapse rho_1 = 0 ----
    print("\n[Test 2] One-step Euler  f_1(x0) = x0 + u(x0,0) = m")
    f1 = euler_map(vel, X0, 1)
    spread1 = float(f1.var(axis=0).sum())
    print("    tr Var(f_1) = %.3e   (should be 0)" % spread1)
    print("    rho_1 = %.6f" % realized_spread(f1, C * 0 + np.array([[0.0, 0.0]])))
    collapse_ok = spread1 < 1e-9

    # ---- Test 3: rho_N rises with N ----
    print("\n[Test 3] rho_N rises with step count")
    N_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    rows = []
    for N in N_LIST:
        fN = euler_map(vel, X0, N)
        rho = float(fN.var(axis=0).sum() / tr_var_x1)
        cov, tot = mode_coverage(fN, C, tol=0.6)
        rows.append(dict(N=N, rho=rho, modes_covered=cov, n_modes=tot))
        print("    N=%-4d rho=%.4f   modes covered %d/%d" % (N, rho, cov, tot))

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
    print("Criteria: endpoint identity holds & rho_1=0 & rho_N monotone & rho_256 > 0.90")
    print("  " + "  ".join("%s=%s" % (k, v) for k, v in report["checks"].items()))
    print("  VERDICT: %s" % report["verdict"])
    print("=" * 68)

    logs = os.path.join(ROOT, "logs")
    os.makedirs(logs, exist_ok=True)
    out = os.path.join(logs, "verify_multistep_theory.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Report written to %s" % out)


if __name__ == "__main__":
    main()
