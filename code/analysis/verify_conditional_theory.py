# -*- coding: utf-8 -*-
"""Conditional coupling-determinism theorem: full validation on a controlled geometry family.

Why a conditional version is required
-------------------------------------
The paper concerns the **policy** pi(a|s), i.e. conditional generation. Every
conclusion of the unconditional version must be re-verified "after fixing the
condition c", because:

  * the discriminative quantity D (fraction of variance unexplained by condition)
    is defined per c;
  * the collapse of the one-step map f_1(., c) must hold per c to be a true collapse;
  * most importantly, this experiment must demonstrate **identical D but wildly
    different rho**.

Controlled design (the core of this script)
-------------------------------------------
Fix the **same** conditional data distribution p(x1|c):
    under condition j, x1 follows a ring mixture of K equal-weight Gaussians,
    ring center = translated b_j + radius R_j rotated ring, component variance sigma^2.
Hence tr Var(x1|j) = R_j^2 + d sigma^2 (closed form), E[x1|j] = b_j.

On this **identical marginal distribution**, only the coupling in the training
objective is changed:

  * independent coupling (the standard CFM / DDPM approach): x0 independent of x1;
  * transport coupling (the essence of reflow / OT-FM / consistency distillation /
    IMLE): x0 and x1 are paired one-to-one by optimal transport, almost deterministic.

Both have identical per-sample data marginals -> D identical (criterion C5),
but the spread-preservation rate rho of the one-step map should differ by an order
of magnitude or more (criterion C4).

Exact conditional velocity field
--------------------------------
As in the unconditional case (linear conditional expectation of a joint Gaussian),
for a given condition j:

    u(x, t, j) = sum_k w_k(x,t,j) [ mu_{j,k} + ((t sigma^2 - (1-t)) / s_t^2) (x - t mu_{j,k}) ]
    w_k(x,t,j) ∝ N(x ; t mu_{j,k}, s_t^2 I),   s_t^2 = (1-t)^2 + t^2 sigma^2

At t = 0, w_k is independent of k (common factor N(x;0,I)), so the bracket reduces
to mu_{j,k} - x, and therefore

    u(x, 0, j) = mean_k mu_{j,k} - x = b_j - x        <- conditional endpoint identity
    f_1(x0, j) = x0 + u(x0, 0, j) = b_j   independent of x0 => rho_1(j) = 0

Using the closed-form velocity field completely removes the confounding factor of
"network fitting error", isolating the "discretization / coupling" mechanism alone.
"""
import json
import os
import sys


def _find_root(d):
    d = os.path.abspath(d)
    while True:
        if os.path.isdir(os.path.join(d, "code")) and os.path.isdir(os.path.join(d, "results")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("project root not found")
        d = parent


ROOT = _find_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code", "src"))

import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

# ---------------------------------------------------------------- experiment parameters
C = 4               # number of conditions
K = 8               # number of modes per condition
SIGMA = 0.25        # intrinsic std of each mode
DIM = 2
N_EXACT = 3000      # samples per condition for the exact velocity-field experiment
N_OT = 600          # samples per condition for the transport-coupling experiment (Hungarian is O(n^3))
N_QUERY = 600       # query points used to estimate rho (kept separate from the training set to avoid overfitting)
K_LIST = (1, 2, 5, 10, 20, 30)     # k for kNN; valid regime requires k <= n/20 = 30
N_LIST = (1, 2, 4, 8, 16, 32, 64, 128, 256)
MODE_TOL = 0.6      # distance tolerance for deciding a mode is "covered"
SEED = 0


# ---------------------------------------------------------------- geometry family
def build_geometry():
    """Return the ring centers (C, K, DIM) per condition, the conditional mean b_j,
    and the closed-form tr Var(x1|j)."""
    centers = np.zeros((C, K, DIM))
    b = np.zeros((C, DIM))
    radii = np.zeros(C)
    for j in range(C):
        # conditional mean shifts with j -- guarantees D < 1 (the conditionally explainable part)
        b[j] = np.array([1.6 * (j / (C - 1)) - 0.8, 0.9 * (j / (C - 1)) - 0.45])
        R = 1.5 + 0.4 * j
        radii[j] = R
        phi = (np.pi / 6.0) * j
        for k in range(K):
            th = k * (2.0 * np.pi / K) + phi
            centers[j, k] = b[j] + R * np.array([np.cos(th), np.sin(th)])
    # closed form: the mean of an equal-weight ring mixture is b_j, hence tr Var(x1|j) = R_j^2 + d sigma^2
    tr_var_cond = radii ** 2 + DIM * SIGMA ** 2
    return centers, b, radii, tr_var_cond


def sample_x1(centers, n_per_cond, rng):
    """Sample from the conditional mixture: the i-th sample under condition j belongs
    to a random mode."""
    out = np.zeros((C, n_per_cond, DIM))
    for j in range(C):
        comp = rng.integers(0, K, size=n_per_cond)
        out[j] = centers[j, comp] + SIGMA * rng.normal(size=(n_per_cond, DIM))
    return out


# ---------------------------------------------------------------- exact velocity field
class ConditionalExactVelocity:
    """Per-condition **exact** marginal CFM velocity field under independent coupling
    (closed form, no training error)."""

    def __init__(self, centers):
        self.centers = np.asarray(centers, dtype=np.float64)   # (C,K,D)

    def __call__(self, x, t, j):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        mu = self.centers[j]                                    # (K,D)
        s2 = (1.0 - t) ** 2 + t ** 2 * SIGMA ** 2
        diff = x[:, None, :] - t * mu[None, :, :]               # (n,K,D)
        logp = -0.5 * (diff ** 2).sum(-1) / s2                  # equal weights, no log w
        logp -= logp.max(axis=1, keepdims=True)
        w = np.exp(logp)
        w /= w.sum(axis=1, keepdims=True)
        coef = (t * SIGMA ** 2 - (1.0 - t)) / s2
        vel = mu[None, :, :] + coef * diff
        return (w[:, :, None] * vel).sum(axis=1)


def euler_map_cond(vel, x0, j, N):
    """N-step explicit Euler with step 1/N, returns f_N(x0, j)."""
    x = np.array(x0, dtype=np.float64, copy=True)
    h = 1.0 / N
    for i in range(N):
        x = x + h * vel(x, i * h, j)
    return x


def mode_coverage(pred, centers_j, tol=MODE_TOL):
    d2 = ((pred[:, None, :] - centers_j[None, :, :]) ** 2).sum(-1)
    return int((np.sqrt(d2.min(axis=0)) < tol).sum())


# ---------------------------------------------------------------- estimators
def knn_uniform(X_train, Y_train, X_query, k):
    """Uniform-weight kNN regression. Uniform weights (not distance-weighted) are
    needed to expose the 1/k variance law."""
    d2 = ((X_query[:, None, :] - X_train[None, :, :]) ** 2).sum(-1)
    idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    return Y_train[idx].mean(axis=1)


def ot_pairing(x0, x1):
    """Optimal-transport pairing: returns the x1 paired one-to-one with x0
    (Hungarian solution to the L2 transport problem)."""
    d2 = ((x0[:, None, :] - x1[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(d2)
    return x1[c]


def tr_var(a):
    return float(np.asarray(a, dtype=np.float64).var(axis=0).sum())


def d_hat(X1_by_cond):
    """D = E_c[tr Var(x1|c)] / tr Var(x1) (fraction of variance unexplained by condition)."""
    within = float(np.mean([tr_var(X1_by_cond[j]) for j in range(C)]))
    total = tr_var(X1_by_cond.reshape(-1, X1_by_cond.shape[-1]))
    return within / total


# ---------------------------------------------------------------- main flow
def main():
    rng = np.random.default_rng(SEED)
    centers, b, radii, tr_var_cond = build_geometry()
    vel = ConditionalExactVelocity(centers)

    print("=" * 72)
    print("Conditional coupling-determinism theorem verification (controlled geometry family)")
    print("  C=%d conditions, each with K=%d ring Gaussian mixture, sigma=%.2f" % (C, K, SIGMA))
    print("  per-condition closed-form tr Var(x1|j) = %s" % np.round(tr_var_cond, 4).tolist())
    print("=" * 72)

    # ================= Test 1: conditional endpoint identity =================
    print("\n[Test 1] Conditional endpoint identity  u(x,0,j) = b_j - x")
    errs = []
    for j in range(C):
        X0 = rng.normal(size=(N_EXACT, DIM))
        u0 = vel(X0, 0.0, j)
        errs.append(float(np.abs(u0 - (b[j] - X0)).max()))
        print("    cond %d: max|u(x,0,j)-(b_j-x)| = %.3e" % (j, errs[-1]))
    endpoint_ok = max(errs) < 1e-9

    # ================= Test 2: per-condition one-step collapse =================
    print("\n[Test 2] One-step map  f_1(x0,j) = b_j  => rho_1(j) = 0")
    X0_all = rng.normal(size=(C, N_EXACT, DIM))
    rho1 = []
    for j in range(C):
        f1 = euler_map_cond(vel, X0_all[j], j, 1)
        s = tr_var(f1)
        print("    cond %d: tr Var(f_1) = %.3e   rho_1 = %.3e   modes covered %d/%d"
              % (j, s, s / tr_var_cond[j], mode_coverage(f1, centers[j]), K))
        rho1.append(s / tr_var_cond[j])
    collapse_ok = max(rho1) < 1e-6

    # ================= Test 3: per-condition rho_N rises monotonically to 1 =================
    print("\n[Test 3] Per-condition rho_N rises with step count")
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

    # ================= Tests 4/5: same D, different coupling =================
    print("\n[Test 4] On the same conditional distribution, only the coupling changes (per-sample data marginals identical)")
    X0_ot = rng.normal(size=(C, N_OT, DIM))
    X1_ind = sample_x1(centers, N_OT, rng)          # independent coupling: x1 independent of x0
    X1_det = np.zeros_like(X1_ind)                  # transport coupling: per-condition Hungarian pairing
    for j in range(C):
        X1_det[j] = ot_pairing(X0_ot[j], X1_ind[j])

    D_ind = d_hat(X1_ind)
    D_det = d_hat(X1_det)
    print("    D(independent)   = %.4f" % D_ind)
    print("    D(transport)     = %.4f" % D_det)
    print("    |D_ind-D_det| = %.4f   (should be ~0: data marginals identical)" % abs(D_ind - D_det))
    d_invariant_ok = abs(D_ind - D_det) < 0.03

    Xq = rng.normal(size=(C, N_QUERY, DIM))
    rows = []
    print("\n    Uniform-weight kNN one-step spread-preservation rate rho_hat(k) (valid regime k <= n/20 = %d)" % (N_OT // 20))
    print("    %-5s %-14s %-14s %-10s %-10s" % ("k", "indep rho", "trans rho", "ratio", "1/k law check"))
    for k in K_LIST:
        num_i = np.mean([tr_var(knn_uniform(X0_ot[j], X1_ind[j], Xq[j], k)) for j in range(C)])
        num_d = np.mean([tr_var(knn_uniform(X0_ot[j], X1_det[j], Xq[j], k)) for j in range(C)])
        den = float(np.mean(tr_var_cond))
        ri, rd = num_i / den, num_d / den
        rows.append(dict(k=k, rho_ind=ri, rho_det=rd, ratio=rd / max(ri, 1e-12),
                         k_times_rho_ind=k * ri))
        print("    %-5d %-14.4f %-14.4f %-10.1f %-10.3f"
              % (k, ri, rd, rd / max(ri, 1e-12), k * ri))

    # judge contrast within the kNN-consistent valid regime (k <= n/20)
    valid = [r for r in rows if r["k"] <= N_OT // 20]
    best = max(valid, key=lambda r: r["ratio"])
    contrast_ok = best["ratio"] > 10.0
    # 1/k law: under independent coupling k * rho_hat(k) should be ~1 (allowing estimation noise)
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
    print("Criteria (all pre-declared, not relaxed post-hoc)")
    for kk, vv in checks.items():
        print("    %-38s %s" % (kk, vv))
    print("  max contrast %.1fx  (k=%d)" % (best["ratio"], best["k"]))
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
    print("Report written to %s" % out)


if __name__ == "__main__":
    main()
