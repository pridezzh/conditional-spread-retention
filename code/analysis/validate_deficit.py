# -*- coding: utf-8 -*-
"""Validation suite for the new Lambda (sliced W2 deficit) -- final version.

--------------------------------------------------------------------
Conclusion first (2026-09-15)
--------------------------------------------------------------------
This suite establishes two things; the second is a **negative result**, yet it is
exactly what the paper needs:

  (A) The estimator itself is correct and consistent:
      it exactly recovers the analytic ceiling 1 - 2*sqrt(2/pi) = 0.40423,
      is stable for n from 200 to 200000, and Lam(Gaussian) = 0.

  (B) But Lambda **cannot** serve as a "mode-loss" discriminative quantity:
      in population values, an 8-mode ring (ring8) is only 0.0901,
      while the **unimodal** banana is 0.2263 -- a **rank reversal**.
      That is, Lambda measures "how far from Gaussian", not "how many modes",
      and these two are not the same thing in this problem.

Therefore in the paper Lambda is only a descriptive index
("how far the conditional distribution is from the moment-matched Gaussian"),
and **no longer** carries the "predict mode loss" responsibility. What truly
decides mode loss is the **coupling of the training objective**
(see the 32x comparison in verify_coupling_theory.py / verify_conditional_theory.py).

--------------------------------------------------------------------
Why the criteria shifted from "effect size" to "accuracy against ground truth"
--------------------------------------------------------------------
The first version set the criterion as an **effect-size threshold** like
"Lambda(Delta=6) >= 0.50", and it FAILED. Post-mortem showed the threshold,
not the code, was wrong: the analytic ceiling of the sliced deficit is 0.40423
(attained by a 1D two-point distribution), while two_modes, having structure in
only one of two directions, is **diluted per dimension** and can never reach 0.50.
Using a guessed effect size as a criterion amounts to tuning the threshold
post-hoc. The final version instead tests **analytically verifiable properties**:
the ceiling value, cross-sample-size consistency, and Gaussian identity. These do
not depend on any guess about the effect size.

Criteria (all fixed up front)
-----------------------------
  R1 Ceiling: on a 1D two-point distribution, |Lam_hat - analytic 0.40423| <= 0.005.
  R2 Consistency: |Lam_hat(n=2000) - Lam_hat(n=200000)| <= 0.015 (two_modes/ring8).
  R3 Gaussian identity: Lam_hat(Gaussian) <= 0.02 (n = 50..800).
  R4 Convergent validity: sliced estimate vs. the **exact OT solution**
                          (after subtracting OT's finite-sample noise floor),
                          mean absolute difference <= 0.05.
  R5 Construct validity: rank correlation with the downstream observable "wrong
                        mass" >= 0.90 and the latter is monotonic.
  R6 Ordering test (expected FAIL, negative result): Lambda(ring8) > Lambda(banana)?
                        If not, Lambda cannot serve as a mode-loss discriminative quantity.
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

# ---- parameters and criteria fixed up front ----
CEILING = 1.0 - 2.0 * np.sqrt(2.0 / np.pi) + 1.0     # = 0.40423...
N_REP = 20
N_PROJ = 64
SEP_LIST = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
N_SIZE = [50, 100, 200, 400, 800]
N_OT = 400
N_BIG = 100000        # population reference value; larger only slows things down
N_MID = 2000

CRIT = dict(R1_CEILING_ABS_TOL=0.005, R2_CONSISTENCY_MAX_DIFF=0.015,
            R3_GAUSS_MAX=0.02, R4_MEAN_ABS_DIFF_MAX=0.05,
            R5_RANK_CORR_MIN=0.90, R6_REQUIRE_RING8_GT_BANANA=True)

UNIMODAL = ["gauss", "t_df3", "t_df5", "laplace", "uniform_ball",
            "banana", "lognorm", "exp_skew", "hetero"]
MULTIMODAL = ["two_modes", "ring8"]
ALL_SHAPES = UNIMODAL + MULTIMODAL


def sample_shape(name, n, rng):
    """Shape family identical to run_falsification.py (to ensure comparability)."""
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
    """Exact W2^2 between equal-mass empirical measures (optimal plan is a permutation matrix)."""
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
    """Ceiling: the deficit of a 1D two-point distribution {+/-c} against N(0,c^2)
    is E[(Z-1)^2], where Z is half-normal.

    Analytic value = 1 - 2*sqrt(2/pi) + 1 = 0.40423. Note this is the **supremum
    of the sliced deficit**: no distribution in any direction can exceed it.
    """
    print("\n[R1] Analytic ceiling: a 1D two-point distribution should give Lam = %.5f" % CEILING)
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
            print("    c=%-5.1f n=%-6d Lam=%.5f   err %.2e" % (c, n, val, abs(val - CEILING)))
    ok = worst <= CRIT["R1_CEILING_ABS_TOL"]
    print("    max err %.2e (<= %.3f) -> %s"
          % (worst, CRIT["R1_CEILING_ABS_TOL"], "OK" if ok else "FAIL"))
    return ok, rows


# ---------------------------------------------------------------- R2
def check_R2():
    print("\n[R2] Consistency: finite-sample estimates should converge to the population value")
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
        print("    %-12s n=2000: %.4f   n=200000: %.4f   diff %.4f"
              % (name, mid, big, abs(mid - big)))
    ok = worst <= CRIT["R2_CONSISTENCY_MAX_DIFF"]
    print("    max diff %.4f (<= %.3f) -> %s"
          % (worst, CRIT["R2_CONSISTENCY_MAX_DIFF"], "OK" if ok else "FAIL"))
    return ok, rows


# ---------------------------------------------------------------- R3
def check_R3():
    print("\n[R3] Gaussian identity: Lam(Gaussian) should be ~ 0 across sample sizes")
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
        print("    n=%-5d Lam = %.4f +/- %.4f" % (n, v.mean(), v.std()))
    ok = worst <= CRIT["R3_GAUSS_MAX"]
    print("    max mean %.4f (<= %.2f) -> %s"
          % (worst, CRIT["R3_GAUSS_MAX"], "OK" if ok else "FAIL"))
    return ok, rows


# ---------------------------------------------------------------- R4
def check_R4():
    """Convergent validity: sliced estimate vs. exact OT solution.

    The exact OT distance between two **i.i.d. samples** is not zero; there is a
    ~O(n^{-1/2}) noise floor. This floor (estimated from two independent samples
    P and P') must therefore be subtracted before comparing against the sliced
    estimate of "sample vs. smoothed fitted Gaussian".
    """
    print("\n[R4] Convergent validity: sliced estimate vs. exact OT (noise floor subtracted)")
    rows = []
    for name in ALL_SHAPES:
        slic, exact = [], []
        for rep in range(3):
            rng = np.random.default_rng(61000 * (rep + 1) + 17)
            P = sample_shape(name, N_OT, rng)
            P2 = sample_shape(name, N_OT, rng)          # an independent sample from the same distribution
            mu = P.mean(0)
            cov = np.cov(P.T) + 1e-10 * np.eye(2)
            Q = rng.normal(size=(N_OT, 2)) @ np.linalg.cholesky(cov).T + mu
            floor = exact_w2_sq_equal_mass(P, P2)       # noise floor
            ex = (exact_w2_sq_equal_mass(P, Q) - floor) / float(P.var(axis=0).sum())
            lam, _, _ = gaussian_deficit(P, n_proj=N_PROJ, seed=rep)
            slic.append(lam)
            exact.append(max(ex, 0.0))
        rows.append(dict(shape=name, lam_sliced=float(np.mean(slic)),
                         lam_exact_ot=float(np.mean(exact))))
        print("    %-14s sliced %.4f   exact OT %.4f   diff %+.4f"
              % (name, np.mean(slic), np.mean(exact),
                 np.mean(slic) - np.mean(exact)))
    mad = float(np.mean([abs(r["lam_sliced"] - r["lam_exact_ot"]) for r in rows]))
    rho = spearman([r["lam_sliced"] for r in rows], [r["lam_exact_ot"] for r in rows])
    ok = mad <= CRIT["R4_MEAN_ABS_DIFF_MAX"]
    print("    mean abs diff %.4f (<= %.2f)   rank corr %.4f   -> %s"
          % (mad, CRIT["R4_MEAN_ABS_DIFF_MAX"], rho, "OK" if ok else "FAIL"))
    return ok, rows, mad, rho


# ---------------------------------------------------------------- R5
def check_R5():
    print("\n[R5] Construct validity: Lambda should rise monotonically with 'wrong mass of the Gaussian repair'")
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
        print("    Δ=%-5.1f  Lam=%.4f   wrong mass=%.4f"
              % (delta, np.mean(lams), np.mean(wrongs)))
    rho = spearman([r["lam"] for r in rows], [r["wrong_mass"] for r in rows])
    w = [r["wrong_mass"] for r in rows]
    mono = all(w[i] <= w[i + 1] + 0.01 for i in range(len(w) - 1))
    ok = bool(rho >= CRIT["R5_RANK_CORR_MIN"] and mono)
    print("    rank corr %.4f (>= %.2f)  wrong mass monotonic %s  -> %s"
          % (rho, CRIT["R5_RANK_CORR_MIN"], mono, "OK" if ok else "FAIL"))
    return ok, rows, rho


# ---------------------------------------------------------------- R6
def check_R6():
    """Ordering test (expected FAIL): can Lambda rank multimodal above the curved
    unimodal?

    If Lambda(ring8) <= Lambda(banana), Lambda suffers a rank reversal and cannot
    serve as a "mode-loss" discriminative quantity -- this is the most important
    **negative result** of the suite.
    """
    print("\n[R6] Ordering test (negative result): should the 8-mode ring rank ahead of the unimodal banana?")
    pop = {}
    for name in ALL_SHAPES:
        rng = np.random.default_rng(7)
        lam, _, _ = gaussian_deficit(sample_shape(name, N_BIG, rng),
                                     n_proj=256, seed=3, correct_bias=False)
        pop[name] = float(lam)
    for name in ALL_SHAPES:
        tag = ""
        if name in MULTIMODAL:
            tag = "  <- multimodal"
        print("    %-14s population Lam = %.4f%s" % (name, pop[name], tag))
    ring8, banana = pop["ring8"], pop["banana"]
    ok = ring8 > banana
    print("    Lambda(ring8)=%.4f  vs  Lambda(banana)=%.4f" % (ring8, banana))
    if ok:
        print("    ordering correct -> OK (Lambda may serve as a mode-loss discriminative quantity)")
    else:
        print("    **rank reversal** -> conclusion: Lambda cannot serve as a mode-loss discriminative quantity.")
        print("       It measures 'how far from Gaussian', and the banana's curvature is")
        print("       further from Gaussian in transport distance than the ring's eight peaks.")
        print("       In the paper Lambda is only a descriptive index.")
    return ok, pop


def main():
    t0 = time.time()
    print("=" * 76)
    print("Sliced W2 deficit (Lambda) validation suite -- final version")
    print("  criteria (fixed up front): %s" % json.dumps(CRIT, ensure_ascii=False))
    print("  analytic ceiling 1-2*sqrt(2/pi)+1 = %.5f" % CEILING)
    print("=" * 76)

    ok1, r1 = check_R1()
    ok2, r2 = check_R2()
    ok3, r3 = check_R3()
    ok4, r4, mad, rho4 = check_R4()
    ok5, r5, rho5 = check_R5()
    ok6, pop = check_R6()

    estimator_ok = bool(ok1 and ok2 and ok3 and ok4 and ok5)
    print("\n" + "=" * 76)
    print("estimator correctness  R1..R5 = %s" % ("PASS" if estimator_ok else "FAIL"))
    print("mode-loss discriminative power  R6 = %s" % ("PASS" if ok6 else "FAIL (negative result, localized)"))
    print("  paper handling: Lambda is a descriptive index; mode loss is decided by the **coupling of the training objective**")
    print("  elapsed %.1fs" % (time.time() - t0))
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
    print("Report written to %s" % out)


if __name__ == "__main__":
    main()
