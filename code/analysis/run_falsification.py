# -*- coding: utf-8 -*-
"""Minimally falsifiable experiment: a false-positive stress test for Lambda on
"non-Gaussian unimodal" distributions.

Tests one of the pre-registered veto conditions for the deficit index Lambda:
that multi-modal separation statistics have excessive false positives on
heavy-tailed / curved unimodal distributions.

Why this must be done first
---------------------------
Lambda's unimodal null hypothesis is a Gaussian whose mean and covariance match
the sample (`_gaussian_null`). The conditional distribution of real data can be
perfectly **unimodal yet clearly non-Gaussian**: heavy-tailed, skewed, a curved
manifold, heteroscedastic, or tightly supported. If such shapes are
systematically misclassified as multimodal, Lambda can no longer distinguish
"structurally lost modes" from "oddly shaped unimodal" distributions, and the
paper's second discrimination axis (benign blur vs. catastrophic mode loss)
becomes meaningless.

This test **needs no training, no GPU, and finishes within minutes**, yet it
decides whether the entire discriminative-quantity line of research holds. So
run it before spending hours rerunning the 162 experiments.

Design
------
Construct a set of **known-unimodal** distributions (including several non-Gaussian
shapes) and a set of **known-multimodal** distributions, call `_select_K`
repeatedly and independently for each, and tally the fraction returning K >= 2:

    unimodal set   -> false positive rate FPR, nominal significance level alpha = 0.05
    multimodal set -> sensitivity TPR, to confirm the test has not degenerated
                      into "always returns K = 1"

Pre-registered criteria (fixed up front, not adjusted to results)
---------------------------------------------------------------
    FAIL_FPR = 0.20   if any unimodal shape's FPR exceeds this -> null fails for that shape
    FAIL_TPR = 0.80   if any multimodal shape's TPR falls below this -> test loses power
    if either is hit, the "discriminative quantity" claim is dropped per the
    pre-registration rule.
"""
import json
import os
import sys
import time


def _find_root(d):
    """Self-healing anchor: walk up until a level containing both code/ and results/."""
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
from discriminant import _select_K, _lam_of  # noqa: E402

# ---- experiment parameters (fixed up front) ----
N_REP = 20
N_SAMP = 200
N_NULL = 49
KMAX = 5
ALPHA = 0.05
FAIL_FPR = 0.20
FAIL_TPR = 0.80

# known-unimodal (incl. Gaussian control and several non-Gaussian shapes)
UNIMODAL = ["gauss", "t_df3", "t_df5", "laplace", "uniform_ball",
            "banana", "lognorm", "exp_skew", "hetero"]
# known-multimodal (for sensitivity, not false positives)
MULTIMODAL = ["two_modes", "ring8"]


def sample_shape(name, n, rng):
    """Generate one shape. All but two_modes / ring8 are unimodal."""
    if name == "gauss":                      # control: true distribution under the null
        return rng.normal(size=(n, 2))
    if name == "t_df3":                      # heavy-tailed, variance barely converges
        return rng.standard_t(3.0, size=(n, 2)) * 0.7
    if name == "t_df5":                      # moderately heavy-tailed
        return rng.standard_t(5.0, size=(n, 2)) * 0.8
    if name == "laplace":                    # sharp peak + heavier tails than Gaussian
        return rng.laplace(size=(n, 2))
    if name == "uniform_ball":               # compact support, much lighter tails than Gaussian
        d = rng.normal(size=(n, 2))
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        r = np.sqrt(rng.uniform(0.0, 1.0, size=(n, 1)))
        return d * r * 2.0
    if name == "banana":                     # curved manifold: 1D curve embedded in 2D
        x1 = rng.normal(scale=1.0, size=(n, 1))
        x2 = x1 ** 2 + rng.normal(scale=0.30, size=(n, 1))
        return np.hstack([x1, x2 - 1.0])
    if name == "lognorm":                    # strongly right-skewed
        z = rng.normal(size=(n, 2))
        return np.exp(0.6 * z) - np.exp(0.18)
    if name == "exp_skew":                   # exponential-type skew
        return rng.exponential(size=(n, 2)) - 1.0
    if name == "hetero":                     # heteroscedastic: spread varies with location
        x1 = rng.normal(size=(n, 1))
        sd = 0.20 + 0.80 * np.abs(x1)
        return np.hstack([x1, rng.normal(scale=sd)])
    if name == "two_modes":                  # true bimodal, well separated
        half = n // 2
        a = rng.normal(-2.5, 0.25, size=(half, 2))
        b = rng.normal(2.5, 0.25, size=(n - half, 2))
        return np.r_[a, b]
    if name == "ring8":                      # 8-mode ring from the paper's synthetic family
        ang = rng.integers(0, 8, size=n)
        th = ang * (2.0 * np.pi / 8.0)
        cen = np.c_[np.cos(th), np.sin(th)] * 3.0
        return cen + rng.normal(scale=0.25, size=(n, 2))
    raise ValueError("unknown shape: %s" % name)


def run_one(name, rep):
    rng = np.random.default_rng(1000 * (rep + 1) + 7)
    R = sample_shape(name, N_SAMP, rng)
    t0 = time.time()
    K = _select_K(R, kmax=KMAX, seed=7919 * (rep + 1), n_null=N_NULL, alpha=ALPHA)
    lam = 0.0
    if K >= 2:
        lam, _ = _lam_of(R, K, seed=7919 * (rep + 1))
    return int(K), float(lam), float(time.time() - t0)


def main():
    report = {
        "experiment": "lambda_false_positive_stress",
        "purpose": "False-positive rate of Lambda's Gaussian unimodal null on "
                   "non-Gaussian unimodal distributions",
        "params": dict(N_REP=N_REP, N_SAMP=N_SAMP, N_NULL=N_NULL,
                       KMAX=KMAX, ALPHA=ALPHA),
        "criteria": dict(FAIL_FPR=FAIL_FPR, FAIL_TPR=FAIL_TPR),
        "unimodal": {},
        "multimodal": {},
    }

    for group, names in (("unimodal", UNIMODAL), ("multimodal", MULTIMODAL)):
        for name in names:
            ks, lams, secs = [], [], []
            for rep in range(N_REP):
                K, lam, dt = run_one(name, rep)
                ks.append(K)
                lams.append(lam)
                secs.append(dt)
                print("  [%s] rep %2d/%d  K=%d  lambda=%.3f  (%.2fs)"
                      % (name, rep + 1, N_REP, K, lam, dt), flush=True)
            ks = np.array(ks)
            rate = float((ks >= 2).mean())
            entry = {
                "K>=2_rate": rate,
                "K_mean": float(ks.mean()),
                "lambda_mean": float(np.mean(lams)),
                "n_rep": N_REP,
                "sec_per_call": float(np.mean(secs)),
            }
            report[group][name] = entry
            print("== %-14s K>=2 rate = %.2f  (mean K %.2f, mean lambda %.3f)"
                  % (name, rate, ks.mean(), np.mean(lams)), flush=True)

    # ---- criteria ----
    fp_fail = {k: v["K>=2_rate"] for k, v in report["unimodal"].items()
               if v["K>=2_rate"] > FAIL_FPR}
    tp_fail = {k: v["K>=2_rate"] for k, v in report["multimodal"].items()
               if v["K>=2_rate"] < FAIL_TPR}
    report["false_positive_failures"] = fp_fail
    report["power_failures"] = tp_fail
    report["verdict"] = "PASS" if not (fp_fail or tp_fail) else "FAIL"

    logs = os.path.join(ROOT, "logs")
    os.makedirs(logs, exist_ok=True)
    out = os.path.join(logs, "falsification_lambda_stress.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 62)
    print("Pre-registered criteria: unimodal FPR > %.2f -> fail; "
          "multimodal TPR < %.2f -> fail" % (FAIL_FPR, FAIL_TPR))
    print("-" * 62)
    for name in UNIMODAL:
        r = report["unimodal"][name]["K>=2_rate"]
        flag = "  <-- false positive too high" if name in fp_fail else ""
        print("  unimodal %-14s FPR = %.2f%s" % (name, r, flag))
    for name in MULTIMODAL:
        r = report["multimodal"][name]["K>=2_rate"]
        flag = "  <-- insufficient power" if name in tp_fail else ""
        print("  multimodal %-14s TPR = %.2f%s" % (name, r, flag))
    print("-" * 62)
    print("VERDICT: %s" % report["verdict"])
    print("Report written to %s" % out)
    print("=" * 62)


if __name__ == "__main__":
    main()
