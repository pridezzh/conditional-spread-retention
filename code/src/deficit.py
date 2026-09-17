# -*- coding: utf-8 -*-
"""A new definition of Lambda: sliced W2 deficit of a conditional distribution
against its moment-matched Gaussian (Gaussian deficit).

**Pure numpy implementation, no sklearn / scipy dependency.**

--------------------------------------------------------------------
1. Why the definition must change (instead of fixing the old implementation's null)
--------------------------------------------------------------------
The old Lambda claimed to test for "multimodality", but it actually tested for
"non-Gaussianity": the null hypothesis was a Gaussian with the same mean and
covariance as the sample. And

        Gaussian ⊊ unimodal

so any **unimodal but non-Gaussian** conditional distribution is systematically
misclassified as multimodal. The stress test
(``code/analysis/run_falsification.py``, pre-registered veto line FPR > 0.20)
measured in practice:

    lognorm 0.85   banana 0.95   uniform_ball 1.00   exp_skew 1.00

all hit the veto line. Note one **counter-intuitive but important** detail: heavy
tails actually make the old test *more conservative* (t_df3 / t_df5 / laplace all
have FPR 0.00) — because heavy tails inflate the "pooled within-cluster variance"
and suppress the separation. So this is not a bias fixable by "tuning a threshold";
rather, **the question the statistic asks is itself wrong**.

This root cause **cannot be cured by changing the null either**: per-coordinate
rank Gaussianization only changes the marginal distribution, not joint structure
such as the banana's curvature or uniform_ball's compact support; a rigorous
multivariate unimodality test would use the Hartigan dip, whose critical values
rely on the asymptotic result that "the uniform distribution is the least favorable
unimodal case", which is hard to calibrate at finite sample sizes, and multivariate
unimodality itself is not a well-defined concept.

--------------------------------------------------------------------
2. The new definition: stop asking "is it multimodal"
--------------------------------------------------------------------
What practitioners really care about is a question **answerable without a null
hypothesis**:

    A one-step map can only output the conditional mean m(c) as a single point.
    The most common remedy is "add a covariance-matched Gaussian noise around m(c)".
    Question: does this remedy actually work?

Write this as a quantity:

    Lam(c) = E_theta [ W2^2( theta#p(·|c) , N(theta'm(c), theta'Sigma(c)theta) ) ]
             / E_theta [ theta' Sigma(c) theta ]

where theta is a random direction on the unit sphere, theta#p denotes the 1-D
projected distribution, and Sigma(c) = Cov(x|c). The denominator is the "average
conditional variance per direction" (= tr Sigma(c)/d).

Properties:
  * Lam ∈ [0, ~1], **scale-invariant**;
  * Lam = 0 iff p(·|c) is Gaussian on **every** 1-D projection;
  * No null hypothesis, no p-value, no FPR/TPR — it is an index that can be read
    directly, so the pre-registered veto line in the audit report that targeted the
    "multimodality test" **no longer applies**.

Why W2 instead of CH / chi-square-type statistics:
    W2 is a **transport distance**: it asks "how far must mass be moved", and its
    sensitivity to tails is polynomial; whereas CH is essentially a between/within
    cluster variance ratio, which heavy tails blow up (or conversely suppress). This
    is exactly the robustness needed against t-distributions, log-normal, and other
    shape effects.

--------------------------------------------------------------------
3. Finite-sample bias and first-order correction
--------------------------------------------------------------------
Estimating W2^2 from empirical quantiles has a positive O(1/n) bias: the extreme
order statistics systematically deviate from the Gaussian quantiles, and the bias
diverges in the tail as 1/phi(z) (truncated at finite n by u = 1/(2n)). This module
uses a **Monte-Carlo baseline from same-size Gaussian samples** for first-order
correction:

    W2^2_corr = W2^2_raw - b(n) * s^2

where b(n) = E[W2^2(n i.i.d. standard normal samples, its moment-matched Gaussian)],
cached by n. By construction the corrected value gives Lam(gauss) = 0 (see test R1
in ``validate_deficit.py``).
"""
import numpy as np

_BASELINE_CACHE = {}


# ---------------------------------------------------------------- inverse normal CDF
def _norm_ppf(u):
    """Inverse standard normal CDF (Acklam rational approximation, relative error
    < 1.2e-9), kept numpy-only."""
    u = np.asarray(u, dtype=np.float64)
    a = np.array([-3.969683028665376e+01, 2.209460984245205e+02,
                  -2.759285104469687e+02, 1.383577518672690e+02,
                  -3.066479806614716e+01, 2.506628277459239e+00])
    b = np.array([-5.447609879822406e+01, 1.615858368580409e+02,
                  -1.556989798598866e+02, 6.680131188771972e+01,
                  -1.328068155288572e+01])
    c = np.array([-7.784894002430293e-03, -3.223964580411365e-01,
                  -2.400758277161838e+00, -2.549732539343734e+00,
                  4.374664141464968e+00, 2.938163982698783e+00])
    d = np.array([7.784695709041462e-03, 3.224671290700398e-01,
                  2.445134137142996e+00, 3.754408661907416e+00])
    out = np.empty_like(u)
    lo = u < 0.02425
    hi = u > 1.0 - 0.02425
    mid = ~(lo | hi)
    if np.any(lo):
        q = np.sqrt(-2.0 * np.log(u[lo]))
        out[lo] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if np.any(hi):
        q = np.sqrt(-2.0 * np.log(1.0 - u[hi]))
        out[hi] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if np.any(mid):
        q = u[mid] - 0.5
        r = q * q
        out[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q \
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return out


# ---------------------------------------------------------------- 1-D W2
def w2_to_gaussian_1d(p):
    """W2^2 of a 1-D sample against its moment-matched Gaussian.

    For the empirical measure (equal mass 1/n), W2^2 = ∫_0^1 (F_n^{-1}(u) - F_g^{-1}(u))^2 du,
    discretized with u_j = (j+1/2)/n, i.e. the mean squared error of the quantiles.

    Returns (w2_squared, sample_variance).
    """
    p = np.sort(np.asarray(p, dtype=np.float64).ravel())
    n = p.size
    if n < 3:
        return 0.0, 0.0
    u = (np.arange(n) + 0.5) / n
    mu = float(p.mean())
    var = float(p.var())
    if not np.isfinite(var) or var <= 1e-300:
        return 0.0, 0.0
    gq = mu + np.sqrt(var) * _norm_ppf(u)
    return float(((p - gq) ** 2).mean()), var


def _baseline(n, seed=91711, n_rep=256):
    """W2^2 of a unit-variance Gaussian under the same pipeline (finite-sample
    positive bias), cached by n.

    n_rep must be large enough: most of this statistic's variance comes from the
    **extreme order statistics** (at u = 1/(2n) the denominator phi(z) is tiny), and
    the distribution is clearly right-skewed; too few repetitions and the Monte-Carlo
    error biases the correction term itself. Measured: at n_rep=24 the baseline differs
    from the independently repeated empirical mean by 34%; it converges at 256.
    """
    key = (int(n), int(n_rep))
    if key not in _BASELINE_CACHE:
        rng = np.random.default_rng(seed)
        tot = 0.0
        for _ in range(n_rep):
            tot += w2_to_gaussian_1d(rng.normal(size=n))[0]
        _BASELINE_CACHE[key] = tot / n_rep
    return _BASELINE_CACHE[key]


# ---------------------------------------------------------------- core statistic
def gaussian_deficit(Z, n_proj=64, seed=0, correct_bias=True, proj_chunk=32):
    """Sliced W2 deficit of a single distribution p.

    Z : (n, d) samples drawn from p.

    Returns (lam, lam_uncorrected, mean_proj_var)
      lam            bias-corrected sliced W2 deficit ∈ [0, ~1]
      lam_uncorrect  uncorrected value (reports how large the correction is)
      mean_proj_var  average variance along random directions (= estimate of tr Sigma / d)

    **Projections must be chunked**: laying out all projection directions at once into
    an (n, n_proj) matrix costs 409 MB at n = 2e5, n_proj = 256, which would blow up
    the process. After chunking, peak memory drops to O(n * proj_chunk).
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n, d = Z.shape
    if n < 8:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(n_proj, d))
    G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-300
    raw = np.empty(n_proj)
    var = np.empty(n_proj)
    for s in range(0, n_proj, proj_chunk):
        e = min(s + proj_chunk, n_proj)
        P = Z @ G[s:e].T                                # (n, chunk)
        for j in range(e - s):
            raw[s + j], var[s + j] = w2_to_gaussian_1d(P[:, j])
        del P
    vbar = float(var.mean())
    if not np.isfinite(vbar) or vbar <= 1e-300:
        return 0.0, 0.0, 0.0
    raw_ratio = float(raw.mean()) / vbar
    lam = raw_ratio
    if correct_bias:
        lam = raw_ratio - _baseline(n)
    return float(np.clip(lam, 0.0, None)), float(raw_ratio), vbar


def conditional_gaussian_deficit(C, X, n_anchor=48, n_nb=150, n_proj=64,
                                 seed=0, pca_dim=32, correct_bias=True,
                                 min_size=40):
    """Conditional version of Lambda: sliced W2 deficit within anchor neighborhoods
    of the condition space.

    Uses the same anchor-neighborhood protocol as the old
    ``conditional_mode_separation`` for easy comparison.

    Returns dict:
      lam           mean of per-anchor Lambda
      lam_std       std across anchors (captures cross-condition heterogeneity)
      lam_raw       uncorrected mean
      per_anchor    per-anchor values
      n_anchor, n_nb
    """
    C = np.asarray(C, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = len(C)
    n_nb = int(min(max(min_size, n_nb), max(min_size, n // 4)))
    Cs = (C - C.mean(axis=0)) / (C.std(axis=0) + 1e-8)   # standardize condition scale
    Xr = X
    if X.shape[1] > pca_dim:                              # reduce dimension first if high-dim
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        Xr = Xc @ U[:, :pca_dim].T
    rng = np.random.default_rng(seed)
    anchor = rng.choice(n, size=min(n_anchor, n), replace=False)
    vals, raws = [], []
    for a in anchor:
        d2 = ((Cs[a][None, :] - Cs) ** 2).sum(-1)
        nb = np.argpartition(d2, n_nb - 1)[:n_nb]
        lam, raw, _ = gaussian_deficit(Xr[nb], n_proj=n_proj,
                                       seed=seed + int(a),
                                       correct_bias=correct_bias)
        vals.append(lam)
        raws.append(raw)
    vals = np.asarray(vals)
    return dict(lam=float(vals.mean()), lam_std=float(vals.std()),
                lam_raw=float(np.mean(raws)), per_anchor=vals,
                n_anchor=int(vals.size), n_nb=n_nb)
