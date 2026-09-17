# -*- coding: utf-8 -*-
"""Discriminants computable before training (paper §4). **Pure numpy, no sklearn**.

(Engineering note: under Windows + MKL, sklearn's KMeans deadlocks in multi-threaded
contexts, so the k-means / kNN / PCA here are all written from scratch, which also
makes the reproduction of the "old method" more transparent.)

Two quantities, both **using only the dataset (c_i, x_i), without training any
generative model**:

  D-hat  **share of target variance left unexplained by the condition** (estimate of
         the "one-step generation error floor")
         = residual variance of conditional-mean regression on a held-out set /
           total target variance.
         Estimator is swappable (kNN / RFF-ridge / linear ridge), default kNN
         (Cover & Hart, 1967), with a "nearest-neighbor pairing estimator" for
         **regressor-free** cross-validation.

  Lam-hat **modal separation of the unexplained (residual) part**
          k-means (Lloyd, 1982 / MacQueen, 1967) on the residual r = x - m_hat(c);
          number of clusters chosen by Calinski-Harabasz criterion (1974) over 1..Kmax;
          Lam = distance of the two nearest cluster centers / (2 × pooled within-cluster
          std); defined as 0 for a single cluster.

Their division of labor (paper Proposition 1 and §4):
  D-hat  determines how large the **error floor** of one step vs. many steps is;
  Lam-hat determines whether this floor shows up as **dropped modes (structured,
         catastrophic)** or **blurring (diffuse, benign)**.
"""
import numpy as np


def _as2(a):
    a = np.asarray(a, dtype=np.float64)
    return a if a.ndim > 1 else a.reshape(-1, 1)


# ------------------------------------------------------------ building blocks
def knn_predict(Ctr, Xtr, Cq, k=10, chunk=256, eps=1e-8):
    """k-nearest-neighbor regression (distance-weighted), chunked to avoid large
    matrices."""
    Ctr, Xtr, Cq = _as2(Ctr), _as2(Xtr), _as2(Cq)
    n, d = Xtr.shape
    k = min(k, max(1, n - 1))
    out = np.empty((len(Cq), d), dtype=np.float64)
    for s in range(0, len(Cq), chunk):
        q = Cq[s:s + chunk]
        d2 = ((q[:, None, :] - Ctr[None, :, :]) ** 2).sum(-1)
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        w = 1.0 / (np.sqrt(np.take_along_axis(d2, idx, axis=1)) + eps)
        xs = Xtr[idx]                                   # (chunk, k, d)
        out[s:s + chunk] = (xs * w[:, :, None]).sum(1) / w.sum(1)[:, None]
    return out


def kmeans_lloyd(X, K, n_init=5, iters=50, seed=0):
    """Lloyd iterations + k-means++ initialization. Returns (centers, labels, inertia)."""
    X = _as2(X)
    n = len(X)
    if K >= n:
        return X.copy(), np.arange(n), 0.0
    rng = np.random.default_rng(seed)
    best = None
    for init in range(n_init):
        # k-means++
        idx0 = rng.integers(0, n)
        cent = [X[idx0]]
        for _ in range(1, K):
            d2 = np.min(((X[:, None, :] - np.array(cent)[None, :, :]) ** 2).sum(-1), axis=1)
            prob = d2 / (d2.sum() + 1e-12)
            cent.append(X[rng.choice(n, p=prob)])
        C = np.array(cent)
        for _ in range(iters):
            d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            lab = d2.argmin(1)
            newC = C.copy()
            for j in range(K):
                m = lab == j
                if m.any():
                    newC[j] = X[m].mean(0)
                else:
                    newC[j] = X[rng.integers(0, n)]
            if np.allclose(newC, C, atol=1e-9):
                C = newC
                break
            C = newC
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d2.argmin(1)
        inertia = float(d2[np.arange(n), lab].sum())
        if best is None or inertia < best[2]:
            best = (C, lab, inertia)
    return best


def calinski_harabasz(X, lab):
    X = _as2(X)
    K = len(np.unique(lab))
    n = len(X)
    if K < 2 or K >= n:
        return 0.0
    gm = X.mean(0)
    cent = np.array([X[lab == j].mean(0) for j in range(K)])
    counts = np.array([(lab == j).sum() for j in range(K)], dtype=float)
    bg = (counts[:, None] * (cent - gm) ** 2).sum()
    wg = float(((X - cent[lab]) ** 2).sum())
    if wg < 1e-12:
        return 0.0
    return float((bg / (K - 1)) / (wg / (n - K)))


def pca_project(X, n_comp=16, fit_on=None):
    """Returns (low-dim representation, mean, projection matrix)."""
    X = _as2(X)
    base = X if fit_on is None else _as2(fit_on)
    mu = base.mean(0)
    U, S, Vt = np.linalg.svd(base - mu, full_matrices=False)
    W = Vt[:min(n_comp, Vt.shape[0])]
    return (X - mu) @ W.T, mu, W


def ridge_fit(Z, Y, alpha=1.0):
    Z = _as2(Z)
    Y = _as2(Y)
    p = Z.shape[1]
    A = Z.T @ Z + alpha * np.eye(p)
    return np.linalg.solve(A, Z.T @ Y)


def rff_features(C, n_comp=256, gamma=None, seed=0):
    C = _as2(C)
    d = C.shape[1]
    if gamma is None:
        gamma = 1.0 / max(d, 1)
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=np.sqrt(2 * gamma), size=(d, n_comp))
    b = rng.uniform(0, 2 * np.pi, size=n_comp)
    return np.sqrt(2.0 / n_comp) * np.cos(C @ W + b)


# ------------------------------------------------------------ discriminants
def fit_conditional_mean(Ctr, Xtr, estimator="knn", k=10, n_rff=256,
                         alpha=1.0, seed=0):
    """Fit E[x|c], return predict(C). Default kNN (Cover & Hart, 1967)."""
    Ctr, Xtr = _as2(Ctr), _as2(Xtr)
    mu_c, sd_c = Ctr.mean(0), Ctr.std(0) + 1e-8
    Ztr = (Ctr - mu_c) / sd_c

    def _prep(C):
        return (_as2(C) - mu_c) / sd_c

    if estimator == "knn":
        return lambda C: knn_predict(Ztr, Xtr, _prep(C), k=k)
    if estimator == "linear":
        Wb = ridge_fit(np.hstack([Ztr, np.ones((len(Ztr), 1))]), Xtr, alpha)

        def _f(C):
            Z = np.hstack([_prep(C), np.ones((len(_as2(C)), 1))])
            return Z @ Wb
        return _f
    if estimator == "rff":
        Ftr = rff_features(Ztr, n_comp=n_rff, seed=seed)
        Wb = ridge_fit(np.hstack([Ftr, np.ones((len(Ftr), 1))]), Xtr, alpha)

        def _f(C):
            F = rff_features(_prep(C), n_comp=n_rff, seed=seed)
            return np.hstack([F, np.ones((len(F), 1))]) @ Wb
        return _f
    if estimator == "mean":
        mu_x = Xtr.mean(0)
        return lambda C: np.tile(mu_x, (len(_as2(C)), 1))
    raise ValueError(estimator)


def d_hat_from_predictor(pred, Cte, Xte, var_total=None):
    Cte, Xte = _as2(Cte), _as2(Xte)
    if var_total is None:
        var_total = Xte.var(axis=0).sum() + 1e-12
    res = Xte - pred(Cte)
    return float((res ** 2).sum(axis=1).mean() / var_total)


def d_hat_pair(C, X, k=1, chunk=512):
    """Regressor-free cross-validation: nearest-neighbor pairing estimator.

    E||x_i - x_nn(i)||^2 ≈ 2 tr Var(x|c) (bias → 0 as neighborhoods shrink),
    so D_pair = mean(d^2) / (2 tr Var(x)).
    """
    C, X = _as2(C), _as2(X)
    # The condition's dimensions may have completely different scales (e.g. image
    # pixels vs. one-hot labels). Must standardize before nearest-neighbor search;
    # otherwise the dimension with the largest numerical range dominates the distance.
    C = (C - C.mean(axis=0)) / (C.std(axis=0) + 1e-8)
    n = len(C)
    tot = np.empty(n)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d2 = ((C[s:e, None, :] - C[None, :, :]) ** 2).sum(-1)
        for i in range(e - s):
            d2[i, s + i] = np.inf          # exclude self
        nb = d2.argmin(1)
        tot[s:e] = ((X[s:e] - X[nb]) ** 2).sum(-1)
    var_total = X.var(axis=0).sum() + 1e-12
    return float(np.clip(tot.mean() / (2 * var_total), 0.0, None))


def _lam_of(Rr, K, seed):
    """Given cluster count K, compute separation Lam = nearest-center distance /
    (2 × pooled within-cluster std)."""
    if K < 2:
        return 0.0, None
    C, lab, _ = kmeans_lloyd(Rr, K, n_init=3, iters=30, seed=seed)
    if len(np.unique(lab)) < K:
        return 0.0, None
    dmin = np.inf
    for i in range(K):
        for j in range(i + 1, K):
            dmin = min(dmin, np.linalg.norm(C[i] - C[j]))
    # "Pooled within-cluster variance" must be weighted by sample count. Averaging
    # clusters with equal weight would give a tiny cluster the same weight as a large
    # one, systematically distorting Lambda when a small cluster is artificially cut.
    within = ((Rr - C[lab]) ** 2).sum(axis=1).mean()
    sbar = np.sqrt(max(within, 1e-12) / Rr.shape[1])
    return float(dmin / (2 * sbar)), (C, lab)


def _best_K_by_ch(Rr, kmax, seed, n_init=2, iters=30):
    """Return the best CH candidate with K>=2 and its score; return (1, 0) when no
    valid candidate exists."""
    kmax = max(1, min(kmax, len(Rr) // 20))
    best_score, best_K = 0.0, 1
    for K in range(2, kmax + 1):
        C, lab, _ = kmeans_lloyd(Rr, K, n_init=n_init, iters=iters, seed=seed + K)
        if len(np.unique(lab)) < K:
            continue
        s = calinski_harabasz(Rr, lab)
        if s > best_score:
            best_score, best_K = s, K
    return best_K, float(best_score)


def _select_K(Rr, kmax, seed, n_null=99, alpha=0.05):
    """Select between K=1 and K>=2 using a Gaussian null test on the "maximum CH
    score".

    Taking the max CH over K=2..Kmax inevitably returns multiple clusters, so the
    old implementation's apparent K=1 was in fact never selectable. This
    implementation includes "searching over several K" in the null too: each Gaussian
    bootstrap sample searches the full K range, and the observed max score is accepted
    as multimodal only when it exceeds the (1-alpha) quantile.
    """
    best_K, observed = _best_K_by_ch(Rr, kmax, seed)
    if best_K == 1 or n_null < 1:
        return best_K
    null_scores = []
    for b in range(n_null):
        rng = np.random.default_rng(seed + 7919 * (b + 1))
        Zn = _gaussian_null(Rr, len(Rr), rng)
        _, score = _best_K_by_ch(
            Zn, kmax, seed + 104729 * (b + 1), n_init=1, iters=20)
        null_scores.append(score)
    threshold = float(np.quantile(null_scores, 1.0 - alpha))
    return int(best_K if observed > threshold else 1)


def _gaussian_null(Rr, n, rng):
    """Gaussian sample matching the given sample's second moments (unimodal null)."""
    mu = Rr.mean(0)
    cov = np.cov(Rr.T) + 1e-10 * np.eye(Rr.shape[1])
    if Rr.shape[0] < 2:
        return Rr
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.maximum(np.diag(cov), 1e-10)))
    return rng.normal(size=(n, Rr.shape[1])) @ L.T + mu


def conditional_mode_separation(C, X, n_anchor=48, n_nb=150, kmax=8, seed=0,
                                pca_dim=16, n_null=99):
    """Modal separation of the **conditional distribution** (core protocol).

    Method: take A anchor contexts in the data, for each anchor take n_nb nearest
    neighbors in the condition space, cluster these n_nb **original target samples**
    (no conditional-mean subtraction needed, so no regressor noise is introduced),
    choose the cluster count by Calinski-Harabasz, and compute
        Lam_a = distance of the two nearest cluster centers / (2 × pooled within-cluster std)
    with a null baseline from a parametric bootstrap against the **same-covariance
    unimodal Gaussian null**.

    Returns (Lam_hat, Lam_null, Lam_p95, frac_multi, K_mean)
    """
    C, X = _as2(C), _as2(X)
    n = len(C)
    n_nb = min(n_nb, max(30, n // 10))
    rng = np.random.default_rng(seed)
    anchor = rng.choice(n, size=min(n_anchor, n), replace=False)
    Xr = X
    if X.shape[1] > pca_dim:
        Xr, _, _ = pca_project(X, n_comp=pca_dim)
    lams, nulls, ks, multi = [], [], [], []
    for a in anchor:
        d2 = ((C[a][None, :] - C) ** 2).sum(-1)
        nb = np.argpartition(d2, n_nb - 1)[:n_nb]
        Z = Xr[nb]
        # K=1 must come from a null test that includes the model-selection step,
        # not from merely comparing K>=2 CH.
        K = _select_K(Z, kmax, seed + int(a), n_null=n_null)
        if K < 2:
            lams.append(0.0)
            nulls.append(0.0)
            ks.append(1)
            multi.append(0.0)
            continue
        lam, _ = _lam_of(Z, K, seed + int(a))
        nl = []
        for b in range(n_null):
            Zn = _gaussian_null(Z, len(Z), np.random.default_rng(seed + 7919 * (int(a) + 1) + b))
            # The significance of choosing K is already corrected by _select_K via the
            # "max CH" null; here we calibrate Lambda's scale at the accepted K to avoid
            # nesting yet another extremely expensive bootstrap test.
            ln, _ = _lam_of(Zn, K, seed + 130363 * (int(a) + 1) + b)
            nl.append(ln)
        nl = np.array([v for v in nl if v is not None])
        lams.append(lam)
        nulls.append(float(nl.mean()) if len(nl) else 0.0)
        ks.append(K)
        multi.append(1.0)  # K>=2 already passed the 5% null test that includes multi-K search
    return (float(np.mean(lams)), float(np.mean(nulls)),
            float(np.mean(multi)), float(np.mean(ks)),
            np.array(lams), np.array(nulls))


def residual_mode_separation(R, kmax=10, seed=0, min_size=20, pca_dim=16,
                             n_null=20):
    """Multimodality and separation of the residual.

    Returns (Lam, K_star, Lam_null_mean, Lam_null_p95, multimodal):
      K_star chosen by Calinski-Harabasz over 1..kmax (Lam=0 for K=1);
      Lam_null_* is the distribution of the same statistic under the **parametric
      bootstrap null** (isotropic Gaussian matching the residual's covariance), used
      to judge "whether this separation exceeds what unimodal noise can produce".
    """
    R = _as2(R)
    n, d = R.shape
    if n < 4 * min_size:
        return 0.0, 1, 0.0, 0.0, False
    Rr = R
    if d > pca_dim:                      # project to principal subspace first if high-dim
        Rr, _, _ = pca_project(R, n_comp=pca_dim)
    kmax = max(1, min(kmax, n // min_size))
    best_score, best_K = 0.0, 1
    for K in range(2, kmax + 1):
        C, lab, _ = kmeans_lloyd(Rr, K, n_init=3, iters=30, seed=seed + K)
        if len(np.unique(lab)) < K:
            continue
        s = calinski_harabasz(Rr, lab)
        if s > best_score:
            best_score, best_K = s, K
    if best_K == 1:
        return 0.0, 1, 0.0, 0.0, False
    Lam, _ = _lam_of(Rr, best_K, seed)
    # Null: unimodal Gaussian matching the residual's second moments
    rng = np.random.default_rng(seed + 12345)
    mu = Rr.mean(0)
    cov = np.cov(Rr.T) + 1e-10 * np.eye(Rr.shape[1])
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.diag(cov)))
    nulls = []
    for b in range(n_null):
        Z = rng.normal(size=(len(Rr), Rr.shape[1])) @ L.T + mu
        lb, _ = _lam_of(Z, best_K, seed + 777 + b)
        nulls.append(lb)
    nulls = np.array(nulls)
    p95 = float(np.percentile(nulls, 95))
    return float(Lam), int(best_K), float(nulls.mean()), p95, bool(Lam > p95)


def marginal_separation(X, kmax=10, seed=0, min_size=20, pca_dim=16):
    """Control quantity: between-cluster variance share of the target **marginal**
    (condition ignored entirely)."""
    X = _as2(X)
    n, d = X.shape
    Xr = X
    if d > pca_dim:
        Xr, _, _ = pca_project(X, n_comp=pca_dim)
    kmax = max(1, min(kmax, n // min_size))
    best_score, best_K, best = 0.0, 1, None
    for K in range(2, kmax + 1):
        C, lab, _ = kmeans_lloyd(Xr, K, n_init=3, iters=30, seed=seed + K)
        if len(np.unique(lab)) < K:
            continue
        s = calinski_harabasz(Xr, lab)
        if s > best_score:
            best_score, best_K, best = s, K, (C, lab)
    if best is None:
        return 0.0, 1
    C, lab = best
    within = ((Xr - C[lab]) ** 2).sum(axis=1).mean()
    # within already sums over all coordinates, so total should also sum the per-dim
    # variance exactly once. The old code additionally multiplied by the dimension,
    # artificially inflating separation for high-dim data and breaking the control.
    total = Xr.var(axis=0).sum() + 1e-12
    return float(np.clip(1.0 - within / total, 0.0, 1.0)), int(best_K)


def compute_discriminant(Ctr, Xtr, Cte, Xte, estimator="knn", k=10, kmax=10,
                         seed=0, n_rff=256, alpha=1.0, with_competitors=True,
                         k_list=(10, 25, 50)):
    """Discriminant + a set of competitor predictors (for the rank-correlation
    comparison in paper Table 2).

    Estimator protocol (important):
      The essence of D is "residual variance share of the **optimal** conditional-mean
      regression", which is Var(x|c)/Var(x) ∈ [0,1]. But any **finite-sample** regressor
      mixes in its own estimation variance Var(m_hat), so
      D_hat ≈ D + Var(m_hat)/Var(x). The most direct case is kNN: the residual contains
      ≈ 1/k_eff of estimation noise, **pushing D_hat above the theoretical upper bound 1**.
      Measured: with K=1 (condition carries no information, the closed-form D is exactly
      1.000), kNN(k=10) gives 1.143 — a +14% positive bias.

      Fixed-k kNN, fixed-feature-count RFF, and linear models cannot all be loosely
      called "consistent estimators". Here we use an internal validation split on the
      training set to pick one candidate, then report D_hat on an independent test set.
      This way the test set bears only a single final evaluation, avoiding the optimistic
      selection bias of the old implementation which "took the minimum of five risks on
      the same test set". Each candidate's test risk is still retained for diagnostics.
    """
    Ctr, Xtr, Cte, Xte = _as2(Ctr), _as2(Xtr), _as2(Cte), _as2(Xte)
    var_total = float(Xte.var(axis=0).sum() + 1e-12)

    specs = [("knn%d" % kk, "knn", {"k": kk}) for kk in k_list]
    specs += [("linear", "linear", {"alpha": alpha}),
              ("rff", "rff", {"n_rff": n_rff, "alpha": alpha})]
    rng = np.random.default_rng(seed + 1709)
    order = rng.permutation(len(Ctr))
    n_val = min(max(200, len(Ctr) // 5), max(1, len(Ctr) // 3))
    val_idx, fit_idx = order[:n_val], order[n_val:]
    if len(fit_idx) < 2:
        fit_idx, val_idx = order, order
    val_total = float(Xtr[val_idx].var(axis=0).sum() + 1e-12)
    selection_scores = {}
    for name, est, kw in specs:
        pred = fit_conditional_mean(Ctr[fit_idx], Xtr[fit_idx], estimator=est,
                                    seed=seed, **kw)
        selection_scores[name] = d_hat_from_predictor(
            pred, Ctr[val_idx], Xtr[val_idx], val_total)
    selected = min(selection_scores, key=selection_scores.get)

    ens = {}
    for name, est, kw in specs:
        pred = fit_conditional_mean(Ctr, Xtr, estimator=est, seed=seed, **kw)
        ens[name] = d_hat_from_predictor(pred, Cte, Xte, var_total)
    mean_pred = fit_conditional_mean(Ctr, Xtr, estimator="mean", seed=seed)
    ens["mean_null"] = d_hat_from_predictor(mean_pred, Cte, Xte, var_total)

    members = {name: value for name, value in ens.items() if name != "mean_null"}
    D_ens = float(members[selected])
    D_hat_knn = float(ens.get("knn%d" % k, ens["knn10"]))

    Lam, lam_null, frac_multi, k_mean, _, _ = conditional_mode_separation(
        Ctr, Xtr, n_anchor=48, n_nb=150, kmax=kmax, seed=seed)
    out = dict(D_hat=D_ens, D_hat_knn=D_hat_knn, D_ens=ens,
               Lam_hat=Lam, Lam_null=lam_null,
               Lam_excess=float(Lam - lam_null),
               Lam_ratio=float(Lam / max(lam_null, 1e-9)),
               K_star=k_mean, frac_multi=frac_multi,
               D_hat_pair=d_hat_pair(Cte, Xte),
               var_total=var_total, dim=int(Xte.shape[1]),
               D_selection_scores=selection_scores,
               D_hat_estimator="validation-selected:%s" % selected)
    if with_competitors:
        out["D_lin"] = ens["linear"]
        out["D_rff"] = ens["rff"]
        sep_m, km = marginal_separation(Xte, kmax=kmax, seed=seed)
        out["sep_marg"] = sep_m
        out["K_marg"] = km
    return out
