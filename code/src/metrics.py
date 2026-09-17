# -*- coding: utf-8 -*-
"""Evaluation metrics: all public, independently implementable measures (no
dependency on any pretrained network).

  sliced_wasserstein   sliced Wasserstein-2 (a special case of the generalized sliced-W
                      distance of Kolouri et al. 2019)
  energy_distance      (equivalent to the first-order energy distance, as a robustness check)
  mode_recall/precision  mode coverage / collapse rate when ground-truth modes are known (toy tasks)
  prdc                improved precision/recall of Kynkaanniemi et al., NeurIPS 2019 (image tasks)
"""
import numpy as np


def conditional_reference_samples(C_ref, X_ref, C_query, n_ref=256, seed=0,
                                  exclude_indices=None):
    """Construct an empirical conditional reference set for each query condition using
    only the observable condition.

    This function deliberately does not receive class labels. The old image-experiment
    implementation picked reference images by the true digit label, even when the
    condition contained no label at all under ``mode='none'``, thus leaking unavailable
    information into the metric.

    Continuous conditions use nearest neighbors in the standardized condition space;
    when the condition is completely constant, samples are drawn independently from the
    whole reference set, corresponding to the empirical marginal. ``exclude_indices[i]``
    can exclude the query sample itself.
    """
    C_ref = np.asarray(C_ref, dtype=np.float64)
    C_query = np.asarray(C_query, dtype=np.float64)
    X_ref = np.asarray(X_ref)
    if C_ref.ndim == 1:
        C_ref = C_ref[:, None]
    if C_query.ndim == 1:
        C_query = C_query[:, None]
    if len(C_ref) != len(X_ref):
        raise ValueError("C_ref and X_ref must have the same number of samples")
    if n_ref < 1:
        raise ValueError("n_ref must be a positive integer")
    n_take = min(int(n_ref), len(C_ref) - (1 if exclude_indices is not None else 0))
    if n_take < 1:
        raise ValueError("insufficient reference-set samples")

    scale = C_ref.std(axis=0)
    active = scale > 1e-12
    center = C_ref.mean(axis=0)
    z_ref = ((C_ref[:, active] - center[active]) / scale[active]
             if np.any(active) else None)
    rng = np.random.default_rng(seed)
    refs = []
    for i, cq in enumerate(C_query):
        banned = None if exclude_indices is None else int(exclude_indices[i])
        if not np.any(active):
            pool = np.arange(len(C_ref))
            if banned is not None:
                pool = pool[pool != banned]
            idx = rng.choice(pool, size=n_take, replace=False)
        else:
            z_q = (cq[active] - center[active]) / scale[active]
            d2 = ((z_ref - z_q) ** 2).sum(axis=1)
            if banned is not None:
                d2[banned] = np.inf
            # The random perturbation only breaks ties among identical conditions; it
            # does not change the ordering of non-tied distances.
            d2 = d2 + rng.uniform(0.0, 1e-12, size=len(d2))
            idx = np.argpartition(d2, n_take - 1)[:n_take]
        refs.append(X_ref[idx])
    return refs


def sliced_wasserstein(A, B, n_proj=1024, seed=0, p=2):
    """Sliced W_2 distance between sample sets (Monte-Carlo over projection directions)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    d = A.shape[1]
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n_proj, d))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    pa = np.sort(A @ W.T, axis=0)
    pb = np.sort(B @ W.T, axis=0)
    n = min(len(A), len(B))
    if len(A) != len(B):
        qa = np.quantile(A @ W.T, np.linspace(0, 1, n), axis=0)
        qb = np.quantile(B @ W.T, np.linspace(0, 1, n), axis=0)
        pa, pb = qa, qb
    val = np.mean(np.abs(pa - pb) ** p) ** (1.0 / p)
    return float(val)


def energy_distance(A, B, seed=0, max_n=2000):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(A) > max_n:
        A = A[rng.choice(len(A), max_n, replace=False)]
    if len(B) > max_n:
        B = B[rng.choice(len(B), max_n, replace=False)]

    def _cross(X, Y):
        return np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1) + 1e-12).mean()

    return float(2 * _cross(A, B) - _cross(A, A) - _cross(B, B))


def mode_stats(samples, centers, tol=None):
    """Mode statistics when ground-truth modes are known.

    Key protocol: a sample counts as "hitting a mode" only if it falls within that
    mode's tolerance ball; otherwise it is recorded as off-mode. Without this step, a
    blob collapsed onto the ring center would be evenly split across modes by "nearest
    center", falsely reporting recall=1.

    Returns dict:
      recall   : share of modes whose empirical mass ≥ half the nominal mass (mode recall)
      drop     : share of modes whose empirical mass < 10% of nominal mass (mode collapse)
      off_frac : share of samples outside every mode's tolerance ball (direct evidence of
                 one-step collapse)
      min_ratio: worst mode's empirical mass / nominal mass
      entropy_ratio : empirical mode-distribution entropy / uniform entropy
    """
    samples = np.asarray(samples, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    K = len(centers)
    d2 = ((samples[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    lab = d2.argmin(axis=1)
    dmin = d2.min(axis=1)
    if tol is None:
        # default tolerance = half the distance between the two nearest mode centers
        cd = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        np.fill_diagonal(cd, np.inf)
        tol = float(cd.min() / 2.0) if K > 1 else np.inf
    off = dmin > tol ** 2
    hit = ~off
    prop = np.bincount(lab[hit], minlength=K) / max(len(samples), 1)
    nominal = 1.0 / K
    recall = float(np.mean(prop >= 0.5 * nominal))
    drop = float(np.mean(prop < 0.1 * nominal))
    min_ratio = float(prop.min() / nominal)
    ent = -(prop[prop > 0] * np.log(prop[prop > 0])).sum()
    return dict(recall=recall, drop=drop, min_ratio=min_ratio,
                off_frac=float(off.mean()),
                entropy_ratio=float(ent / np.log(K)) if K > 1 else 0.0,
                prop=prop.tolist())


def mode_stats_band(samples, centers, radius, sigma, band=3.0, min_share=0.01):
    """Mode statistics for the ring-mixture task (**σ-scale ball** protocol, replacing the
    tolerance ball).

    Why change the protocol (two pitfalls found in this project's measurements):
      1) The old tolerance was "half the nearest mode spacing", which shrinks with K. At
         K=16, sep_ratio=6, the spacing between adjacent modes is only ~2.34σ on the
         2R·sin(pi/16)*sigma scale, so half-spacing ≈1.17σ is **smaller than the mode's
         own radial scale**, and true samples were misclassified as off-mode (under the old
         protocol K=16 recall falsely dropped to 0.75~0.88).
      2) At K=1 there is no "nearest mode spacing", so the old protocol degenerated into
         "nearest center" assignment, **exactly masking** the most important failure mode —
         one-step sampling collapse (under the old protocol K=1 reported recall=1.00).

    New protocol: tolerance is a **σ-scale ball** independent of K, tol = sigma*(band + sqrt(d)):
      off_frac      : share of samples whose distance to the nearest mode center > tol
                      —— one-step collapse (falling inside the ring / between modes) is
                      recorded directly here
      recall_band   : after assigning in-ball samples to the nearest center, share of modes
                      whose empirical mass ≥ min_share
      drop_band     : 1 - recall_band
      entropy_ratio : normalized entropy of the in-ball empirical mode distribution
      radial_off    : share with | ||x|| - R | > tol (cross-check, immune to angular misalignment)
    """
    samples = np.asarray(samples, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    K = len(centers)
    d = samples.shape[1]
    tol = float(sigma * (band + np.sqrt(d)))
    d2 = ((samples[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    lab = d2.argmin(axis=1)
    dmin = np.sqrt(d2[np.arange(len(samples)), lab])
    on = dmin <= tol
    rad = np.linalg.norm(samples, axis=1)
    radial_off = float(np.mean(np.abs(rad - radius) > tol))
    off_frac = float(1.0 - on.mean())
    out = dict(tol=tol, off_frac=off_frac, radial_off=radial_off,
               on_band=bool(on.mean() >= min_share),
               band_frac=float(on.mean()),
               recall_band=0.0, drop_band=1.0, min_share_ratio=0.0,
               entropy_ratio=0.0)
    if not out["on_band"]:
        out["drop_band"] = 1.0
        return out
    prop = np.bincount(lab[on], minlength=K) / int(on.sum())   # in-ball empirical mass
    recall = float(np.mean(prop >= min_share))
    ent = -(prop[prop > 0] * np.log(prop[prop > 0])).sum()
    out.update(recall_band=recall, drop_band=1.0 - recall,
               min_share_ratio=float(prop.min() / (1.0 / K)),
               entropy_ratio=float(ent / np.log(K)) if K > 1 else 0.0,
               prop_band=prop.tolist())
    return out


def conditional_sw(true_samples, gen_samples, n_proj=512, seed=0):
    """Conditional SW: compute SW per context first, then average over contexts.

    true_samples: (n_ctx, m, d); gen_samples: (n_ctx, m, d)
    """
    true_samples = np.asarray(true_samples, dtype=np.float64)
    gen_samples = np.asarray(gen_samples, dtype=np.float64)
    vals = [sliced_wasserstein(true_samples[i], gen_samples[i],
                               n_proj=n_proj, seed=seed + i)
            for i in range(len(true_samples))]
    return float(np.mean(vals)), float(np.std(vals))


def _pairwise_dist(X, Y, chunk=512):
    """Chunked computation of ||x-y||^2 to avoid opening a full n×n matrix at once."""
    n, m = len(X), len(Y)
    out = np.empty((n, m), dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = ((X[s:e, None, :] - Y[None, :, :]) ** 2).sum(-1)
    return np.sqrt(np.maximum(out, 0.0))


def prdc(real, fake, k=5, seed=0, max_n=3000):
    """Precision / recall / density / coverage of Kynkaanniemi et al., NeurIPS 2019.

    precision: share of generated samples falling inside a real-sample k-neighborhood ball
               (generation quality)
    recall   : share of real samples covered by a generated-sample neighborhood ball
               (mode coverage; dropped modes → decrease)
    density  : average number of real balls covering each generated sample / k
    coverage : share of real balls containing at least one generated sample
    """
    real = np.asarray(real, dtype=np.float64)
    fake = np.asarray(fake, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(real) > max_n:
        real = real[rng.choice(len(real), max_n, replace=False)]
    if len(fake) > max_n:
        fake = fake[rng.choice(len(fake), max_n, replace=False)]
    nr, nf = len(real), len(fake)
    kk = min(k, nr - 1, nf - 1)

    # k-neighborhood radius of real / generated samples themselves (excluding self)
    Drr = _pairwise_dist(real, real)
    Dff = _pairwise_dist(fake, fake)
    r_real = np.sort(Drr, axis=1)[:, kk]         # kk-th nearest (column 0 is self)
    r_fake = np.sort(Dff, axis=1)[:, kk]

    Drf = _pairwise_dist(real, fake)             # (nr, nf)
    # precision: for each fake, find nearest real and check if it lies in its ball
    j = Drf.argmin(axis=0)                       # nearest real for each fake
    d_fake = Drf[j, np.arange(nf)]
    precision = float(np.mean(d_fake <= r_real[j]))
    # recall: for each real, find nearest fake
    i = Drf.argmin(axis=1)                       # nearest fake for each real
    d_real = Drf[np.arange(nr), i]
    recall = float(np.mean(d_real <= r_fake[i]))
    # density: how many real balls cover each fake / k
    cnt = (Drf <= r_real[:, None]).sum(axis=0)
    density = float(np.mean(cnt / kk))
    # coverage: share of real balls covering at least one fake
    coverage = float(np.mean((Drf <= r_real[:, None]).sum(axis=1) > 0))
    return dict(precision=precision, recall=recall, density=density,
                coverage=coverage)


def pca_features(X, n_comp=64, seed=0, fit_on=None):
    """PCA projection (numpy SVD implementation, no sklearn). When fit_on is given,
    fit the basis on it.

    Returns (low-dim representation of the base samples, low-dim representation of X).
    """
    X = np.asarray(X, dtype=np.float64)
    base = X if fit_on is None else np.asarray(fit_on, dtype=np.float64)
    mu = base.mean(0)
    U, S, Vt = np.linalg.svd(base - mu, full_matrices=False)
    W = Vt[:min(n_comp, Vt.shape[0])]
    return (base - mu) @ W.T, (X - mu) @ W.T
